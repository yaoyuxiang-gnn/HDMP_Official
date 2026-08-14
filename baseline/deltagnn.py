"""DeltaGNN 基线（arXiv:2501.06002, Graph Neural Network with Information Flow Control）。
"""
from .common import *


def _mean_agg(feat, ei, n):
    """邻居均值聚合（edge_index + index_add_ 实现，规避 MPS 的 torch.sparse.mm）。
    约定 edge_index[0]=源、edge_index[1]=目标，目标节点聚合源节点（不加自环）；
    空图或无邻居节点输出 0 向量（与论文"孤立节点 delta 由其自身定义"一致）。
    """
    out = torch.zeros_like(feat)
    if ei.size(1) == 0:
        return out
    src, dst = ei[0], ei[1]
    out.index_add_(0, dst, feat[src])
    deg = torch.zeros(n, device=feat.device, dtype=feat.dtype)
    deg.index_add_(0, dst, torch.ones(ei.size(1), device=feat.device, dtype=feat.dtype))
    return out / deg.clamp(min=1).unsqueeze(1)


def _filter_edges(ei, score, k):
    """滤边算子 Θ：按边两端点 IFS 的较小值升序，删除分数最低的 k 条边（自环不删）。
    k<=0 或空图时为 no-op，直接返回原 edge_index。
    """
    e = ei.size(1)
    if e == 0 or k <= 0:
        return ei
    s_e = torch.minimum(score[ei[0]], score[ei[1]])          # 边分数 = 两端点 IFS 较小值
    s_e = s_e.masked_fill(ei[0] == ei[1], float("inf"))      # 自环置 +inf 保证保留
    drop = torch.argsort(s_e)[: min(k, e)]                   # 升序取最低 k 条删除
    keep = torch.ones(e, dtype=torch.bool, device=ei.device)
    keep[drop] = False
    return ei[:, keep]


class DeltaGNNNet(nn.Module):
    """DeltaGNN：带信息流控制（IFC）的 GCN 同质分支 + 异质凝聚分支 + JK 读出。
    """
    def __init__(self, din, dh, nc, T=3, theta0=0.0, theta_max=0.3, eta=5e-3,
                 top_c=500, L_he=2, dropout=0.5):
        super().__init__()
        self.T, self.L_he = T, L_he
        self.theta_max, self.eta, self.top_c = float(theta_max), float(eta), int(top_c)
        self.dropout = dropout
        self.dh = dh
        # ---- 同质分支：T 个 GCN 层（首层 din->dh，其余 dh->dh）----
        dims = [din] + [dh] * T
        self.convs = nn.ModuleList(GCNConv(dims[i], dims[i + 1]) for i in range(T))
        # ---- 异质凝聚分支：L_he 层，自/邻变换参数分离（ψs 与 ψ）----
        # 首层不聚合（式 5），故邻变换 ψ 只为第 2..L_he 层创建（L_he-1 个）
        he_dims = [din] + [dh] * L_he
        self.psi_s = nn.ModuleList(nn.Linear(he_dims[i], he_dims[i + 1]) for i in range(L_he))
        self.psi_n = nn.ModuleList(nn.Linear(he_dims[i], he_dims[i + 1]) for i in range(1, L_he))
        # ---- 读出：同质 JK 拼接 -> Linear；异质各层拼接 -> Linear；合并 -> 分类 ----
        self.lin_ho = nn.Linear(T * dh, dh)
        self.lin_he = nn.Linear(L_he * dh, dh)
        self.lin_out = nn.Linear(2 * dh, nc)
        # ---- IFC 状态（register_buffer 持久化、随 .to(device) 迁移、不反传）----
        self.register_buffer("theta", torch.tensor(float(theta0)))       # 当前滤边比例 θ
        self.register_buffer("u_prev", torch.tensor(float("-inf")))      # 上一轮 utility（爬山比较用）
        # Welford 增量统计（沿层序 t=1..T 累计，每次 forward 重新初始化）：
        self.register_buffer("d_mean", torch.zeros(0))   # 每节点一阶 delta 的均值 Δ̄
        self.register_buffer("a_mean", torch.zeros(0))   # 每节点二阶 delta 的均值（Welford 内部量）
        self.register_buffer("a_var", torch.zeros(0))    # 每节点二阶 delta 的方差 V[a]
        # 诊断信息（普通属性，不入 state_dict）：最近一次 forward 各次滤边后的剩余边数
        self.dbg_edge_counts = []

    def _ifs(self, m, ei, n, t, d_prev):
        """第 t 层的信息流分数：由变换特征 M 计算 Δ、a，Welford 更新后得 S。

        全程 detach（统计量不参与反传）。返回 (S, Δᵗ)。
        """
        with torch.no_grad():
            agg = _mean_agg(m.detach(), ei, n)               # 当前过滤图上的邻居均值（无邻居为 0）
            d_t = (agg - m.detach()).norm(p=2, dim=1)        # 一阶 delta Δᵗ（速度）
            a_t = (d_t - d_prev).abs() if t > 1 else torch.zeros_like(d_t)  # 二阶 delta（加速度）
            # Welford 增量更新（样本数 n=t）：均值 Δ̄（式 2）
            self.d_mean += (d_t - self.d_mean) / t
            # 方差 V[a]（式 3，总体方差的增量形式，内部同时维护 ā）
            diff1 = a_t - self.a_mean
            self.a_mean += diff1 / t
            self.a_var += (diff1 * (a_t - self.a_mean) - self.a_var) / t
            self.a_var.clamp_(min=0)                         # 数值噪声防护
            s = (self.a_var + 1.0) / (self.d_mean + 1.0)     # IFS（式 4，l=m=1）
        return s, d_t

    def forward(self, x, ei):
        n = x.size(0)
        dev, dt = x.device, x.dtype
        # 每次 forward 重新初始化沿层序的 Welford 统计（buffer 持久化，仅重置数值）
        self.d_mean = torch.zeros(n, device=dev, dtype=dt)
        self.a_mean = torch.zeros(n, device=dev, dtype=dt)
        self.a_var = torch.zeros(n, device=dev, dtype=dt)
        self.dbg_edge_counts = []

        # ---------- 同质分支：T 个 GCN 层 + 层间 IFC 滤边 ----------
        h = x
        ei_cur = ei
        d_prev = None
        s = torch.ones(n, device=dev, dtype=dt)  # 兜底：T=0 时不会用到
        xs = []
        for t in range(1, self.T + 1):
            conv = self.convs[t - 1]
            m = conv.lin(h)                      # 该层线性变换输出（聚合前），用于 IFS
            s, d_prev = self._ifs(m, ei_cur, n, t, d_prev)
            h = conv(h, ei_cur)                  # GCN 聚合（GCNConv 内部加自环）
            h = F.relu(h)                        # ϕ = ReLU
            h = F.dropout(h, self.dropout, self.training)
            xs.append(h)
            if t < self.T:                       # 层间交错滤边：结果供下一层使用
                ei_cur = _filter_edges(ei_cur, s, int(self.theta.item() * ei_cur.size(1)))
                self.dbg_edge_counts.append(ei_cur.size(1))

        # ---------- θ 爬山更新（仅训练模式）：U = 末层 IFS 均值 ----------
        if self.training:
            u = s.mean()
            if u > self.u_prev:
                self.theta = torch.clamp(self.theta + self.eta, max=self.theta_max)
            self.u_prev = u

        # ---------- 异质凝聚分支：top-C 全连接有向图（无自环）上的异质聚合 ----------
        c = min(self.top_c, n)
        top = torch.topk(s.detach(), c).indices            # 末层 IFS 最高的 C 个节点
        h_he = torch.zeros(n, self.dh, device=dev, dtype=dt)  # 非 top 节点 h_he=0
        hc = x[top]                                        # X⁰ = 原始输入特征
        hs = []
        for l in range(self.L_he):
            self_t = self.psi_s[l](hc)                     # 自变换 ψs
            if l == 0:
                hn = F.relu(self_t)                        # 首层不聚合：X¹ = ϕ(ψs(X⁰))
            else:
                nei = self.psi_n[l - 1](hc)                # 邻变换 ψ（第 2 层起才有）
                # 全连接有向图（无自环）上的均值聚合 = (总和 − 自身)/(C−1)
                agg = (nei.sum(dim=0, keepdim=True) - nei) / max(c - 1, 1)
                hn = F.relu(agg + self_t)                  # Xᵗ⁺¹ = ϕ(mean_agg(ψ(Xᵗ)) + ψs(Xᵗ))
            hn = F.dropout(hn, self.dropout, self.training)
            hs.append(hn)
            hc = hn
        h_he[top] = self.lin_he(torch.cat(hs, dim=1))      # 各层输出拼接 -> Linear
        h_ho = self.lin_ho(torch.cat(xs, dim=1))           # [X¹;…;Xᵀ] -> h_ho
        return self.lin_out(torch.cat([h_ho, h_he], dim=1))
