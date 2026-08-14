"""ie-HGCN (TKDE 2021)。"""
from .common import *
from .sehgnn import _row_normalize, _spmm

class IeHGCNBlock(nn.Module):
    """单个目标类型 Omega 的 ie-HGCN block：把各邻居类型 + 自身投影到公共空间，
    object-level 用行归一化 mean，type-level 用注意力（query 来自自身）融合。"""
    def __init__(self, g, omega, in_dims, dout, da=64):
        super().__init__()
        self.omega = omega
        self.neigh = [et for et in g.edge_types if et[2] == omega]  # Γ->Ω 关系
        self.w_self = nn.Linear(in_dims[omega], dout)
        self.w_rel = nn.ModuleDict({f"{et[0]}->{et[1]}": nn.Linear(in_dims[et[0]], dout) for et in self.neigh})
        self.wq = nn.Linear(dout, da)
        self.wk = nn.Linear(dout, da)
        self.wa = nn.Linear(da * 2, 1)
        # faithfulness 探针挂钩：capture_attn=True 时记录 type-level 注意力（候选均值）
        self.capture_attn = False
        self.last_attn = None  # [n_cand]，下标 0 为 self，其余与 self.neigh 同序

    def forward(self, g, h, rel_mask=None):
        # h: {type: [N_type, in_dim]}，输出 omega 类型的新表示 [N_omega, dout]
        # rel_mask: 探针用，按 "src->dst" 类型对静默指定关系通道（None = 正常行为）。
        # 注意：HGB 图中所有关系名均为 "to"，按关系名掩码会退化成全图掩码，
        # 必须以 (src_type, dst_type) 类型对为粒度（与 ie-HGCN 的 type-level 粒度一致）。
        z_self = self.w_self(h[self.omega])                       # [N_om, dout]
        cands = [z_self]
        for et in self.neigh:
            s, r, d = et
            if rel_mask is not None and f"{s}->{d}" in rel_mask:
                continue  # 探针：静默该关系通道
            key = f"{s}->{r}"
            proj = self.w_rel[key](h[s])                          # [N_s, dout]
            ei = g[et].edge_index.to(proj.device)                 # 边索引移到特征 device（GPU）
            ei_n, w = _row_normalize(ei, g[s].num_nodes, g[d].num_nodes)
            z_rel = _spmm(ei_n, w, proj, g[d].num_nodes)          # [N_om, dout]
            cands.append(z_rel)
        # type-level 注意力：query 来自 z_self
        q = self.wq(z_self)                                       # [N_om, da]
        scores = []
        for z in cands:
            k = self.wk(z)
            e = F.elu(self.wa(torch.cat([k, q], dim=-1)))         # [N_om, 1]
            scores.append(e)
        attn = torch.softmax(torch.cat(scores, dim=-1), dim=-1)   # [N_om, n_cand]
        if self.capture_attn:
            self.last_attn = attn.detach().mean(dim=0)            # [n_cand] 节点均值
        stacked = torch.stack(cands, dim=1)                       # [N_om, n_cand, dout]
        out = (attn.unsqueeze(-1) * stacked).sum(dim=1)           # [N_om, dout]
        return F.elu(out)




class IeHGCNNet(nn.Module):
    """ie-HGCN：每层对每个类型一个 block，堆叠 N 层，最后目标类型接分类头。
    隐式评估所有 ≤N 跳的元路径（Theorem 1），type-level 注意力提供可解释性。"""
    def __init__(self, g, tgt, nc, n_layers=3, dh=64, da=64):
        super().__init__()
        self.tgt = tgt
        self.node_types = list(g.node_types)
        in_dims = {nt: g[nt].x.size(1) for nt in self.node_types}
        # 输入投影：统一到 dh
        self.in_proj = nn.ModuleDict({nt: nn.Linear(in_dims[nt], dh) for nt in self.node_types})
        # 每层：每个类型一个 block（输入 dh 输出 dh）
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            dims = {nt: dh for nt in self.node_types}
            self.layers.append(nn.ModuleDict({nt: IeHGCNBlock(g, nt, dims, dh, da) for nt in self.node_types}))
        self.out = nn.Linear(dh, nc)
        self.drop = nn.Dropout(0.5)

    def forward(self, g, xdict, rel_mask=None):
        # rel_mask: 探针用，按关系名静默指定关系通道（None = 正常行为，不影响基线结果）
        h = {nt: self.drop(F.elu(self.in_proj[nt](xdict[nt]))) for nt in self.node_types}
        for layer in self.layers:
            # 每层所有类型都更新（深层元路径评估需要中间类型表示）
            new_h = {}
            for nt in self.node_types:
                new_h[nt] = layer[nt](g, h, rel_mask=rel_mask)
            h = {nt: self.drop(new_h[nt]) for nt in self.node_types}
        return self.out(h[self.tgt])


# ---------------- DiffMG (KDD 2021)：可微 meta graph 搜索 ----------------
