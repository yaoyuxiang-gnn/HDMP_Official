"""GTN (NeurIPS 2019)。"""
from .common import *
from .sehgnn import _row_normalize, _spmm


def _edge_norm(ei, num_src, num_dst):
    """对称归一化 D^-1/2 A D^-1/2（含自环），返回 (edge_index, weight)。GTN 用对称归一化。"""
    device = ei.device
    loop = torch.arange(num_dst, device=device).repeat(2, 1)
    ei_full = torch.cat([ei, loop], dim=1) if num_src == num_dst else ei
    deg = torch.zeros(num_dst, device=device).index_add_(0, ei_full[1], torch.ones(ei_full.size(1), device=device))
    deg_inv_sqrt = deg.clamp(min=1).pow(-0.5)
    w = deg_inv_sqrt[ei_full[0]] * deg_inv_sqrt[ei_full[1]]
    return ei_full, w




class GTNNet(nn.Module):
    """GTN：学关系权重 → 软组合邻接（2层 GTLayer 得 metapath 邻接）→ GCN。
    大图友好：邻接用稀疏 (edge_index, weight) 表示，前向用 index_add 稀疏乘；
    训练时对目标节点 mini-batch 分块（由 run_one_seed 控制），避免全量稠密化。"""
    def __init__(self, g, tgt, nc, num_channels=2, dh=64, n_layers=2):
        super().__init__()
        self.tgt = tgt
        self.node_types = list(g.node_types)
        self.n_rel = len(g.edge_types)
        self.num_channels = num_channels
        self.n_layers = n_layers
        # 每层 GTLayer：num_channels 组关系权重（softmax 归一化）
        self.weights = nn.ParameterList([
            nn.Parameter(torch.randn(num_channels, self.n_rel) * 0.1) for _ in range(n_layers)
        ])
        # 输入投影（各类型统一到 dh）
        self.lin = nn.ModuleDict({nt: nn.Linear(g[nt].x.size(1), dh) for nt in self.node_types})
        self.out = nn.Linear(dh, nc)
        # 预存各关系的稀疏邻接（对称归一化 + 自环），运行期按 (s,r,d) 索引
        self.adjs = {}  # et -> (ei, w, num_src, num_dst)
        # faithfulness 探针挂钩：被静默的关系下标集合（None = 正常行为，不影响基线结果）
        self.rel_mask = None

    def build_adjs(self, g):
        for et in g.edge_types:
            s, r, d = et
            ei = g[et].edge_index
            # 同类型(s==d)用对称归一化含自环；异类型用目标端行归一化
            if s == d:
                ei_n, w = _edge_norm(ei, g[s].num_nodes, g[d].num_nodes)
            else:
                ei_n, w = _row_normalize(ei, g[s].num_nodes, g[d].num_nodes)
            self.adjs[et] = (ei_n, w, g[s].num_nodes, g[d].num_nodes)

    def _combine(self, layer_idx):
        """按学习到的关系权重，软组合各关系邻接 → 每 channel 一个组合邻接（稀疏）。"""
        w = torch.softmax(self.weights[layer_idx], dim=1)  # [num_channels, n_rel]
        combos = []
        for c in range(self.num_channels):
            # 收集各关系按权重的加权和（仅当两端类型一致才能矩阵相加）
            # GTN 实际是对同构邻接加权；异构图简化：对每个 (s==d) 的关系加权组合
            combos.append(w[c])
        return combos

    def forward(self, xdict, eidict, node_slices, batch_idx=None):
        """前向：类型投影 → 沿软组合关系传播 n_layers 次 → 目标类型分类。
        batch_idx: 目标节点的 mini-batch 下标（None 则全量）。大图用稀疏乘逐关系传播。"""
        h = {nt: self.lin[nt](xdict[nt]) for nt in self.node_types}
        for li in range(self.n_layers):
            w = torch.softmax(self.weights[li], dim=1)  # [C, n_rel]
            # 对每个类型，沿所有指向它的关系做加权聚合（channel 0 主通道）
            new_h = {}
            for nt in self.node_types:
                agg = h[nt].clone()  # 自身（残差）
                for ri, et in enumerate(self.adjs.keys()):
                    s, r, d = et
                    if d != nt:
                        continue
                    if self.rel_mask is not None and ri in self.rel_mask:
                        continue  # 探针：静默该关系通道
                    ei, ew, ns, nd = self.adjs[et]
                    src_h = h[s].to(ei.device)
                    msg = _spmm(ei, ew, src_h, nd)  # [N_nt, dh]
                    agg = agg + w[0, ri] * msg
                new_h[nt] = F.relu(agg)
            h = new_h
        logits = self.out(h[self.tgt])
        if batch_idx is not None:
            return logits[batch_idx]
        return logits