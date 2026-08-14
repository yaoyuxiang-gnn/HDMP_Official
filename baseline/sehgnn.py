"""SeHGNN (AAAI 2023) 与共享稀疏工具。"""
from .common import *


def _row_normalize(ei, num_src, num_dst):
    """对二部边索引做行归一化（目标端 degree^-1），返回 (edge_index, edge_weight)。
    在 ei 的 device 上创建 tensor，避免 cuda/cpu 混用。"""
    deg = torch.zeros(num_dst, device=ei.device).index_add_(0, ei[1], torch.ones(ei.size(1), device=ei.device))
    deg = deg.clamp(min=1)
    w = 1.0 / deg[ei[1]]
    return ei, w


def _spmm(ei, w, x, num_dst):
    """稀疏乘：out[i] = sum_j w_ij * x[j]，ei=[2,E]（src=j, dst=i）。"""
    out = torch.zeros(num_dst, x.size(1), device=x.device, dtype=x.dtype)
    msg = x[ei[0]] * w.unsqueeze(-1)
    out.index_add_(0, ei[1], msg)
    return out


def build_target_metapaths(g, tgt, max_len=2):
    """枚举以 tgt 结尾的 ≤max_len 跳关系序列（基于 schema），返回 rel_seq 列表。
    每个 rel_seq 是 edge_type 元组序列（每跳一个 (s,r,d)）。仅保留落在 tgt 的路径。"""
    paths = []
    # 1 跳：(X -> tgt)
    for et in g.edge_types:
        s, r, d = et
        if d == tgt:
            paths.append((et,))
    # 2 跳：(X -> Y -> tgt)
    one_hop_src = set(et[0] for et in g.edge_types if et[2] == tgt)
    for et1 in g.edge_types:
        s1, r1, d1 = et1
        for et2 in g.edge_types:
            s2, r2, d2 = et2
            if d2 == tgt and s2 == d1 and s1 != tgt:  # 避免目标类型自身成环
                paths.append((et1, et2))
    # 去重（保序）
    seen, uniq = set(), []
    for p in paths:
        if p not in seen:
            seen.add(p); uniq.append(p)
    return uniq


def metapath_agg_feat(g, tgt, rel_seq):
    """对一条元路径（关系序列），从 tgt 反向逐跳 mean 聚合，得到 tgt 节点的聚合特征。
    实现：从路径起点类型的特征出发，沿 rel_seq 正向传播到 tgt。"""
    # 起点类型
    start_type = rel_seq[0][0]
    x = g[start_type].x
    cur = x
    for et in rel_seq:
        s, r, d = et
        ei = g[et].edge_index  # [2, E]: src=s, dst=d
        ei_n, w = _row_normalize(ei, g[s].num_nodes, g[d].num_nodes)
        cur = _spmm(ei_n, w, cur, g[d].num_nodes)
    return cur  # [N_tgt, feat_dim_of_start]


class SeHGNNNet(nn.Module):
    """SeHGNN：对每条元路径预计算 mean 聚合特征（无参数、仅一次），
    训练时只做特征投影 + Transformer 语义融合 + 分类。单层结构、长元路径扩感受野。"""
    def __init__(self, g, tgt, dh, nc, max_len=2):
        super().__init__()
        self.tgt = tgt
        self.rel_seqs = build_target_metapaths(g, tgt, max_len=max_len)
        # 每条元路径一个特征投影（把聚合特征映到 dh）
        self.projs = nn.ModuleList()
        for rs in self.rel_seqs:
            din = g[rs[0][0]].x.size(1)
            self.projs.append(nn.Linear(din, dh))
        # Transformer 融合：把各元路径的 dh 特征当 token 序列
        enc_layer = nn.TransformerEncoderLayer(d_model=dh, nhead=4, dim_feedforward=dh*2,
                                               dropout=0.5, batch_first=True, activation="gelu")
        self.fusion = nn.TransformerEncoder(enc_layer, num_layers=1)
        self.out = nn.Linear(dh, nc)

    def precompute(self, g):
        """预计算每条元路径的 mean 聚合特征（在 CPU/GPU 一次性完成）。"""
        self.mp_feats = [metapath_agg_feat(g, self.tgt, rs) for rs in self.rel_seqs]

    def forward(self, batch_idx=None):
        # 每条元路径：聚合特征 -> 投影 [B, P, dh]（batch_idx 分块防大图 OOM）
        dev = self.out.weight.device
        if batch_idx is not None:
            # mp_feats 可能在 CPU（大图）：用 CPU 索引取出，再移到 device 投影
            idx_cpu = batch_idx.cpu()
            feats = [f[idx_cpu].to(dev) for f in self.mp_feats]
        else:
            feats = [f.to(dev) for f in self.mp_feats]
        toks = torch.stack([self.projs[i](f) for i, f in enumerate(feats)], dim=1)  # [B, P, dh]
        fused = self.fusion(toks)          # [B, P, dh]
        h = fused.mean(dim=1)              # [B, dh] 跨元路径均值
        return self.out(h)