"""LMSPS (NeurIPS 2024) 基线移植版 —— HGB 的 ACM / DBLP 节点分类。
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn

# (数据集, 类型字母映射, 搜好的元路径, 目标类型字母)
LMSPS_CONFIGS = {
    "acm": {
        "rename": {'paper': 'P', 'author': 'A', 'subject': 'C', 'term': 'T'},
        "paths": ['PPP', 'PAPP', 'PCPA', 'PCPP', 'PPPC', 'PPPP', 'PAPAP', 'PAPPP',
                  'PCPAP', 'PCPPA', 'PPAPA', 'PPAPC', 'PPAPP', 'PAPAPA', 'PAPCPA',
                  'PAPPAP', 'PAPPCP', 'PAPPPP', 'PCPAPP', 'PCPCPP', 'PCPPAP',
                  'PCPPPP', 'PPAPAP', 'PPAPCP', 'PPAPPA', 'PPAPPP', 'PPCPAP',
                  'PPPAPA', 'PPPCPA', 'PPPPPP'],
        "target": 'P',
        "dropout": 0.55, "n_layers_2": 2,
    },
    "dblp": {
        "rename": {'author': 'A', 'paper': 'P', 'term': 'T', 'venue': 'V'},
        "paths": ['AP', 'APT', 'APVP', 'APAPA', 'APTPA', 'APTPT', 'APTPV', 'APVPA',
                  'APVPV', 'APAPAP', 'APAPTP', 'APAPVP', 'APTPTP', 'APTPVP',
                  'APVPTP', 'APAPAPV', 'APAPTPA', 'APAPTPV', 'APAPVPA', 'APAPVPT',
                  'APAPVPV', 'APTPAPA', 'APTPAPT', 'APTPTPT', 'APTPVPV', 'APVPAPT',
                  'APVPTPA', 'APVPVPA', 'APVPVPT', 'APVPVPV'],
        "target": 'A',
        "dropout": 0.5, "n_layers_2": 2, "residual": True,
    },
}


def build_norm_adjs(g, rename):
    """每个有向关系的行归一化(目标端 deg^-1) CSR，约定 out = A @ x_src。"""
    adjs = {}
    for (s, r, d) in g.edge_types:
        ei = g[(s, r, d)].edge_index.numpy()
        S, D = rename[s], rename[d]
        ns, nd = g[s].num_nodes, g[d].num_nodes
        A = sp.coo_matrix((np.ones(ei.shape[1]), (ei[1], ei[0])), shape=(nd, ns)).tocsr()
        deg = np.asarray(A.sum(1)).ravel()
        deg[deg == 0] = 1
        adjs[(S, D)] = (sp.diags(1.0 / deg) @ A).tocsr()
    return adjs


def precompute(feats, adjs, paths, tgt):
    """逐跳 mean 传播(等价官方 hg_propagate)，返回终点为目标类型的 {path: [N_t, d]}。
    缓存公共前缀避免重复计算。"""
    cache = {}

    def rec(p):
        if p in cache:
            return cache[p]
        if len(p) == 1:
            cache[p] = feats[p]
            return cache[p]
        prev = rec(p[:-1])
        A = adjs.get((p[-2], p[-1]))
        cache[p] = None if (prev is None or A is None) else A @ prev
        return cache[p]

    out = {}
    for p in paths:
        f = rec(p)
        if f is not None and p[-1] == tgt:
            out[p] = np.asarray(f, dtype=np.float32)
    return out


# ---- LMSPS 模型（官方 model.py 的 withoutLabel 精简等价版） ----
class Conv1d1x1(nn.Module):
    def __init__(self, cin, cout, groups, bias=True):
        super().__init__()
        self.groups = groups
        self.W = nn.Parameter(torch.randn(groups, cin, cout)) if groups > 1 else nn.Parameter(torch.randn(cin, cout))
        self.bias = nn.Parameter(torch.zeros(groups, cout) if groups > 1 else torch.zeros(1, cout)) if bias else None

    def forward(self, x):  # x: [B, C(groups), D]
        if self.groups == 1:
            return torch.einsum('bcm,mn->bcn', x, self.W) + self.bias
        # 官方 model.py:54 是 `+ self.bias`（[groups, cout] 广播到 [B, groups, cout] 末两维）
        return torch.einsum('bcm,cmn->bcn', x, self.W) + self.bias


class LMSPSNet(nn.Module):
    """withoutLabel 配置：residual + bns，无 label 特征，nfeat=embed_size。

    官方 model.py 中所有输入特征（含目标类型）都先经 embeding 投影到统一 nfeat，
    再进入 feat_project 与 res_fc（res_fc 作用于已投影的 nfeat 维目标特征）。
    因此 tgt_feat 的原始维度可能与 nfeat 不同，需要 tgt_embed 投影。
    """
    def __init__(self, in_dims, nfeat, hidden, nclass, num_meta,
                 dropout=0.5, input_drop=0.1, n_layers_2=2, residual=True, bns=True,
                 tgt_dim=None):
        super().__init__()
        # 目标类型特征投影：与官方 embeding[tgt_type] 对应（原始维 -> nfeat）
        self.tgt_embed = nn.Linear(tgt_dim, nfeat, bias=False) if (tgt_dim and tgt_dim != nfeat) else None
        self.res_fc = nn.Linear(nfeat, hidden, bias=False) if residual else None
        self.embed = nn.ParameterDict({
            k: nn.Parameter(torch.empty(d, nfeat).uniform_(-0.5, 0.5))
            for k, d in in_dims.items() if d != nfeat})
        self.feat_project = nn.Sequential(
            Conv1d1x1(nfeat, hidden, num_meta), nn.LayerNorm([num_meta, hidden]),
            nn.PReLU(), nn.Dropout(dropout),
            Conv1d1x1(hidden, hidden, num_meta), nn.LayerNorm([num_meta, hidden]),
            nn.PReLU(), nn.Dropout(dropout))
        self.concat_project = nn.Linear(num_meta * hidden, hidden)

        def blk():
            return ([nn.BatchNorm1d(hidden), nn.PReLU(), nn.Dropout(dropout)] if bns
                    else [nn.PReLU(), nn.Dropout(dropout)])
        layers = []
        for _ in range(n_layers_2 - 1):
            layers += [nn.Linear(hidden, hidden, bias=not bns)] + blk()
        layers += [nn.Linear(hidden, nclass, bias=False), nn.BatchNorm1d(nclass)]
        self.lr_output = nn.Sequential(*layers)
        self.prelu = nn.PReLU()
        self.dropout = nn.Dropout(dropout)
        self.input_drop = nn.Dropout(input_drop)
        gain = nn.init.calculate_gain("relu")
        nn.init.xavier_uniform_(self.concat_project.weight, gain=gain)
        nn.init.zeros_(self.concat_project.bias)
        if residual:
            nn.init.xavier_uniform_(self.res_fc.weight, gain=gain)

    def forward(self, x_list, tgt_feat):
        proj = [(x @ self.embed[k] if k in self.embed else x) for k, x in x_list]
        x = self.input_drop(torch.stack(proj, dim=1))   # [B, num_meta, nfeat]
        x = self.feat_project(x)
        x = self.concat_project(x.reshape(x.size(0), -1))
        if self.res_fc is not None:
            t = self.tgt_embed(tgt_feat) if self.tgt_embed is not None else tgt_feat
            x = x + self.res_fc(t)
        return self.lr_output(self.dropout(self.prelu(x)))
