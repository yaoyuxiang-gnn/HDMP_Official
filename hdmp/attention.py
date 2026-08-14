"""双层注意力模块：节点级 GAT 聚合 + 元路径级注意力。

不依赖 torch_scatter：用 scatter_reduce_ / index_add_ 实现边上 softmax 与聚合。
复合邻接以 edge_index（目标类型局部 id 空间）形式预计算后传入。
"""
from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


def edge_softmax(scores: torch.Tensor, dst: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """对每条边的打分按目标节点分组做 softmax（数值稳定版）。
    scores: [E]；dst: [E] 每条边的目标节点 id。
    """
    max_per = torch.full((num_nodes,), float("-inf"), device=scores.device, dtype=scores.dtype)
    max_per.scatter_reduce_(0, dst, scores, reduce="amax", include_self=True)
    exp = torch.exp(scores - max_per[dst])
    denom = torch.zeros(num_nodes, device=scores.device, dtype=scores.dtype)
    denom.index_add_(0, dst, exp)
    return exp / denom[dst].clamp(min=1e-12)


class NodeLevelAttention(nn.Module):
    """节点级注意力：对 M 条粗筛元路径批量执行 GAT 聚合。
    """
    def __init__(self, K: int, d: int, negative_slope: float = 0.2, num_heads: int = 1):
        super().__init__()
        self.K, self.d, self.H = K, d, num_heads
        assert d % num_heads == 0, f"d={d} 必须能被 num_heads={num_heads} 整除"
        self.dk = d // num_heads                                         # 每头维度
        # 每头独立变换矩阵与打分向量
        self.W = nn.Parameter(torch.empty(K, num_heads, self.dk, self.dk))
        self.att = nn.Parameter(torch.empty(K, num_heads, 2 * self.dk))
        self.W_in = nn.Parameter(torch.empty(K, num_heads, d, self.dk))  # 输入 d -> 每头 dk
        nn.init.xavier_uniform_(self.W)
        nn.init.xavier_uniform_(self.att)
        nn.init.xavier_uniform_(self.W_in)
        self.leaky = nn.LeakyReLU(negative_slope)

    def forward(self, h_src: torch.Tensor, h_dst: torch.Tensor,
                edge_indices: List[torch.Tensor], path_idx: torch.Tensor) -> torch.Tensor:
        """异构双分 GAT 聚合：源节点（任意类型，全局 id）→ 目标节点（目标类型，局部 id）。
        h_src: [N, d] 全部节点的表示（源侧，全局 id 空间）；
        h_dst: [N_t, d] 目标节点表示（目标侧，局部 id 空间）；
        edge_indices: 长度 M 的 [2, E_k] 列表，ei[0]=源（全局 id），ei[1]=目标（局部 id）；
        path_idx: [M] 每条粗筛路径在候选全集中的身份索引。
        返回 Z [N_t, M, d]：每条元路径下的目标节点表示（多头拼接回 d 维）。
        """
        M = path_idx.numel()
        W_in = self.W_in[path_idx]                                       # [M, H, d, dk]
        W = self.W[path_idx]                                             # [M, H, dk, dk]
        att = self.att[path_idx]                                         # [M, H, 2dk]
        # 先映射到每头 dk 维，再做头内变换
        hs = torch.einsum("nd,mhde->mhne", h_src, W_in)                  # [M, H, N, dk]
        hd = torch.einsum("nd,mhde->mhne", h_dst, W_in)                  # [M, H, N_t, dk]
        Wh_src = torch.einsum("mhne,mhef->mhnf", hs, W)                  # [M, H, N, dk]
        Wh_dst = torch.einsum("mhne,mhef->mhnf", hd, W)                  # [M, H, N_t, dk]
        Zs = []
        for k, ei in enumerate(edge_indices):
            src, dst = ei[0], ei[1]
            # 逐头聚合
            z_heads = []
            for h in range(self.H):
                pair = torch.cat([Wh_dst[k, h][dst], Wh_src[k, h][src]], dim=-1)  # [E_k, 2dk]
                score = self.leaky(pair @ att[k, h])                              # [E_k]
                alpha = edge_softmax(score, dst, h_dst.size(0))
                z = torch.zeros_like(Wh_dst[k, h])
                z.index_add_(0, dst, alpha.unsqueeze(-1) * Wh_src[k, h][src])
                z_heads.append(z)                                                 # [N_t, dk]
            Zs.append(torch.cat(z_heads, dim=-1))                                 # [N_t, d]
        return torch.stack(Zs, dim=1)                                    # [N_t, M, d]


class MetaPathLevelAttention(nn.Module):
    """元路径级注意力（§4.2，含语义闭环）：打分同时输入节点表示与元路径嵌入。"""
    def __init__(self, d: int, d_P: int):
        super().__init__()
        self.W_sem = nn.Linear(d + d_P, d)
        self.q = nn.Linear(d, 1, bias=False)

    def forward(self, Z: torch.Tensor, e_tilde: torch.Tensor):
        """Z: [N_t, M, d]；e_tilde: [M, d_P]。
        返回 logits a [N_t, M] 与归一化权重 omega [N_t, M]。
        """
        M = Z.size(1)
        e = e_tilde.unsqueeze(0).expand(Z.size(0), M, -1)            # [N_t, M, d_P]
        a = self.q(torch.tanh(self.W_sem(torch.cat([Z, e], dim=-1)))).squeeze(-1)
        omega = F.softmax(a, dim=-1)
        return a, omega
