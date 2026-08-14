"""元路径语义编码模块"""
from __future__ import annotations

import math
from typing import List
import torch
import torch.nn as nn


def sinusoidal_pe(max_len: int, dim: int) -> torch.Tensor:
    """标准正弦位置编码，返回 [max_len, dim]。"""
    pos = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
    div = torch.exp(torch.arange(0, dim, 2, dtype=torch.float) * (-math.log(10000.0) / dim))
    pe = torch.zeros(max_len, dim)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe


class MetaPathEncoder(nn.Module):
    """将元路径 token 序列（含 CLS 位）编码为 d_P 维语义向量（§3.1）。
    输入张量约定：tokens [K, L]，第 0 位为 CLS（id = num_types），
    其余为类型 id；slot 标记 0=节点类型位、1=边类型位、2=CLS。
    """
    def __init__(self, num_node_types: int, num_edge_types: int, d_P: int,
                 L_T: int, max_len: int, n_heads: int = 4, attn_dropout: float = 0.1):
        super().__init__()
        self.num_node_types = num_node_types               # T_A
        num_types = num_node_types + num_edge_types        # 词表 = T_A + T_R
        self.cls_id = num_types
        self.type_emb = nn.Embedding(num_types + 1, d_P)   # +1: CLS
        self.slot_emb = nn.Embedding(3, d_P)
        self.register_buffer("pe", sinusoidal_pe(max_len + 1, d_P))
        # attn_dropout: Transformer 注意力权重 dropout。MPS 后端的
        # scaled_dot_product_attention 不支持 dropout>0（抛 NotImplementedError），
        # 因此在 MPS 设备上训练时需显式传 attn_dropout=0；CPU/CUDA 默认 0.1 不变，
        # 保证已复现的小数据集(ACM/DBLP/IMDB)结果不受影响。
        layer = nn.TransformerEncoderLayer(
            d_model=d_P, nhead=n_heads, dim_feedforward=4 * d_P,
            dropout=attn_dropout, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=L_T)

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # tokens: [K, L]（已含 CLS 位）；mask: [K, L]（True=有效位）
        # 槽位判断：token id ∈ [T_A, T_A+T_R) 为边类型位；id == cls_id 为 CLS；其余为节点类型位
        is_edge = mask & (tokens >= self.num_node_types) & (tokens != self.cls_id)
        slot = torch.zeros_like(tokens)
        slot[is_edge] = 1
        slot[tokens == self.cls_id] = 2
        x = self.type_emb(tokens) + self.slot_emb(slot) + self.pe[: tokens.size(1)]
        x = self.encoder(x, src_key_padding_mask=~mask)
        return x[:, 0]  # CLS 位输出 [K, d_P]


class InstanceContext(nn.Module):
    """实例上下文增强（§3.2，含 R2 修复：类型特定投影）。

    对每种节点类型学习一个到公共维度 d_P 的投影，再对路径实例两级均值池化，
    与语义编码拼接融合为最终的元路径表示 ẽ。
    """

    def __init__(self, feat_dims: List[int], d_P: int):
        super().__init__()
        self.projs = nn.ModuleList([nn.Linear(da, d_P) for da in feat_dims])
        self.fuse = nn.Linear(2 * d_P, d_P)

    def project_all(self, features: List[torch.Tensor], type_offsets: torch.Tensor) -> torch.Tensor:
        """将全部节点特征投影到公共维度并拼成 [N, d_P] 全局表。"""
        outs = []
        for a, feat in enumerate(features):
            outs.append(self.projs[a](feat))
        return torch.cat(outs, dim=0)

    def forward(self, e_sem: torch.Tensor, proj_table: torch.Tensor,
                inst_idx: torch.Tensor, inst_mask: torch.Tensor) -> torch.Tensor:
        # inst_idx: [K, S, L]；inst_mask: [K, S, L]
        f = proj_table[inst_idx]                       # [K, S, L, d_P]
        m = inst_mask.unsqueeze(-1).float()            # [K, S, L, 1]
        ctx = (f * m).sum(dim=(1, 2)) / m.sum(dim=(1, 2)).clamp(min=1.0)
        return self.fuse(torch.cat([e_sem, ctx], dim=-1))  # ẽ [K, d_P]
