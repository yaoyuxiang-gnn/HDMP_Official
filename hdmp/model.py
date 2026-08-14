"""HDMP 主模型（技术方案 §3-§6 的端到端编排）。

使用前需先离线准备（见 prepare() 或 train.py 中的等价流程）：
1. 枚举 + 剪枝得到候选集；2. 构建层次关系对；3. 采样路径实例；
4. 对粗筛后的候选预计算复合邻接（转换为目标类型局部 id 空间的 edge_index）。
"""
from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import MetaPathLevelAttention, NodeLevelAttention
from .encoding import InstanceContext, MetaPathEncoder
from .losses import hierarchical_loss, stabilization_loss
from .selection import CascadeScorer, GumbelGate, top_rho_score


class HDMPModel(nn.Module):
    """端到端可微元路径自动发现模型。"""

    def __init__(self, num_node_types: int, num_edge_types: int, feat_dims: List[int],
                 num_classes: int, K: int, max_tokens: int, cfg: Dict):
        super().__init__()
        d, d_P, M = cfg["d"], cfg["d_P"], cfg["M"]
        self.cfg = cfg
        self.K = K
        self.encoder = MetaPathEncoder(num_node_types, num_edge_types, d_P,
                                       cfg["L_T"], max_len=max_tokens,
                                       attn_dropout=cfg.get("attn_dropout", 0.1))
        self.instance_ctx = InstanceContext(feat_dims, d_P)
        self.target_proj = nn.Linear(feat_dims[0], d)          # 目标类型特征 -> 隐藏维
        self.n_layers = cfg.get("n_layers", 1)                       # 节点注意力堆叠层数
        self.node_att = NodeLevelAttention(K, d, num_heads=cfg.get("num_heads", 1))  # 参数按路径身份索引
        if self.n_layers > 1:
            # 第二层：以第一层输出为目标侧表示再聚合（高阶语义，追平多层 GAT 的表示能力）
            self.node_att2 = NodeLevelAttention(K, d, num_heads=cfg.get("num_heads", 1))
        self.sem_att = MetaPathLevelAttention(d, d_P)
        self.gate = GumbelGate(cfg["tau_init"], cfg["tau_final"], cfg["anneal_rate"])
        self.scorer = CascadeScorer(d_P, d)
        self.dropout = nn.Dropout(cfg.get("dropout", 0.0))                   # 特征/融合 dropout
        self.classifier = nn.Linear(d, num_classes)

        # 缓冲区（由外部离线预计算后 register）
        self.register_buffer("tokens", torch.empty(0, dtype=torch.long))      # [K, L]（含 CLS）
        self.register_buffer("token_mask", torch.empty(0, dtype=torch.bool))
        self.register_buffer("inst_idx", torch.empty(0, dtype=torch.long))    # [K, S, l_max+1]
        self.register_buffer("inst_mask", torch.empty(0, dtype=torch.bool))
        self.register_buffer("pairs1", torch.empty(0, 2, dtype=torch.long))
        self.register_buffer("pairs2", torch.empty(0, 2, dtype=torch.long))
        self.shortlist_idx: torch.Tensor = torch.arange(min(M, K))            # 当前粗筛候选
        self.edge_indices: List[torch.Tensor] = []                            # 每条候选的 [2, E_k]
        self._anneal_epoch: int | None = None                                 # 退火完成时的 epoch（budget 爬升计时）
        # T2 对照实验：固定门控掩码 [M]（0/1）。设置后 forward 绕过 Gumbel 门控学习，
        # 直接用该掩码作为 z_used——random-subset（随机选B条）/ oracle-expert（专家路径全开）
        # 用于归因"选择机制 vs 残差/聚合"的贡献差异。默认 None 不影响主流程。
        self.fixed_z: torch.Tensor | None = None

    def set_fixed_gate(self, mask: torch.Tensor | None) -> None:
        """设置/清除固定门控掩码（T2 对照实验用）。"""
        self.fixed_z = mask

    # ---------------- 离线准备 ----------------
    def load_prepared(self, tokens, token_mask, inst_idx, inst_mask, pairs1, pairs2) -> None:
        self.tokens, self.token_mask = tokens, token_mask
        self.inst_idx, self.inst_mask = inst_idx, inst_mask
        self.pairs1, self.pairs2 = pairs1, pairs2

    def set_shortlist(self, idx: torch.Tensor, edge_indices: List[torch.Tensor]) -> None:
        self.shortlist_idx = idx
        self.edge_indices = edge_indices

    # ---------------- 编码 ----------------
    def encode_paths(self, features: List[torch.Tensor], type_offsets: torch.Tensor) -> torch.Tensor:
        """Transformer 语义编码 + 实例上下文融合，返回 ẽ [K, d_P]。"""
        e_sem = self.encoder(self.tokens, self.token_mask)
        proj_table = self.instance_ctx.project_all(features, type_offsets)
        return self.instance_ctx(e_sem, proj_table, self.inst_idx, self.inst_mask)

    # ---------------- 前向 ----------------
    def forward(self, features: List[torch.Tensor], type_offsets: torch.Tensor,
                target_feat: torch.Tensor, epoch: int, training: bool = True):
        cfg = self.cfg
        e_tilde = self.encode_paths(features, type_offsets)                  # [K, d_P]
        # 源侧：全节点经类型投影到 d_P 维的统一表示（全局 id 空间）
        h_src = self.instance_ctx.project_all(features, type_offsets)        # [N, d_P]
        # 目标侧：目标节点自身表示（加 dropout 提升泛化、降方差）
        h_dst = self.dropout(self.target_proj(target_feat))                  # [N_t, d]

        Z = self.node_att(h_src, h_dst, self.edge_indices, self.shortlist_idx)  # [N_t, M, d]
        if self.n_layers > 1:
            # 第二层：把第一层聚合结果（路径维均值）作为新的目标侧查询，捕获高阶邻居
            h_dst2 = F.elu(Z.mean(dim=1)) + h_dst                            # [N_t, d] 残差
            Z = Z + self.node_att2(h_src, self.dropout(h_dst2), self.edge_indices, self.shortlist_idx)
        e_short = e_tilde[self.shortlist_idx]                                # [M, d_P]
        logits, omega = self.sem_att(Z, e_short)                             # [N_t, M]

        frozen = epoch <= cfg["t_warm"]
        anneal_done = self.gate.temperature(epoch) <= cfg["tau_final"] + 1e-9
        # 全局评分 = 注意力 top-ρ 聚合 + 级联打分器偏置（使 scorer 获得下游梯度，§5.4）
        g = top_rho_score(logits, cfg["rho"])                                # [M]
        h_bar = h_dst.mean(dim=0)
        g = g + self.scorer(e_tilde, h_bar)[self.shortlist_idx]              # [M]
        if self.fixed_z is not None:
            # T2 对照：固定门控（random-subset / oracle-expert），绕过 Gumbel 学习，
            # 训练/推理都用预设的 0/1 掩码，隔离"选择机制"与"残差+聚合"的贡献。
            z_used = self.fixed_z.to(g.device).float()
            z_soft = z_used
        else:
            z_used, z_soft = self.gate(g, epoch, training, frozen=frozen, use_st=anneal_done)

        # 融合表示：门控加权的注意力融合 + 目标节点自身特征残差连接。
        # 残差让模型保底用好自身特征（HAN 的语义聚合同样是"自身 + 邻居"），
        # 防止门控/注意力把强特征稀释（对特征主导的数据集如 IMDB 尤其关键）。
        neigh = torch.einsum("nm,nmd->nd", z_used.unsqueeze(0) * omega, Z)   # [N_t, d]
        # 残差连接（消融开关 no_residual=True 时关闭，验证残差的必要性）
        fused = neigh if cfg.get("no_residual", False) else (neigh + h_dst)
        fused = self.dropout(fused)
        out = self.classifier(fused)                                         # [N_t, C]

        loss_reg = (
            hierarchical_loss(e_tilde, self.pairs1, self.pairs2, cfg["lambda_h1"], cfg["lambda_h2"])
            + cfg["lambda_s"] * stabilization_loss(logits)
        )
        aux = {"logits": logits, "omega": omega, "g": g, "z_soft": z_soft, "e_tilde": e_tilde}
        return out, loss_reg, aux

    # ---------------- 推理 ----------------
    @torch.no_grad()
    def selected_paths(self, features, type_offsets, target_feat, B: int) -> torch.Tensor:
        """返回全局评分 top-B 的候选索引（相对 shortlist_idx 的位置）。"""
        self.eval()
        e_tilde = self.encode_paths(features, type_offsets)
        h_src = self.instance_ctx.project_all(features, type_offsets)
        h_dst = self.target_proj(target_feat)
        Z = self.node_att(h_src, h_dst, self.edge_indices, self.shortlist_idx)
        logits, _ = self.sem_att(Z, e_tilde[self.shortlist_idx])
        g = top_rho_score(logits, self.cfg["rho"])
        return torch.topk(g, k=min(B, g.numel())).indices


def cross_entropy_on_mask(out: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor,
                          class_weight: torch.Tensor | None = None) -> torch.Tensor:
    return F.cross_entropy(out[mask], labels[mask], weight=class_weight)
