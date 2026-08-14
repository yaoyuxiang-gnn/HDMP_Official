"""可微子集选择模块：top-ρ 全局评分、Gumbel-Sigmoid 门控、
Straight-Through 估计、温度退火与级联粗筛打分器。"""
from __future__ import annotations

import math
from collections import defaultdict

import torch
import torch.nn as nn


def top_rho_score(logits: torch.Tensor, rho: float) -> torch.Tensor:
    """top-ρ 分位数聚合的全局重要性评分（§5.1，修复 v1 均值稀释问题）。

    logits: [N_t, M]（逐节点的元路径打分）；返回 g [M]。
    """
    q = torch.quantile(logits, rho, dim=0, keepdim=True)      # [1, M]
    mask = logits >= q
    return (logits * mask).sum(dim=0) / mask.sum(dim=0).clamp(min=1)


class GumbelGate(nn.Module):
    """Gumbel-Sigmoid 可微门控
    噪声 ε = G0 - G1（G 为独立 Gumbel），即二元 Gumbel-Softmax 的等价形式。
    训练：软门控 → 退火 → ST 硬门控；推理：直接对全局评分取 top-B。
    """

    def __init__(self, tau_init: float = 1.0, tau_final: float = 0.1, anneal_rate: float = 3e-3):
        super().__init__()
        self.tau_init, self.tau_final, self.anneal_rate = tau_init, tau_final, anneal_rate

    def temperature(self, step: int) -> float:
        return max(self.tau_final, self.tau_init * math.exp(-self.anneal_rate * step))

    def forward(self, g: torch.Tensor, step: int, training: bool,
                frozen: bool = False, use_st: bool = False, warm_bias: float = 0.0):
        """返回 (z_used, z_soft)。
        frozen: warm-up 阶段固定为 1；use_st: 退火完成后启用 Straight-Through；
        warm_bias: 暖启动偏置，解冻初期为正值并线性衰减至 0，
        使门控从 warm-up 的 z≡1 平滑过渡到随机采样，避免突变崩溃。
        """
        if frozen or not training:
            if frozen:
                return torch.ones_like(g), torch.ones_like(g)
            z = torch.sigmoid(g / self.temperature(step))
            return z, z
        u1 = torch.rand_like(g).clamp(1e-10, 1 - 1e-10)
        u2 = torch.rand_like(g).clamp(1e-10, 1 - 1e-10)
        g0 = -torch.log(-torch.log(u1))
        g1 = -torch.log(-torch.log(u2))
        z_soft = torch.sigmoid((g + warm_bias + g0 - g1) / self.temperature(step))
        if use_st:
            z_hard = (z_soft > 0.5).float()
            z_used = z_hard + z_soft - z_soft.detach()   # ST：前向硬、反向软
        else:
            z_used = z_soft
        return z_used, z_soft


class CascadeScorer(nn.Module):
    """级联粗筛打分器（§5.4）：仅用元路径嵌入与目标类型节点均值表示打分，无图计算。"""
    def __init__(self, d_P: int, d: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_P + d, d_P), nn.ReLU(), nn.Linear(d_P, 1)
        )

    def forward(self, e_tilde: torch.Tensor, h_bar: torch.Tensor) -> torch.Tensor:
        """e_tilde: [K, d_P]；h_bar: [d]。返回 s [K]。"""
        K = e_tilde.size(0)
        h = h_bar.unsqueeze(0).expand(K, -1)
        return self.mlp(torch.cat([e_tilde, h], dim=-1)).squeeze(-1)


def stratified_shortlist(rel_seqs, coverage: torch.Tensor, M: int, T_R: int = 4) -> torch.Tensor:
    """分层配额粗筛（方案 A）：按 hop 长度分层，每层按比例分配名额，层内按实例覆盖率排序。
    解决覆盖率启发式与元路径语义负相关的问题：短路径的复合邻接天然更稠密、
    覆盖率更高，纯覆盖率 top-M 会系统性筛掉所有长路径之外的短路径（含专家路径）。
    分层配额保证每个 hop 长度都有代表进入候选空间，让可微选择机制公平比较。
    """
    by_hop = defaultdict(list)
    for i, rels in enumerate(rel_seqs):
        by_hop[len(rels)].append(i)
    hops = sorted(by_hop)
    # 配额：各层至少 1 条；短路径层（hop<=2）至少覆盖 T_R 条以保证每种语义关系都有代表
    quota = {h: 1 for h in hops}
    for h in hops:
        if h <= 2:
            quota[h] = max(1, min(T_R, len(by_hop[h])))
    used = sum(quota.values())
    remain = M - used
    if remain > 0:
        total = sum(len(by_hop[h]) for h in hops)
        for h in hops:
            quota[h] += int(round(remain * len(by_hop[h]) / total))
    # 层内选择：先保证语义多样性（每种"关系签名"至少 1 条），再按覆盖率补足配额。
    # 关系签名 = 路径用到的关系类型集合（忽略方向与顺序），
    # 避免 cite 类路径因覆盖率天然高于 author/subject 类而独占短路径层名额。
    def rel_signature(rels):
        # 逆关系 r >= T_R 归一为其原关系 id，使正/逆视为同一语义关系
        return tuple(sorted({r if r < T_R else r - T_R for r in rels}))

    selected = []
    for h in hops:
        layer = sorted(by_hop[h], key=lambda i: float(coverage[i]), reverse=True)
        seen_sig = set()
        picked = []
        # 第一遍：每种关系签名取覆盖率最高的一条
        for i in layer:
            sig = rel_signature(rel_seqs[i])
            if sig not in seen_sig:
                seen_sig.add(sig)
                picked.append(i)
            if len(picked) >= quota[h]:
                break
        # 第二遍：若配额未满，按覆盖率继续补足
        if len(picked) < quota[h]:
            for i in layer:
                if i not in picked:
                    picked.append(i)
                if len(picked) >= quota[h]:
                    break
        selected.extend(picked)
    # 若配额取完仍不足 M（某些层候选太少），按覆盖率从剩余中补足
    if len(selected) < M:
        rest = [i for h in hops for i in by_hop[h] if i not in selected]
        rest.sort(key=lambda i: float(coverage[i]), reverse=True)
        selected.extend(rest[: M - len(selected)])
    return torch.tensor(selected[:M], dtype=torch.long)
