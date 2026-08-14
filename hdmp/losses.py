"""损失函数模块"""
from __future__ import annotations
import torch


def hierarchical_loss(e_tilde: torch.Tensor, pairs1: torch.Tensor, pairs2: torch.Tensor,
                      lambda_h1: float, lambda_h2: float) -> torch.Tensor:
    """层次一致性软正则：Δl=1 强约束、Δl=2 弱约束。
    pairs1 / pairs2: [num_pairs, 2]，每行 (父路径 idx, 子路径 idx)。
    """
    loss = e_tilde.new_zeros(())
    if pairs1.numel() > 0 and lambda_h1 > 0:
        diff = e_tilde[pairs1[:, 0]] - e_tilde[pairs1[:, 1]]
        loss = loss + lambda_h1 * (diff ** 2).sum(dim=-1).mean()
    if pairs2.numel() > 0 and lambda_h2 > 0:
        diff = e_tilde[pairs2[:, 0]] - e_tilde[pairs2[:, 1]]
        loss = loss + lambda_h2 * (diff ** 2).sum(dim=-1).mean()
    return loss


def stabilization_loss(logits: torch.Tensor) -> torch.Tensor:
    """logits 尺度稳定正则"""
    return logits.abs().mean()
