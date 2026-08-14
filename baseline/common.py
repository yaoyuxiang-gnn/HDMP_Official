"""baselines 包：共享数据加载、指标与训练工具"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch_geometric.nn import SAGEConv, GATv2Conv, HGTConv, HANConv, GCNConv
from torch_geometric.datasets import HGBDataset, OGB_MAG


def load(name, root="./data"):
    if name.lower() in ("ogbn_mag", "ogbn-mag", "mag"):
        import torch_geometric.transforms as T
        ds = OGB_MAG(root="./data/ogbn_mag", preprocess="metapath2vec", transform=T.ToUndirected())
        g = ds[0]
        tgt = "paper"
        gen = torch.Generator().manual_seed(0)
        for nt in g.node_types:
            x = g[nt].get("x", None)
            if x is None:
                g[nt].x = torch.randn(g[nt].num_nodes, 64, generator=gen) * 0.1
            g[nt].x = g[nt].x.float()
            g[nt].x = g[nt].x / g[nt].x.norm(dim=1, keepdim=True).clamp(min=1e-12)
        nc = int(g[tgt].y.max().item()) + 1
        return g, tgt, nc
    ds = HGBDataset(root=root, name=name)
    g = ds[0]
    target = ds._target_node_type if hasattr(ds, "_target_node_type") else g.node_types[0]
    tgt = None
    for nt in g.node_types:
        if hasattr(g[nt], "y") and g[nt].y is not None and hasattr(g[nt], "train_mask"):
            tgt = nt
            break
    for nt in g.node_types:
        if not hasattr(g[nt], "x") or g[nt].x is None:
            n = g[nt].num_nodes
            if n <= 20000:
                g[nt].x = torch.eye(n)
            else:
                gen = torch.Generator().manual_seed(0)
                g[nt].x = torch.randn(n, 128, generator=gen)
    yy = g[tgt].y
    nc = yy.size(1) if yy.dim() == 2 else int(yy.max().item()) + 1
    return g, tgt, nc


def masks(g, tgt, val_ratio=0.2, seed=0):
    if hasattr(g[tgt], "val_mask") and g[tgt].val_mask is not None:
        return g[tgt].train_mask.clone(), g[tgt].val_mask.clone(), g[tgt].test_mask.clone()
    y = g[tgt].y
    if y.dim() == 2:
        y = y.argmax(dim=1)
    num_classes = int(y.max().item()) + 1
    train_mask = g[tgt].train_mask.clone()
    test_mask = g[tgt].test_mask.clone()
    rng = np.random.default_rng(seed)
    val_mask = torch.zeros_like(train_mask)
    tr_idx = torch.nonzero(train_mask, as_tuple=False).view(-1).numpy()
    for c in range(num_classes):
        c_idx = tr_idx[y.numpy()[tr_idx] == c]
        n_val = max(1, int(round(len(c_idx) * val_ratio)))
        chosen = rng.choice(c_idx, size=n_val, replace=False)
        val_mask[torch.from_numpy(chosen)] = True
    train_mask = train_mask & ~val_mask
    return train_mask, val_mask, test_mask


def all_metrics(logits, y, mask):
    """Micro-F1（单标签=Accuracy）、Macro-F1、Macro-AUC（OvR）。"""
    from sklearn.metrics import roc_auc_score
    yt = y[mask].cpu().numpy()
    pred = logits[mask].argmax(-1).cpu().numpy()
    prob = torch.softmax(logits[mask], dim=-1).cpu().numpy()
    nc = logits.size(-1)
    try:
        if len(np.unique(yt)) == nc:
            auc = float(roc_auc_score(yt, prob, multi_class="ovr", average="macro",
                                      labels=list(range(nc))))
        else:
            # MAG 按年份划分，test 中部分冷门类零样本：列子集 + 行重归一化
            # （OvR AUC 是逐类排序指标，行归一为列内单调变换，AUC 值不变）
            present = np.unique(yt)
            p = prob[:, present]
            p = p / p.sum(axis=1, keepdims=True)
            auc = float(roc_auc_score(yt, p, multi_class="ovr", average="macro",
                                      labels=present))
    except Exception:
        auc = float("nan")
    return {
        "macro_f1": float(f1_score(yt, pred, average="macro")),
        "micro_f1": float(f1_score(yt, pred, average="micro")),
        "auc": auc,
    }


def to_target_homo(g, tgt):
    """抽取目标类型节点及其间所有边（经任意关系），同质化为无向图。"""
    ei_list = []
    tgt_nodes = g[tgt].num_nodes
    for et in g.edge_types:
        s, r, d = et
        ei = g[et].edge_index
        if s == tgt and d == tgt:
            ei_list.append(ei)
    if ei_list:
        ei = torch.cat(ei_list, dim=1)
        ei = torch.cat([ei, ei.flip(0)], dim=1)  # 无向
    else:
        ei = torch.empty(2, 0, dtype=torch.long)
    x = g[tgt].x
    return x, ei


def train_full(model, feat_in, ei_in, get_out, y, tr, va, te, device, epochs=150, lr=5e-3):
    model = model.to(device)
    y = y.to(device); tr, va, te = tr.to(device), va.to(device), te.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=5e-4)
    for ep in range(1, epochs + 1):
        model.train()
        out = get_out(model, feat_in, ei_in)
        loss = F.cross_entropy(out[tr], y[tr])
        opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        out = get_out(model, feat_in, ei_in)
    return all_metrics(out, y, te)


