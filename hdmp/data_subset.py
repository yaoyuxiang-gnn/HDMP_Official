"""大图子图采样模块
用途：对 ogbn-mag 这类大图，抽取指定数量目标节点的连通诱导子图，
用于"方法随图规模变化"的可扩展性实验（1K/10K/50K/100K 目标节点 ×
Macro-F1/Micro-F1/Macro-AUC 三指标）。
"""
from __future__ import annotations

from typing import List

import numpy as np
import torch

from .data import HetGraph


def _stratified_seeds(labels: np.ndarray, n_target: int,
                      rng: np.random.Generator, min_per_class: int) -> np.ndarray:
    """类别分层采样目标节点：先保证每类至少 min_per_class 个样本，
    剩余名额按类别分布比例分配。返回选中的目标类型**局部 id**。"""
    n_t = len(labels)
    classes, counts = np.unique(labels, return_counts=True)
    nc = len(classes)
    # 每类先保底 min_per_class（若该类总数不足则全取）
    chosen = []
    for c in classes:
        idx_c = np.nonzero(labels == c)[0]
        take = min(min_per_class, len(idx_c))
        chosen.append(rng.choice(idx_c, size=take, replace=False))
    base = np.concatenate(chosen) if chosen else np.array([], dtype=np.int64)
    if len(base) >= n_target:
        # 保底已超额，随机下采样到 n_target（保持类别覆盖）
        rng.shuffle(base)
        return np.sort(base[:n_target])
    # 剩余名额按比例从未选节点中补
    remaining = n_target - len(base)
    chosen_set = np.zeros(n_t, dtype=bool)
    chosen_set[base] = True
    rest_idx = np.nonzero(~chosen_set)[0]
    # 按类别频率为权重做加权采样（保持类别分布）
    rest_labels = labels[rest_idx]
    freq = counts / counts.sum()
    w = freq[np.searchsorted(classes, rest_labels)]
    w = w / w.sum()
    extra = rng.choice(rest_idx, size=min(remaining, len(rest_idx)), replace=False, p=w)
    return np.sort(np.concatenate([base, extra]))


def subset_graph(graph: HetGraph, n_target: int, seed: int = 0,
                 aux_per_target: int = 20, min_per_class: int = 5,
                 stratified: bool = True) -> HetGraph:
    """抽取以 n_target 个目标节点为核心的诱导子图，返回接口一致的 HetGraph。
    """
    rng = np.random.default_rng(seed)
    off0 = int(graph.type_offsets[graph.target_type].item())
    labels_np = graph.labels.numpy()
    if stratified:
        seeds_local = _stratified_seeds(labels_np, n_target, rng, min_per_class)
    else:
        seeds_local = np.sort(rng.choice(len(labels_np),
                                         size=min(n_target, len(labels_np)), replace=False))
    seeds = seeds_local + off0   # 转全局 id

    # 目标节点 = 种子
    keep_set = np.zeros(graph.num_nodes, dtype=bool)
    keep_set[seeds] = True

    # 辅助邻居（向量化）：只保留**非目标类型**节点，避免把其他目标节点当辅助引入。
    # 第 1 跳：种子的非目标邻居（author/field 等）；第 2 跳：这些辅助节点的非目标
    # 邻居（覆盖 paper->author->institution 这类 2 跳才到达的类型）。
    type_of_node_np = graph.type_of_node.numpy()
    is_target = type_of_node_np == graph.target_type
    seed_set = keep_set.copy()

    def nontarget_neighbors(node_mask):
        out = []
        for ei in graph.edge_index:
            e = ei.numpy()
            out.append(e[1][node_mask[e[0]]])
            out.append(e[0][node_mask[e[1]]])
        if not out:
            return np.array([], dtype=np.int64)
        nb = np.concatenate(out)
        return nb[(~node_mask[nb]) & (~is_target[nb])]

    hop1 = nontarget_neighbors(seed_set)
    aux_all = [hop1]
    if hop1.size:
        hop1_set = np.zeros(graph.num_nodes, dtype=bool)
        hop1_set[hop1] = True
        hop2 = nontarget_neighbors(hop1_set)          # institution 等
        aux_all.append(hop2)
    aux_all = np.concatenate(aux_all) if aux_all else np.array([], dtype=np.int64)
    if aux_all.size:
        cnt = np.bincount(aux_all, minlength=graph.num_nodes)
        cand = np.nonzero(cnt)[0]
        order = np.argsort(-cnt[cand], kind="stable")
        cap = n_target * aux_per_target
        aux_nodes = cand[order[:cap]]
        keep_set[aux_nodes] = True
    keep_global = np.nonzero(keep_set)[0]

    off = int(graph.type_offsets[graph.target_type].item())
    keep_tgt_global = keep_global[type_of_node_np[keep_global] == graph.target_type]
    n_t = len(keep_tgt_global)

    # 新全局 id 映射
    new_id = np.full(graph.num_nodes, -1, dtype=np.int64)
    new_id[keep_global] = np.arange(len(keep_global))

    # 重建 features / type_of_node / local_id / type_offsets
    features: List[torch.Tensor] = []
    type_of_node_list = []
    local_id_list = []
    offsets = [0]
    for t in range(graph.T_A):
        lo, hi = int(graph.type_offsets[t]), int(graph.type_offsets[t + 1])
        node_ids = np.arange(lo, hi)
        kept = node_ids[keep_set[node_ids]]
        features.append(graph.features[t][kept - lo])
        type_of_node_list.append(np.full(len(kept), t, dtype=np.int64))
        local_id_list.append(np.arange(len(kept), dtype=np.int64))
        offsets.append(offsets[-1] + len(kept))
    type_of_node = torch.from_numpy(np.concatenate(type_of_node_list))
    local_id = torch.from_numpy(np.concatenate(local_id_list))
    type_offsets = torch.tensor(offsets)

    # 重建 edge_index（只保留两端都在 keep 的边，重映射 id）
    edge_index: List[torch.LongTensor] = []
    for ei in graph.edge_index:
        e = ei.numpy()
        m = keep_set[e[0]] & keep_set[e[1]]
        edge_index.append(torch.from_numpy(new_id[e[:, m]]))

    # 标签与 mask：目标类型的局部 id（在新空间）
    new_tgt_lo = int(type_offsets[graph.target_type])
    new_tgt_global = keep_global[graph.type_of_node[keep_global].numpy() == graph.target_type]
    old_local_of_new_tgt = new_tgt_global - off
    labels = graph.labels[torch.from_numpy(old_local_of_new_tgt)]
    train_mask = graph.train_mask[torch.from_numpy(old_local_of_new_tgt)]
    val_mask = graph.val_mask[torch.from_numpy(old_local_of_new_tgt)]
    test_mask = graph.test_mask[torch.from_numpy(old_local_of_new_tgt)]

    return HetGraph(
        node_types=graph.node_types,
        edge_types=graph.edge_types,
        directed=graph.directed,
        type_of_node=type_of_node,
        local_id=local_id,
        type_offsets=type_offsets,
        features=features,
        edge_index=edge_index,
        target_type=graph.target_type,
        labels=labels,
        num_classes=graph.num_classes,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
    )
