"""候选元路径枚举、剪枝与层次关系构建（对应技术方案 §2 与 §3.3）。

元路径的统一表示：token 元组 (A1, R1, A2, ..., Rl, A_{l+1})，
其中节点类型位取 0..T_A-1，边类型位取 T_A..T_A+T_R-1（边类型加了 T_A 偏移）。
关系 id 序列（rel_seq）= token 元组中奇数位置减 T_A。
"""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import torch

from .data import HetGraph, composite_adjacency

MetaPath = Tuple[int, ...]  # token 元组表示


def _tokenize(node_seq: Sequence[int], rel_seq: Sequence[int], T_A: int) -> MetaPath:
    toks: List[int] = []
    for i, r in enumerate(rel_seq):
        toks.append(node_seq[i])
        toks.append(T_A + r)
    toks.append(node_seq[-1])
    return tuple(toks)


def rel_seq_of(path: MetaPath, T_A: int) -> Tuple[int, ...]:
    return tuple(t - T_A for t in path[1::2])


def enumerate_metapaths(graph: HetGraph, l_max: int,
                        start_type: int = None, end_type: int = None) -> List[MetaPath]:
    """在模式图上约束 DFS 枚举候选元路径（技术方案 §2.1）。

    start_type / end_type 默认为任务目标类型（节点分类场景下元路径两端同类）。
    """
    T_A = graph.T_A
    s_type = graph.target_type if start_type is None else start_type
    e_type = graph.target_type if end_type is None else end_type

    # 模式图邻接：src_type -> [(rel, dst_type)]，并为有向关系补充逆关系
    # 逆关系 id 约定：r_inv = r + T_R（复合邻接取原关系邻接的转置，见 data.relation_adjacency）
    schema_adj: Dict[int, List[Tuple[int, int]]] = {a: [] for a in range(T_A)}
    for a, r, b in graph.schema_triples():
        schema_adj[a].append((r, b))
        schema_adj[b].append((r + graph.T_R, a))   # 逆关系

    results: List[MetaPath] = []

    def dfs(node_seq: List[int], rel_seq: List[int]) -> None:
        l = len(rel_seq)
        if l >= 1 and node_seq[-1] == e_type:
            results.append(_tokenize(node_seq, rel_seq, T_A))
        if l >= l_max:
            return
        for r, b in schema_adj[node_seq[-1]]:
            # 简单环控制：避免同一节点类型连续重复出现导致的组合爆炸
            dfs(node_seq + [b], rel_seq + [r])

    dfs([s_type], [])
    # 去重（同一 token 序列可能经不同 DFS 分支到达）
    return sorted(set(results))


def prune_no_instance(graph: HetGraph, paths: List[MetaPath]) -> List[MetaPath]:
    """实例存在性剪枝（技术方案 §2.2 第 1 条）：复合邻接非零元为零的候选剔除。

    精确全量复合邻接（小数据集 ACM/DBLP/IMDB/Freebase 用）。大图（OGBN-MAG）
    的存在性检查由 hdmp.mag_edges.path_has_instance（向量化采样）承担，不走此函数。
    """
    def adj_fn(p):
        return composite_adjacency(graph, rel_seq_of(p, graph.T_A))
    kept = []
    for p in paths:
        if adj_fn(p).nnz > 0:
            kept.append(p)
    return kept


def prune_inverse_dup(graph: HetGraph, paths: List[MetaPath]) -> List[MetaPath]:
    """逆路径去重（§2.2 第 2 条）：无方向关系的逆序对仅保留字典序较小者。"""
    if all(graph.directed):
        return paths
    seen = set()
    kept = []
    for p in paths:
        rev = tuple(reversed(p))
        key = min(p, rev)
        if key in seen:
            continue
        seen.add(key)
        kept.append(p)
    return kept


def pad_tokens(paths: List[MetaPath], num_types: int, pad_id: int = None):
    """将变长 token 序列对齐为张量（左侧补 CLS 由编码器处理，此处仅右填充）。

    返回 tokens [K, L_max_tokens] 与 mask [K, L_max_tokens]（True=有效）。
    """
    L = max(len(p) for p in paths)
    pad = num_types if pad_id is None else pad_id
    tokens = torch.full((len(paths), L), pad, dtype=torch.long)
    mask = torch.zeros(len(paths), L, dtype=torch.bool)
    for i, p in enumerate(paths):
        tokens[i, : len(p)] = torch.tensor(p, dtype=torch.long)
        mask[i, : len(p)] = True
    return tokens, mask


def build_hierarchy(paths: List[MetaPath], max_delta: int = 2):
    """构建连续子序列父子关系（技术方案 §3.3 定义 3）。

    返回 pairs1、pairs2：形状均为 [num_pairs, 2] 的 (父, 子) 索引对，
    分别对应 hop 长度差 Δl=1 与 Δl=2。
    """
    def sub_delta(child: MetaPath, parent: MetaPath) -> int:
        """若 child 是 parent 的连续子序列，返回 hop 长度差，否则 0。"""
        lp, lc = len(parent), len(child)
        if lc >= lp:
            return 0
        for start in range(0, lp - lc + 1, 2):  # 节点类型位必在偶数位置
            if parent[start:start + lc] == child:
                return (lp - lc) // 2
        return 0

    pairs1, pairs2 = [], []
    for pi, parent in enumerate(paths):
        for ci, child in enumerate(paths):
            dl = sub_delta(child, parent)
            if dl == 1:
                pairs1.append((pi, ci))
            elif dl == 2 and max_delta >= 2:
                pairs2.append((pi, ci))
    return (torch.tensor(pairs1, dtype=torch.long).view(-1, 2),
            torch.tensor(pairs2, dtype=torch.long).view(-1, 2))
