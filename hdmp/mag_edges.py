"""大图（OGBN-MAG 级）元路径复合邻接的向量化采样构建（extended work）。
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from .data import HetGraph


def build_relation_csrs(graph: HetGraph) -> List[Tuple[np.ndarray, np.ndarray]]:
    """为 2*T_R 种关系（含逆关系）构建全局 id 空间的 CSR 邻接表。

    返回 [(indptr int64 [N+1], indices int32 [E_r])]，r >= T_R 为逆关系（转置）。
    """
    n = graph.num_nodes
    csrs = []
    for r in range(2 * graph.T_R):
        base = r if r < graph.T_R else r - graph.T_R
        ei = graph.edge_index[base].numpy()
        src, dst = (ei[0], ei[1]) if r < graph.T_R else (ei[1], ei[0])
        order = np.argsort(src, kind="stable")
        src_s, dst_s = src[order], dst[order].astype(np.int32)
        counts = np.bincount(src_s, minlength=n)
        indptr = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
        csrs.append((indptr, dst_s))
    return csrs


def _hop(cur_s: np.ndarray, cur_n: np.ndarray, indptr: np.ndarray,
         indices: np.ndarray, fanout: int, rng: np.random.Generator,
         n_nodes: int) -> Tuple[np.ndarray, np.ndarray]:
    """frontier 推进一跳：每个当前节点有放回抽 fanout 个后继，按 (节点,后继) 去重。

    cur_s: [F] 起点局部 id；cur_n: [F] 当前节点全局 id。
    返回新的 (cur_s, cur_n)。有放回抽样 + 去重与均匀无放回近似一致，
    是邻居采样的标准工程做法（避免为 hub 节点物化全部邻居）。
    """
    deg = indptr[cur_n + 1] - indptr[cur_n]
    keep = deg > 0
    cur_s, cur_n, deg = cur_s[keep], cur_n[keep], deg[keep]
    if cur_n.size == 0:
        return cur_s, cur_n
    k = np.minimum(deg, fanout)
    draws = (rng.random((cur_n.size, fanout)) * deg[:, None]).astype(np.int64)  # [F, fanout]
    valid = np.arange(fanout)[None, :] < k[:, None]
    nb = indices[indptr[cur_n][:, None] + draws].astype(np.int64)  # [F, fanout]
    fs = np.repeat(cur_s, fanout)[valid.ravel()]
    fc = np.repeat(cur_n, fanout)[valid.ravel()]
    fn = nb.ravel()[valid.ravel()]
    # 按 (当前节点, 后继) 去重，避免同一后继被重复推进
    key = fc * n_nodes + fn
    _, first = np.unique(key, return_index=True)
    return fs[first], fn[first]


def sampled_path_edges(graph: HetGraph, rel_seq, rel_csrs, fanout: int,
                       seed: int = 0, chunk: int = 10000,
                       max_starts: int = None,
                       max_end_per_start: int = 64,
                       max_deg: int = 8) -> Tuple[np.ndarray, np.ndarray]:
    """单条元路径的采样复合邻接（CSR-by-dst）。

    rel_seq: 关系 id 序列（可含逆关系 id >= T_R）；语义与原版一致——从目标类型
    节点出发沿 rel_seq 到达元路径邻居（两端均为目标类型）。
    max_starts: 仅做存在性检查时限制起点数（None=全部目标节点）。
    max_end_per_start: 每个起点最多保留的终点数（防多跳 fanout^L 膨胀）。
    max_deg: 每个目标（dst）节点最多保留的元路径邻居数（train 语义上限）。

    返回 (indptr [N_t+1] int64, src [E] int64 全局 id)。
    """
    rng = np.random.default_rng(seed)
    n = graph.num_nodes
    tgt = graph.target_global_ids().numpy()
    n_t = tgt.numel() if hasattr(tgt, "numel") else len(tgt)
    if max_starts is not None and len(tgt) > max_starts:
        tgt = rng.choice(tgt, size=max_starts, replace=False)
        tgt = np.sort(tgt)
    pos = np.full(n, -1, dtype=np.int64)      # 全局 id -> 目标局部 id（非目标类型为 -1）
    pos[tgt] = np.arange(len(tgt))

    all_src, all_dst = [], []
    for c0 in range(0, len(tgt), chunk):
        starts = tgt[c0:c0 + chunk]
        cur_s = pos[starts]                    # 起点局部 id
        cur_n = starts.astype(np.int64)        # 当前节点全局 id
        for r in rel_seq:
            indptr, indices = rel_csrs[r]
            cur_s, cur_n = _hop(cur_s, cur_n, indptr, indices, fanout, rng, n)
            if cur_n.size == 0:
                break
        if cur_n.size == 0:
            continue
        # 终点须为目标类型
        tl = pos[cur_n]
        keep = tl >= 0
        cur_s, tl = cur_s[keep], tl[keep]
        if cur_s.size == 0:
            continue
        # (起点, 终点) 去重
        key = cur_s * n + tl
        _, first = np.unique(key, return_index=True)
        cur_s, tl = cur_s[first], tl[first]
        # 每起点终点数上限
        order = np.argsort(cur_s, kind="stable")
        cur_s, tl = cur_s[order], tl[order]
        uniq, st_idx, cnts = np.unique(cur_s, return_index=True, return_counts=True)
        sel = []
        for u, st, c in zip(uniq, st_idx, cnts):
            if c > max_end_per_start:
                sel.append(st + rng.choice(c, max_end_per_start, replace=False))
            else:
                sel.append(np.arange(st, st + c))
        sel = np.concatenate(sel)
        all_src.append(tgt[cur_s[sel]])          # src = 元路径起点（全局 id）
        all_dst.append(tl[sel])
    if not all_src:
        return np.zeros(n_t + 1, dtype=np.int64), np.zeros(0, dtype=np.int64)
    src_g = np.concatenate(all_src).astype(np.int64)
    dst_l = np.concatenate(all_dst).astype(np.int64)
    # (dst, src) 全局去重后按 dst 分组
    key = dst_l * n + src_g
    _, first = np.unique(key, return_index=True)
    src_g, dst_l = src_g[first], dst_l[first]
    order = np.argsort(dst_l, kind="stable")
    src_g, dst_l = src_g[order], dst_l[order]
    counts = np.bincount(dst_l, minlength=n_t)
    # 每 dst 邻居上限 max_deg：超出随机保留（向量化：组内随机键排序取前 max_deg）
    if (counts > max_deg).any():
        rng2 = np.random.default_rng(seed + 1)
        rand = rng2.random(src_g.size)
        order2 = np.lexsort((rand, dst_l))           # 组内按随机键排序
        start_of = np.searchsorted(dst_l[order2], dst_l[order2], side="left")
        rank = np.arange(src_g.size) - start_of       # 组内名次
        keep = order2[rank < max_deg]
        keep.sort()
        src_g, dst_l = src_g[keep], dst_l[keep]
        counts = np.bincount(dst_l, minlength=n_t)
    indptr = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    return indptr, src_g


def build_candidate_edges(graph: HetGraph, rel_seqs, shortlist, fanout: int,
                          max_deg: int, seed: int = 0,
                          cache: Dict[int, Tuple[np.ndarray, np.ndarray]] = None):
    """为 shortlist 中的候选构建/复用采样复合邻接（按候选 id 缓存）。"""
    if cache is None:
        cache = {}
    rel_csrs = build_relation_csrs(graph)
    out = []
    for i in shortlist.tolist():
        if i not in cache:
            cache[i] = sampled_path_edges(graph, rel_seqs[i], rel_csrs,
                                          fanout=fanout, max_deg=max_deg, seed=seed + i)
        out.append(cache[i])
    return out, cache


def path_has_instance(graph: HetGraph, rel_seq, rel_csrs, fanout: int = 3,
                      max_starts: int = 512, seed: int = 0) -> bool:
    """存在性检查（prune_no_instance 的大图替代）：采样少量起点判断是否非空。"""
    indptr, src = sampled_path_edges(graph, rel_seq, rel_csrs, fanout=fanout,
                                     seed=seed, max_starts=max_starts,
                                     max_end_per_start=8, max_deg=8)
    return src.size > 0
