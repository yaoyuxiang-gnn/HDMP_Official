"""数据模块：异构图数据结构、合成图生成、邻接与路径实例预计算。

全局节点 id 空间：所有类型的节点共享一个连续 id 区间，
通过 type_of_node / local_id / type_offsets 在全局 id 与 (类型, 局部 id) 之间换算。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import scipy.sparse as sp
import torch


@dataclass
class HetGraph:
    """异构图内存表示。"""

    node_types: List[str]               # 节点类型名
    edge_types: List[str]               # 边类型名
    directed: List[bool]                # 每种边类型是否有方向
    type_of_node: torch.LongTensor      # [N] 每个节点的类型 id
    local_id: torch.LongTensor          # [N] 节点在其类型内的局部 id
    type_offsets: torch.LongTensor      # [T_A + 1] 各类型在全局 id 空间的起始偏移
    features: List[torch.Tensor]        # 每种类型 [N_a, d_a]
    edge_index: List[torch.LongTensor]  # 每种关系 [2, E_r]，全局 id
    target_type: int                    # 下游任务的目标节点类型
    labels: torch.LongTensor            # [N_target]
    num_classes: int
    train_mask: torch.BoolTensor        # [N_target]
    val_mask: torch.BoolTensor
    test_mask: torch.BoolTensor

    @property
    def num_nodes(self) -> int:
        return int(self.type_of_node.size(0))

    @property
    def T_A(self) -> int:
        return len(self.node_types)

    @property
    def T_R(self) -> int:
        return len(self.edge_types)

    def target_global_ids(self) -> torch.LongTensor:
        return torch.nonzero(self.type_of_node == self.target_type, as_tuple=False).view(-1)

    def schema_triples(self):
        """从边列表提取模式图三元组 (src_type, rel, dst_type)。"""
        triples = set()
        for r, ei in enumerate(self.edge_index):
            if ei.numel() == 0:
                continue
            pair = torch.stack([self.type_of_node[ei[0]], self.type_of_node[ei[1]]])
            for a, b in pair.unique(dim=1).t().tolist():
                triples.add((a, r, b))
        return sorted(triples)


def _split_masks(n: int, seed: int = 0):
    """60/20/20 划分目标节点。"""
    g = np.random.default_rng(seed)
    perm = g.permutation(n)
    n_tr, n_va = int(0.6 * n), int(0.2 * n)
    tr = np.zeros(n, bool); va = np.zeros(n, bool); te = np.zeros(n, bool)
    tr[perm[:n_tr]] = True; va[perm[n_tr:n_tr + n_va]] = True; te[perm[n_tr + n_va:]] = True
    return (torch.from_numpy(tr), torch.from_numpy(va), torch.from_numpy(te))


def make_synthetic_graph(seed: int = 0) -> HetGraph:
    """生成可学习的合成异构图（用于无数据依赖的自测）。
    """
    g = torch.Generator().manual_seed(seed)
    rng = np.random.default_rng(seed)
    n_t, n_1, n_2 = 300, 240, 180
    d0, d1, d2, nc = 24, 32, 16, 3

    prototypes = torch.randn(nc, d0, generator=g) * 1.2
    labels = torch.randint(0, nc, (n_t,), generator=g)

    # target 自身特征：类别原型 + 中等噪声（探针约 0.7~0.85，留出聚合提升空间）
    f0 = prototypes[labels] + 0.8 * torch.randn(n_t, d0, generator=g)
    f1 = torch.randn(n_1, d1, generator=g)                 # aux1：纯结构桥梁，无类别信息
    f2 = torch.randn(n_2, d2, generator=g)

    # 每个类别分配一组“类别桥梁” aux1 节点；同类 target 共享桥梁以形成同配两跳边
    bridges_per_class = n_1 // nc                            # 每类 80 个桥梁
    class_bridge = {c: list(range(c * bridges_per_class, (c + 1) * bridges_per_class)) for c in range(nc)}

    # R0: target -> aux1。每个 target 主要连接其类别的桥梁（同配），少量随机噪声边。
    s, d = [], []
    for v in range(n_t):
        c = int(labels[v])
        n_signal = int(rng.integers(3, 6))                   # 3~5 条同类桥梁边
        sig = rng.choice(class_bridge[c], size=min(n_signal, bridges_per_class), replace=False)
        n_noise = int(rng.integers(0, 3))                    # 0~2 条跨类噪声边
        noise = rng.integers(0, n_1, size=n_noise) if n_noise else []
        nbrs = list(sig) + list(noise)
        s.extend([v] * len(nbrs)); d.extend(nbrs)
    e0 = torch.tensor([s, d], dtype=torch.long)
    # R0 边转换到全局 id 空间：target 源偏移 0，aux1 目标偏移 n_t
    e0 = e0 + torch.tensor([[0], [n_t]])
    # R1: 1->2、R2: 0->2 随机连边（局部 id）
    e1 = torch.stack([torch.randint(0, n_1, (900,), generator=g),
                      torch.randint(0, n_2, (900,), generator=g)])
    e2 = torch.stack([torch.randint(0, n_t, (700,), generator=g),
                      torch.randint(0, n_2, (700,), generator=g)])

    counts = [n_t, n_1, n_2]
    offsets = torch.tensor([0, n_t, n_t + n_1, n_t + n_1 + n_2])
    type_of_node = torch.cat([torch.full((c,), a, dtype=torch.long) for a, c in enumerate(counts)])
    local_id = torch.cat([torch.arange(c) for c in counts])
    # R1/R2 的边需在全局 id 空间：类型-1 偏移 n_t，类型-2 偏移 n_t+n_1
    e1 = e1 + torch.tensor([[n_t], [n_t + n_1]])
    e2 = e2 + torch.tensor([[0], [n_t + n_1]])

    tr, va, te = _split_masks(n_t, seed)
    return HetGraph(
        node_types=["target", "aux1", "aux2"],
        edge_types=["r0", "r1", "r2"],
        directed=[True, True, True],
        type_of_node=type_of_node, local_id=local_id, type_offsets=offsets,
        features=[f0, f1, f2], edge_index=[e0, e1, e2],
        target_type=0, labels=labels, num_classes=nc,
        train_mask=tr, val_mask=va, test_mask=te,
    )


def relation_adjacency(graph: HetGraph, r: int) -> sp.csr_matrix:
    """第 r 种关系的全局邻接矩阵（scipy CSR）。

    r >= T_R 表示逆关系（r - T_R 的反向），取其邻接的转置。
    """
    n = graph.num_nodes
    if r >= graph.T_R:
        A = relation_adjacency(graph, r - graph.T_R)
        return A.transpose().tocsr()
    ei = graph.edge_index[r].numpy()
    return sp.coo_matrix((np.ones(ei.shape[1]), (ei[0], ei[1])), shape=(n, n)).tocsr()


def composite_adjacency(graph: HetGraph, rel_seq) -> sp.csr_matrix:
    """元路径（关系 id 序列）的复合邻接 A_P = prod A_Ri，二值化。"""
    A = relation_adjacency(graph, rel_seq[0])
    for r in rel_seq[1:]:
        A = A @ relation_adjacency(graph, r)
    A.data[:] = 1.0
    A.eliminate_zeros()
    return A.tocsr()


def sample_path_instances(graph: HetGraph, rel_seqs, S: int, l_max: int, seed: int = 0,
                          max_starts: int = None):
    """为每条元路径采样至多 S 条路径实例（随机游走）。

    返回 inst_idx [K, S, l_max+1]（全局 id）、inst_mask [K, S, l_max+1]（bool）。
    采样失败的槽位 mask 全 False，模型端安全跳过。

    max_starts（extended work，默认 None）：大图加速——每条候选最多扫描
    max_starts 个候选起点后即停（凑够 S 条实例通常远早于此），避免在
    大图上对每条候选遍历全部节点。仅影响实例采样数量，不影响模型主体。
    """
    rng = np.random.default_rng(seed)
    n = graph.num_nodes
    # 构建 2*T_R 种关系的前向邻接表：r < T_R 为原关系，r >= T_R 为其逆关系
    adj = []
    for r in range(2 * graph.T_R):
        base = r if r < graph.T_R else r - graph.T_R
        ei = graph.edge_index[base].numpy()
        # 用 numpy 直接构建 CSR 邻接，比逐边 append 的纯 Python list 快得多（大图关键）
        if r < graph.T_R:
            src, dst = ei[0], ei[1]
        else:
            src, dst = ei[1], ei[0]
        order = np.argsort(src, kind="stable")
        src_s, dst_s = src[order], dst[order]
        counts = np.bincount(src_s, minlength=n)
        indptr = np.concatenate([[0], np.cumsum(counts)])
        adj.append((indptr.astype(np.int64), dst_s.astype(np.int64)))

    K = len(rel_seqs)
    inst_idx = np.zeros((K, S, l_max + 1), dtype=np.int64)
    inst_mask = np.zeros((K, S, l_max + 1), dtype=bool)
    for k, rels in enumerate(rel_seqs):
        indptr0, idx0 = adj[rels[0]]
        starts = np.nonzero(np.diff(indptr0) > 0)[0]
        rng.shuffle(starts)
        if max_starts is not None:
            starts = starts[:max_starts]
        cnt = 0
        for v in starts:
            if cnt >= S:
                break
            inst, cur, ok = [v], v, True
            for r in rels:
                indptr_r, idx_r = adj[r]
                lo, hi = indptr_r[cur], indptr_r[cur + 1]
                if hi <= lo:
                    ok = False
                    break
                nxts = idx_r[lo:hi]
                cur = int(rng.choice(nxts))
                inst.append(cur)
            if ok:
                inst_idx[k, cnt, : len(inst)] = inst
                inst_mask[k, cnt, : len(inst)] = True
                cnt += 1
    return torch.from_numpy(inst_idx), torch.from_numpy(inst_mask)


def _edges_equal_flip(ei1: torch.Tensor, ei2: torch.Tensor) -> bool:
    """判断 ei1 的边集是否与 ei2 的翻转边集相同（即两种边类型互逆）。"""
    if ei1.shape != ei2.shape:
        return False
    e1 = set(map(tuple, ei1.t().tolist()))
    e2 = set(map(tuple, ei2.flip(0).t().tolist()))
    return e1 == e2


def load_hgb(name: str, root: str = "./data", val_ratio: float = 0.2, seed: int = 0) -> HetGraph:
    """加载 HGB 数据集（ACM/DBLP/IMDB）并转换为 HetGraph。
    """
    try:
        from torch_geometric.datasets import HGBDataset  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "加载 HGB 需要 torch_geometric：pip install torch_geometric"
        ) from exc

    data = HGBDataset(root=root, name=name)[0]
    node_types: List[str] = list(data.node_types)
    type_idx = {t: i for i, t in enumerate(node_types)}

    # ---- 目标类型（唯一携带 y 的类型）与标签 ----
    target_name = next(t for t in node_types if data[t].get("y", None) is not None)
    target_type = type_idx[target_name]
    y = data[target_name].y
    if y.dim() == 2:  # one-hot -> 类别索引（IMDB）
        y = y.argmax(dim=1)
    num_classes = int(y.max().item()) + 1

    # ---- 合并互逆边对为逻辑关系 ----
    # raw: [(src_type, rel_name, dst_type, edge_index)]
    raw = [(et[0], et[1], et[2], data[et].edge_index) for et in data.edge_types]
    used = [False] * len(raw)
    logical = []  # (src_type_idx, dst_type_idx, edge_index_local, rel_name)
    for i, (a, r1, b, ei1) in enumerate(raw):
        if used[i]:
            continue
        used[i] = True
        # 查找互逆对 (b, r2, a)
        for j in range(i + 1, len(raw)):
            a2, r2, b2, ei2 = raw[j]
            if used[j] or a2 != b or b2 != a:
                continue
            if _edges_equal_flip(ei1, ei2):
                used[j] = True
                break
        logical.append((type_idx[a], type_idx[b], ei1, f"{a}-{r1}-{b}"))

    ONEHOT_MAX = 20000
    RANDOM_FEAT_DIM = 128
    features: List[torch.Tensor] = []
    for t in node_types:
        x = data[t].get("x", None)
        if x is None:
            n = data[t].num_nodes
            if n <= ONEHOT_MAX:
                x = torch.eye(n)
            else:
                g = torch.Generator().manual_seed(0)
                x = torch.randn(n, RANDOM_FEAT_DIM, generator=g)
        x = x.float()
        x = x / x.norm(dim=1, keepdim=True).clamp(min=1e-12)
        features.append(x)

    counts = [data[t].num_nodes for t in node_types]
    offsets = [0]
    for c in counts:
        offsets.append(offsets[-1] + c)
    type_offsets = torch.tensor(offsets)
    type_of_node = torch.cat([torch.full((c,), a, dtype=torch.long) for a, c in enumerate(counts)])
    local_id = torch.cat([torch.arange(c) for c in counts])

    edge_index: List[torch.Tensor] = []
    edge_names: List[str] = []
    for src_a, dst_b, ei, rel_name in logical:
        off = torch.tensor([[type_offsets[src_a].item()], [type_offsets[dst_b].item()]])
        edge_index.append(ei + off)
        edge_names.append(rel_name)

    train_mask = data[target_name].train_mask.clone()
    test_mask = data[target_name].test_mask.clone()
    g = np.random.default_rng(seed)
    val_mask = torch.zeros_like(train_mask)
    tr_idx = torch.nonzero(train_mask, as_tuple=False).view(-1).numpy()
    for c in range(num_classes):
        c_idx = tr_idx[y.numpy()[tr_idx] == c]
        n_val = max(1, int(round(len(c_idx) * val_ratio)))
        chosen = g.choice(c_idx, size=n_val, replace=False)
        val_mask[torch.from_numpy(chosen)] = True
    train_mask = train_mask & ~val_mask

    return HetGraph(
        node_types=node_types,
        edge_types=edge_names,
        directed=[True] * len(edge_names),   # 逆关系由枚举层派生
        type_of_node=type_of_node, local_id=local_id, type_offsets=type_offsets,
        features=features, edge_index=edge_index,
        target_type=target_type, labels=y, num_classes=num_classes,
        train_mask=train_mask, val_mask=val_mask, test_mask=test_mask,
    )


def load_ogbn_mag(root: str = "./data/ogbn_mag", embed_dim: int = 64, seed: int = 0) -> HetGraph:
    """加载 ogbn-mag（OGB 大规模异构引用网络）并转换为 HetGraph。
    图结构：paper / author / institution / field_of_study 四类节点，
    关系含 author-writes-paper、paper-cites-paper、author-affiliated-institution、
    paper-has_topic-field。目标类型 paper，标签为 venue（349 类），按年份官方划分。
    """
    from torch_geometric.datasets import OGB_MAG  # type: ignore

    data = OGB_MAG(root=root, preprocess="metapath2vec")[0]
    node_types: List[str] = list(data.node_types)
    type_idx = {t: i for i, t in enumerate(node_types)}

    target_name = "paper"
    target_type = type_idx[target_name]
    y = data[target_name].y.view(-1)
    num_classes = int(y.max().item()) + 1

    # ---- 特征：paper 用自带 768 维，其他类型用固定维随机嵌入 ----
    g = torch.Generator().manual_seed(seed)
    features: List[torch.Tensor] = []
    for t in node_types:
        x = data[t].get("x", None)
        if x is None:
            n = data[t].num_nodes
            x = torch.randn(n, embed_dim, generator=g) * 0.1   # 可学习投影的随机初始化
        x = x.float()
        x = x / x.norm(dim=1, keepdim=True).clamp(min=1e-12)
        features.append(x)

    raw = [(et[0], et[1], et[2], data[et].edge_index) for et in data.edge_types]
    used = [False] * len(raw)
    logical = []
    for i, (a, r1, b, ei1) in enumerate(raw):
        if used[i]:
            continue
        used[i] = True
        for j in range(i + 1, len(raw)):
            a2, r2, b2, ei2 = raw[j]
            if used[j] or a2 != b or b2 != a:
                continue
            if _edges_equal_flip(ei1, ei2):
                used[j] = True
                break
        logical.append((type_idx[a], type_idx[b], ei1, f"{a}-{r1}-{b}"))

    counts = [data[t].num_nodes for t in node_types]
    offsets = [0]
    for c in counts:
        offsets.append(offsets[-1] + c)
    type_offsets = torch.tensor(offsets)
    type_of_node = torch.cat([torch.full((c,), a, dtype=torch.long) for a, c in enumerate(counts)])
    local_id = torch.cat([torch.arange(c) for c in counts])

    edge_index: List[torch.Tensor] = []
    edge_names: List[str] = []
    for src_a, dst_b, ei, rel_name in logical:
        off = torch.tensor([[type_offsets[src_a].item()], [type_offsets[dst_b].item()]])
        edge_index.append(ei + off)
        edge_names.append(rel_name)

    # ---- OGB 官方划分（按年份）----
    train_mask = data[target_name].train_mask
    val_mask = data[target_name].val_mask
    test_mask = data[target_name].test_mask

    return HetGraph(
        node_types=node_types,
        edge_types=edge_names,
        directed=[True] * len(edge_names),
        type_of_node=type_of_node, local_id=local_id, type_offsets=type_offsets,
        features=features, edge_index=edge_index,
        target_type=target_type, labels=y, num_classes=num_classes,
        train_mask=train_mask, val_mask=val_mask, test_mask=test_mask,
    )
