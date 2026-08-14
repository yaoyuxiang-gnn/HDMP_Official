"""DiffMG (KDD 2021)。"""
from .common import *


def build_full_graph(g):
    """把异构图转成 DiffMG 的全图视角：所有节点类型拼接成 N×N 大图，
    每种边类型一张行归一化稀疏邻接 (A_r + I)，末尾追加 I(skip) 和 O(zero)。
    返回 (feats[N,F], types[N], adjs[list of sparse], node_slices{tgt: (start,end)})。"""
    node_types = list(g.node_types)
    sizes = [g[nt].num_nodes for nt in node_types]
    offsets = np.cumsum([0] + sizes).tolist()          # 各类型在大图中的起始下标
    N = sum(sizes)
    # 特征不拼接（各类型维度不同），用列表；类型 id 拼接
    feat_list = [g[nt].x for nt in node_types]         # 各类型特征 [N_t, F_t]
    types = torch.cat([torch.full((sizes[i],), i, dtype=torch.long) for i in range(len(node_types))])
    node_slices = {nt: (offsets[i], offsets[i] + sizes[i]) for i, nt in enumerate(node_types)}

    # 每种边类型一张 N×N 稀疏邻接（含自环 + 行归一化）
    adjs = []
    I_sp = torch.sparse_coo_tensor(torch.arange(N).repeat(2, 1), torch.ones(N), (N, N)).coalesce()
    for et in g.edge_types:
        s, r, d = et
        ei = g[et].edge_index.clone()
        ei[0] += node_slices[s][0]                      # 源节点全局下标
        ei[1] += node_slices[d][0]                      # 目标节点全局下标
        ei = torch.cat([ei, torch.arange(N).repeat(2, 1)], dim=1)  # + 自环
        vals = torch.ones(ei.size(1))
        A = torch.sparse_coo_tensor(ei, vals, (N, N)).coalesce()
        # 行归一化（按目标端 degree）
        deg = torch.zeros(N).index_add_(0, A.indices()[1], A.values())
        deg = deg.clamp(min=1)
        w = A.values() / deg[A.indices()[1]]
        adjs.append(torch.sparse_coo_tensor(A.indices(), w, (N, N)).coalesce())
    adjs.append(I_sp)                                   # skip op（单位阵）
    adjs.append(torch.sparse_coo_tensor(torch.empty(2, 0, dtype=torch.long), torch.empty(0), (N, N)).coalesce())  # zero op
    return feat_list, types, adjs, node_slices, node_types


class DiffMGCell(nn.Module):
    """DiffMG 的 DAG Cell：H(k) = Σ_{i<k} f̄(H(i))，每条 link 沿选中边类型 spmm。"""
    def __init__(self, n_adjs, cstr, K):
        super().__init__()
        self.K = K
        self.cstr = cstr                                # 任务相关边类型下标（含 identity）
        self.n_adjs = n_adjs

    def forward(self, h0, adjs, msg_seq, msg_res):
        states = [h0]
        off = 0
        for k in range(self.K):
            seq = torch.spmm(adjs[msg_seq[k]], states[k])
            res = 0
            for j in range(k):
                res = res + torch.spmm(adjs[msg_res[off + j]], states[j])
            off += k
            states.append(seq + res)
        return states[self.K]




class DiffMGNet(nn.Module):
    """DiffMG：类型特定投影 + DAG Cell（导出固定的 meta graph）+ LayerNorm + 分类头。
    本实现：搜索由独立的 search 函数完成，此处用导出的固定 msg_seq/msg_res 前向。"""
    def __init__(self, g, tgt, nc, node_types, K=4, dh=64):
        super().__init__()
        self.tgt = tgt
        self.node_types = node_types
        self.K = K
        self.ws = nn.ModuleList([nn.Linear(g[nt].x.size(1), dh) for nt in node_types])
        self.affine = nn.Linear(dh, dh)
        self.ln = nn.LayerNorm(dh)
        self.out = nn.Linear(dh, nc)
        self.drop = nn.Dropout(0.5)

    def forward(self, feat_list, types, adjs, cell, msg_seq, msg_res, node_slices, N):
        # 类型特定投影到公共空间：从 feat_list 按类型取特征，放入全图 hid 的对应切片
        dh = self.affine.out_features
        hid = torch.zeros(N, dh, device=feat_list[0].device)
        for t, nt in enumerate(self.node_types):
            s, e = node_slices[nt]
            hid[s:e] = self.ws[t](feat_list[t])
        hid = self.drop(torch.tanh(hid))
        h0 = self.affine(hid)
        z = cell(h0, adjs, msg_seq, msg_res)
        z = F.gelu(self.ln(z))
        return self.out(z)


def diffmg_search(feat_list, types, adjs, cstr, K, n_adjs, tr, va, y_tgt, tgt_slice,
                  node_types, in_dims, device, node_slices, N, dh=64, search_epochs=50, eps0=0.3, seed=0,
                  return_lams=False):
    """DiffMG 搜索阶段：bi-level 优化架构参数（ε-greedy 采样单 op），导出最优 meta graph。"""
    torch.manual_seed(seed); np.random.seed(seed)
    # 架构参数：主链 K 条（前 K-1 条全候选、最后一条 cstr），跳链 K(K-1)/2 条
    # 架构参数用 nn.Parameter（叶子节点），×1e-3 通过初始化实现
    lam_seq = [nn.Parameter(torch.randn(n_adjs - 1, device=device) * 1e-3) for _ in range(K - 1)]
    lam_seq.append(nn.Parameter(torch.randn(len(cstr), device=device) * 1e-3))
    lam_res = [nn.Parameter(torch.randn(n_adjs, device=device) * 1e-3) for _ in range(K * (K - 1) // 2)]
    # 模型权重：类型投影 + affine
    ws = nn.ModuleList([nn.Linear(in_dims[nt], dh) for nt in node_types]).to(device)
    affine = nn.Linear(dh, dh).to(device)
    params = list(ws.parameters()) + list(affine.parameters())
    opt_w = torch.optim.Adam(params, lr=5e-3, weight_decay=5e-4)
    opt_a = torch.optim.Adam(lam_seq + lam_res, lr=3e-4)
    nc_out = y_tgt.max().item() + 1
    head = nn.Linear(dh, nc_out).to(device)
    opt_w = torch.optim.Adam(params + list(head.parameters()), lr=5e-3, weight_decay=5e-4)

    ts, te = tgt_slice
    eps = eps0
    for ep in range(search_epochs):
        # ε-greedy 采样：主链前 K-1 从全候选（排 zero），最后一条从 cstr；跳链全候选
        def sample_op(lam, n_cand):
            if np.random.uniform() < eps:
                return np.random.randint(n_cand)
            return int(torch.argmax(torch.softmax(lam, dim=0)).item())
        idx_seq = [sample_op(lam_seq[k], n_adjs - 1) for k in range(K - 1)]
        idx_seq.append(sample_op(lam_seq[K - 1], len(cstr)))
        idx_res = [sample_op(lam_res[i], n_adjs) for i in range(K * (K - 1) // 2)]

        def fwd():
            hid = torch.zeros(N, dh, device=device)
            for t, nt in enumerate(node_types):
                s, e = node_slices[nt]
                hid[s:e] = ws[t](feat_list[t])
            hid = torch.tanh(hid)
            states = [affine(hid)]
            off = 0
            for k in range(K):
                # 主链：乘 softmax 权重（可微 argmax）
                if k < K - 1:
                    w_sel = torch.softmax(lam_seq[k], dim=0)[idx_seq[k]]
                    aidx = idx_seq[k]
                else:
                    w_sel = torch.softmax(lam_seq[K - 1], dim=0)[idx_seq[K - 1]]
                    aidx = cstr[idx_seq[K - 1]]
                seq = w_sel * torch.spmm(adjs[aidx], states[k])
                res = 0
                for j in range(k):
                    w_r = torch.softmax(lam_res[off + j], dim=0)[idx_res[off + j]]
                    res = res + w_r * torch.spmm(adjs[idx_res[off + j]], states[j])
                off += k
                states.append(seq + res)
            z = states[K]
            return head(z)[ts:te]

        # Phase 1: 更新权重（train loss）
        for o in (opt_w, opt_a): o.zero_grad()
        out = fwd()
        loss_w = F.cross_entropy(out[tr], y_tgt[tr])
        loss_w.backward()
        opt_w.step()
        # Phase 2: 更新架构参数（val loss）
        for o in (opt_w, opt_a): o.zero_grad()
        out = fwd()
        loss_a = F.cross_entropy(out[va], y_tgt[va])
        loss_a.backward()
        opt_a.step()
        eps *= 0.9

    # 导出：argmax（eps=0）
    with torch.no_grad():
        msg_seq = [int(torch.argmax(torch.softmax(lam_seq[k], dim=0)).item()) for k in range(K - 1)]
        msg_seq.append(cstr[int(torch.argmax(torch.softmax(lam_seq[K - 1], dim=0)).item())])
        msg_res = [int(torch.argmax(torch.softmax(l, dim=0)).item()) for l in lam_res]
    if return_lams:
        return msg_seq, msg_res, lam_seq, lam_res
    return msg_seq, msg_res
