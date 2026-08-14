"""Tokenphormer (AAAI 2025) 同构基线：Structure-aware Multi-token Graph Transformer。
"""
import math
from .common import *

class TokenphormerNet(nn.Module):
    """Tokenphormer：self/hop-token + 混合游走 walk-token 的多 token 图 Transformer。

    __init__ 只建参数；图相关的预计算（ÂᵏX 与游走索引）在 prepare(x, ei) 完成。
    无边图（ei 为 [2,0]）自动退化：hop-token 全等于 X、游走停在自身，前向不报错。"""

    def __init__(self, din, dh, nc, n_hop=3, n_walk=8, walk_len=6, n_layers=2, dropout=0.1):
        super().__init__()
        self.din, self.dh, self.nc = din, dh, nc
        self.n_hop = n_hop        # hop-token 数（不含 self-token），论文固定为 3
        self.n_walk = n_walk      # 每节点游走条数 m（4 种类型轮转混合）
        self.walk_len = walk_len  # 游走序列长度（含起始节点自身，位置 0）
        self.n_layers = n_layers
        self.dropout = dropout
        # 3. token 投影：self/hop 共用一个，walk 用一个
        self.proj_hop = nn.Linear(din, dh)
        self.proj_walk = nn.Linear(din, dh)
        # 4. Transformer 主干：Pre-LN，单头（论文设定），FFN 维度 4·dh
        enc_layer = nn.TransformerEncoderLayer(
            d_model=dh, nhead=1, dim_feedforward=4 * dh, dropout=dropout,
            batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers,
                                             enable_nested_tensor=False)
        # 5. readout 前的 final LayerNorm（官方代码的 final_ln）
        self.final_ln = nn.LayerNorm(dh)
        # attention readout 向量 w_a ∈ R^{dh} 与分类头
        self.w_a = nn.Parameter(torch.empty(dh))
        self.head = nn.Linear(dh, nc)
        # 6. 官方代码的参数初始化：所有 Linear 权重 ~ N(0, 0.02/√n_layers)，偏置置 0
        self.apply(self._init_params)
        nn.init.normal_(self.w_a, std=0.02 / math.sqrt(n_layers))
        # prepare 填充的缓存（CPU）：_hop [n_hop+1, N, F]、_walks [N, m, walk_len]、_x [N, F]
        self._hop = None
        self._walks = None
        self._x = None

    def _init_params(self, module):
        """官方代码（model.py 的 init_params）的初始化策略：
        nn.Linear 权重 ~ N(0, 0.02/√n_layers)，偏置置 0；小初始化抑制训练初期的
        大步长震荡（官方另配 warmup 调度器，此处只负责初始化）。"""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02 / math.sqrt(self.n_layers))
            if module.bias is not None:
                module.bias.data.zero_()

    # ---------------- 预计算：hop-token 与 walk-token ----------------
    @torch.no_grad()
    def prepare(self, x, ei, seed=0):
        """预计算 hop token（ÂᵏX，迭代 spmm）与每节点 m 条混合游走索引。

        x 在 CPU 传入即可（内部统一转 CPU float32）；ei 为 [2, E]（可为 0 条边）。
        缓存均保留在 CPU，forward 时按 batch gather 再移到参数设备。seed 控制游走生成。"""
        x = x.detach().cpu().float()
        ei = ei.detach().cpu().long()
        n = x.size(0)
        # 1) 对称归一化邻接 Â = D^{-1/2}(A+I)D^{-1/2} 的 (row, col, ew)，预计算一次复用。
        #    先去原始自环（统一由 +I 加入）并按 (row,col) 去重多重边，保证度与权重正确。
        if ei.numel() > 0:
            row, col = ei[0], ei[1]
            keep = row != col
            row, col = row[keep], col[keep]
        else:
            row = col = torch.empty(0, dtype=torch.long)
        key = torch.unique(row * n + col)
        row, col = key // n, key % n                       # 去重后的边（无向）
        ar = torch.arange(n)
        r2 = torch.cat([row, ar])
        c2 = torch.cat([col, ar])                          # A + I 的稀疏坐标
        deg = torch.bincount(r2, minlength=n).float().clamp(min=1.0)
        ew = deg[r2].pow(-0.5) * deg[c2].pow(-0.5)         # D^{-1/2}(A+I)D^{-1/2} 边权
        # 2) 迭代 spmm：H_0 = X，H_k = Â H_{k-1}（edge_index + index_add_ 手写 spmm，
        #    MPS 上 torch.sparse.mm 支持不全，统一走该路径；无边时 Â=I，H_k 恒等于 X）
        hops = [x]
        h = x
        for _ in range(self.n_hop):
            msg = h[c2] * ew.unsqueeze(-1)
            nxt = torch.zeros_like(h)
            nxt.index_add_(0, r2, msg)
            h = nxt
            hops.append(h)
        self._hop = torch.stack(hops, dim=0)               # [n_hop+1, N, F]（CPU 缓存）
        self._x = x
        # 3) 游走索引生成：邻接表用去重边、不含自环（论文消除了游走自环概率），CSR 化
        order = torch.argsort(row)
        nbr = col[order].numpy()
        src_sorted = row[order].numpy()
        wdeg = np.bincount(src_sorted, minlength=n).astype(np.int64)
        wstart = np.zeros(n, dtype=np.int64)
        wstart[1:] = np.cumsum(wdeg)[:-1]
        self._walks = torch.from_numpy(self._gen_walks(n, wdeg, wstart, nbr, seed))
        return self

    # ---------------- 游走生成（numpy/CPU 向量化） ----------------
    @staticmethod
    def _sample_nb(rng, cur, deg, start, nbr, forbid=None, max_retry=8):
        """对一组当前节点均匀采样邻居（向量化）。
        forbid：禁止命中的节点（如前一节点，用于非回溯）；邻居数 > 1 时重试躲避，
        度 = 1 无处可去时接受回溯。孤立节点（度 0）停在原地（cur 不变）。"""
        d = deg[cur]
        nxt = cur.copy()
        act = d > 0
        if not act.any():
            return nxt
        c = cur[act]
        da = d[act].astype(np.int64)
        sa = start[c]
        r = (rng.random(c.shape[0]) * da).astype(np.int64)
        cand = nbr[sa + r]
        if forbid is not None:
            fb = forbid[act]
            bad = (cand == fb) & (da > 1)
            for _ in range(max_retry):
                if not bad.any():
                    break
                idx = np.nonzero(bad)[0]
                r2 = (rng.random(idx.shape[0]) * da[idx]).astype(np.int64)
                cand[idx] = nbr[sa[idx] + r2]
                bad = (cand == fb) & (da > 1)
            # 重试后仍命中（邻居去重后实际只剩 forbid）：接受回溯
        nxt[act] = cand
        return nxt

    def _step_njw(self, rng, cur, prev, deg, start, nbr, nonback, max_retry=8):
        """NJW/NBNJW 单步：2-hop 邻域内跳跃（向量化）。
        先均匀跳到邻居 u、再从 u 均匀跳到 w（等价于均匀单步转移下的 2 步概率传播
        采样，即论文 k-step probability propagation 的 k=2 实现）。拒绝采样：
        w == cur（论文消除自环概率）；NBNJW 额外要求 w != prev（非回溯）。
        重试失败回退为 1-hop（非回溯）采样，保证非孤立节点必移动。"""
        d = deg[cur]
        nxt = cur.copy()
        act = d > 0
        if not act.any():
            return nxt
        c = cur[act]
        pv = prev[act]
        u = self._sample_nb(rng, c, deg, start, nbr)          # 1-hop（邻接表无自环，u != c）
        w = self._sample_nb(rng, u, deg, start, nbr)          # 2-hop
        bad = (w == c)
        if nonback:
            bad = bad | (w == pv)
        for _ in range(max_retry):
            if not bad.any():
                break
            u_b = self._sample_nb(rng, c[bad], deg, start, nbr)
            w_b = self._sample_nb(rng, u_b, deg, start, nbr)
            w[bad] = w_b
            bad2 = (w == c)
            if nonback:
                bad2 = bad2 | (w == pv)
            bad = bad2
        if bad.any():
            # 回退：1-hop（NBNJW 时非回溯）采样；度=1 链端可能只能回到 prev，接受
            fb = pv if nonback else None
            w[bad] = self._sample_nb(rng, c[bad], deg, start, nbr,
                                     forbid=fb[bad] if fb is not None else None)
        nxt[act] = w
        return nxt

    def _gen_walks(self, n, deg, start, nbr, seed):
        """生成全部游走索引 [N, m, walk_len]（numpy int64）。
        槽位 j 的游走类型为 j%4（轮转混合）：0=URW，1=NBRW，2=NJW，3=NBNJW；
        位置 0 恒为起始节点自身；无边节点的游走全部停在自身。"""
        rng = np.random.default_rng(seed)
        walks = np.empty((n, self.n_walk, self.walk_len), dtype=np.int64)
        walks[:, :, 0] = np.arange(n)[:, None]
        for t in range(4):
            slots = np.arange(t, self.n_walk, 4)              # 该类型占据的游走槽位
            if slots.size == 0:
                continue
            s_t = slots.size
            cur = walks[:, slots, 0].reshape(-1).copy()       # [N*s_t]（节点主序）
            prev = np.full(n * s_t, -1, dtype=np.int64)
            for s in range(1, self.walk_len):
                if t == 0:                                    # URW：均匀随机
                    nxt = self._sample_nb(rng, cur, deg, start, nbr)
                elif t == 1:                                  # NBRW：非回溯
                    nxt = self._sample_nb(rng, cur, deg, start, nbr, forbid=prev)
                elif t == 2:                                  # NJW：2-hop 邻域跳跃
                    nxt = self._step_njw(rng, cur, prev, deg, start, nbr, nonback=False)
                else:                                         # NBNJW：非回溯 NJW
                    nxt = self._step_njw(rng, cur, prev, deg, start, nbr, nonback=True)
                walks[:, slots, s] = nxt.reshape(n, s_t)
                prev, cur = cur, nxt
        return walks

    # ---------------- 前向：按节点索引子集（mini-batch） ----------------
    def forward(self, idx=None):
        """对 idx 指定的节点子集输出 logits [B, nc]；idx=None 时全量。
        依赖 prepare 的 CPU 缓存：按 batch gather 后移到参数所在设备计算。"""
        if self._hop is None or self._walks is None:
            raise RuntimeError("TokenphormerNet 需先调用 prepare(x, ei) 再 forward。")
        dev = self.head.weight.device
        if idx is None:
            idx_cpu = torch.arange(self._x.size(0))
        else:
            idx_cpu = idx.detach().cpu().long()
        b = idx_cpu.size(0)
        # self/hop token：从 [n_hop+1, N, F] 按 batch 取出 -> [B, n_hop+1, F]
        hop = self._hop[:, idx_cpu, :].permute(1, 0, 2).to(dev)
        # walk token：gather 游走经过节点的特征并沿游走维取均值 -> [B, m, F]
        wids = self._walks[idx_cpu]                           # [B, m, walk_len]
        wf = self._x[wids.reshape(-1)]
        wf = wf.view(b, self.n_walk, self.walk_len, self.din).mean(dim=2).to(dev)
        # 3. 投影并拼接为 token 序列 [B, K, dh]，K = (n_hop+1) + m
        tok = torch.cat([self.proj_hop(hop), self.proj_walk(wf)], dim=1)
        # 4. Transformer 主干 + final LayerNorm（官方代码设定）
        h = self.final_ln(self.encoder(tok))                  # [B, K, dh]
        # 5. attention readout + 分类头
        alpha = torch.softmax(h @ self.w_a, dim=1)            # [B, K]
        h_fin = (h * alpha.unsqueeze(-1)).sum(dim=1)          # [B, dh]
        return self.head(h_fin)
