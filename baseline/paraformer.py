"""ParaFormer (WSDM 2026)：Generalized PageRank Graph Transformer 同构基线。
"""
import math
from .common import *


class ParaFormerNet(nn.Module):
    """ParaFormer：广义 PageRank 注意力（GPA）+ 辅助 GCN 融合的同构图 Transformer。
    输入为目标类型子图特征 x[N, F] 与 edge_index ei（common.to_target_homo 产出，
    ei 可为 0 条边的空图，此时辅助 GCN 退化为含自环的线性变换，前向不报错）。
    小图（N ≤ scalable_threshold）走精确 GPA，大图走 S-GPA 线性注意力近似。"""

    def __init__(self, din, dh, nc, K=10, beta=0.5, dropout=0.5,
                 scalable_threshold=8192, init_alpha=0.3):
        super().__init__()
        self.K = K                      # GPR 截断阶数（论文/官方默认 10）
        self.beta = beta                # 局部（GCN）/ 全局（GPA）融合权重
        self.dropout = dropout
        self.scalable_threshold = scalable_threshold
        # 1. 输入投影（官方：fc -> LayerNorm -> ReLU -> Dropout）
        self.in_proj = nn.Linear(din, dh)
        self.ln_in = nn.LayerNorm(dh)
        # 2. GPA 的 Q/K/V 投影（精确路径与 S-GPA 路径共享参数）
        self.w_q = nn.Linear(dh, dh)
        self.w_k = nn.Linear(dh, dh)
        self.w_v = nn.Linear(dh, dh)
        # GPR 权重 γ（可学习标量）：官方 PPR 初始化
        # γ_k = α(1-α)^k（k=0..K-1），γ_K = (1-α)^K
        temp = init_alpha * (1.0 - init_alpha) ** np.arange(K + 1)
        temp[-1] = (1.0 - init_alpha) ** K
        self.gamma = nn.Parameter(torch.tensor(temp, dtype=torch.float))
        # 4. GPA 输出后的 LayerNorm（官方 use_bn=True）
        self.ln_out = nn.LayerNorm(dh)
        # 5. 辅助 GCN 分支：独立输入投影 + 2 层 GCNConv（残差）
        self.gnn_proj = nn.Linear(din, dh)
        self.gcn1 = GCNConv(dh, dh)
        self.gcn2 = GCNConv(dh, dh)
        # 6. 分类头
        self.head = nn.Linear(dh, nc)

    def _gpa_exact(self, q, k, v):
        """精确 GPA：Â = softmax(QKᵀ/√dh)（行归一），Z = Σ_k γ_k·ÂᵏV。
        迭代计算 ÂᵏV = Â @ (Âᵏ⁻¹V)，绝不对 [N,N] 矩阵显式求幂。"""
        dh = q.size(-1)
        a = torch.softmax(q @ k.transpose(0, 1) / math.sqrt(dh), dim=-1)  # [N, N]
        v0 = F.dropout(F.relu(v), self.dropout, self.training)   # 官方：k=0 项
        z = self.gamma[0] * v0             # Â⁰V = V
        p = v
        for i in range(1, self.K + 1):
            p = a @ p                      # p = ÂⁱV
            # 官方对每步传播结果施加 Dropout
            z = z + self.gamma[i] * F.dropout(p, self.dropout, self.training)
        return z

    def _gpa_scalable(self, q, k, v):
        """S-GPA 线性注意力近似（论文 Algorithm 1 / Eq.13-15）：
        Q̃ = softmax(Q, dim=-1)（按特征维），K̃ = softmax(K, dim=0)（按节点维，
        与官方代码一致的 Shen et al. 归一化方式），预计算 G = K̃ᵀQ̃ ∈ [dh, dh]、
        M₀ = K̃ᵀV，则 ÂᵏV = Q̃(K̃ᵀQ̃)ᵏ⁻¹(K̃ᵀV)（=(Q̃K̃ᵀ)ᵏV，结合律）；
        迭代 M ← G·M、Z += γ_k·Q̃M。k=0 项恒等（Â⁰V = V），直接用 V。
        K̃ 按节点维归一是数值稳定的关键（谱半径 O(1)，幂次不爆炸）。
        注：迭代顺序按论文公式（左乘 G）；官方 gpr_attention.py 的 einsum
        顺序与之等价地覆盖同一近似族，此处以论文公式为准。"""
        qt = torch.softmax(q, dim=-1)      # [N, dh]（按特征维归一）
        kt = torch.softmax(k, dim=0)       # [N, dh]（按节点维归一，官方 dim=0）
        g = kt.transpose(0, 1) @ qt        # G = K̃ᵀQ̃ ∈ [dh, dh]
        m = kt.transpose(0, 1) @ v         # M₀ = K̃ᵀV ∈ [dh, dh]
        v0 = F.dropout(F.relu(v), self.dropout, self.training)
        z = self.gamma[0] * v0             # k=0 项：Â⁰V = V
        for i in range(1, self.K + 1):
            # 第 i 项：Q̃Gⁱ⁻¹M₀（官方对传播结果施加 Dropout）
            z = z + self.gamma[i] * F.dropout(qt @ m, self.dropout, self.training)
            m = g @ m                          # M ← G·M
        return z

    def forward(self, x, ei):
        n = x.size(0)
        # 1. 输入投影（官方：fc -> LayerNorm -> ReLU -> Dropout）
        h = F.dropout(F.relu(self.ln_in(self.in_proj(x))),
                      self.dropout, self.training)
        # 2/3. GPA：大图走 S-GPA 线性近似，小图走精确注意力
        q, k, v = self.w_q(h), self.w_k(h), self.w_v(h)
        if n > self.scalable_threshold:
            z = self._gpa_scalable(q, k, v)
        else:
            z = self._gpa_exact(q, k, v)
        # 4. 残差 + LayerNorm + ReLU + Dropout（官方：0.5 残差系数）
        z = 0.5 * z + 0.5 * h
        z = F.dropout(F.relu(self.ln_out(z)), self.dropout, self.training)
        # 5. 辅助 GCN 融合：Ẑ = (1-β)·Z + β·GCN(x, ei)（空图时 GCN 仅剩自环，仍有效）
        g = F.dropout(F.relu(self.gnn_proj(x)), self.dropout, self.training)
        g1 = F.dropout(F.relu(self.gcn1(g, ei)), self.dropout, self.training)
        g = self.gcn2(g1, ei) + g          # 残差（官方 gnn_use_residual）
        z = (1.0 - self.beta) * z + self.beta * g
        # 6. 分类头
        return self.head(z)
