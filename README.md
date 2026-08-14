# 项目简介

论文《Learning to Choose the Relations That Matter: Differentiable Meta-Path Discovery for Heterogeneous Graph Neural Networks》的 Official PyTorch 实现。


## 目录结构
```
├── hdmp/                # 方法核心包
│   ├── data.py          # HetGraph 数据结构、HGB/OGBN-MAG 加载、实例采样
│   ├── data_subset.py   # 大图诱导子图采样（可扩展性实验）
│   ├── enumeration.py   # 候选枚举、剪枝、层次关系构建
│   ├── encoding.py      # Transformer 语义编码、实例上下文
│   ├── attention.py     # 双层注意力
│   ├── selection.py     # top-ρ 评分、Gumbel 门控、级联粗筛、分层配额 shortlist
│   ├── losses.py        # 层次一致性正则、logits 尺度稳定正则
│   ├── mag_edges.py     # MAG 候选复合邻接构建（向量化采样）
│   └── model.py         # HDMPModel 端到端编排
├── configs/             # 数据集配置 + ablation/ scalability/ 分组
├── train.py             # 统一训练入口（小数据集全量；OGBN-MAG 委托 train_mag）
├── train_mag.py         # 大图 mini-batch 训练器（由 train.py 调用，不单独运行）
└── requirements.txt
```

## 快速开始

```bash
pip install -r requirements.txt

# 合成图自测（无需下载数据，验证全流程）
python train.py --synthetic --epochs 30

# 真实数据集（HGB：acm / dblp / imdb / freebase）
python train.py --config configs/acm.yaml
python analysis/evaluate.py --config configs/acm.yaml --checkpoint checkpoints/hdmp_acm_seed0.pt

# 大图 OGBN-MAG（mini-batch，自动委托 train_mag）
python train.py --config configs/ogbn_mag.yaml
```
