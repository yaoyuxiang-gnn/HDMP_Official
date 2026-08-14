"""HDMP 训练入口。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.metrics import f1_score

from hdmp.data import HetGraph, composite_adjacency, load_hgb, make_synthetic_graph, sample_path_instances
from hdmp.enumeration import (build_hierarchy, enumerate_metapaths, pad_tokens,
                              prune_inverse_dup, prune_no_instance, rel_seq_of)
from hdmp.model import HDMPModel, cross_entropy_on_mask
from hdmp.selection import stratified_shortlist  # noqa: F401  (analysis/ 脚本经本模块转引)


def macro_f1(out, labels, mask):
    pred = out[mask].argmax(dim=-1).cpu().numpy()
    return f1_score(labels[mask].cpu().numpy(), pred, average="macro")


def all_metrics(out, labels, mask):
    """Micro-F1（单标签=Accuracy）、Macro-F1、Macro-AUC（OvR）。"""
    from sklearn.metrics import roc_auc_score
    yt = labels[mask].cpu().numpy()
    pred = out[mask].argmax(dim=-1).cpu().numpy()
    prob = torch.softmax(out[mask], dim=-1).cpu().numpy()
    nc = out.size(-1)
    try:
        auc = float(roc_auc_score(yt, prob, multi_class="ovr", average="macro", labels=list(range(nc))))
    except Exception:
        auc = float("nan")
    return {"macro_f1": float(f1_score(yt, pred, average="macro")),
            "micro_f1": float(f1_score(yt, pred, average="micro")), "auc": auc}


def prepare(graph: HetGraph, cfg: dict):
    """离线准备：枚举→剪枝→层次关系→实例采样→CLS 前缀 token 张量。

    小数据集（HGB 等）精确全量预计算；大图（OGBN-MAG）由 train_mag.prepare_mag
    用向量化采样处理，不走此函数。模型主体完全不变。
    """
    paths = enumerate_metapaths(graph, cfg["l_max"])
    paths = prune_inverse_dup(graph, paths)
    paths = prune_no_instance(graph, paths)
    if not paths:
        raise RuntimeError("剪枝后候选集为空，请放宽 l_max 或检查图连通性")
    pairs1, pairs2 = build_hierarchy(paths)
    rel_seqs = [rel_seq_of(p, graph.T_A) for p in paths]
    inst_idx, inst_mask = sample_path_instances(
        graph, rel_seqs, cfg["S"], cfg["l_max"],
        max_starts=cfg.get("inst_max_starts"))
    cls = graph.T_A + 2 * graph.T_R  # CLS 的 token id（词表含 2T_R 种关系：正向 + 逆向）
    tokens, mask = pad_tokens(paths, num_types=cls, pad_id=cls)
    cls_col = torch.full((tokens.size(0), 1), cls, dtype=torch.long)
    tokens = torch.cat([cls_col, tokens], dim=1)
    mask = torch.cat([torch.ones(tokens.size(0), 1, dtype=torch.bool), mask], dim=1)
    return paths, rel_seqs, tokens, mask, inst_idx, inst_mask, pairs1, pairs2


def build_shortlist_edges(graph: HetGraph, rel_seqs, idx, max_deg: int = 64, seed: int = 0) -> list:
    """对粗筛候选预计算复合邻接（精确全量，小数据集用），构造异构双分边索引。

    约定：ei[0] = 源节点（任意类型，保留全局 id）；ei[1] = 目标节点（目标类型，局部 id）。
    只保留目标端落在目标类型上的边。
    max_deg：每个目标节点的最大元路径邻居数（GraphSAGE 式随机采样），
    防止高稠密复合邻接（如 DBLP 的 APA，nnz 达数百万）导致内存爆炸。
    大图（OGBN-MAG）的采样邻接由 train_mag.EdgeBank / hdmp.mag_edges 处理，不走此函数。
    """
    rng = np.random.default_rng(seed)
    tgt = graph.target_global_ids()
    pos = torch.full((graph.num_nodes,), -1, dtype=torch.long)
    pos[tgt] = torch.arange(tgt.numel())
    edges = []
    for i in idx.tolist():
        A = composite_adjacency(graph, rel_seqs[i]).tocoo()
        ei = np.stack([A.row, A.col])                    # [2, E]：全局 id
        dst_local = pos[torch.from_numpy(ei[1])].numpy() # 目标端映射为局部 id
        keep = dst_local >= 0
        src_g, dst_l = ei[0][keep], dst_local[keep]
        # 按目标节点分组做邻居上限采样
        order = np.argsort(dst_l, kind="stable")
        src_g, dst_l = src_g[order], dst_l[order]
        uniq, starts = np.unique(dst_l, return_index=True)
        sel = []
        for u, st in zip(uniq, starts):
            en = st + np.searchsorted(dst_l[st:], u, side="right")
            cnt = en - st
            if cnt > max_deg:
                sel.append(st + rng.choice(cnt, size=max_deg, replace=False))
            else:
                sel.append(np.arange(st, en))
        sel = np.concatenate(sel)
        edges.append(torch.stack([torch.from_numpy(src_g[sel]), torch.from_numpy(dst_l[sel])]))
    return edges


def train_one_seed(graph: HetGraph, cfg: dict, seed: int, device: str) -> dict:
    torch.manual_seed(seed)
    # 小数据集（ACM/DBLP/IMDB/Freebase）走原 model.forward 全量前向，完全可复现。
    # 大图（OGBN-MAG）已由 run_mag 单独处理（mini-batch），不走此函数。
    def fwd(model, features, offsets, target_feat, epoch, training):
        return model(features, offsets, target_feat, epoch, training=training)
    (paths, rel_seqs, tokens, mask, inst_idx, inst_mask, pairs1, pairs2) = prepare(graph, cfg)
    K = len(paths)
    feat_dims = [f.size(1) for f in graph.features]
    model = HDMPModel(graph.T_A, 2 * graph.T_R, feat_dims, graph.num_classes,
                      K, tokens.size(1), cfg).to(device)
    model.load_prepared(tokens.to(device), mask.to(device), inst_idx.to(device),
                        inst_mask.to(device), pairs1.to(device), pairs2.to(device))

    # 初始粗筛：分层配额（按 hop 长度分层，保证短路径/专家路径进入候选空间）
    M = min(cfg["M"], K)
    coverage = inst_mask.sum(dim=(1, 2)).float()
    short = stratified_shortlist(rel_seqs, coverage, M, T_R=graph.T_R)
    ablation = cfg.get("ablation", "none")
    fixed_gate = None

    if ablation == "random":
        B = min(cfg["B"], K)
        gen = torch.Generator().manual_seed(cfg.get("seed", 0))
        short = torch.randperm(K, generator=gen)[:B]
        fixed_gate = torch.ones(len(short))
        print(f"  [T2/random] 随机选 {B} 条候选作 shortlist，门控固定全开", flush=True)
    elif ablation == "oracle":
        type_names = graph.node_types
        expert = cfg.get("expert_paths", [])
        expert_sets = {tuple(t) for t in expert}
        keep = []
        for i, p in enumerate(paths):
            node_seq = tuple(type_names[t] for t in p[0::2])  # 偶数位为节点类型
            if node_seq in expert_sets:
                keep.append(i)
        if not keep:
            raise RuntimeError(f"oracle 模式未匹配到任何专家路径，请检查 expert_paths 配置: {expert_sets}")
        short = torch.tensor(keep, dtype=torch.long)
        fixed_gate = torch.ones(len(short))
        print(f"  [T2/oracle] 候选限定为 {len(short)} 条专家路径，门控固定全开", flush=True)
    elif ablation == "bottom":
        B = min(cfg["B"], K)
        short = torch.argsort(coverage)[:B]   # coverage 升序，取最低 B 条
        fixed_gate = torch.ones(len(short))
        print(f"  [T7/bottom] 候选限定为 coverage 最低的 {B} 条路径，门控固定全开", flush=True)
    elif ablation == "equal":
        fixed_gate = torch.full((len(short),), 1.0 / len(short))
        print(f"  [P0/equal] {len(short)} 条候选等权 1/M 聚合，无门控学习", flush=True)
    elif ablation == "randgate":
        B = min(cfg["B"], len(short))
        gen = torch.Generator().manual_seed(cfg.get("seed", 0))
        gate_mask = torch.zeros(len(short))
        sel = torch.randperm(len(short), generator=gen)[:B]
        gate_mask[sel] = 1.0
        fixed_gate = gate_mask
        print(f"  [P0/randgate] 同候选集随机选 {B} 条门控全开，无门控学习", flush=True)

    model.set_shortlist(short, [e.to(device) for e in build_shortlist_edges(
        graph, rel_seqs, short, max_deg=cfg.get("max_deg", 64))])
    if fixed_gate is not None:
        model.set_fixed_gate(fixed_gate.to(device))

    enc_params = list(model.encoder.parameters())
    other_params = [p for n, p in model.named_parameters() if not n.startswith("encoder.")]
    opt = torch.optim.AdamW([
        {"params": enc_params, "lr": cfg["lr"] * cfg["encoder_lr_scale"]},
        {"params": other_params, "lr": cfg["lr"]},
    ], weight_decay=cfg["weight_decay"])

    features = [f.to(device) for f in graph.features]
    offsets = graph.type_offsets.to(device)
    target_feat = graph.features[graph.target_type].to(device)
    labels = graph.labels.to(device)
    tr, va = graph.train_mask.to(device), graph.val_mask.to(device)

    eval_per_epoch = bool(cfg.get("eval_test_per_epoch", False))
    te = graph.test_mask.to(device) if eval_per_epoch else None
    history: list = []
    class_weight = None
    if cfg.get("class_weight", False):
        cnt = torch.bincount(labels[tr], minlength=graph.num_classes).float().clamp(min=1)
        class_weight = (cnt.sum() / cnt)
        class_weight = (class_weight / class_weight.mean()).to(device)  # 归一使均值≈1
    best_va, best_state, bad, best_epoch = -1.0, None, 0, cfg["t_warm"] + 1
    va_ema = None
    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        # 级联粗筛周期性更新
        if ablation == "none" and K > M and epoch > cfg["t_warm"] + 20 and epoch % 20 == 1:
            with torch.no_grad():
                e_tilde = model.encode_paths(features, offsets)
                h_bar = model.target_proj(target_feat).mean(dim=0)
                s_all = model.scorer(e_tilde, h_bar)
                keep = M // 2
                old_top = model.shortlist_idx.to(device)[
                    torch.topk(s_all[model.shortlist_idx.to(device)], k=min(keep, M)).indices
                ]
                cand = torch.topk(s_all, k=min(K, M * 2)).indices.to(device)
                new_part = cand[~torch.isin(cand, old_top)][: M - old_top.numel()]
                short = torch.cat([old_top, new_part]).cpu()
            model.set_shortlist(short, [e.to(device) for e in build_shortlist_edges(
                graph, rel_seqs, short, max_deg=cfg.get("max_deg", 64))])

        out, loss_reg, _ = fwd(model, features, offsets, target_feat, epoch, training=True)
        loss = cross_entropy_on_mask(out, labels, tr, class_weight) + loss_reg
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
        opt.step()

        model.eval()
        with torch.no_grad():
            out, _, _ = fwd(model, features, offsets, target_feat, epoch, training=False)
            va_f1 = macro_f1(out, labels, va)
            if eval_per_epoch:
                m_va = all_metrics(out, labels, va)
                m_te = all_metrics(out, labels, te)
                history.append({
                    "epoch": epoch, "loss": float(loss.item()),
                    "val_macro_f1": m_va["macro_f1"], "val_micro_f1": m_va["micro_f1"],
                    "val_auc": m_va["auc"],
                    "test_macro_f1": m_te["macro_f1"], "test_micro_f1": m_te["micro_f1"],
                    "test_auc": m_te["auc"],
                })
        va_ema = va_f1 if va_ema is None else 0.6 * va_ema + 0.4 * va_f1
        if va_ema > best_va:
            best_va, bad, best_epoch = va_ema, 0, epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if not cfg.get("fixed_epochs", False) and bad >= cfg["patience"]:
                break
        if eval_per_epoch:
            print(f"  seed {seed} | epoch {epoch:3d} | loss {loss.item():.4f} | "
                  f"val MaF1 {history[-1]['val_macro_f1']:.4f} | test MaF1 {history[-1]['test_macro_f1']:.4f}",
                  flush=True)
        elif epoch % 10 == 0 or epoch == 1:
            print(f"  seed {seed} | epoch {epoch:3d} | loss {loss.item():.4f} | val F1 {va_f1:.4f} | ema {va_ema:.4f}")

    fixed = cfg.get("fixed_epochs", False)
    if not fixed and best_state is not None:
        model.load_state_dict(best_state)
        eval_epoch = best_epoch
    else:
        eval_epoch = epoch  # 最终 epoch：门控已退火完成，状态稳定
    model.eval()
    with torch.no_grad():
        out, _, aux = fwd(model, features, offsets, target_feat, eval_epoch, training=False)
        te = graph.test_mask.to(device)
        m_all = all_metrics(out, labels, te)
        result = {
            "seed": seed,
            "test_macro_f1": m_all["macro_f1"],
            "test_micro_f1": m_all["micro_f1"],
            "test_auc": m_all["auc"],
            "val_macro_f1": best_va,
            "best_epoch": eval_epoch,
            "K": K,
            "selected": [paths[model.shortlist_idx[i].item()] for i in
                         model.selected_paths(features, offsets, target_feat, cfg["B"]).tolist()],
        }
        if eval_per_epoch:
            result["history"] = history
    ckpt = Path("checkpoints"); ckpt.mkdir(exist_ok=True)
    # ablation 模式的 checkpoint 带 ablation 名，避免覆盖主模型
    ab = cfg.get("ablation", "none")
    tag = f"_{ab}" if ab not in ("none", None, "") else ""
    torch.save(model.state_dict(), ckpt / f"hdmp_{cfg.get('dataset', 'syn')}{tag}_seed{seed}.pt")
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--eval-test-per-epoch", action="store_true",
                    help="每个 epoch 同时记录 val/test 指标（仅观测，不改训练动态与超参数）")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config)) if args.config else yaml.safe_load(
        open(Path(__file__).parent / "configs" / "acm.yaml"))
    if args.epochs:
        cfg["epochs"] = args.epochs
    if args.eval_test_per_epoch:
        cfg["eval_test_per_epoch"] = True

    # 大图（OGBN-MAG）：委托给 train_mag.run_mag 的 mini-batch 训练器。
    # 小数据集（HGB 等）继续走下方全量路径，结果完全可复现。
    is_mag = (not args.synthetic) and cfg["dataset"].lower() in ("ogbn_mag", "ogbn-mag", "mag")
    if is_mag and cfg.get("subset_targets") is None:
        cfg["dataset"] = "ogbn_mag"
        from train_mag import run_mag
        summary = run_mag(cfg, args.device)
        out_dir = Path("results"); out_dir.mkdir(exist_ok=True)
        # 单 seed 配置（并行多进程各跑一个 seed）时文件名带 seed 后缀，避免互相覆盖
        if len(cfg.get("seeds", [])) == 1:
            out_path = out_dir / f"ogbn_mag_seed{cfg['seeds'][0]}.json"
        else:
            out_path = out_dir / "ogbn_mag_main.json"
        out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str))
        print(f"\n=== ogbn_mag 主结果 ===")
        print(f"Macro-F1 = {summary['test_macro_f1_mean']:.4f} +/- {summary['test_macro_f1_std']:.4f}")
        print(f"Micro-F1 = {summary['test_micro_f1_mean']:.4f} +/- {summary['test_micro_f1_std']:.4f}")
        if summary["test_auc_mean"] is not None:
            print(f"AUC      = {summary['test_auc_mean']:.4f} +/- {summary['test_auc_std']:.4f}")
        print(f"已保存 {out_path}")
        return

    if args.synthetic:
        cfg["dataset"] = "synthetic"
        graph = make_synthetic_graph(seed=0)
    elif is_mag:
        # subset_targets 启用时走子图可扩展性实验（小图，沿用全量路径）。
        cfg["dataset"] = "ogbn_mag"
        from hdmp.data import load_ogbn_mag
        graph = load_ogbn_mag(embed_dim=cfg.get("embed_dim", 64))
        from hdmp.data_subset import subset_graph
        n_sub = int(cfg["subset_targets"])
        graph = subset_graph(graph, n_sub, seed=cfg.get("subset_seed", 0),
                             min_per_class=int(cfg.get("min_per_class", 5)))
        cfg["dataset"] = f"ogbn_mag_sub{n_sub}"
        print(f"[subset] 抽取 {n_sub} 目标节点(每类保底{cfg.get('min_per_class',5)})的诱导子图: "
              f"总节点 {graph.num_nodes}, 目标节点 {graph.labels.numel()}")
    else:
        graph = load_hgb(cfg["dataset"])

    seeds = cfg.get("seeds", [0])
    results = [train_one_seed(graph, cfg, s, args.device) for s in seeds]
    f1s = [r["test_macro_f1"] for r in results]
    mean_f1 = float(np.mean(f1s))
    std_f1 = float(np.std(f1s))
    mi_f1s = [r["test_micro_f1"] for r in results]
    aucs = [r["test_auc"] for r in results if not np.isnan(r["test_auc"])]
    summary = {
        "dataset": cfg["dataset"],
        "seeds": seeds,
        "per_seed": results,
        "test_macro_f1_mean": mean_f1,
        "test_macro_f1_std": std_f1,
        "test_micro_f1_mean": float(np.mean(mi_f1s)),
        "test_micro_f1_std": float(np.std(mi_f1s)),
        "test_auc_mean": float(np.mean(aucs)) if aucs else None,
        "test_auc_std": float(np.std(aucs)) if aucs else None,
    }
    out_dir = Path("results"); out_dir.mkdir(exist_ok=True)
    # ablation（对照实验）与逐 epoch 观测模式输出独立文件名，避免覆盖主结果 main.json
    ab = cfg.get("ablation", "none")
    if cfg.get("eval_test_per_epoch", False):
        fname = f"{cfg['dataset']}_epoch_curve.json"
    elif ab == "none":
        fname = f"{cfg['dataset']}_main.json"
    else:
        fname = f"{cfg['dataset']}_ab_{ab}.json"
    out_path = out_dir / fname
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str))
    print(f"\n=== {cfg['dataset']} 主结果 ===")
    print(f"per-seed Macro-F1: {[round(x,4) for x in f1s]}")
    print(f"Macro-F1 = {mean_f1:.4f} +/- {std_f1:.4f}  (n={len(f1s)})")
    print(f"Micro-F1 = {summary['test_micro_f1_mean']:.4f} +/- {summary['test_micro_f1_std']:.4f}")
    if summary["test_auc_mean"] is not None:
        print(f"AUC      = {summary['test_auc_mean']:.4f} +/- {summary['test_auc_std']:.4f}")
    print(f"已保存 {out_path}")


if __name__ == "__main__":
    main()
