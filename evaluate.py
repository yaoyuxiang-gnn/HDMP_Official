"""HDMP 评估与可解释性输出脚本。

加载训练好的 checkpoint，报告测试集 Macro/Micro F1，
并输出选中的元路径（可读类型序列）及其全局重要性排序（技术方案 §8）。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # 仓库根（hdmp/、train.py）

import torch
import yaml
from sklearn.metrics import f1_score

from hdmp.data import load_hgb, make_synthetic_graph
from hdmp.model import HDMPModel
from train import build_shortlist_edges, prepare, stratified_shortlist


def readable_path(path, graph) -> str:
    """将 token 序列渲染为可读元路径，如 target -[r0]-> aux1 -[r1]-> aux2。"""
    parts = []
    for i, t in enumerate(path):
        if i % 2 == 0:
            parts.append(graph.node_types[t])
        else:
            r = t - graph.T_A
            if r >= graph.T_R:  # 逆关系
                parts.append(f"<-[{graph.edge_types[r - graph.T_R]}]-")
            else:
                parts.append(f"-[{graph.edge_types[r]}]->")
    return " ".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    graph = make_synthetic_graph(seed=0) if args.synthetic else load_hgb(cfg["dataset"])
    device = args.device

    (paths, rel_seqs, tokens, mask, inst_idx, inst_mask, pairs1, pairs2) = prepare(graph, cfg)
    feat_dims = [f.size(1) for f in graph.features]
    model = HDMPModel(graph.T_A, 2 * graph.T_R, feat_dims, graph.num_classes,
                      len(paths), tokens.size(1), cfg).to(device)
    model.load_prepared(tokens.to(device), mask.to(device), inst_idx.to(device),
                        inst_mask.to(device), pairs1.to(device), pairs2.to(device))
    # 与训练一致的初始粗筛：分层配额（按 hop 长度分层）
    coverage = inst_mask.sum(dim=(1, 2)).float()
    short = stratified_shortlist(rel_seqs, coverage, min(cfg["M"], len(paths)))
    model.set_shortlist(short, [e.to(device) for e in build_shortlist_edges(graph, rel_seqs, short)])
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    features = [f.to(device) for f in graph.features]
    offsets = graph.type_offsets.to(device)
    target_feat = graph.features[graph.target_type].to(device)
    labels = graph.labels.to(device)
    te = graph.test_mask.to(device)

    with torch.no_grad():
        out, _, aux = model(features, offsets, target_feat, cfg["epochs"], training=False)
        pred = out[te].argmax(dim=-1).cpu().numpy()
        gold = labels[te].cpu().numpy()
        print(f"Test Macro-F1: {f1_score(gold, pred, average='macro'):.4f}")
        print(f"Test Micro-F1: {f1_score(gold, pred, average='micro'):.4f}")

        g = aux["g"]
        top = torch.topk(g, k=min(cfg["B"], g.numel()))
        print("\n选中的元路径（按全局重要性降序）:")
        ranking = []
        for rank, (score, i) in enumerate(zip(top.values.tolist(), top.indices.tolist()), 1):
            p = paths[model.shortlist_idx[i].item()]
            readable = readable_path(p, graph)
            print(f"  {rank:2d}. g={score:+.4f}  {readable}")
            ranking.append({"rank": rank, "score": score, "metapath": readable})
        with open("selected_metapaths.json", "w", encoding="utf-8") as f:
            json.dump(ranking, f, ensure_ascii=False, indent=2)
        print("\n已保存 selected_metapaths.json")


if __name__ == "__main__":
    main()
