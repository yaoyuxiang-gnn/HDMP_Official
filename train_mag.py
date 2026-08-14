"""HDMP 的大图（OGBN-MAG 级）mini-batch 训练器（extended work）。
与 train.py 的关系：方法主体（两级注意力、top-ρ 评分、Gumbel 门控、层次正则、
级联粗筛）完全复用 hdmp 模块权重与超参；仅把"全量目标节点前向"替换为
"目标节点 mini-batch 前向"，采样复合邻接用 mag_edges 的向量化实现——这是大图
可扩展性工程，与 NeighborLoader 之于 GCN 同类。
不直接作为脚本运行；由 train.py 在检测到大图配置（含 mb_targets）时调用 run_mag。
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score

from hdmp.data import load_ogbn_mag, sample_path_instances
from hdmp.enumeration import (build_hierarchy, enumerate_metapaths, pad_tokens,
                              prune_inverse_dup, rel_seq_of)
from hdmp.mag_edges import build_candidate_edges, build_relation_csrs, path_has_instance
from hdmp.model import HDMPModel
from hdmp.attention import edge_softmax
from hdmp.selection import stratified_shortlist, top_rho_score
from hdmp.losses import hierarchical_loss, stabilization_loss


def macro_f1(out, labels, mask_idx):
    pred = out.argmax(dim=-1).cpu().numpy()
    return f1_score(labels[mask_idx].cpu().numpy(), pred, average="macro")


def prepare_mag(graph, cfg):
    """离线准备：枚举→逆去重→存在性剪枝（向量化采样版）→层次→实例采样→tokens。"""
    t0 = time.time()
    paths = enumerate_metapaths(graph, cfg["l_max"])
    paths = prune_inverse_dup(graph, paths)
    rel_csrs = build_relation_csrs(graph)
    kept = [p for p in paths
            if path_has_instance(graph, rel_seq_of(p, graph.T_A), rel_csrs,
                                 fanout=3, max_starts=512)]
    if not kept:
        raise RuntimeError("存在性剪枝后候选集为空")
    paths = kept
    print(f"[prepare] 枚举+剪枝: K={len(paths)} ({time.time()-t0:.0f}s)", flush=True)
    pairs1, pairs2 = build_hierarchy(paths)
    rel_seqs = [rel_seq_of(p, graph.T_A) for p in paths]
    inst_idx, inst_mask = sample_path_instances(
        graph, rel_seqs, cfg["S"], cfg["l_max"],
        max_starts=cfg.get("inst_max_starts", 2000))
    cls = graph.T_A + 2 * graph.T_R
    tokens, mask = pad_tokens(paths, num_types=cls, pad_id=cls)
    cls_col = torch.full((tokens.size(0), 1), cls, dtype=torch.long)
    tokens = torch.cat([cls_col, tokens], dim=1)
    mask = torch.cat([torch.ones(tokens.size(0), 1, dtype=torch.bool), mask], dim=1)
    print(f"[prepare] 实例采样完成 ({time.time()-t0:.0f}s)", flush=True)
    return paths, rel_seqs, rel_csrs, tokens, mask, inst_idx, inst_mask, pairs1, pairs2


class EdgeBank:
    """shortlist 候选的采样复合邻接（CSR-by-dst），torch 形式，按候选 id 缓存。
    磁盘缓存到 data/ogbn_mag/edge_cache/，跨运行复用（构建一次约 30 min）。"""

    def __init__(self, graph, rel_seqs, rel_csrs, fanout, max_deg,
                 cache_dir="data/ogbn_mag/edge_cache"):
        self.graph, self.rel_seqs, self.rel_csrs = graph, rel_seqs, rel_csrs
        self.fanout, self.max_deg = fanout, max_deg
        self.cache = {}
        self.dir = Path(cache_dir)
        self.dir.mkdir(parents=True, exist_ok=True)

    def _disk_path(self, i):
        return self.dir / f"cand{i}_f{self.fanout}_d{self.max_deg}.npz"

    def get(self, shortlist):
        out = []
        for i in shortlist.tolist():
            if i not in self.cache:
                fp = self._disk_path(i)
                if fp.exists():
                    z = np.load(fp)
                    indptr, src = z["indptr"], z["src"]
                else:
                    t0 = time.time()
                    indptr, src = build_candidate_edges(
                        self.graph, self.rel_seqs, torch.tensor([i]), self.fanout,
                        self.max_deg, seed=0, cache={})[0][0]
                    np.savez(fp, indptr=indptr, src=src)
                    print(f"    [edges] cand {i}: E={src.size} ({time.time()-t0:.0f}s, cached)", flush=True)
                self.cache[i] = (torch.from_numpy(indptr), torch.from_numpy(src))
            out.append(self.cache[i])
        return out


def forward_batch(model, batch_cpu, batch_dev, proj, e_tilde, edges_t, epoch, training):
    cfg = model.cfg
    att = model.node_att
    sl = model.shortlist_idx.to(e_tilde.device)
    M = sl.numel()
    B = batch_dev.numel()
    device = e_tilde.device
    # H==1 时 W_in[p,0]@W[p,0] 合并为单矩阵，省一次全边变换（结合律，数学等价）
    W_eff = att.W_in[sl, 0] @ att.W[sl, 0]                 # [M, d, dk]
    h_dst = model.dropout(model.target_proj(model.target_feat_dev[batch_dev]))  # [B, d]
    Z = h_dst.new_zeros((B, M, att.dk))
    # 训练用 checkpoint 逐候选重算：autograd 只需保留单条候选的边级中间张量，
    # 峰值显存从 M×E×dk 降到 E×dk（与 hdmp/minibatch.py 的 use_ckpt 策略一致）。
    from torch.utils.checkpoint import checkpoint

    def one_path(proj, h_dst, W_k, a_k, indptr, src):
        cnt = indptr[batch_cpu + 1] - indptr[batch_cpu]
        total = int(cnt.sum())
        if total == 0:
            return h_dst.new_zeros((B, att.dk))
        cum = cnt.cumsum(0) - cnt
        base = torch.repeat_interleave(indptr[batch_cpu] - cum, cnt)
        eidx = base + torch.arange(total, dtype=torch.long)
        src_e = src[eidx].to(device)
        dst_e = torch.repeat_interleave(torch.arange(B, dtype=torch.long), cnt).to(device)
        Wh_s = proj[src_e] @ W_k                           # [E, dk]
        Wh_d = h_dst @ W_k                                 # [B, dk]
        pair = torch.cat([Wh_d[dst_e], Wh_s], dim=-1)
        score = att.leaky(pair @ a_k)
        alpha = edge_softmax(score, dst_e, B)
        z = h_dst.new_zeros((B, att.dk))
        z.index_add_(0, dst_e, alpha.unsqueeze(-1) * Wh_s)
        return z

    for k in range(M):
        indptr, src = edges_t[k]
        if training:
            Z[:, k] = checkpoint(one_path, proj, h_dst, W_eff[k], att.att[sl[k], 0],
                                   indptr, src, use_reentrant=False)
        else:
            Z[:, k] = one_path(proj, h_dst, W_eff[k], att.att[sl[k], 0], indptr, src)
    e_short = e_tilde[sl]
    logits, omega = model.sem_att(Z, e_short)
    frozen = epoch <= cfg["t_warm"]
    anneal_done = model.gate.temperature(epoch) <= cfg["tau_final"] + 1e-9
    g = top_rho_score(logits, cfg["rho"])
    h_bar = h_dst.mean(dim=0)
    g = g + model.scorer(e_tilde, h_bar)[sl]
    if model.fixed_z is not None:
        z_used = model.fixed_z.to(device).float()
    else:
        z_used, _ = model.gate(g, epoch, training, frozen=frozen, use_st=anneal_done)
    neigh = torch.einsum("nm,nmd->nd", z_used.unsqueeze(0) * omega, Z)
    fused = neigh + h_dst                                  # 残差连接（同原版）
    fused = model.dropout(fused)
    out = model.classifier(fused)
    loss_reg = (hierarchical_loss(e_tilde, model.pairs1, model.pairs2,
                                  cfg["lambda_h1"], cfg["lambda_h2"])
                + cfg["lambda_s"] * stabilization_loss(logits))
    return out, loss_reg, logits


@torch.no_grad()
def eval_logits(model, node_idx, proj, e_tilde, edges_t, epoch, mb):
    """分块推理：返回 node_idx 上拼接的 logits [n, C]。"""
    model.eval()
    outs = []
    for i in range(0, node_idx.numel(), mb):
        b_cpu = node_idx[i:i + mb]
        b_dev = b_cpu.to(e_tilde.device)
        out, _, _ = forward_batch(model, b_cpu, b_dev, proj, e_tilde, edges_t,
                                  epoch, training=False)
        outs.append(out.cpu())
    return torch.cat(outs, dim=0)


def train_one_seed(graph, cfg, seed, device, prep, bank):
    torch.manual_seed(seed)
    np.random.seed(seed)
    (paths, rel_seqs, tokens, mask, inst_idx, inst_mask, pairs1, pairs2) = prep
    K = len(paths)
    feat_dims = [f.size(1) for f in graph.features]
    model = HDMPModel(graph.T_A, 2 * graph.T_R, feat_dims, graph.num_classes,
                      K, tokens.size(1), cfg).to(device)
    model.load_prepared(tokens.to(device), mask.to(device), inst_idx.to(device),
                        inst_mask.to(device), pairs1.to(device), pairs2.to(device))
    model.target_feat_dev = graph.features[graph.target_type].to(device)

    M = min(cfg["M"], K)
    coverage = inst_mask.sum(dim=(1, 2)).float()
    short = stratified_shortlist(rel_seqs, coverage, M, T_R=graph.T_R)
    edges_t = bank.get(short)
    model.set_shortlist(short, [])          # edge_indices 由 edges_t 提供（CSR 形式）

    enc_params = list(model.encoder.parameters())
    other = [p for n, p in model.named_parameters() if not n.startswith("encoder.")]
    opt = torch.optim.AdamW([
        {"params": enc_params, "lr": cfg["lr"] * cfg["encoder_lr_scale"]},
        {"params": other, "lr": cfg["lr"]},
    ], weight_decay=cfg["weight_decay"])

    features_dev = [f.to(device) for f in graph.features]
    offsets_dev = graph.type_offsets.to(device)
    labels = graph.labels
    tr_idx = torch.nonzero(graph.train_mask, as_tuple=False).view(-1)
    va_idx = torch.nonzero(graph.val_mask, as_tuple=False).view(-1)
    te_idx = torch.nonzero(graph.test_mask, as_tuple=False).view(-1)
    mb = cfg.get("mb_targets", 16384)

    def proj_and_encode():
        proj = model.instance_ctx.project_all(features_dev, offsets_dev)
        e_sem = model.encoder(model.tokens, model.token_mask)
        e_tilde = model.instance_ctx(e_sem, proj, model.inst_idx, model.inst_mask)
        return proj, e_tilde

    best_va, va_ema = -1.0, None
    epochs = cfg["epochs"]
    for epoch in range(1, epochs + 1):
        t_ep = time.time()
        model.train()
        # 级联粗筛周期性更新（与 train.py 同调度：t_warm+20 后每 20 轮）
        if K > M and epoch > cfg["t_warm"] + 20 and epoch % 20 == 1:
            with torch.no_grad():
                proj, e_tilde = proj_and_encode()
                h_bar = model.target_proj(model.target_feat_dev[tr_idx[:50000].to(device)]).mean(dim=0)
                s_all = model.scorer(e_tilde, h_bar)
                keep = M // 2
                old_top = model.shortlist_idx.to(device)[
                    torch.topk(s_all[model.shortlist_idx.to(device)], k=min(keep, M)).indices]
                cand = torch.topk(s_all, k=min(K, M * 2)).indices.to(device)
                new_part = cand[~torch.isin(cand, old_top)][: M - old_top.numel()]
                short = torch.cat([old_top, new_part]).cpu()
            edges_t = bank.get(short)
            model.set_shortlist(short, [])

        perm = torch.randperm(tr_idx.numel())
        ep_loss, nb = 0.0, 0
        for i in range(0, tr_idx.numel(), mb):
            b_cpu = tr_idx[perm[i:i + mb]]
            b_dev = b_cpu.to(device)
            proj, e_tilde = proj_and_encode()
            out, loss_reg, _ = forward_batch(model, b_cpu, b_dev, proj, e_tilde,
                                             edges_t, epoch, training=True)
            loss = torch.nn.functional.cross_entropy(out, labels[b_cpu].to(device)) + loss_reg
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            opt.step()
            ep_loss += loss.item(); nb += 1

        # 验证（EMA 平滑，协议与 train.py 一致）
        with torch.no_grad():
            proj, e_tilde = proj_and_encode()
            out_va = eval_logits(model, va_idx, proj, e_tilde, edges_t, epoch, mb)
            va_f1 = macro_f1(out_va, labels, va_idx)
        va_ema = va_f1 if va_ema is None else 0.6 * va_ema + 0.4 * va_f1
        best_va = max(best_va, va_ema)
        if epoch % 5 == 0 or epoch == 1:
            print(f"  seed {seed} | epoch {epoch:3d} | loss {ep_loss/max(nb,1):.4f} | "
                  f"val F1 {va_f1:.4f} | ema {va_ema:.4f} | {time.time()-t_ep:.0f}s", flush=True)

    # 固定 epoch 协议：用最终状态评估（与 train.py fixed_epochs 一致）
    with torch.no_grad():
        proj, e_tilde = proj_and_encode()
        out_te = eval_logits(model, te_idx, proj, e_tilde, edges_t, epochs, mb)
        m_all = all_metrics_full(out_te, labels[te_idx])
        # 选路：全量目标分块 logits → top-ρ → top-B（与 selected_paths 一致）
        sl = model.shortlist_idx
        logits_all = []
        for i in range(0, graph.labels.numel(), mb):
            b_cpu = torch.arange(i, min(i + mb, graph.labels.numel()))
            b_dev = b_cpu.to(device)
            _, _, lg = forward_batch(model, b_cpu, b_dev, proj, e_tilde, edges_t,
                                     epochs, training=False)
            logits_all.append(lg.cpu())
        g = top_rho_score(torch.cat(logits_all, dim=0).to(device), cfg["rho"])
        topB = torch.topk(g, k=min(cfg["B"], g.numel())).indices
        selected = [paths[sl[i].item()] for i in topB.tolist()]
    return {"seed": seed, "test_macro_f1": m_all["macro_f1"],
            "test_micro_f1": m_all["micro_f1"], "test_auc": m_all["auc"],
            "val_macro_f1": best_va, "K": K, "selected": selected}


def all_metrics_full(out, y):
    """out [n, C] 与 y [n] 直接对齐的指标计算。"""
    from sklearn.metrics import roc_auc_score
    yt = y.cpu().numpy()
    pred = out.argmax(dim=-1).cpu().numpy()
    prob = torch.softmax(out, dim=-1).cpu().numpy()
    try:
        present = np.unique(yt)
        p = prob[:, present]
        p = p / p.sum(axis=1, keepdims=True)
        auc = float(roc_auc_score(yt, p, multi_class="ovr",
                                  average="macro", labels=present))
    except Exception:
        auc = float("nan")
    return {"macro_f1": float(f1_score(yt, pred, average="macro")),
            "micro_f1": float(f1_score(yt, pred, average="micro")), "auc": auc}


def run_mag(cfg: dict, device: str) -> dict:
    """大图（OGBN-MAG）mini-batch 训练主流程，返回与 train.py 一致的 summary dict。"""
    t0 = time.time()
    graph = load_ogbn_mag(embed_dim=cfg.get("embed_dim", 64))
    print(f"[data] MAG 加载: N={graph.num_nodes}, 目标={graph.labels.numel()}, "
          f"关系={graph.T_R}, 类别={graph.num_classes} ({time.time()-t0:.0f}s)", flush=True)

    prep_full = prepare_mag(graph, cfg)
    (paths, rel_seqs, rel_csrs, tokens, mask, inst_idx, inst_mask, pairs1, pairs2) = prep_full
    bank = EdgeBank(graph, rel_seqs, rel_csrs,
                    fanout=cfg.get("neighbor_fanout", 15),
                    max_deg=cfg.get("max_deg", 8),
                    cache_dir=cfg.get("edge_cache_dir", "data/ogbn_mag/edge_cache"))
    # 初始 shortlist 的边在第一个 seed 前预建（seed=0 采样，跨 seed 共享，与原版一致）
    coverage = inst_mask.sum(dim=(1, 2)).float()
    short0 = stratified_shortlist(rel_seqs, coverage, min(cfg["M"], len(paths)), T_R=graph.T_R)
    t0 = time.time()
    bank.get(short0)
    print(f"[edges] 初始 shortlist 边构建完成 ({time.time()-t0:.0f}s)", flush=True)

    prep = (paths, rel_seqs, tokens, mask, inst_idx, inst_mask, pairs1, pairs2)
    seeds = cfg.get("seeds", [0])
    results = [train_one_seed(graph, cfg, s, device, prep, bank) for s in seeds]
    f1s = [r["test_macro_f1"] for r in results]
    mis = [r["test_micro_f1"] for r in results]
    aucs = [r["test_auc"] for r in results if not np.isnan(r["test_auc"])]
    return {
        "dataset": cfg.get("dataset", "ogbn_mag"), "seeds": seeds, "per_seed": results,
        "test_macro_f1_mean": float(np.mean(f1s)), "test_macro_f1_std": float(np.std(f1s)),
        "test_micro_f1_mean": float(np.mean(mis)), "test_micro_f1_std": float(np.std(mis)),
        "test_auc_mean": float(np.mean(aucs)) if aucs else None,
        "test_auc_std": float(np.std(aucs)) if aucs else None,
    }
