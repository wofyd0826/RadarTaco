"""Diagnose the gap between hard argmax accuracy and actual soft-gate quality.

The "router accuracy" (~90.7% for v6c) is top-1 argmax match with hard GT bin.
But the model actually uses top_k=2 renormalized SOFT MIXTURE for expert
selection. This script measures:

  For each MoE token, per block, per sample:
    hard_top1_correct    : argmax(router_probs) == argmax(gt_frac)   [current metric]
    hard_top2_recall     : argmax(gt_frac) ∈ topk(router_probs, 2)   [right specialist survived?]
    soft_l1_probs        : |router_probs - gt_frac|.sum()            [distributional match]
    soft_l1_topk_gate    : |router_topk_gate - gt_frac_topk_norm|.sum() [actual gate used vs ideal]

Buckets:
  A. hard_top1_correct=True   → sanity check
  B. hard_top1_wrong AND hard_top2_recall=True   ← "near miss" — right specialist still in top-2
  C. hard_top1_wrong AND hard_top2_recall=False  ← catastrophic — right specialist excluded

Reports:
  - Fraction of tokens in each bucket, per depth range
  - Distribution of soft_l1 within each bucket
  - Adjacency pattern of misroutes (near↔mid vs near↔far)
  - Weighted actual-gate quality: how much of the "wrong routes" are actually
    quantitatively close to the oracle gate

Usage:
  python tests/analyze_soft_vs_hard_routing_v6c.py --n 1000
"""
import argparse, json, os, sys
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from tqdm import tqdm

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, ROOT)

from src.dataset.nuscenes import NuScenesRadarDepthDataset
from src.model.radar_fusion import MoEFusionBlock
from scripts.train import _build_model

device = "cuda" if torch.cuda.is_available() else "cpu"


def load_model(run_dir, ckpt="best.pt"):
    cfg = OmegaConf.load(os.path.join(run_dir, "config.yaml"))
    m = _build_model(cfg, float(cfg.dataset.max_depth), int(cfg.dataset.max_radar_points))
    sd = torch.load(os.path.join(run_dir, ckpt), map_location=device, weights_only=False)
    m.load_state_dict(sd["model"])
    print(f"  loaded {run_dir}/{ckpt}  epoch={sd.get('epoch','?')}")
    return m.eval().to(device), cfg


def _collect_moe_blocks(model):
    out = []
    for l in range(len(model.radar_fusion.node_blocks)):
        nb = model.radar_fusion.node_blocks[l]
        eb = model.radar_fusion.edge_blocks[l]
        if isinstance(nb, MoEFusionBlock): out.append((f"L{l}_node", nb))
        if isinstance(eb, MoEFusionBlock): out.append((f"L{l}_edge", eb))
    return out


def compute_pixel_fraction(depth_gt_dense, tok_hw, bins):
    """Return (B, K, h, w) per-token pixel-bin fraction; sums to 1 along K."""
    K = len(bins) - 1
    edges = torch.tensor(bins, device=depth_gt_dense.device, dtype=torch.float32)
    ds = depth_gt_dense.squeeze(1)
    pix_bin = torch.zeros_like(ds, dtype=torch.long)
    for i in range(K):
        pix_bin = torch.where((ds >= float(edges[i])) & (ds < float(edges[i+1])),
                              pix_bin.new_tensor(i), pix_bin)
    pix_bin = torch.where(ds >= float(edges[-1]), pix_bin.new_tensor(K-1), pix_bin)
    onehot = F.one_hot(pix_bin, num_classes=K).permute(0, 3, 1, 2).float()
    return F.adaptive_avg_pool2d(onehot, tok_hw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default="output/radartaco_moe_stage2_v6c_confshared")
    ap.add_argument("--ckpt", default="best.pt")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--split", default="test")
    ap.add_argument("--out-json", default="output/analysis/soft_vs_hard_routing_v6c.json")
    args = ap.parse_args()

    model, cfg = load_model(args.run_dir, args.ckpt)
    router_bins = list(cfg.model.moe_bins)
    top_k = int(cfg.model.get("moe_top_k", 2))
    K = len(router_bins) - 1
    print(f"router_bins={router_bins}   K={K}   top_k={top_k}")

    split_file = os.path.join(cfg.dataset.data_root, "splits", f"{args.split}.txt")
    ds = NuScenesRadarDepthDataset(
        data_root=cfg.dataset.data_root, split_file=split_file,
        dense_gt_dir=cfg.dataset.dense_gt_dir,
        radar_3d_dir=cfg.dataset.radar_3d_dir,
        night_ids_file=cfg.dataset.night_ids_file,
        max_radar_points=int(cfg.dataset.max_radar_points),
        max_depth=float(cfg.dataset.max_depth),
        min_depth=float(cfg.dataset.min_depth),
        augmentation=False,
    )
    n = min(args.n, len(ds))
    idxs = list(range(n)) if n == len(ds) else np.linspace(0, len(ds)-1, n).astype(int).tolist()
    print(f"\n{args.split} — {len(idxs):,}/{len(ds):,} samples\n")

    moe_blocks = _collect_moe_blocks(model)
    print(f"MoE blocks: {[nm for nm, _ in moe_blocks]}\n")

    # Bucket names.
    # A: top1 correct
    # B: top1 wrong, but top2 recall (right specialist still in top-k)
    # C: top1 wrong AND top2 miss (catastrophic)
    bucket_names = ["A_top1_correct", "B_top1_wrong_top2_ok", "C_top1_wrong_top2_miss"]
    # Per-bucket counts and soft_l1 accumulators.
    # counts[bucket][gt_depth_bin] = count
    counts = {b: np.zeros(K, dtype=np.int64) for b in bucket_names}
    # For soft_l1 distributions we accumulate list of values (subsample if too many).
    soft_l1_probs_by_bucket    = {b: [] for b in bucket_names}
    soft_l1_topk_gate_by_bucket = {b: [] for b in bucket_names}
    # Misroute adjacency: for top1_wrong tokens, record (gt_bin, router_top1)
    adjacency = np.zeros((K, K), dtype=np.int64)

    # Sampling to keep memory in check.
    SAMPLE_STRIDE = 20    # keep every 20th token's soft_l1 for distribution stats

    tok_counter = 0

    with torch.no_grad():
        for i in tqdm(idxs, desc="samples", leave=False):
            s = ds[int(i)]
            rgb = s["rgb_norm"].unsqueeze(0).to(device)
            rp  = s["radar_points"].unsqueeze(0).to(device)
            rm  = s["radar_mask"].unsqueeze(0).to(device)
            dgd = s["depth_gt_dense"].unsqueeze(0).to(device)

            out = model(rgb, rp, rm)
            if not (isinstance(out, dict) and out.get("router_logits")):
                continue

            for (name, blk), rl in zip(moe_blocks, out["router_logits"]):
                # rl: (B, K, h, w)
                probs = F.softmax(rl, dim=1)                                  # (B, K, h, w)
                # top-k gate (renormalized), same as model uses.
                _, idx_topk = probs.topk(top_k, dim=1)
                keep = torch.zeros_like(probs).scatter_(1, idx_topk, 1.0)
                gate = probs * keep
                gate = gate / gate.sum(dim=1, keepdim=True).clamp_min(1e-8)   # (B, K, h, w)

                # GT pixel fraction (soft) and hard argmax.
                gt_frac = compute_pixel_fraction(dgd, probs.shape[-2:], router_bins)
                gt_hard = gt_frac.argmax(dim=1)                               # (B, h, w)
                # GT top-k gate (pixel-fraction restricted to top-k of GT, renormalized).
                # This is the ideal soft gate the model would use under OracleSoft.
                _, gt_idx_topk = gt_frac.topk(top_k, dim=1)
                gt_keep = torch.zeros_like(gt_frac).scatter_(1, gt_idx_topk, 1.0)
                gt_gate = gt_frac * gt_keep
                gt_gate = gt_gate / gt_gate.sum(dim=1, keepdim=True).clamp_min(1e-8)

                pred_top1 = probs.argmax(dim=1)                               # (B, h, w)
                # top-k recall: is gt_hard in top-k of probs?
                topk_mask = (idx_topk == gt_hard.unsqueeze(1))                # (B, top_k, h, w)
                top2_recall = topk_mask.any(dim=1)                            # (B, h, w) — True if GT bin ∈ top-k

                top1_correct = (pred_top1 == gt_hard)                         # (B, h, w)

                # Per-token soft_l1 (full probs vs GT full fraction)
                sl1_probs = (probs - gt_frac).abs().sum(dim=1)                # (B, h, w)
                # Per-token soft_l1 between actual gate used and ideal GT gate
                sl1_gate  = (gate - gt_gate).abs().sum(dim=1)                 # (B, h, w)

                # Flatten to iterate (per token).
                gt_hard_f     = gt_hard.reshape(-1).cpu().numpy()
                pred_top1_f   = pred_top1.reshape(-1).cpu().numpy()
                top1_ok_f     = top1_correct.reshape(-1).cpu().numpy()
                top2_ok_f     = top2_recall.reshape(-1).cpu().numpy()
                sl1_probs_f   = sl1_probs.reshape(-1).cpu().numpy()
                sl1_gate_f    = sl1_gate.reshape(-1).cpu().numpy()

                for ti in range(len(gt_hard_f)):
                    gtb = int(gt_hard_f[ti])
                    prb = int(pred_top1_f[ti])
                    if top1_ok_f[ti]:
                        bucket = "A_top1_correct"
                    elif top2_ok_f[ti]:
                        bucket = "B_top1_wrong_top2_ok"
                        adjacency[gtb, prb] += 1
                    else:
                        bucket = "C_top1_wrong_top2_miss"
                        adjacency[gtb, prb] += 1

                    counts[bucket][gtb] += 1
                    # Subsample soft_l1 storage.
                    if tok_counter % SAMPLE_STRIDE == 0:
                        soft_l1_probs_by_bucket[bucket].append(float(sl1_probs_f[ti]))
                        soft_l1_topk_gate_by_bucket[bucket].append(float(sl1_gate_f[ti]))
                    tok_counter += 1

    # ---- Report ----
    total = sum(counts[b].sum() for b in bucket_names)
    print("\n" + "=" * 100)
    print("Soft vs Hard routing quality  (v6c stage 2)")
    print("=" * 100)
    print(f"Total tokens analyzed: {total:,}\n")

    print("Bucket distribution (all tokens)")
    print("-" * 100)
    print(f"  {'Bucket':<28} {'Count':>14} {'%_total':>10}   {'Near':>10} {'Mid':>10} {'Far':>10}")
    for b in bucket_names:
        c = counts[b].sum()
        pct = 100.0 * c / max(total, 1)
        by_bin = counts[b]
        pcts = [100.0 * by_bin[k] / max(by_bin.sum(), 1) for k in range(K)]
        print(f"  {b:<28} {c:>14,} {pct:>9.2f}%   "
              + " ".join(f"{p:>9.1f}%" for p in pcts))
    print()

    # Fraction of top1-wrong tokens that are "salvageable" (bucket B).
    top1_wrong = counts["B_top1_wrong_top2_ok"].sum() + counts["C_top1_wrong_top2_miss"].sum()
    salvage_pct = 100.0 * counts["B_top1_wrong_top2_ok"].sum() / max(top1_wrong, 1)
    print(f"Among top1-WRONG tokens: {salvage_pct:.1f}% still have GT bin in top-{top_k}")
    print()

    # Soft L1 distribution per bucket
    def _quantiles(arr):
        if len(arr) == 0:
            return {q: float("nan") for q in (0.5, 0.75, 0.9, 0.95, 0.99)}
        a = np.asarray(arr)
        return {q: float(np.quantile(a, q)) for q in (0.5, 0.75, 0.9, 0.95, 0.99)}

    print("Soft L1 distribution (|router_probs - gt_frac|.sum(dim=K)) per bucket")
    print("-" * 100)
    print(f"  {'Bucket':<28} {'N_sampled':>10} {'mean':>8} {'p50':>8} {'p75':>8} {'p90':>8} {'p95':>8} {'p99':>8}")
    for b in bucket_names:
        arr = soft_l1_probs_by_bucket[b]
        q = _quantiles(arr)
        mean = float(np.mean(arr)) if arr else float("nan")
        print(f"  {b:<28} {len(arr):>10,} {mean:>8.3f} "
              + " ".join(f"{q[qk]:>8.3f}" for qk in (0.5, 0.75, 0.9, 0.95, 0.99)))
    print()

    print("Soft L1 distribution (|router_topk_gate - gt_topk_gate|) — ACTUAL gate used")
    print("-" * 100)
    print(f"  {'Bucket':<28} {'N_sampled':>10} {'mean':>8} {'p50':>8} {'p75':>8} {'p90':>8} {'p95':>8} {'p99':>8}")
    for b in bucket_names:
        arr = soft_l1_topk_gate_by_bucket[b]
        q = _quantiles(arr)
        mean = float(np.mean(arr)) if arr else float("nan")
        print(f"  {b:<28} {len(arr):>10,} {mean:>8.3f} "
              + " ".join(f"{q[qk]:>8.3f}" for qk in (0.5, 0.75, 0.9, 0.95, 0.99)))
    print()

    # Threshold-based "practically close" test on ACTUAL gate.
    # If |gate - gt_gate|.sum() < 0.3, the mixture is very close to oracle;
    # < 0.6 is reasonably close; > 1.0 is far off.
    print("Practical closeness on ACTUAL gate (top_k gate vs GT top_k gate)")
    print("-" * 100)
    print(f"  {'Bucket':<28} {'N_sampled':>10} {'<0.3(close)':>13} {'<0.6(ok)':>10} {'>1.0(bad)':>10}")
    for b in bucket_names:
        arr = np.asarray(soft_l1_topk_gate_by_bucket[b])
        if len(arr) == 0:
            print(f"  {b:<28} {'0':>10}")
            continue
        p_close = 100.0 * (arr < 0.3).mean()
        p_ok    = 100.0 * (arr < 0.6).mean()
        p_bad   = 100.0 * (arr > 1.0).mean()
        print(f"  {b:<28} {len(arr):>10,} {p_close:>12.2f}% {p_ok:>9.2f}% {p_bad:>9.2f}%")
    print()

    # Adjacency of misroutes (GT bin vs router top-1) for top1-wrong tokens
    print("Misroute adjacency  (rows: GT bin, cols: router top-1)")
    print(f"  Only top1-wrong tokens counted.  K={K} bins.")
    print("-" * 60)
    labels = ["near", "mid", "far"] if K == 3 else [f"bin{k}" for k in range(K)]
    header = "  " + " " * 8 + "  ".join(f"{l:>10}" for l in labels)
    print(header)
    for gt_b in range(K):
        row = f"  gt={labels[gt_b]:<5} " + "  ".join(f"{adjacency[gt_b, pr]:>10,}" for pr in range(K))
        print(row)
    print()

    # Row-normalized adjacency (given the GT bin was b, when the router was wrong,
    # what was the router's top-1?)
    print("Row-normalized adjacency  (given wrong, distribution of router top-1)")
    print("-" * 60)
    print(header)
    for gt_b in range(K):
        row_sum = adjacency[gt_b].sum()
        if row_sum == 0:
            continue
        row = f"  gt={labels[gt_b]:<5} " + "  ".join(
            f"{100.0*adjacency[gt_b, pr]/row_sum:>9.1f}%" for pr in range(K))
        print(row)
    print()

    # Persist
    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    dump = {
        "run_dir": args.run_dir,
        "ckpt": args.ckpt,
        "n_samples": len(idxs),
        "top_k": top_k,
        "K": K,
        "router_bins": router_bins,
        "counts_by_bucket_gt_bin": {b: counts[b].tolist() for b in bucket_names},
        "adjacency_wrong_only": adjacency.tolist(),
        "soft_l1_probs_summary": {
            b: {"n": len(soft_l1_probs_by_bucket[b]),
                "mean": float(np.mean(soft_l1_probs_by_bucket[b])) if soft_l1_probs_by_bucket[b] else None,
                **{f"p{int(q*100)}": float(np.quantile(soft_l1_probs_by_bucket[b], q))
                                    if soft_l1_probs_by_bucket[b] else None
                   for q in (0.5, 0.75, 0.9, 0.95, 0.99)}}
            for b in bucket_names
        },
        "soft_l1_topk_gate_summary": {
            b: {"n": len(soft_l1_topk_gate_by_bucket[b]),
                "mean": float(np.mean(soft_l1_topk_gate_by_bucket[b])) if soft_l1_topk_gate_by_bucket[b] else None,
                **{f"p{int(q*100)}": float(np.quantile(soft_l1_topk_gate_by_bucket[b], q))
                                    if soft_l1_topk_gate_by_bucket[b] else None
                   for q in (0.5, 0.75, 0.9, 0.95, 0.99)}}
            for b in bucket_names
        },
    }
    with open(args.out_json, "w") as f:
        json.dump(dump, f, indent=2)
    print(f"Saved → {args.out_json}")


if __name__ == "__main__":
    main()
