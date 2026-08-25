"""Per-image least-squares scale+shift alignment to LiDAR GT, then MAE/RMSE.

For each test image:
  1. Load pred (depth_refined / depth_refined_grid uint16 ×256).
  2. Load sparse LiDAR (depth_lidar uint16 ×256).
  3. Build valid mask: gt ∈ (min_depth, max_depth) AND pred ∈ (eps, 65535-1)
     so 0 (invalid sentinel) and 65535 (saturation) pred pixels are excluded.
  4. Solve closed-form LS: argmin_{s, b} Σ (s·pred_i + b − gt_i)²
     using normal equations on the lidar-valid pixels.
  5. pred_aligned = s · pred + b
  6. Compute MAE/RMSE per range bin, on the same lidar-valid pixels.

Run on multiple pred dirs side by side; print a combined table.
"""
import argparse
import csv
import json
import os
import sys

import numpy as np
from PIL import Image
from tqdm import tqdm

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from src.evaluation.metrics import compute_range_metrics, RANGE_BINS_OVERALL, RANGE_BINS_FAR  # noqa: E402


def load_depth_uint16(path, scale=256.0):
    return np.asarray(Image.open(path), dtype=np.float32) / scale


def ls_align(pred, gt, mask, eps=1e-7):
    """Closed-form (s, b) = argmin Σ_mask (s·pred + b − gt)²."""
    p = pred[mask].astype(np.float64)
    g = gt[mask].astype(np.float64)
    if len(p) < 2:
        return 1.0, 0.0
    # Normal equations: [Σp²  Σp; Σp  N] · [s; b] = [Σpg; Σg]
    n = len(p)
    sp = p.sum()
    sg = g.sum()
    spp = (p * p).sum()
    spg = (p * g).sum()
    det = spp * n - sp * sp
    if abs(det) < eps:
        return 1.0, 0.0
    s = (spg * n - sg * sp) / det
    b = (sg - s * sp) / n
    return float(s), float(b)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pred_dirs", nargs="+", required=True,
                   help="One or more pred dirs, evaluated side by side.")
    p.add_argument("--labels", nargs="+", default=None,
                   help="Optional labels for each pred dir (defaults to basename).")
    p.add_argument("--data_root", required=True)
    p.add_argument("--split", required=True)
    p.add_argument("--gt_dir", default="depth_lidar")
    p.add_argument("--min_depth", type=float, default=1e-3)
    p.add_argument("--max_depth", type=float, default=100.0)
    p.add_argument("--pred_scale", type=float, default=256.0)
    p.add_argument("--gt_scale", type=float, default=256.0)
    p.add_argument("--exclude_pred_sat", action="store_true", default=True,
                   help="Exclude pred pixels equal to 65535 (uint16 saturation).")
    p.add_argument("--no_exclude_pred_sat", dest="exclude_pred_sat",
                   action="store_false")
    p.add_argument("--no_align", action="store_true",
                   help="Skip per-image LS alignment; report MAE/RMSE of raw "
                        "pred against LiDAR at LiDAR-valid pixels.")
    p.add_argument("--out_csv", default=None,
                   help="Optional path to save a per-sample csv (one row per "
                        "(label, sample_id)).")
    p.add_argument("--limit", type=int, default=0)
    args = p.parse_args()

    labels = args.labels or [os.path.basename(d.rstrip("/")) for d in args.pred_dirs]
    assert len(labels) == len(args.pred_dirs), \
        "--labels must match --pred_dirs length"

    samples = []
    with open(args.split) as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            samples.append(ln.split("\t")[0])
    if args.limit:
        samples = samples[:args.limit]
    print(f"# samples: {len(samples)}  splits: {args.split}")
    print(f"# pred dirs:")
    for lab, d in zip(labels, args.pred_dirs):
        print(f"  {lab}: {d}")
    print(f"# gt_dir: {args.data_root}/{args.gt_dir}")
    print(f"# range bins overall: {RANGE_BINS_OVERALL}")
    print(f"# range bins far:     {RANGE_BINS_FAR}\n")

    # Per-dir per-sample metrics. Aggregate by mean over samples per range bin.
    range_bins = RANGE_BINS_OVERALL + RANGE_BINS_FAR
    bin_keys = [f"{d_min}-{d_max}m" for d_min, d_max in range_bins]
    metric_keys = ["mae", "rmse"]

    rows = []  # per-sample csv
    summaries = {lab: {bk: {mk: [] for mk in metric_keys} for bk in bin_keys}
                 for lab in labels}
    # Also report per-sample scale & shift stats
    align_stats = {lab: {"scale": [], "shift": []} for lab in labels}
    pixel_counts = {lab: 0 for lab in labels}
    skipped = {lab: 0 for lab in labels}

    for sid in tqdm(samples):
        gt_path = os.path.join(args.data_root, args.gt_dir, f"{sid}.png")
        if not os.path.exists(gt_path):
            for lab in labels:
                skipped[lab] += 1
            continue
        gt = load_depth_uint16(gt_path, args.gt_scale)
        for lab, pred_dir in zip(labels, args.pred_dirs):
            pred_path = os.path.join(pred_dir, f"{sid}.png")
            if not os.path.exists(pred_path):
                skipped[lab] += 1
                continue
            pred_raw = np.asarray(Image.open(pred_path))
            pred = pred_raw.astype(np.float32) / args.pred_scale
            if pred.shape != gt.shape:
                skipped[lab] += 1
                continue
            # Valid mask: lidar-valid AND pred not invalid/saturated.
            valid = (gt > args.min_depth) & (gt < args.max_depth) & np.isfinite(gt)
            valid &= np.isfinite(pred) & (pred_raw != 0)
            if args.exclude_pred_sat:
                valid &= (pred_raw != 65535)
            if int(valid.sum()) < 10:
                skipped[lab] += 1
                continue
            if args.no_align:
                s, b = 1.0, 0.0
                pred_aligned = pred
            else:
                s, b = ls_align(pred, gt, valid)
                pred_aligned = s * pred + b
            # Per-range metrics on lidar-valid pixels (same valid mask)
            r = compute_range_metrics(
                pred_aligned, gt, valid, range_bins,
                min_depth=args.min_depth, max_depth=args.max_depth,
                metric_keys=metric_keys,
            )
            row = {"label": lab, "sample_id": sid, "scale": s, "shift": b,
                   "n_valid": int(valid.sum())}
            for bk in bin_keys:
                for mk in metric_keys:
                    v = r[bk][mk]
                    if np.isfinite(v):
                        summaries[lab][bk][mk].append(v)
                    row[f"{bk}/{mk}"] = v
            rows.append(row)
            align_stats[lab]["scale"].append(s)
            align_stats[lab]["shift"].append(b)
            pixel_counts[lab] += int(valid.sum())

    # Aggregate: mean over samples
    print("\n" + "=" * 90)
    print(f"{'label':<24} {'range':<10} {'N':>6} {'MAE':>9} {'RMSE':>9}")
    print("-" * 90)
    for lab in labels:
        for bk in bin_keys:
            vals_mae = summaries[lab][bk]["mae"]
            vals_rmse = summaries[lab][bk]["rmse"]
            n = len(vals_mae)
            mae = np.mean(vals_mae) if vals_mae else float("nan")
            rmse = np.mean(vals_rmse) if vals_rmse else float("nan")
            print(f"{lab:<24} {bk:<10} {n:>6} {mae:>9.4f} {rmse:>9.4f}")
        print("-" * 90)

    # Alignment param summary
    print("\nAlignment params (per-sample):")
    print(f"{'label':<24} {'N':>6} {'scale mean':>12} {'scale med':>12} "
          f"{'shift mean':>12} {'shift med':>12}")
    for lab in labels:
        sc = np.asarray(align_stats[lab]["scale"])
        sh = np.asarray(align_stats[lab]["shift"])
        if len(sc) == 0:
            continue
        print(f"{lab:<24} {len(sc):>6} {sc.mean():>12.4f} {np.median(sc):>12.4f} "
              f"{sh.mean():>12.4f} {np.median(sh):>12.4f}")

    print("\nSkipped (missing pred / shape mismatch / too few inliers):")
    for lab in labels:
        print(f"  {lab}: {skipped[lab]}")

    if args.out_csv:
        os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
        with open(args.out_csv, "w", newline="") as f:
            if rows:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w.writeheader()
                for r in rows:
                    w.writerow(r)
        print(f"\nWrote per-sample csv: {args.out_csv}")


if __name__ == "__main__":
    main()
