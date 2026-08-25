"""Evaluate a refined dense-depth PNG dir against sparse LiDAR GT on ZJU.

For each sample_id in the split, computes |pred - lidar| at LiDAR-valid
pixels and aggregates MAE / RMSE / Rel / delta1 over depth ranges.

Compares one or more refined dirs (e.g. gt_refined, gt_refined_grid) side
by side so you can see which ROE mode best preserves LiDAR-anchored
metric.

Usage:
    python3 scripts/eval_zju_refined_vs_lidar.py \\
      --split /data/public/ZJU-4DRadarCam/data/test.txt \\
      --data_root /data/public/ZJU-4DRadarCam/data \\
      --gt_dir gt \\
      --pred_dirs gt_refined gt_refined_grid \\
      --out_csv /workspace/RadarTaco/output/zju_refined_eval/per_sample.csv \\
      --out_summary /workspace/RadarTaco/output/zju_refined_eval/summary.txt
"""
import argparse
import csv
import os
import sys
from collections import defaultdict

import numpy as np
from PIL import Image
from tqdm import tqdm

RANGES = [(0, 10), (10, 20), (20, 30), (30, 50), (50, 80), (0, 50), (0, 80), (0, 100)]


def read_depth(path: str, scale_unit: float) -> np.ndarray | None:
    if not os.path.exists(path):
        return None
    arr = np.asarray(Image.open(path), dtype=np.float32) / scale_unit
    return arr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--gt_dir", default="gt", help="sparse LiDAR GT dir")
    ap.add_argument("--pred_dirs", nargs="+", required=True,
                    help="one or more refined dense-depth dirs")
    ap.add_argument("--scale_unit", type=float, default=256.0)
    ap.add_argument("--min_depth", type=float, default=0.5)
    ap.add_argument("--max_depth", type=float, default=100.0)
    ap.add_argument("--out_csv", default=None)
    ap.add_argument("--out_summary", default=None)
    args = ap.parse_args()

    with open(args.split) as f:
        sids = [ln.strip().split("\t")[0] for ln in f if ln.strip()]
    print(f"[plan] {len(sids)} samples, {len(args.pred_dirs)} pred dirs")

    # per-pred accumulators: list of (sample_id, range_key, mae, rmse, rel, d1, n_pix)
    per_sample_rows = []
    # cumulative sums per (pred, range): [sum_abs, sum_sq, sum_rel, sum_d1, sum_n]
    agg = {pd: {r: np.zeros(5, dtype=np.float64) for r in RANGES}
           for pd in args.pred_dirs}
    n_skipped = defaultdict(int)

    for sid in tqdm(sids):
        gt = read_depth(os.path.join(args.data_root, args.gt_dir, f"{sid}.png"),
                        args.scale_unit)
        if gt is None:
            for pd in args.pred_dirs:
                n_skipped[pd] += 1
            continue
        valid_gt = (gt > args.min_depth) & (gt < args.max_depth) & np.isfinite(gt)
        if not valid_gt.any():
            for pd in args.pred_dirs:
                n_skipped[pd] += 1
            continue

        for pd in args.pred_dirs:
            pred = read_depth(os.path.join(args.data_root, pd, f"{sid}.png"),
                              args.scale_unit)
            if pred is None or pred.shape != gt.shape:
                n_skipped[pd] += 1
                continue
            valid = valid_gt & (pred > args.min_depth) & (pred < args.max_depth) \
                    & np.isfinite(pred)
            if not valid.any():
                n_skipped[pd] += 1
                continue
            err = np.abs(pred[valid] - gt[valid])
            gt_v = gt[valid]
            sq = err * err
            rel = err / np.maximum(gt_v, args.min_depth)
            ratio = np.maximum(pred[valid] / gt_v, gt_v / pred[valid])
            d1 = (ratio < 1.25).astype(np.float64)

            # per-sample summary across the union 0-80m range
            mask_80 = valid_gt & (gt < 80.0)
            if mask_80.any():
                pred_80 = pred[mask_80]
                gt_80 = gt[mask_80]
                e80 = np.abs(pred_80 - gt_80)
                per_sample_rows.append({
                    "sample_id": sid, "pred_dir": pd,
                    "mae_0_80": float(e80.mean()),
                    "rmse_0_80": float(np.sqrt((e80 ** 2).mean())),
                    "n_pix_0_80": int(mask_80.sum()),
                })

            for r in RANGES:
                lo, hi = r
                m = (gt_v >= lo) & (gt_v < hi)
                if not m.any():
                    continue
                agg[pd][r][0] += err[m].sum()
                agg[pd][r][1] += sq[m].sum()
                agg[pd][r][2] += rel[m].sum()
                agg[pd][r][3] += d1[m].sum()
                agg[pd][r][4] += m.sum()

    # --- aggregate ---
    summary = []
    for pd in args.pred_dirs:
        summary.append(f"=== {pd} ===")
        summary.append(f"  skipped: {n_skipped[pd]}")
        summary.append("  range        mae      rmse     rel      d1      n_pix")
        for r in RANGES:
            s_abs, s_sq, s_rel, s_d1, n = agg[pd][r]
            if n == 0:
                summary.append(f"  {r[0]:>3}-{r[1]:<3}m   (no pixels)")
                continue
            mae = s_abs / n
            rmse = np.sqrt(s_sq / n)
            rel = s_rel / n
            d1 = s_d1 / n
            summary.append(
                f"  {r[0]:>3}-{r[1]:<3}m  {mae:7.4f}  {rmse:7.4f}  "
                f"{rel:6.4f}  {d1:6.4f}  {int(n):>12d}"
            )
        summary.append("")

    text = "\n".join(summary)
    print()
    print(text)

    if args.out_summary:
        os.makedirs(os.path.dirname(args.out_summary), exist_ok=True)
        with open(args.out_summary, "w") as f:
            f.write(text + "\n")
        print(f"[ok] wrote {args.out_summary}")

    if args.out_csv:
        os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
        with open(args.out_csv, "w", newline="") as f:
            w = csv.DictWriter(
                f, fieldnames=["sample_id", "pred_dir",
                               "mae_0_80", "rmse_0_80", "n_pix_0_80"])
            w.writeheader()
            for row in per_sample_rows:
                w.writerow(row)
        print(f"[ok] wrote {args.out_csv}  ({len(per_sample_rows)} rows)")


if __name__ == "__main__":
    main()
