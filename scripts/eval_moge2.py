#!/usr/bin/env python3
"""Evaluate precomputed MoGe-2 depth maps on the nuScenes test split using
RadarTaco's DepthEvaluator (same range bins / day-night split as eval.py).

Predictions are expected to be uint16 PNGs (×256 m) at
    <pred_dir>/<sample_id>.png
matching the depth_lidar storage convention. GT is the sparse single-frame
LiDAR depth at
    <data_root>/depth_lidar/<sample_id>.png
The valid mask is exactly where depth_lidar > 0, which is RadarTaco's
test-time evaluation convention.

Usage:
    python scripts/eval_moge2.py \
        --pred_dir /data/public/nuScenes/derived/depth_moge2 \
        --data_root /data/public/nuScenes/derived \
        --split /data/public/nuScenes/derived/splits/test.txt \
        --night_ids /data/public/nuScenes/derived/splits/night_ids.txt \
        --out_dir /workspace/RadarTaco/output/moge2_eval_test
"""
import argparse
import csv
import json
import logging
import os
import sys

import numpy as np
from PIL import Image
from tqdm import tqdm

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from src.evaluation.metrics import DepthEvaluator  # noqa: E402

logging.basicConfig(format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                    level=logging.INFO)
logger = logging.getLogger("moge2.eval")


def load_depth_uint16(path: str, scale: float = 256.0) -> np.ndarray:
    return np.asarray(Image.open(path), dtype=np.float32) / scale


def load_night_ids(path):
    if not path or not os.path.exists(path):
        return set()
    with open(path) as f:
        return {ln.strip() for ln in f if ln.strip()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pred_dir", required=True,
                   help="<dir>/<sample_id>.png predictions (uint16 ×scale).")
    p.add_argument("--data_root", required=True,
                   help="nuScenes derived root (with depth_lidar/).")
    p.add_argument("--split", required=True)
    p.add_argument("--night_ids", default=None)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--pred_scale", type=float, default=256.0)
    p.add_argument("--gt_scale", type=float, default=256.0)
    p.add_argument("--min_depth", type=float, default=1e-3)
    p.add_argument("--max_depth", type=float, default=100.0)
    p.add_argument("--gt_dir", default="depth_lidar",
                   help="GT dir under data_root. Defaults to depth_lidar "
                        "(sparse single-frame lidar; matches RadarTaco "
                        "test-time eval).")
    p.add_argument("--limit", type=int, default=0)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    night_ids = load_night_ids(args.night_ids)

    samples = []
    with open(args.split) as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            sid = ln.split("\t")[0]
            samples.append(sid)
    if args.limit:
        samples = samples[:args.limit]
    logger.info(f"eval {len(samples)} samples  night_ids={len(night_ids)}")

    evaluator = DepthEvaluator(min_depth=args.min_depth, max_depth=args.max_depth)
    all_metrics = []
    per_sample_rows = []
    missing_pred = missing_gt = 0

    for sid in tqdm(samples):
        pred_path = os.path.join(args.pred_dir, f"{sid}.png")
        gt_path = os.path.join(args.data_root, args.gt_dir, f"{sid}.png")
        if not os.path.exists(pred_path):
            missing_pred += 1
            continue
        if not os.path.exists(gt_path):
            missing_gt += 1
            continue

        pred = load_depth_uint16(pred_path, args.pred_scale)
        gt = load_depth_uint16(gt_path, args.gt_scale)
        if pred.shape != gt.shape:
            logger.warning(f"shape mismatch for {sid}: pred {pred.shape} vs gt {gt.shape}")
            continue
        # valid mask follows RadarTaco's _make_depth_tensor convention
        valid_mask = (gt > args.min_depth) & (gt < args.max_depth) & np.isfinite(gt)

        is_night = sid in night_ids
        m = evaluator.evaluate_sample(pred, gt, valid_mask, is_night=is_night)
        all_metrics.append(m)

        row = {"sample_id": sid, "is_night": is_night}
        for cat, ranges in m.items():
            if not isinstance(ranges, dict):
                continue
            for rname, mets in ranges.items():
                if not isinstance(mets, dict):
                    continue
                for k, v in mets.items():
                    row[f"{cat}/{rname}/{k}"] = v
        per_sample_rows.append(row)

    if missing_pred or missing_gt:
        logger.warning(f"missing pred={missing_pred}  missing gt={missing_gt}")

    if not all_metrics:
        raise SystemExit("no samples evaluated; check paths and --pred_dir.")

    agg = evaluator.aggregate_metrics(all_metrics)
    with open(os.path.join(args.out_dir, "metrics.json"), "w") as f:
        json.dump(agg, f, indent=2)
    with open(os.path.join(args.out_dir, "summary.txt"), "w") as f:
        f.write(f"# MoGe-2 eval on nuScenes (samples={len(all_metrics)})\n")
        f.write(f"# pred_dir = {args.pred_dir}\n")
        f.write(f"# gt_dir   = {args.data_root}/{args.gt_dir}\n\n")
        for category, ranges in agg.items():
            f.write(f"=== {category} ===\n")
            for rname, metrics in ranges.items():
                line = "  " + rname + "  " + "  ".join(
                    f"{k}={v:.4f}" for k, v in metrics.items()
                )
                f.write(line + "\n")
            f.write("\n")

    if per_sample_rows:
        fieldnames = []
        seen = set()
        for r in per_sample_rows:
            for k in r:
                if k not in seen:
                    seen.add(k); fieldnames.append(k)
        path = os.path.join(args.out_dir, "per_sample.csv")
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            w.writeheader()
            for r in per_sample_rows:
                w.writerow(r)

    head = agg["overall"].get("0-80m", {})
    head100 = agg["overall"].get("0-100m", {})
    night = agg.get("night", {}).get("0-80m", {})
    far = agg.get("far", {}).get("50-80m", {})
    far100 = agg.get("far", {}).get("80-100m", {})
    logger.info(
        f"DONE  0-80m MAE={head.get('mae', float('nan')):.2f}/RMSE={head.get('rmse', float('nan')):.2f}  "
        f"|  0-100m MAE={head100.get('mae', float('nan')):.2f}/RMSE={head100.get('rmse', float('nan')):.2f}  "
        f"|  far 50-80m MAE={far.get('mae', float('nan')):.2f}  80-100m MAE={far100.get('mae', float('nan')):.2f}  "
        f"|  night MAE={night.get('mae', float('nan')):.2f}"
    )


if __name__ == "__main__":
    main()
