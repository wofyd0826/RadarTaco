#!/usr/bin/env python3
"""Compute MAE / RMSE of a dense depth dir vs depth_lidar, evaluated at
LiDAR pixels only, binned by max evaluation distance (POLAR convention).

Reports both per-split (train, val) and combined.
"""
import argparse
import os
import sys

import numpy as np
from PIL import Image
from tqdm import tqdm


def load_split(path: str):
    ids = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ids.append(line.split("\t")[0])
    return ids


def eval_split(dense_dir, lidar_dir, sample_ids, max_depths, min_eval=0.5):
    sums_abs = {d: 0.0 for d in max_depths}
    sums_sq = {d: 0.0 for d in max_depths}
    counts = {d: 0 for d in max_depths}
    n_missing = 0
    for sid in tqdm(sample_ids, dynamic_ncols=True):
        dp = os.path.join(dense_dir, f"{sid}.png")
        lp = os.path.join(lidar_dir, f"{sid}.png")
        if not os.path.exists(dp) or not os.path.exists(lp):
            n_missing += 1
            continue
        dense = np.asarray(Image.open(dp), dtype=np.float32) / 256.0
        lidar = np.asarray(Image.open(lp), dtype=np.float32) / 256.0
        for md in max_depths:
            m = (lidar > min_eval) & (lidar < md)
            if not m.any():
                continue
            diff = dense[m].astype(np.float64) - lidar[m].astype(np.float64)
            sums_abs[md] += np.abs(diff).sum()
            sums_sq[md] += (diff * diff).sum()
            counts[md] += int(m.sum())
    return sums_abs, sums_sq, counts, n_missing


def report(label, sums_abs, sums_sq, counts):
    print(f"\n--- {label} ---")
    print(f"  {'max_d':>6} | {'MAE (m)':>10} | {'RMSE (m)':>10} | {'#px':>14}")
    for md in sorted(sums_abs.keys()):
        n = counts[md]
        if n == 0:
            print(f"  {md:>6.0f} | {'NA':>10} | {'NA':>10} | {0:>14}")
            continue
        mae = sums_abs[md] / n
        rmse = (sums_sq[md] / n) ** 0.5
        print(f"  {md:>6.0f} | {mae:>10.4f} | {rmse:>10.4f} | {n:>14,d}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dense_dir", default="depth_edge_res",
                   help="subdir name under --data_root (e.g. depth_edge_res)")
    p.add_argument("--lidar_dir", default="depth_lidar")
    p.add_argument("--data_root", default="/data/public/nuScenes/derived")
    p.add_argument("--split_dir", default="/data/public/nuScenes/derived/splits")
    p.add_argument("--max_depths", type=float, nargs="+", default=[50, 70, 80])
    args = p.parse_args()

    dense_dir = os.path.join(args.data_root, args.dense_dir)
    lidar_dir = os.path.join(args.data_root, args.lidar_dir)

    train_ids = load_split(os.path.join(args.split_dir, "train.txt"))
    val_ids   = load_split(os.path.join(args.split_dir, "val.txt"))
    print(f"dense_dir : {dense_dir}")
    print(f"lidar_dir : {lidar_dir}")
    print(f"train     : {len(train_ids):,d} samples")
    print(f"val       : {len(val_ids):,d} samples")
    print(f"max_depths: {args.max_depths}")

    print("\nProcessing train ...")
    sa_t, ss_t, c_t, miss_t = eval_split(dense_dir, lidar_dir, train_ids, args.max_depths)
    print("\nProcessing val ...")
    sa_v, ss_v, c_v, miss_v = eval_split(dense_dir, lidar_dir, val_ids, args.max_depths)

    report(f"TRAIN ({len(train_ids)-miss_t} samples, {miss_t} missing)", sa_t, ss_t, c_t)
    report(f"VAL   ({len(val_ids)-miss_v} samples, {miss_v} missing)", sa_v, ss_v, c_v)

    # Combined
    sa_a = {d: sa_t[d] + sa_v[d] for d in args.max_depths}
    ss_a = {d: ss_t[d] + ss_v[d] for d in args.max_depths}
    c_a  = {d: c_t[d]  + c_v[d]  for d in args.max_depths}
    report(f"COMBINED ({len(train_ids)+len(val_ids)-miss_t-miss_v} samples)", sa_a, ss_a, c_a)


if __name__ == "__main__":
    main()
