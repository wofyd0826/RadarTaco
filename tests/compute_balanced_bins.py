"""Compute balanced depth bin edges from actual pixel distribution.

Samples N images from train split, computes pixel-weighted percentiles of
depth_gt_dense, and reports several candidate balanced-bin partitions.
"""
import argparse
import os
import sys
import time

import numpy as np
from omegaconf import OmegaConf

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, ROOT)

from src.dataset.nuscenes import NuScenesRadarDepthDataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--out", default="output/analysis/balanced_bins.txt")
    ap.add_argument("--split", default="train")
    args = ap.parse_args()

    cfg = OmegaConf.load(
        "output/shape_lidar_grad_shape_edge_res_fix/config.yaml")
    split_file = os.path.join(
        cfg.dataset.data_root, "splits", f"{args.split}.txt")
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
    print(f"{args.split} dataset: {len(ds)} samples", flush=True)

    idxs = np.linspace(0, len(ds) - 1, args.n).astype(int)
    all_valid = []
    t0 = time.time()
    for i, idx in enumerate(idxs):
        it = ds[int(idx)]
        dgt = it["depth_gt_dense"].numpy().squeeze()
        valid = (dgt > 0) & (dgt < 100.0)
        all_valid.append(dgt[valid].astype(np.float32))
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{args.n}  ({time.time()-t0:.1f}s)", flush=True)
    all_v = np.concatenate(all_valid)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    lines = []
    def _p(s=""):
        print(s, flush=True)
        lines.append(s)

    _p(f"\ntotal valid pixels sampled: {len(all_v):,}")
    _p(f"depth stats: min={all_v.min():.2f} max={all_v.max():.2f} "
       f"mean={all_v.mean():.2f} median={np.median(all_v):.2f}")

    _p("\n=== Current bins [0, 20, 50, 100] pixel share ===")
    for lo, hi in [(0.0, 20.0), (20.0, 50.0), (50.0, 100.0)]:
        n = int(((all_v >= lo) & (all_v < hi)).sum())
        _p(f"  [{lo:>5.1f}, {hi:>5.1f}):  {n:>10,d}  "
           f"({100*n/len(all_v):>6.2f}%)")

    _p("\n=== Balanced partitions ===")
    for K, name in [(3, "3-bin"), (4, "4-bin"), (5, "5-bin"), (6, "6-bin")]:
        pcts = [100 * i / K for i in range(1, K)]
        edges = np.percentile(all_v, pcts)
        bin_edges = [0.0] + [float(e) for e in edges] + [100.0]
        edge_str = ", ".join(f"{e:.1f}" for e in bin_edges)
        _p(f"\n{name} equal-population edges: [{edge_str}]")
        for i in range(K):
            lo, hi = bin_edges[i], bin_edges[i + 1]
            n = int(((all_v >= lo) & (all_v < hi)).sum())
            _p(f"  [{lo:>5.1f}, {hi:>6.1f}):  {n:>10,d}  "
               f"({100*n/len(all_v):>6.2f}%)")

    _p("\n=== Full percentile table (for custom bin design) ===")
    for p in [5, 10, 15, 20, 25, 30, 33, 40, 50, 60, 66, 70, 75, 80, 85, 90, 95, 99]:
        _p(f"  p{p:>3d}: {np.percentile(all_v, p):>6.2f} m")

    with open(args.out, "w") as f:
        f.write("\n".join(lines))
    print(f"\nsaved: {args.out}", flush=True)


if __name__ == "__main__":
    main()
