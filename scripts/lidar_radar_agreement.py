#!/usr/bin/env python3
"""How well do radar depths agree with lidar GT?

Pools every (radar_depth, lidar_gt_depth) pair within 5 px across all 6019
val samples and reports:
  - global stats: mean/median |Δ|, signed bias (radar - lidar), Pearson r
  - by lidar GT depth bin
  - by pixel offset (1, 2, 3, 5 px)
  - scatter plot of radar_depth vs lidar_gt_depth

Output:
  output/baseline/eval_val/analysis_acc/lidar_radar_pairs.npz       (raw pairs)
  output/baseline/eval_val/analysis_acc/lidar_radar_agreement.csv   (binned table)
  output/baseline/eval_val/analysis_acc/lidar_radar_scatter.png
"""
import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from src.dataset.nuscenes import NuScenesRadarDepthDataset            # noqa: E402

CFG_PATH = os.path.join(ROOT, "output/baseline/config.yaml")
OUT_DIR = os.path.join(ROOT, "output/baseline/eval_val/analysis_acc")
NPZ_PATH = os.path.join(OUT_DIR, "lidar_radar_pairs.npz")
TBL_PATH = os.path.join(OUT_DIR, "lidar_radar_agreement.csv")
PNG_PATH = os.path.join(OUT_DIR, "lidar_radar_scatter.png")
RADIUS_PX = 5
R2 = RADIUS_PX ** 2


def collect_pairs(radar_xy, radar_d, gt_hw, valid_mask, H, W):
    """Return three arrays: lidar_d, radar_d, pixel_dist for all pairs in 5 px."""
    lid = []; rad = []; pdist = []
    for (rx, ry), rd in zip(radar_xy, radar_d):
        x0 = max(0, int(np.floor(rx - RADIUS_PX)))
        x1 = min(W, int(np.ceil(rx + RADIUS_PX)) + 1)
        y0 = max(0, int(np.floor(ry - RADIUS_PX)))
        y1 = min(H, int(np.ceil(ry + RADIUS_PX)) + 1)
        sub = valid_mask[y0:y1, x0:x1]
        if not sub.any():
            continue
        ys, xs = np.where(sub)
        ys_g = ys + y0; xs_g = xs + x0
        d2 = (xs_g - rx) ** 2 + (ys_g - ry) ** 2
        keep = d2 <= R2
        if not keep.any():
            continue
        ys_g = ys_g[keep]; xs_g = xs_g[keep]
        gt_d = gt_hw[ys_g, xs_g]
        lid.append(gt_d)
        rad.append(np.full_like(gt_d, rd))
        pdist.append(np.sqrt(d2[keep]))
    if not lid:
        return np.zeros(0), np.zeros(0), np.zeros(0)
    return np.concatenate(lid), np.concatenate(rad), np.concatenate(pdist)


def main():
    cfg = OmegaConf.load(CFG_PATH)
    ds = NuScenesRadarDepthDataset(
        data_root=cfg.dataset.data_root,
        split_file=cfg.dataset.split_val,
        dense_gt_dir=cfg.dataset.get("dense_gt_dir", "depth_acc"),
        radar_3d_dir=cfg.dataset.get("radar_3d_dir", "radar_3d"),
        night_ids_file=cfg.dataset.get("night_ids_file", None),
        max_radar_points=int(cfg.dataset.max_radar_points),
        max_depth=float(cfg.dataset.max_depth),
        min_depth=float(cfg.dataset.min_depth),
        resize_to_hw=None, augmentation=False,
    )
    print(f"val samples: {len(ds)}")
    loader = DataLoader(ds, batch_size=1, shuffle=False,
                        num_workers=int(cfg.dataset.num_workers), pin_memory=False)

    L, R, P = [], [], []
    t0 = time.time()
    for batch in tqdm(loader, total=len(loader)):
        rmask = batch["radar_mask"][0].numpy().astype(bool)
        rpts = batch["radar_points"][0].numpy()[rmask]
        if rpts.shape[0] == 0:
            continue
        radar_xy = rpts[:, 3:5]; radar_d = rpts[:, 5]
        gt = batch["depth_gt_lidar"][0, 0].numpy()
        v = batch["valid_mask_lidar"][0, 0].numpy().astype(bool)
        H, W = gt.shape
        l, r, p = collect_pairs(radar_xy, radar_d, gt, v, H, W)
        if l.size:
            L.append(l); R.append(r); P.append(p)
    L = np.concatenate(L); R = np.concatenate(R); P = np.concatenate(P)
    dt = time.time() - t0
    print(f"\ncollected {L.size:,} (radar, lidar) pairs in {dt:.1f}s")
    os.makedirs(OUT_DIR, exist_ok=True)
    np.savez(NPZ_PATH, lidar=L, radar=R, pix_dist=P)
    print(f"wrote {NPZ_PATH}")

    # Global stats
    diff = R - L              # signed: positive = radar overshoots lidar
    abs_diff = np.abs(diff)
    pearson_r = float(np.corrcoef(L, R)[0, 1])
    print("\n=== global agreement ===")
    print(f"  n pairs       : {L.size:,}")
    print(f"  Pearson r     : {pearson_r:.4f}")
    print(f"  mean |Δ|      : {abs_diff.mean():.3f} m")
    print(f"  median |Δ|    : {np.median(abs_diff):.3f} m")
    print(f"  RMSE          : {np.sqrt((diff**2).mean()):.3f} m")
    print(f"  signed bias   : mean={diff.mean():+.3f}  median={np.median(diff):+.3f}")
    print(f"  |Δ| <= 1 m    : {(abs_diff <= 1.0).mean()*100:.1f}%")
    print(f"  |Δ| <= 2 m    : {(abs_diff <= 2.0).mean()*100:.1f}%")
    print(f"  |Δ| <= 5 m    : {(abs_diff <= 5.0).mean()*100:.1f}%")
    print(f"  lidar depth   : mean={L.mean():.2f}  median={np.median(L):.2f}")
    print(f"  radar depth   : mean={R.mean():.2f}  median={np.median(R):.2f}")

    # By lidar depth bin
    bins = [0, 5, 10, 20, 30, 50, 100]
    rows = []
    for i in range(len(bins) - 1):
        m = (L >= bins[i]) & (L < bins[i + 1])
        if not m.any():
            continue
        rows.append({
            "depth_bin": f"[{bins[i]}, {bins[i+1]})",
            "n": int(m.sum()),
            "mae": float(abs_diff[m].mean()),
            "rmse": float(np.sqrt((diff[m]**2).mean())),
            "bias": float(diff[m].mean()),
            "pearson_r": float(np.corrcoef(L[m], R[m])[0, 1]),
            "pct_within_1m": float((abs_diff[m] <= 1).mean()*100),
            "pct_within_5m": float((abs_diff[m] <= 5).mean()*100),
        })
    by_depth = pd.DataFrame(rows)

    # By pixel offset
    rows = []
    for thr in [1, 2, 3, 5]:
        m = P <= thr
        if not m.any(): continue
        rows.append({
            "max_pix_dist": thr, "n": int(m.sum()),
            "mae": float(abs_diff[m].mean()),
            "rmse": float(np.sqrt((diff[m]**2).mean())),
            "bias": float(diff[m].mean()),
            "pearson_r": float(np.corrcoef(L[m], R[m])[0, 1]),
        })
    by_pix = pd.DataFrame(rows)

    print("\n=== agreement by lidar depth bin ===")
    print(by_depth.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print("\n=== agreement by pixel offset cutoff ===")
    print(by_pix.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    by_depth.to_csv(TBL_PATH, index=False)
    print(f"\nwrote {TBL_PATH}")

    # Scatter (downsampled for plotting)
    if L.size > 100000:
        idx = np.random.RandomState(0).choice(L.size, 100000, replace=False)
        Ld, Rd = L[idx], R[idx]
    else:
        Ld, Rd = L, R
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(Ld, Rd, s=1, alpha=0.05, color="#1f77b4")
    lim = 100
    ax.plot([0, lim], [0, lim], "r-", lw=1, label="y = x")
    ax.set_xlabel("lidar GT depth (m)")
    ax.set_ylabel("radar depth (m)")
    ax.set_xlim(0, lim); ax.set_ylim(0, lim); ax.set_aspect("equal")
    ax.set_title(f"radar vs lidar (5 px colocated, n={L.size:,}, Pearson r={pearson_r:.3f})")
    ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout()
    fig.savefig(PNG_PATH, dpi=120, bbox_inches="tight"); plt.close(fig)
    print(f"wrote {PNG_PATH}")


if __name__ == "__main__":
    sys.exit(main())
