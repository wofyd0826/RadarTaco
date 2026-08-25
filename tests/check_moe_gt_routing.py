"""Verify that _compute_router_gt correctly assigns each training sample
to the depth-bin whose pixels dominate, and check bin-assignment balance
across the whole train split (=how much training signal each expert
actually gets in Stage 1)."""
import os
import sys

import numpy as np
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, ROOT)

from src.dataset.nuscenes import NuScenesRadarDepthDataset  # noqa: E402
from src.model.radartaco import RadarTaco                    # noqa: E402


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    bins = (0.0, 20.0, 50.0, 100.0)
    labels = ["near[0-20)", "mid[20-50)", "far[50-100)"]

    # Load the same dataset settings the trainer uses
    ds = NuScenesRadarDepthDataset(
        data_root="/data/public/nuScenes/derived",
        split_file="/data/public/nuScenes/derived/splits/train.txt",
        dense_gt_dir="depth_edge_res",
        radar_3d_dir="radar_3d",
        night_ids_file="/data/public/nuScenes/derived/splits/night_ids.txt",
        max_radar_points=128, max_depth=100.0, min_depth=1e-3,
        augmentation=False,
    )
    print(f"train samples: {len(ds)}")

    # ---- (A) verify _compute_router_gt on a few real samples ----
    print("\n=== (A) Per-sample GT check on 8 real samples ===")
    print(f"{'idx':>5} {'near_frac':>10} {'mid_frac':>10} {'far_frac':>10} {'GT_bin':>10}")
    for k in range(8):
        idx = k * (len(ds) // 8)
        s = ds[idx]
        d = s["depth_gt_dense"].unsqueeze(0).to(device)     # (1,1,H,W)
        gt_bin = RadarTaco._compute_router_gt(d, bins)[0].item()
        n = float(((d > 0) & (d < 20)).float().mean())
        m = float(((d >= 20) & (d < 50)).float().mean())
        f = float((d >= 50).float().mean())
        print(f"{idx:>5} {n:>10.3f} {m:>10.3f} {f:>10.3f} {labels[gt_bin]:>10}")

    # ---- (B) whole train split bin distribution ----
    print("\n=== (B) GT-bin assignment across FULL train split ===")
    counts = np.zeros(3, dtype=np.int64)
    for i in tqdm(range(len(ds)), desc="scanning"):
        d = ds[i]["depth_gt_dense"].unsqueeze(0)            # cpu OK
        gt_bin = RadarTaco._compute_router_gt(d, bins)[0].item()
        counts[gt_bin] += 1
    total = int(counts.sum())
    print(f"\nbin-assignment counts (n={total}):")
    for k, l in enumerate(labels):
        pct = 100.0 * counts[k] / total
        bar = "█" * int(pct)
        print(f"  {l:>13}   {counts[k]:>6,}  ({pct:5.1f}%)  {bar}")

    print("\n→ expert training-signal ratio in Stage-1 teacher-forcing:")
    print(f"  Expert  Effective train samples/epoch")
    for k, l in enumerate(labels):
        print(f"  {l:>13}    {counts[k]:>6,}")

    # Guidance if very unbalanced
    m = counts.max() / counts.min() if counts.min() > 0 else float("inf")
    print(f"\nmax/min imbalance ratio: {m:.1f}×")
    if m > 5:
        print("  ⚠️  strong imbalance — under-represented expert may under-train.")
    if counts.min() < 500:
        print(f"  ⚠️  under-represented expert sees < 500 samples/epoch — very sparse.")


if __name__ == "__main__":
    main()
