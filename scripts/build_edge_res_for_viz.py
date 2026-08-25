"""Generate depth_edge_res for a small list of test-split indices only.

Reuses the same edge-aware residual reconstruction as
`build_dense_gt_edge_residual.py` but reads from `test.txt` and writes
only for the requested sample indices — enough to unblock oracle-combine
visualisation on the test split (which lacks the pre-computed
depth_edge_res that trainval scenes have).
"""
import argparse
import contextlib
import io
import os
import sys

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, ROOT)

from src.dataset.nuscenes import NuScenesRadarDepthDataset          # noqa: E402
from scripts.poisson_dense_gt_prototype import (                    # noqa: E402
    edge_aware_residual_reconstruct_gpu,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indices", nargs="+", type=int, required=True,
                    help="test.txt sample indices to build")
    ap.add_argument("--out_dir", default="/data/public/nuScenes/derived/depth_edge_res")
    ap.add_argument("--data_root", default="/data/public/nuScenes/derived")
    ap.add_argument("--split_file",
                    default="/data/public/nuScenes/derived/splits/test.txt")
    ap.add_argument("--shape_dir", default="depth_refined",
                    help="DAv2-derived dense source (available for test)")
    ap.add_argument("--sigma_s", type=float, default=20.0)
    ap.add_argument("--sigma_r", type=float, default=1.0)
    ap.add_argument("--K", type=int, default=200)
    ap.add_argument("--guide", default="depth", choices=["depth", "depth_log", "rgb"])
    ap.add_argument("--max_depth", type=float, default=100.0)
    ap.add_argument("--min_depth", type=float, default=1e-3)
    args = ap.parse_args()

    ds = NuScenesRadarDepthDataset(
        data_root=args.data_root,
        split_file=args.split_file,
        dense_gt_dir=args.shape_dir,
        radar_3d_dir="radar_3d",
        night_ids_file=os.path.join(args.data_root, "splits/night_ids.txt"),
        max_radar_points=128,
        max_depth=args.max_depth,
        min_depth=args.min_depth,
        augmentation=False,
    )
    print(f"Split      : {args.split_file}  ({len(ds)} samples)")
    print(f"Shape dir  : {args.shape_dir}")
    print(f"Params     : σ_s={args.sigma_s} σ_r={args.sigma_r} K={args.K} guide={args.guide}")
    print(f"Indices    : {args.indices}")

    sink = io.StringIO()
    for i in tqdm(args.indices):
        sample = ds[i]
        sample_id = sample["sample_id"]
        out_path = os.path.join(args.out_dir, f"{sample_id}.png")
        if os.path.exists(out_path):
            print(f"skip (exists): {out_path}")
            continue
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        dense_gt = sample["depth_gt_dense"][0].numpy()
        lidar    = sample["depth_gt_lidar"][0].numpy()
        mask_l   = sample["valid_mask_lidar"][0].numpy().astype(bool)
        mask_d   = sample["valid_mask_dense"][0].numpy().astype(bool)
        rgb      = sample["rgb_norm"].numpy()
        mask_l   = mask_l & mask_d

        with contextlib.redirect_stdout(sink):
            dense_new = edge_aware_residual_reconstruct_gpu(
                dense_gt, lidar, mask_l, rgb,
                sigma_s=args.sigma_s, sigma_r=args.sigma_r,
                K=args.K, guide=args.guide,
            )
        sink.seek(0); sink.truncate(0)
        dense_new = np.where(mask_d, dense_new, dense_gt)
        arr = np.clip(np.round(dense_new * 256.0), 0, 65535).astype(np.uint16)
        Image.fromarray(arr).save(out_path)
        print(f"wrote: {out_path}")


if __name__ == "__main__":
    main()
