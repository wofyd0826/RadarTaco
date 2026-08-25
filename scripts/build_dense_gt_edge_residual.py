#!/usr/bin/env python3
"""Generate full-dataset dense GT using GPU edge-aware residual interp
(B depth-guided, σ_s=20, σ_r=1, default INVALID_FILL=dense_gt).

Output structure (matches existing depth_lidar / depth_acc layout):
    <out_dir>/scene_XXXX/{cam_basename}.png   uint16 × 256 → meters

Resumable: existing PNGs are skipped.
"""
import argparse
import contextlib
import io
import os
import sys

import numpy as np
import torch
from hydra import initialize_config_dir, compose
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.dataset.nuscenes import NuScenesRadarDepthDataset
from scripts.poisson_dense_gt_prototype import (
    edge_aware_residual_reconstruct_gpu,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", required=True,
                   help="e.g. /data/public/nuScenes/derived/depth_edge_d20r1")
    p.add_argument("--data_root", default="/data/public/nuScenes/derived")
    p.add_argument("--split", choices=["train", "val", "test"], default="train")
    p.add_argument("--shape_dir", default="depth_refined")
    p.add_argument("--sigma_s", type=float, default=20.0)
    p.add_argument("--sigma_r", type=float, default=1.0)
    p.add_argument("--K", type=int, default=200)
    p.add_argument("--guide", default="depth", choices=["depth", "depth_log", "rgb"])
    p.add_argument("--max_depth", type=float, default=100.0)
    p.add_argument("--min_depth", type=float, default=1e-3)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    with initialize_config_dir(
        config_dir=os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "config")),
        version_base=None,
    ):
        cfg = compose(config_name="config", overrides=[
            "+experiment=shape_lidar_grad_cos",
            f"dataset.dense_gt_dir={args.shape_dir}",
        ])

    ds_cfg = cfg.dataset
    split_file = getattr(ds_cfg, f"split_{args.split}")
    ds = NuScenesRadarDepthDataset(
        data_root=args.data_root,
        split_file=split_file,
        dense_gt_dir=args.shape_dir,
        lidar_gt_dir="depth_lidar",
        radar_3d_dir=ds_cfg.get("radar_3d_dir", "radar_3d"),
        max_radar_points=int(ds_cfg.max_radar_points),
        max_depth=args.max_depth,
        min_depth=args.min_depth,
        augmentation=False, photometric_aug=None,
    )

    print(f"Output     : {args.out_dir}")
    print(f"Split      : {args.split} ({len(ds)} samples)")
    print(f"Shape dir  : {args.shape_dir}")
    print(f"Params     : σ_s={args.sigma_s}  σ_r={args.sigma_r}  K={args.K}  guide={args.guide}")
    print(f"GPU        : {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only'}")

    n_done = 0
    n_skip = 0
    sink = io.StringIO()
    pbar = tqdm(range(len(ds)), dynamic_ncols=True)
    for i in pbar:
        sample = ds[i]
        sample_id = sample["sample_id"]                  # 'scene_XXXX/cam_basename'
        out_path = os.path.join(args.out_dir, f"{sample_id}.png")
        if os.path.exists(out_path):
            n_skip += 1
            continue
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        dense_gt = sample["depth_gt_dense"][0].numpy()
        lidar = sample["depth_gt_lidar"][0].numpy()
        mask_l = sample["valid_mask_lidar"][0].numpy().astype(bool)
        mask_d = sample["valid_mask_dense"][0].numpy().astype(bool)
        rgb = sample["rgb_norm"].numpy()
        mask_l = mask_l & mask_d                          # (A)

        with contextlib.redirect_stdout(sink):
            dense_new = edge_aware_residual_reconstruct_gpu(
                dense_gt, lidar, mask_l, rgb,
                sigma_s=args.sigma_s, sigma_r=args.sigma_r,
                K=args.K, guide=args.guide,
            )
        sink.seek(0); sink.truncate(0)
        # (C) default fill: dense_gt 그대로 (sky 등 100m cap 영역)
        dense_new = np.where(mask_d, dense_new, dense_gt)
        # Save as uint16 × 256 (matches existing depth_lidar / depth_acc encoding)
        arr = np.clip(np.round(dense_new * 256.0), 0, 65535).astype(np.uint16)
        Image.fromarray(arr).save(out_path)
        n_done += 1
        if n_done % 50 == 0:
            pbar.set_postfix(done=n_done, skipped=n_skip)

    print(f"\nDone. {n_done} written, {n_skip} skipped (already existed).")


if __name__ == "__main__":
    main()
