#!/usr/bin/env python3
"""For every sample in the val split, compute the within-5-px radar
comparison using BOTH depth_lidar and depth_acc as GT, plus per-sample
GT statistics (count, min depth, max depth) for each.

Output: output/baseline/eval_val/all_val_lidar_vs_acc.csv  (one row per sample)
"""
import csv
import os
import sys
import time

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from src.dataset.nuscenes import NuScenesRadarDepthDataset            # noqa: E402
from src.model.radartaco import RadarTaco                             # noqa: E402

CKPT = os.path.join(ROOT, "output/baseline/best.pt")
CFG_PATH = os.path.join(ROOT, "output/baseline/config.yaml")
OUT_CSV = os.path.join(ROOT, "output/baseline/eval_val/all_val_lidar_vs_acc.csv")
RADIUS_PX = 5
R2 = RADIUS_PX ** 2

FIELDS = [
    "sample_id", "is_night",
    # lidar
    "lidar_n_gt", "lidar_gt_min", "lidar_gt_max",
    "lidar_n", "lidar_radar_mae", "lidar_pred_mae",
    # acc
    "acc_n_gt", "acc_gt_min", "acc_gt_max",
    "acc_n", "acc_radar_mae", "acc_pred_mae",
]


def collect_within5(radar_xy, radar_d, gt_hw, valid_mask, pred_hw, H, W):
    """Vectorized: for each radar point, find GT pixels in its 5-px disk."""
    if radar_xy.shape[0] == 0:
        return 0, float("nan"), float("nan")
    r_errs = []
    p_errs = []
    for (rx, ry), rd in zip(radar_xy, radar_d):
        x0 = max(0, int(np.floor(rx - RADIUS_PX)))
        x1 = min(W, int(np.ceil(rx + RADIUS_PX)) + 1)
        y0 = max(0, int(np.floor(ry - RADIUS_PX)))
        y1 = min(H, int(np.ceil(ry + RADIUS_PX)) + 1)
        sub = valid_mask[y0:y1, x0:x1]
        if not sub.any():
            continue
        ys, xs = np.where(sub)
        ys_g = ys + y0
        xs_g = xs + x0
        d2 = (xs_g - rx) ** 2 + (ys_g - ry) ** 2
        keep = d2 <= R2
        if not keep.any():
            continue
        ys_g = ys_g[keep]; xs_g = xs_g[keep]
        gt_d = gt_hw[ys_g, xs_g]
        pr_d = pred_hw[ys_g, xs_g]
        r_errs.append(np.abs(rd - gt_d))
        p_errs.append(np.abs(pr_d - gt_d))
    if not r_errs:
        return 0, float("nan"), float("nan")
    re = np.concatenate(r_errs); pe = np.concatenate(p_errs)
    return int(re.size), float(re.mean()), float(pe.mean())


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
                        num_workers=int(cfg.dataset.num_workers), pin_memory=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = RadarTaco(
        radar_encoder_name=cfg.model.radar_encoder,
        max_depth=float(cfg.dataset.max_depth),
        max_radar_points=int(cfg.dataset.max_radar_points),
        k_neighbors=int(cfg.model.k_neighbors),
        a_l=tuple(cfg.model.a_l),
        radar_channels=tuple(cfg.model.radar_channels),
        attn_heads=int(cfg.model.attn_heads),
        mlp_hidden=int(cfg.model.get("mlp_hidden", 128)),
        pretrained_image_encoder=False,
        output_mode=str(cfg.model.get("output_mode", "metric")),
        min_depth_clip=float(cfg.model.get("min_depth_clip", 0.5)),
        multi_scale=bool(cfg.model.get("multi_scale", False)),
        multi_scale_levels=tuple(cfg.model.get("multi_scale_levels", (2, 4, 8, 16))),
    ).to(device).eval()
    ckpt = torch.load(CKPT, map_location=device)
    model.load_state_dict(ckpt["model"], strict=True)
    print(f"loaded {CKPT} (epoch={ckpt.get('epoch', '?')})")

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    f = open(OUT_CSV, "w", newline="")
    w = csv.DictWriter(f, fieldnames=FIELDS); w.writeheader()

    t0 = time.time()
    for batch in tqdm(loader, total=len(loader)):
        sid = batch["sample_id"][0]
        rgb = batch["rgb_norm"].to(device, non_blocking=True)
        rpts = batch["radar_points"].to(device, non_blocking=True)
        rmask = batch["radar_mask"].to(device, non_blocking=True)
        with torch.inference_mode():
            pred = model(rgb, rpts, rmask)
        pred_hw = pred[0, 0].float().cpu().numpy()

        rmask_np = batch["radar_mask"][0].cpu().numpy().astype(bool)
        rpts_np = batch["radar_points"][0].cpu().numpy()[rmask_np]
        radar_xy = rpts_np[:, 3:5]; radar_d = rpts_np[:, 5]

        gt_l = batch["depth_gt_lidar"][0, 0].cpu().numpy()
        v_l = batch["valid_mask_lidar"][0, 0].cpu().numpy().astype(bool)
        gt_a = batch["depth_gt_dense"][0, 0].cpu().numpy()
        v_a = batch["valid_mask_dense"][0, 0].cpu().numpy().astype(bool)
        H, W = gt_l.shape

        n_gt_l = int(v_l.sum())
        if n_gt_l > 0:
            ld = gt_l[v_l]
            gt_l_min = float(ld.min()); gt_l_max = float(ld.max())
        else:
            gt_l_min = gt_l_max = float("nan")
        n_gt_a = int(v_a.sum())
        if n_gt_a > 0:
            ad = gt_a[v_a]
            gt_a_min = float(ad.min()); gt_a_max = float(ad.max())
        else:
            gt_a_min = gt_a_max = float("nan")

        n_l, rmae_l, pmae_l = collect_within5(radar_xy, radar_d, gt_l, v_l, pred_hw, H, W)
        n_a, rmae_a, pmae_a = collect_within5(radar_xy, radar_d, gt_a, v_a, pred_hw, H, W)

        w.writerow({
            "sample_id": sid,
            "is_night": bool(batch["is_night"][0].item()),
            "lidar_n_gt": n_gt_l, "lidar_gt_min": gt_l_min, "lidar_gt_max": gt_l_max,
            "lidar_n": n_l, "lidar_radar_mae": rmae_l, "lidar_pred_mae": pmae_l,
            "acc_n_gt": n_gt_a, "acc_gt_min": gt_a_min, "acc_gt_max": gt_a_max,
            "acc_n": n_a, "acc_radar_mae": rmae_a, "acc_pred_mae": pmae_a,
        })
    f.close()
    dt = time.time() - t0
    print(f"\nwrote {OUT_CSV}  ({len(ds)} rows, {dt:.1f}s)")


if __name__ == "__main__":
    sys.exit(main())
