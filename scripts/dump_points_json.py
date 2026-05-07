#!/usr/bin/env python3
"""Dump aligned (x, y, depth) for GT lidar points, model predictions at GT
locations, and radar points — for the 2 best / 2 worst MAE samples of the
RadarTaco baseline checkpoint on the test split.

Usage:
    python scripts/dump_points_json.py
"""
import json
import os
import sys

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from src.dataset.nuscenes import NuScenesRadarDepthDataset            # noqa: E402
from src.model.radartaco import RadarTaco                             # noqa: E402

CKPT = os.path.join(ROOT, "output/baseline/best.pt")
CFG_PATH = os.path.join(ROOT, "output/baseline/config.yaml")
OUT_PATH = os.path.join(ROOT, "output/baseline/eval_test/points_compare.json")

# (sample_id, label, mae_0_80m) — picked from per_sample.csv
SELECTED = [
    ("scene_10062/n008-2018-08-28-15-47-40-0400__CAM_FRONT__1535486104912404",
     "best_1", 0.1549),
    ("scene_10062/n008-2018-08-28-15-47-40-0400__CAM_FRONT__1535486107362404",
     "best_2", 0.1565),
    ("scene_10142/n015-2018-11-14-19-45-36+0800__CAM_FRONT__1542196185362460",
     "worst_1", 7.1501),
    ("scene_10142/n015-2018-11-14-19-45-36+0800__CAM_FRONT__1542196183862460",
     "worst_2", 5.5351),
]


def main():
    cfg = OmegaConf.load(CFG_PATH)
    split_file = cfg.dataset.split_test

    ds = NuScenesRadarDepthDataset(
        data_root=cfg.dataset.data_root,
        split_file=split_file,
        dense_gt_dir=cfg.dataset.get("dense_gt_dir", "depth_acc"),
        radar_3d_dir=cfg.dataset.get("radar_3d_dir", "radar_3d"),
        night_ids_file=cfg.dataset.get("night_ids_file", None),
        max_radar_points=int(cfg.dataset.max_radar_points),
        max_depth=float(cfg.dataset.max_depth),
        min_depth=float(cfg.dataset.min_depth),
        resize_to_hw=None, augmentation=False,
    )
    sid2idx = {s["sample_id"]: i for i, s in enumerate(ds.samples)}
    missing = [sid for sid, _, _ in SELECTED if sid not in sid2idx]
    if missing:
        raise SystemExit(f"sample_id not in split: {missing}")

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

    out = {
        "checkpoint": os.path.relpath(CKPT, ROOT),
        "split": "test",
        "channel_layout": {
            "radar_points_columns": ["front_m", "left_m", "up_m",
                                     "x_pix", "y_pix", "depth_m"],
            "gt_pred_point": ["x_pix", "y_pix", "depth_m"],
        },
        "samples": [],
    }

    for sid, label, csv_mae in SELECTED:
        sample = ds[sid2idx[sid]]
        rgb_norm = sample["rgb_norm"].unsqueeze(0).to(device)
        radar_points = sample["radar_points"].unsqueeze(0).to(device)
        radar_mask = sample["radar_mask"].unsqueeze(0).to(device)
        with torch.inference_mode():
            pred = model(rgb_norm, radar_points, radar_mask)
        pred_hw = pred[0, 0].float().cpu().numpy()
        gt_hw = sample["depth_gt_lidar"][0].numpy()
        valid = sample["valid_mask_lidar"][0].numpy().astype(bool)

        ys, xs = np.where(valid)
        gt_d = gt_hw[ys, xs]
        pred_d = pred_hw[ys, xs]
        ae = np.abs(pred_d - gt_d)
        # MAE recomputed on 0-80m so we can confirm the CSV pick.
        m_080 = (gt_d > 0) & (gt_d < 80.0)
        mae_080 = float(ae[m_080].mean()) if m_080.any() else float("nan")

        gt_points = [
            {"x": int(x), "y": int(y), "depth": float(d)}
            for x, y, d in zip(xs, ys, gt_d)
        ]
        pred_points = [
            {"x": int(x), "y": int(y), "depth": float(d), "abs_err": float(e)}
            for x, y, d, e in zip(xs, ys, pred_d, ae)
        ]

        rmask = sample["radar_mask"].numpy().astype(bool)
        rpts = sample["radar_points"].numpy()[rmask]   # (Nr, 6)
        radar_points_out = [
            {"x": float(p[3]), "y": float(p[4]), "depth": float(p[5]),
             "front_m": float(p[0]), "left_m": float(p[1]), "up_m": float(p[2])}
            for p in rpts
        ]

        out["samples"].append({
            "label": label,
            "sample_id": sid,
            "is_night": bool(sample["is_night"].item()),
            "image_hw": list(gt_hw.shape),
            "csv_mae_0_80m": csv_mae,
            "recomputed_mae_0_80m": mae_080,
            "n_gt_points": len(gt_points),
            "n_radar_points": len(radar_points_out),
            "gt_points": gt_points,
            "pred_points": pred_points,
            "radar_points": radar_points_out,
        })
        print(f"{label:8s}  {sid}  gt={len(gt_points):5d}  "
              f"radar={len(radar_points_out):3d}  mae_csv={csv_mae:.3f}  "
              f"mae_recalc={mae_080:.3f}")

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
