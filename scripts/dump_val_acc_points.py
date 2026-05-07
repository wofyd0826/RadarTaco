#!/usr/bin/env python3
"""Pick best 10 / worst 10 val samples by overall/0-80m/mae, run inference,
and dump per-sample points (gt from depth_acc, pred at gt locations, radar)
plus a flat CSV (one row per gt pixel with nearest radar attached).

Outputs:
    output/baseline/eval_val/points_compare_acc.json
    output/baseline/eval_val/points_compare_acc/{label}.csv
    output/baseline/eval_val/points_compare_acc/all.csv
"""
import csv
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from src.dataset.nuscenes import NuScenesRadarDepthDataset            # noqa: E402
from src.model.radartaco import RadarTaco                             # noqa: E402

CKPT = os.path.join(ROOT, "output/baseline/best.pt")
CFG_PATH = os.path.join(ROOT, "output/baseline/config.yaml")
EVAL_DIR = os.path.join(ROOT, "output/baseline/eval_val")
PER_SAMPLE_CSV = os.path.join(EVAL_DIR, "per_sample.csv")
JSON_OUT = os.path.join(EVAL_DIR, "points_compare_acc.json")
CSV_DIR = os.path.join(EVAL_DIR, "points_compare_acc")

K = 10  # best K + worst K


def pick_samples():
    df = pd.read_csv(PER_SAMPLE_CSV)
    df = df.dropna(subset=["overall/0-80m/mae"])
    df = df.sort_values("overall/0-80m/mae")
    best = df.head(K)[["sample_id", "overall/0-80m/mae"]].values.tolist()
    worst = df.tail(K)[["sample_id", "overall/0-80m/mae"]].values.tolist()
    selected = []
    for i, (sid, mae) in enumerate(best, 1):
        selected.append((sid, f"best_{i:02d}", float(mae)))
    for i, (sid, mae) in enumerate(reversed(worst), 1):
        selected.append((sid, f"worst_{i:02d}", float(mae)))
    return selected


def make_csv_rows(sample):
    gt = sample["gt_points"]
    pred = sample["pred_points"]
    radar = sample["radar_points"]
    if len(radar) == 0:
        rxy = np.zeros((0, 2), dtype=np.float32)
        rd = np.zeros((0,), dtype=np.float32)
    else:
        rxy = np.array([[r["x"], r["y"]] for r in radar], dtype=np.float32)
        rd = np.array([r["depth"] for r in radar], dtype=np.float32)
    rows = []
    for g, p in zip(gt, pred):
        x, y = g["x"], g["y"]
        row = {
            "sample_id": sample["sample_id"], "label": sample["label"],
            "x": x, "y": y,
            "gt_depth": g["depth"], "pred_depth": p["depth"],
            "pred_abs_err": p["abs_err"],
        }
        if len(radar) == 0:
            row.update({k: "" for k in ("nearest_radar_x", "nearest_radar_y",
                                        "nearest_radar_depth", "radar_pix_dist",
                                        "radar_depth_abs_err")})
        else:
            d = np.hypot(rxy[:, 0] - x, rxy[:, 1] - y)
            j = int(np.argmin(d))
            row.update({
                "nearest_radar_x": float(rxy[j, 0]),
                "nearest_radar_y": float(rxy[j, 1]),
                "nearest_radar_depth": float(rd[j]),
                "radar_pix_dist": float(d[j]),
                "radar_depth_abs_err": float(abs(rd[j] - g["depth"])),
            })
        rows.append(row)
    return rows


FIELDS = [
    "sample_id", "label", "x", "y",
    "gt_depth", "pred_depth", "pred_abs_err",
    "nearest_radar_x", "nearest_radar_y", "nearest_radar_depth",
    "radar_pix_dist", "radar_depth_abs_err",
]


def main():
    cfg = OmegaConf.load(CFG_PATH)
    selected = pick_samples()
    print(f"selected {len(selected)} samples (top/bottom {K} by 0-80m MAE)")

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
    sid2idx = {s["sample_id"]: i for i, s in enumerate(ds.samples)}

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

    out_json = {
        "checkpoint": os.path.relpath(CKPT, ROOT),
        "split": "val",
        "gt_source": "depth_acc (depth_gt_dense)",
        "channel_layout": {
            "radar_points_columns": ["front_m", "left_m", "up_m",
                                     "x_pix", "y_pix", "depth_m"],
            "gt_pred_point": ["x_pix", "y_pix", "depth_m"],
        },
        "samples": [],
    }
    all_rows = []
    os.makedirs(CSV_DIR, exist_ok=True)

    for sid, label, csv_mae in selected:
        if sid not in sid2idx:
            print(f"  SKIP {sid} (not in val split)")
            continue
        sample = ds[sid2idx[sid]]
        rgb = sample["rgb_norm"].unsqueeze(0).to(device)
        rpts = sample["radar_points"].unsqueeze(0).to(device)
        rmask = sample["radar_mask"].unsqueeze(0).to(device)
        with torch.inference_mode():
            pred = model(rgb, rpts, rmask)
        pred_hw = pred[0, 0].float().cpu().numpy()
        gt_hw = sample["depth_gt_dense"][0].numpy()
        valid = sample["valid_mask_dense"][0].numpy().astype(bool)

        ys, xs = np.where(valid)
        gt_d = gt_hw[ys, xs]; pred_d = pred_hw[ys, xs]
        ae = np.abs(pred_d - gt_d)
        m_080 = (gt_d > 0) & (gt_d < 80.0)
        mae_080 = float(ae[m_080].mean()) if m_080.any() else float("nan")

        gt_points = [{"x": int(x), "y": int(y), "depth": float(d)}
                     for x, y, d in zip(xs, ys, gt_d)]
        pred_points = [{"x": int(x), "y": int(y), "depth": float(d), "abs_err": float(e)}
                       for x, y, d, e in zip(xs, ys, pred_d, ae)]

        rmask_np = sample["radar_mask"].numpy().astype(bool)
        rpts_np = sample["radar_points"].numpy()[rmask_np]
        radar_points = [
            {"x": float(p[3]), "y": float(p[4]), "depth": float(p[5]),
             "front_m": float(p[0]), "left_m": float(p[1]), "up_m": float(p[2])}
            for p in rpts_np
        ]

        sample_dict = {
            "label": label, "sample_id": sid,
            "is_night": bool(sample["is_night"].item()),
            "image_hw": list(gt_hw.shape),
            "csv_mae_0_80m_lidar": csv_mae,
            "recomputed_mae_0_80m_acc": mae_080,
            "n_gt_points": len(gt_points),
            "n_radar_points": len(radar_points),
            "gt_points": gt_points,
            "pred_points": pred_points,
            "radar_points": radar_points,
        }
        out_json["samples"].append(sample_dict)
        rows = make_csv_rows(sample_dict)
        all_rows.extend(rows)

        path = os.path.join(CSV_DIR, f"{label}.csv")
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS); w.writeheader(); w.writerows(rows)

        print(f"  {label:8s}  {sid}  gt={len(gt_points):5d}  radar={len(radar_points):3d}  "
              f"lidar_mae={csv_mae:.3f}  acc_mae={mae_080:.3f}")

    with open(JSON_OUT, "w") as f:
        json.dump(out_json, f, indent=2)
    print(f"wrote {JSON_OUT}  ({os.path.getsize(JSON_OUT)/1024:.1f} KB)")

    path_all = os.path.join(CSV_DIR, "all.csv")
    with open(path_all, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS); w.writeheader(); w.writerows(all_rows)
    print(f"wrote {path_all}  ({len(all_rows)} rows)")


if __name__ == "__main__":
    sys.exit(main())
