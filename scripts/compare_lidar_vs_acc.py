#!/usr/bin/env python3
"""For the same 20 val samples (best 10 / worst 10 by lidar 0-80m MAE),
compute the within-5-px radar comparison using BOTH depth_lidar (sparse,
single-frame) and depth_acc (accumulated dense) as GT, and report per-sample
n / radar MAE / pred MAE side by side.
"""
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
OUT_CSV = os.path.join(EVAL_DIR, "analysis_acc/lidar_vs_acc_radar5px.csv")
RADIUS_PX = 5
K = 10


def pick_samples():
    df = pd.read_csv(PER_SAMPLE_CSV).dropna(subset=["overall/0-80m/mae"])
    df = df.sort_values("overall/0-80m/mae")
    best = df.head(K)[["sample_id", "overall/0-80m/mae"]].values.tolist()
    worst = df.tail(K)[["sample_id", "overall/0-80m/mae"]].values.tolist()
    out = []
    for i, (sid, mae) in enumerate(best, 1):
        out.append((sid, f"best_{i:02d}", float(mae)))
    for i, (sid, mae) in enumerate(reversed(worst), 1):
        out.append((sid, f"worst_{i:02d}", float(mae)))
    return out


def collect_within5(radar_xy, radar_d, gt_hw, valid_mask, pred_hw, H, W):
    """For each radar projection, accumulate (radar, GT) pairs within 5 px."""
    radar_errs, pred_errs = [], []
    for (rx, ry), rd in zip(radar_xy, radar_d):
        x0 = max(0, int(np.floor(rx - RADIUS_PX)))
        x1 = min(W, int(np.ceil(rx + RADIUS_PX)) + 1)
        y0 = max(0, int(np.floor(ry - RADIUS_PX)))
        y1 = min(H, int(np.ceil(ry + RADIUS_PX)) + 1)
        ys, xs = np.where(valid_mask[y0:y1, x0:x1])
        if ys.size == 0:
            continue
        ys_g = ys + y0; xs_g = xs + x0
        d2 = (xs_g - rx) ** 2 + (ys_g - ry) ** 2
        keep = d2 <= RADIUS_PX ** 2
        if not keep.any():
            continue
        ys_g = ys_g[keep]; xs_g = xs_g[keep]
        gt_d = gt_hw[ys_g, xs_g]
        pr_d = pred_hw[ys_g, xs_g]
        radar_errs.extend(np.abs(rd - gt_d).tolist())
        pred_errs.extend(np.abs(pr_d - gt_d).tolist())
    re = np.array(radar_errs); pe = np.array(pred_errs)
    return {
        "n": int(len(re)),
        "radar_mae": float(re.mean()) if len(re) else float("nan"),
        "pred_mae": float(pe.mean()) if len(pe) else float("nan"),
    }


def main():
    cfg = OmegaConf.load(CFG_PATH)
    selected = pick_samples()
    print(f"selected {len(selected)} samples")

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

    rows = []
    for sid, label, csv_mae in selected:
        if sid not in sid2idx:
            print(f"  SKIP {sid}"); continue
        sample = ds[sid2idx[sid]]
        rgb = sample["rgb_norm"].unsqueeze(0).to(device)
        rpts = sample["radar_points"].unsqueeze(0).to(device)
        rmask = sample["radar_mask"].unsqueeze(0).to(device)
        with torch.inference_mode():
            pred = model(rgb, rpts, rmask)
        pred_hw = pred[0, 0].float().cpu().numpy()

        rmask_np = sample["radar_mask"].numpy().astype(bool)
        rpts_np = sample["radar_points"].numpy()[rmask_np]
        radar_xy = rpts_np[:, 3:5]; radar_d = rpts_np[:, 5]

        gt_lidar = sample["depth_gt_lidar"][0].numpy()
        v_lidar = sample["valid_mask_lidar"][0].numpy().astype(bool)
        gt_acc = sample["depth_gt_dense"][0].numpy()
        v_acc = sample["valid_mask_dense"][0].numpy().astype(bool)
        H, W = gt_lidar.shape

        s_lidar = collect_within5(radar_xy, radar_d, gt_lidar, v_lidar, pred_hw, H, W)
        s_acc = collect_within5(radar_xy, radar_d, gt_acc, v_acc, pred_hw, H, W)

        rows.append({
            "label": label, "sample_id": sid,
            "lidar_n": s_lidar["n"],
            "lidar_radar_mae": s_lidar["radar_mae"],
            "lidar_pred_mae": s_lidar["pred_mae"],
            "acc_n": s_acc["n"],
            "acc_radar_mae": s_acc["radar_mae"],
            "acc_pred_mae": s_acc["pred_mae"],
        })
        print(f"  {label:9s}  lidar n={s_lidar['n']:4d} radar={s_lidar['radar_mae']:6.2f} pred={s_lidar['pred_mae']:6.2f}  "
              f"|  acc n={s_acc['n']:5d} radar={s_acc['radar_mae']:6.2f} pred={s_acc['pred_mae']:6.2f}")

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    df.to_csv(OUT_CSV, index=False)

    print()
    print(df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    grp = df.copy()
    grp["group"] = np.where(grp["label"].str.startswith("best_"), "best", "worst")
    rollup = grp.groupby("group").agg(
        lidar_n_total=("lidar_n", "sum"),
        lidar_radar_mae=("lidar_radar_mae", "mean"),
        lidar_pred_mae=("lidar_pred_mae", "mean"),
        acc_n_total=("acc_n", "sum"),
        acc_radar_mae=("acc_radar_mae", "mean"),
        acc_pred_mae=("acc_pred_mae", "mean"),
    ).reset_index()
    print("\n=== group rollup (mean of per-sample MAE) ===")
    print(rollup.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"\nwrote {OUT_CSV}")


if __name__ == "__main__":
    sys.exit(main())
