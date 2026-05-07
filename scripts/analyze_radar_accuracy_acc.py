#!/usr/bin/env python3
"""Same radar-accuracy analysis as analyze_radar_accuracy.py but with the
accumulated GT (`depth_acc`, i.e. `depth_gt_dense`) as ground truth instead
of the sparse single-frame `depth_lidar`. depth_acc has many more valid
pixels, so the within-5px sample size grows substantially.

For each radar projection we collect every depth_acc pixel within 5 px and
treat that as a GT-radar colocated comparison. Pred is sampled at those
same pixels.
"""
import os
import sys

import numpy as np
import torch
from omegaconf import OmegaConf

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from src.dataset.nuscenes import NuScenesRadarDepthDataset            # noqa: E402
from src.model.radartaco import RadarTaco                             # noqa: E402

CKPT = os.path.join(ROOT, "output/baseline/best.pt")
CFG_PATH = os.path.join(ROOT, "output/baseline/config.yaml")
OUT_CSV = os.path.join(ROOT, "output/baseline/eval_test/analysis/radar_accuracy_within5px_acc.csv")
RADIUS_PX = 5

SELECTED = [
    ("scene_10062/n008-2018-08-28-15-47-40-0400__CAM_FRONT__1535486104912404", "best_1"),
    ("scene_10062/n008-2018-08-28-15-47-40-0400__CAM_FRONT__1535486107362404", "best_2"),
    ("scene_10142/n015-2018-11-14-19-45-36+0800__CAM_FRONT__1542196185362460", "worst_1"),
    ("scene_10142/n015-2018-11-14-19-45-36+0800__CAM_FRONT__1542196183862460", "worst_2"),
]


def main():
    cfg = OmegaConf.load(CFG_PATH)
    ds = NuScenesRadarDepthDataset(
        data_root=cfg.dataset.data_root,
        split_file=cfg.dataset.split_test,
        dense_gt_dir=cfg.dataset.get("dense_gt_dir", "depth_acc"),  # = depth_acc
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
    for sid, label in SELECTED:
        sample = ds[sid2idx[sid]]
        rgb = sample["rgb_norm"].unsqueeze(0).to(device)
        rpts = sample["radar_points"].unsqueeze(0).to(device)
        rmask = sample["radar_mask"].unsqueeze(0).to(device)
        with torch.inference_mode():
            pred = model(rgb, rpts, rmask)
        pred_hw = pred[0, 0].float().cpu().numpy()
        gt_hw = sample["depth_gt_dense"][0].numpy()       # depth_acc
        valid_dense = sample["valid_mask_dense"][0].numpy().astype(bool)
        H, W = gt_hw.shape

        radar_pts_full = sample["radar_points"].numpy()
        rmask_np = sample["radar_mask"].numpy().astype(bool)
        radar_pts = radar_pts_full[rmask_np]              # (Nr, 6)
        radar_xy = radar_pts[:, 3:5]                      # (Nr, 2) pixel
        radar_d = radar_pts[:, 5]                         # depth

        # Collect every (radar, gt-pixel) pair within 5 px.
        radar_errs = []
        pred_errs = []
        gt_depths = []
        radar_depths_paired = []
        for (rx, ry), rd in zip(radar_xy, radar_d):
            x0 = max(0, int(np.floor(rx - RADIUS_PX)))
            x1 = min(W, int(np.ceil(rx + RADIUS_PX)) + 1)
            y0 = max(0, int(np.floor(ry - RADIUS_PX)))
            y1 = min(H, int(np.ceil(ry + RADIUS_PX)) + 1)
            ys, xs = np.where(valid_dense[y0:y1, x0:x1])
            if ys.size == 0:
                continue
            ys_g = ys + y0
            xs_g = xs + x0
            d2 = (xs_g - rx) ** 2 + (ys_g - ry) ** 2
            keep = d2 <= RADIUS_PX ** 2
            if not keep.any():
                continue
            ys_g = ys_g[keep]; xs_g = xs_g[keep]
            gt_d = gt_hw[ys_g, xs_g]
            pr_d = pred_hw[ys_g, xs_g]
            radar_errs.extend(np.abs(rd - gt_d).tolist())
            pred_errs.extend(np.abs(pr_d - gt_d).tolist())
            gt_depths.extend(gt_d.tolist())
            radar_depths_paired.extend([float(rd)] * gt_d.size)

        n = len(radar_errs)
        re = np.array(radar_errs); pe = np.array(pred_errs)
        gt_arr = np.array(gt_depths); rd_arr = np.array(radar_depths_paired)
        n_gt_total = int(valid_dense.sum())
        rows.append({
            "label": label,
            "sample_id": sid,
            "n_gt_total_dense": n_gt_total,
            "n_pairs_within5px": n,
            "coverage_pct": 100.0 * n / max(1, n_gt_total),
            "gt_depth_mean": float(gt_arr.mean()) if n else float("nan"),
            "radar_depth_mean": float(rd_arr.mean()) if n else float("nan"),
            "radar_depth_mae": float(re.mean()) if n else float("nan"),
            "radar_depth_rmse": float(np.sqrt((re ** 2).mean())) if n else float("nan"),
            "pred_mae_same": float(pe.mean()) if n else float("nan"),
            "pred_rmse_same": float(np.sqrt((pe ** 2).mean())) if n else float("nan"),
        })
        print(f"{label:8s}  pairs={n:5d}  radar_MAE={re.mean():.3f}  pred_MAE={pe.mean():.3f}")

    import pandas as pd
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print()
    print(df.to_string(index=False))
    print(f"\nwrote {OUT_CSV}")


if __name__ == "__main__":
    sys.exit(main())
