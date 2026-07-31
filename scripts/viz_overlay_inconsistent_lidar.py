"""Overlay LiDAR points where |dense_gt - lidar| > threshold onto pred.

Per sample produces:
  [ rgb | pred-alone | pred + INCONSISTENT lidar (|dense-lidar|>th) |
    pred + CONSISTENT lidar (|dense-lidar|<th) | pred + ALL valid lidar ]

The third panel is the central one — it shows where the stripe-driving
LiDAR points actually sit relative to the model's prediction.
Inconsistent points are colored by their LiDAR depth (warm cmap on cool
pred background), so a glance reveals whether stripes happen exactly at
the rows containing these points.

Usage:
    CUDA_VISIBLE_DEVICES=3 python3 scripts/viz_overlay_inconsistent_lidar.py \\
      --ckpt /workspace/RadarTaco/output/shape_lidar/best.pt \\
      --split test --n_samples 12 --threshold 2.0 \\
      --out_dir /workspace/RadarTaco/output/shape_lidar/viz_inconsistent_th2
"""
import argparse
import os
import sys

import cv2
import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from src.dataset.nuscenes import NuScenesRadarDepthDataset
from src.evaluation.viz import build_grid, colorize_depth, rgb_to_uint8
from src.model.radartaco import RadarTaco
from src.model.rgb_only import RGBOnlyDepth


def _dilate_sparse(values: np.ndarray, valid: np.ndarray, point_radius: int):
    k = 2 * point_radius + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    vd = cv2.dilate(np.where(valid, values, 0.0).astype(np.float32), kernel)
    md = cv2.dilate(valid.astype(np.uint8), kernel).astype(bool)
    return vd, md


def overlay_points(bg_rgb: np.ndarray, values: np.ndarray, valid: np.ndarray,
                   vmax: float, cmap: str, point_radius: int) -> np.ndarray:
    v_d, m_d = _dilate_sparse(values, valid, point_radius)
    fg = colorize_depth(v_d, valid_mask=m_d, vmin=0.0, vmax=vmax, cmap=cmap)
    out = bg_rgb.copy()
    out[m_d] = fg[m_d]
    return out


def build_model(cfg, device):
    name = str(cfg.model.get("name", "radartaco")).lower()
    if name == "rgb_only":
        m = RGBOnlyDepth(
            max_depth=float(cfg.dataset.max_depth),
            pretrained_image_encoder=False,
            output_mode=str(cfg.model.get("output_mode", "metric")),
            min_depth_clip=float(cfg.model.get("min_depth_clip", 0.5)),
            multi_scale=bool(cfg.model.get("multi_scale", False)),
            multi_scale_levels=tuple(cfg.model.get("multi_scale_levels", (2, 4, 8, 16))),
        )
    else:
        m = RadarTaco(
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
            use_aux_branch=bool(cfg.model.get("use_aux_branch", False)),
        )
    return m.to(device).eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--n_samples", type=int, default=12)
    ap.add_argument("--stride", type=int, default=0)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--threshold", type=float, default=2.0,
                    help="|dense_gt - lidar| threshold (meters).")
    ap.add_argument("--pred_cmap", default="viridis")
    ap.add_argument("--inconsistent_cmap", default="hot",
                    help="cmap for inconsistent LiDAR points (depth-colored).")
    ap.add_argument("--consistent_cmap", default="cool",
                    help="cmap for consistent LiDAR points.")
    ap.add_argument("--point_radius", type=int, default=4)
    ap.add_argument("--vmax", type=float, default=80.0)
    ap.add_argument("--err_vmax", type=float, default=10.0,
                    help="Saturation cap (m) for mismatch-colored dots in panels 3/4.")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    cfg = OmegaConf.load(os.path.join(os.path.dirname(args.ckpt), "config.yaml"))
    model = build_model(cfg, device)
    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model"], strict=True)
    print(f"[ok] loaded {args.ckpt} (epoch={ckpt.get('epoch', '?')})")

    split_file = os.path.join(cfg.dataset.data_root, "splits", f"{args.split}.txt")
    ds = NuScenesRadarDepthDataset(
        data_root=cfg.dataset.data_root,
        split_file=split_file,
        dense_gt_dir=cfg.dataset.get("dense_gt_dir", "depth_acc"),
        radar_3d_dir=cfg.dataset.get("radar_3d_dir", "radar_3d"),
        night_ids_file=cfg.dataset.get("night_ids_file", None),
        rel_depth_dir=None,
        rel_depth_dropout_prob=0.0,
        max_radar_points=int(cfg.dataset.max_radar_points),
        max_depth=float(cfg.dataset.max_depth),
        min_depth=float(cfg.dataset.min_depth),
        resize_to_hw=None, augmentation=False,
    )
    n_total = len(ds)
    if args.stride > 0:
        idxs = list(range(0, n_total, args.stride))[: args.n_samples]
    else:
        idxs = np.linspace(0, n_total - 1, args.n_samples).astype(int).tolist()
    print(f"[plan] {len(idxs)} / {n_total} samples (idxs={idxs[:8]}...)")
    print(f"[cfg] threshold={args.threshold} m  point_radius={args.point_radius}")

    n_incon_total, n_cons_total, n_valid_total = 0, 0, 0

    for k, i in enumerate(idxs):
        sample = ds[i]
        batch = {kk: (vv.unsqueeze(0).to(device) if torch.is_tensor(vv) else vv)
                 for kk, vv in sample.items()}
        with torch.inference_mode():
            pred = model(batch["rgb_norm"], batch["radar_points"], batch["radar_mask"],
                         batch.get("rel_depth"))
        rgb = rgb_to_uint8(batch["rgb_norm"][0].cpu().numpy())
        lidar = batch["depth_gt_lidar"][0, 0].cpu().numpy()
        mask_lidar = batch["valid_mask_lidar"][0, 0].cpu().numpy().astype(bool)
        dense = batch["depth_gt_dense"][0, 0].cpu().numpy()
        mask_dense = batch["valid_mask_dense"][0, 0].cpu().numpy().astype(bool)
        pn = pred[0, 0].float().cpu().numpy()

        valid_both = mask_lidar & mask_dense
        mismatch = np.abs(dense - lidar)
        inconsistent_mask = valid_both & (mismatch > args.threshold)
        consistent_mask = valid_both & (mismatch <= args.threshold)

        n_incon = int(inconsistent_mask.sum())
        n_cons = int(consistent_mask.sum())
        n_valid = int(valid_both.sum())
        n_incon_total += n_incon
        n_cons_total += n_cons
        n_valid_total += n_valid
        frac = (n_incon / max(n_valid, 1)) * 100.0

        pred_alone = colorize_depth(pn, vmin=0.0, vmax=args.vmax, cmap=args.pred_cmap)

        overlay_incon = overlay_points(pred_alone, mismatch, inconsistent_mask,
                                       vmax=args.err_vmax, cmap=args.inconsistent_cmap,
                                       point_radius=args.point_radius)
        overlay_cons = overlay_points(pred_alone, mismatch, consistent_mask,
                                      vmax=args.err_vmax, cmap=args.consistent_cmap,
                                      point_radius=args.point_radius)
        overlay_all = overlay_points(pred_alone, lidar, mask_lidar,
                                     vmax=args.vmax, cmap=args.inconsistent_cmap,
                                     point_radius=args.point_radius)

        panel = build_grid([rgb, pred_alone, overlay_incon, overlay_cons, overlay_all])
        sid = sample.get("sample_id", f"idx_{i:04d}")
        safe_sid = str(sid).replace("/", "__")
        out_p = os.path.join(args.out_dir, f"{k:02d}_{safe_sid}.png")
        Image.fromarray(panel).save(out_p)
        print(f"  [{k+1}/{len(idxs)}] {safe_sid}  "
              f"inconsistent={n_incon:>5d}/{n_valid:<6d} ({frac:5.1f}%)  "
              f"-> {os.path.basename(out_p)}")

    frac_all = (n_incon_total / max(n_valid_total, 1)) * 100.0
    print(f"[done] panels: rgb | pred | pred+INCONSISTENT (|d-l|>{args.threshold}, "
          f"colored by |d-l| up to {args.err_vmax}m) "
          f"| pred+CONSISTENT (|d-l|<={args.threshold}, colored by |d-l|) "
          f"| pred+ALL (colored by LiDAR depth up to {args.vmax}m)")
    print(f"[stats] across {len(idxs)} samples: "
          f"inconsistent={n_incon_total} / valid_both={n_valid_total} ({frac_all:.2f}%)")


if __name__ == "__main__":
    main()
