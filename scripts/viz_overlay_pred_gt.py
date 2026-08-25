"""Overlay pred + sparse LiDAR GT on the same canvas with distinct cmaps.

Per sample produces a single panel:
    [ rgb | pred (viridis) + GT (hot) dots | pred-alone | gt-alone ]
The middle panel is the overlay — pred dense as background, GT as dilated
colored dots on top using a contrasting colormap. Both share the same vmax
so a quick glance reveals where colors disagree.

Usage:
    CUDA_VISIBLE_DEVICES=3 python3 scripts/viz_overlay_pred_gt.py \\
      --ckpt /workspace/RadarTaco/output/shape_lidar_w1.0_smooth0.5/best.pt \\
      --split test --n_samples 8 \\
      --out_dir /workspace/RadarTaco/output/shape_lidar_w1.0_smooth0.5/viz_overlay
"""
import argparse
import os
import sys

import cv2
import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image
from torch.utils.data import DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from src.dataset.nuscenes import NuScenesRadarDepthDataset
from src.evaluation.viz import build_grid, colorize_depth, rgb_to_uint8
from src.model.radartaco import RadarTaco
from src.model.rgb_only import RGBOnlyDepth


def _dilate_sparse(values: np.ndarray, valid: np.ndarray, point_radius: int):
    """Dilate sparse `values` (masked by `valid`) into disks of point_radius."""
    k = 2 * point_radius + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    vd = cv2.dilate(np.where(valid, values, 0.0).astype(np.float32), kernel)
    md = cv2.dilate(valid.astype(np.uint8), kernel).astype(bool)
    return vd, md


def overlay_gt_on_pred(pred: np.ndarray, gt: np.ndarray, gt_valid: np.ndarray,
                       vmax: float, gt_cmap: str = "hot",
                       pred_cmap: str = "viridis",
                       point_radius: int = 4) -> np.ndarray:
    """Pred (dense, cool cmap) as background, GT depth (sparse, warm cmap) on top."""
    bg = colorize_depth(pred, vmin=0.0, vmax=vmax, cmap=pred_cmap)
    gt_d, gt_m = _dilate_sparse(gt, gt_valid, point_radius)
    fg = colorize_depth(gt_d, valid_mask=gt_m, vmin=0.0, vmax=vmax, cmap=gt_cmap)
    out = bg.copy()
    out[gt_m] = fg[gt_m]
    return out


def overlay_err_on_pred(pred: np.ndarray, gt: np.ndarray, gt_valid: np.ndarray,
                        vmax: float, err_vmax: float,
                        err_cmap: str = "Reds",
                        pred_cmap: str = "viridis",
                        point_radius: int = 4,
                        dense: bool = False) -> np.ndarray:
    """Pred (dense) as background, |pred-gt| at GT pixels colored on top.

    With dense=False, GT is treated as sparse points (dilated to disks).
    With dense=True, error is shown at every valid pixel without dilation
    (use for dense GT where almost every pixel is valid).
    """
    bg = colorize_depth(pred, vmin=0.0, vmax=vmax, cmap=pred_cmap)
    err = np.where(gt_valid, np.abs(pred - gt), 0.0)
    if dense:
        err_d, err_m = err, gt_valid
    else:
        err_d, err_m = _dilate_sparse(err, gt_valid, point_radius)
    fg = colorize_depth(err_d, valid_mask=err_m, vmin=0.0, vmax=err_vmax, cmap=err_cmap)
    out = bg.copy()
    out[err_m] = fg[err_m]
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
    ap.add_argument("--n_samples", type=int, default=8)
    ap.add_argument("--stride", type=int, default=0,
                    help="If >0, take every `stride`-th sample (else evenly spaced).")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--gt_cmap", default="hot")
    ap.add_argument("--pred_cmap", default="viridis")
    ap.add_argument("--err_cmap", default="Reds")
    ap.add_argument("--point_radius", type=int, default=4)
    ap.add_argument("--vmax", type=float, default=80.0)
    ap.add_argument("--err_vmax", type=float, default=10.0,
                    help="Max abs error (m) for color saturation.")
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
    print(f"[plan] {len(idxs)} samples out of {n_total} (idxs={idxs})")

    for k, i in enumerate(idxs):
        sample = ds[i]
        batch = {kk: (vv.unsqueeze(0).to(device) if torch.is_tensor(vv) else vv)
                 for kk, vv in sample.items()}
        with torch.inference_mode():
            pred = model(batch["rgb_norm"], batch["radar_points"], batch["radar_mask"],
                         batch.get("rel_depth"))
        rgb = rgb_to_uint8(batch["rgb_norm"][0].cpu().numpy())
        gn = batch["depth_gt_lidar"][0, 0].cpu().numpy()
        mn = batch["valid_mask_lidar"][0, 0].cpu().numpy().astype(bool)
        gn_dense = batch["depth_gt_dense"][0, 0].cpu().numpy()
        mn_dense = batch["valid_mask_dense"][0, 0].cpu().numpy().astype(bool)
        pn = pred[0, 0].float().cpu().numpy()

        overlay_gt = overlay_gt_on_pred(pn, gn, mn, vmax=args.vmax,
                                        gt_cmap=args.gt_cmap, pred_cmap=args.pred_cmap,
                                        point_radius=args.point_radius)
        overlay_err = overlay_err_on_pred(pn, gn, mn, vmax=args.vmax,
                                          err_vmax=args.err_vmax,
                                          err_cmap=args.err_cmap,
                                          pred_cmap=args.pred_cmap,
                                          point_radius=args.point_radius)
        # |dense_gt - lidar_gt| at sparse LiDAR pixels — diagnoses dense GT
        # quality (how off is the refined pseudo-GT relative to its LiDAR anchor).
        # Background is the dense GT itself (not pred) so this panel reads as
        # "where does the dense pseudo-GT disagree with its LiDAR anchor".
        valid_both = mn & mn_dense
        bg = colorize_depth(gn_dense, valid_mask=mn_dense,
                            vmin=0.0, vmax=args.vmax, cmap=args.pred_cmap)
        err_dg = np.where(valid_both, np.abs(gn_dense - gn), 0.0)
        err_d, err_m = _dilate_sparse(err_dg, valid_both, args.point_radius)
        fg = colorize_depth(err_d, valid_mask=err_m, vmin=0.0, vmax=args.err_vmax,
                            cmap=args.err_cmap)
        overlay_err_dense_vs_lidar = bg.copy()
        overlay_err_dense_vs_lidar[err_m] = fg[err_m]

        pred_alone = colorize_depth(pn, vmin=0.0, vmax=args.vmax, cmap=args.pred_cmap)
        panel = build_grid([rgb, pred_alone, overlay_gt, overlay_err,
                            overlay_err_dense_vs_lidar])
        sid = sample.get("sample_id", f"idx_{i:04d}")
        safe_sid = str(sid).replace("/", "__")
        out_p = os.path.join(args.out_dir, f"{k:02d}_{safe_sid}.png")
        Image.fromarray(panel).save(out_p)
        print(f"  [{k+1}/{len(idxs)}] saved {out_p}")

    print(f"[done] cmap: pred={args.pred_cmap} (dense), "
          f"GT={args.gt_cmap} (depth dots), error={args.err_cmap} "
          f"(|err| dots, vmax={args.err_vmax}m)")


if __name__ == "__main__":
    main()
