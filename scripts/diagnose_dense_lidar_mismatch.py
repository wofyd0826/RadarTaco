"""Diagnose whether stripes in pred correlate with |dense_gt - lidar| mismatch.

Hypothesis (user):
    dense_gt and sparse LiDAR disagree at some LiDAR pixels. When the loss
    has both `w_lidar · L1(pred, lidar)` and `w_shape · ShapeLoss(pred, dense)`,
    at those mismatched pixels pred is pulled toward lidar while neighboring
    (non-LiDAR) pixels are pulled toward dense_gt's smooth shape. The
    resulting per-row jump = horizontal stripe.

This script aggregates three signals across the test split:

  1. Per-row mean `|dense_gt - lidar|` at LiDAR-valid pixels — does
     mismatch concentrate at specific y-rows (= beams)?
  2. Per-row mean `|pred - lidar|` vs `|pred - dense_gt|` — does the
     model split the difference at mismatch rows?
  3. Per-sample Spearman correlation between row-aggregated
     `|dense - lidar|` and row-aggregated pred discontinuity (a proxy
     for stripe magnitude). High +ρ supports the hypothesis.

Also dumps a small viz panel for N samples showing where mismatch is
spatially concentrated.

Usage:
    CUDA_VISIBLE_DEVICES=3 python3 scripts/diagnose_dense_lidar_mismatch.py \\
      --checkpoint output/shape_lidar/best.pt \\
      --split test --n_viz 8 \\
      --dense_dir depth_refined_grid
"""
import argparse
import os
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image
from scipy.stats import spearmanr
from torch.utils.data import DataLoader
from tqdm import tqdm

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from src.dataset.nuscenes import NuScenesRadarDepthDataset
from src.evaluation.viz import build_grid, colorize_depth, rgb_to_uint8
from src.model.radartaco import RadarTaco
from src.model.rgb_only import RGBOnlyDepth


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


def vertical_jump(pred: np.ndarray) -> np.ndarray:
    """Per-row mean of |∂pred/∂y| — proxy for vertical-stripe magnitude.

    A clean pred has small vertical gradient everywhere. A pred with a
    horizontal stripe at row R has a large jump at row R (and R+1).
    Returns array of shape (H,) giving the row-wise mean |jump|.
    """
    dy = np.abs(np.diff(pred, axis=0))            # (H-1, W)
    pad = np.zeros((1, pred.shape[1]), dtype=dy.dtype)
    dy_full = np.concatenate([dy, pad], axis=0)   # (H, W)
    return dy_full.mean(axis=1)


def per_row_stat(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Mean of `values` per y-row over valid pixels. Returns (H,)."""
    H = values.shape[0]
    out = np.zeros(H, dtype=np.float32)
    cnt = valid.sum(axis=1).astype(np.float32)    # per-row count
    sum_ = (values * valid).sum(axis=1).astype(np.float32)
    nz = cnt > 0
    out[nz] = sum_[nz] / cnt[nz]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--dense_dir", default="depth_refined_grid",
                    help="Reference dense GT to compare against LiDAR.")
    ap.add_argument("--out_dir", default=None,
                    help="Default: <ckpt_dir>/diagnose_dense_lidar/")
    ap.add_argument("--n_viz", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0,
                    help="If >0, only process this many samples (debug).")
    ap.add_argument("--max_depth", type=float, default=100.0)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg_p = os.path.join(os.path.dirname(args.checkpoint), "config.yaml")
    cfg = OmegaConf.load(cfg_p)
    out_dir = args.out_dir or os.path.join(
        os.path.dirname(args.checkpoint), "diagnose_dense_lidar")
    os.makedirs(out_dir, exist_ok=True)
    viz_dir = os.path.join(out_dir, "viz")
    os.makedirs(viz_dir, exist_ok=True)

    model = build_model(cfg, device)
    ck = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ck["model"], strict=True)
    print(f"[ok] loaded {args.checkpoint} (epoch={ck.get('epoch','?')})")

    split_file = os.path.join(cfg.dataset.data_root, "splits", f"{args.split}.txt")
    # Evaluate against canonical sparse single-frame LiDAR.
    ds = NuScenesRadarDepthDataset(
        data_root=cfg.dataset.data_root,
        split_file=split_file,
        dense_gt_dir=args.dense_dir,
        lidar_gt_dir="depth_lidar",
        radar_3d_dir=cfg.dataset.get("radar_3d_dir", "radar_3d"),
        night_ids_file=cfg.dataset.get("night_ids_file", None),
        max_radar_points=int(cfg.dataset.max_radar_points),
        max_depth=float(cfg.dataset.max_depth),
        min_depth=float(cfg.dataset.min_depth),
        resize_to_hw=None, augmentation=False,
    )
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=4,
                        pin_memory=True)
    N = len(ds) if not args.limit else min(args.limit, len(ds))
    print(f"[plan] processing {N} samples")

    H_canon = W_canon = None
    row_sum_dense_lidar = None       # Σ per-row |dense-lidar| at LiDAR
    row_sum_pred_lidar = None        # Σ per-row |pred-lidar| at LiDAR
    row_sum_pred_dense = None        # Σ per-row |pred-dense| over valid dense
    row_sum_pred_jump = None         # Σ per-row |∂pred/∂y|
    row_cnt = None                   # #samples contributing to each row
    rho_list = []                    # per-sample Spearman ρ(dense-lidar, pred-jump)
    per_sample_rows = []             # (sample_id, n_anchors, rho, mean_mismatch, mean_jump, mean_pred_lidar_err)
    n_anchors = []                   # per-sample LiDAR point count

    viz_idx = set(np.linspace(0, N - 1, args.n_viz).astype(int).tolist())

    pbar = tqdm(loader, total=N)
    for i, batch in enumerate(pbar):
        if i >= N:
            break
        batch = {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
                 for k, v in batch.items()}
        with torch.inference_mode():
            pred = model(batch["rgb_norm"], batch["radar_points"],
                         batch["radar_mask"], batch.get("rel_depth"))
        pn = pred[0, 0].float().cpu().numpy()
        gn = batch["depth_gt_lidar"][0, 0].cpu().numpy()
        mn = batch["valid_mask_lidar"][0, 0].cpu().numpy().astype(bool)
        gd = batch["depth_gt_dense"][0, 0].cpu().numpy()
        md = batch["valid_mask_dense"][0, 0].cpu().numpy().astype(bool)

        if row_sum_dense_lidar is None:
            H_canon, W_canon = pn.shape
            row_sum_dense_lidar = np.zeros(H_canon, dtype=np.float64)
            row_sum_pred_lidar = np.zeros(H_canon, dtype=np.float64)
            row_sum_pred_dense = np.zeros(H_canon, dtype=np.float64)
            row_sum_pred_jump = np.zeros(H_canon, dtype=np.float64)
            row_cnt = np.zeros(H_canon, dtype=np.int64)

        # --- per-row signals ---
        vl_both = mn & md                                  # LiDAR + dense both valid
        diff_dense_lidar = np.where(vl_both, np.abs(gd - gn), 0.0)
        diff_pred_lidar = np.where(mn, np.abs(pn - gn), 0.0)
        diff_pred_dense = np.where(md, np.abs(pn - gd), 0.0)
        jump = vertical_jump(pn)                           # (H,)

        row_dl = per_row_stat(diff_dense_lidar, vl_both)
        row_pl = per_row_stat(diff_pred_lidar, mn)
        row_pd = per_row_stat(diff_pred_dense, md)

        row_sum_dense_lidar += row_dl
        row_sum_pred_lidar += row_pl
        row_sum_pred_dense += row_pd
        row_sum_pred_jump += jump
        row_cnt += 1
        n_anchors.append(int(mn.sum()))

        # Spearman corr only on rows with at least 5 LiDAR pixels
        cnt_l = mn.sum(axis=1)
        valid_rows = cnt_l >= 5
        rho_val = float("nan")
        if valid_rows.sum() >= 30:
            try:
                rho_val, _ = spearmanr(row_dl[valid_rows], jump[valid_rows])
                if np.isfinite(rho_val):
                    rho_list.append(float(rho_val))
            except Exception:
                pass
        # Per-sample row for scatter plotting
        sid = batch.get("sample_id", [f"idx_{i:06d}"])
        sid = sid[0] if isinstance(sid, (list, tuple)) else str(sid)
        per_sample_rows.append({
            "sample_id": str(sid),
            "n_anchors": int(mn.sum()),
            "rho": float(rho_val) if np.isfinite(rho_val) else float("nan"),
            "mean_mismatch_at_lidar": float(np.abs(gd[vl_both] - gn[vl_both]).mean()) if vl_both.any() else float("nan"),
            "mean_jump": float(jump.mean()),
            "mean_pred_lidar_err": float(np.abs(pn[mn] - gn[mn]).mean()) if mn.any() else float("nan"),
        })

        # --- viz for a few samples ---
        if i in viz_idx:
            rgb = rgb_to_uint8(batch["rgb_norm"][0].cpu().numpy())
            kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

            def dilate_for_viz(vals, mask):
                d = cv2.dilate(np.where(mask, vals, 0.0).astype(np.float32), kern)
                m = cv2.dilate(mask.astype(np.uint8), kern).astype(bool)
                return d, m

            dl_d, dl_m = dilate_for_viz(np.abs(gd - gn), vl_both)
            pl_d, pl_m = dilate_for_viz(np.abs(pn - gn), mn)
            panel = build_grid([
                rgb,
                colorize_depth(pn, vmax=80, cmap="viridis"),                # pred dense
                colorize_depth(dl_d, valid_mask=dl_m, vmax=5,
                               cmap="Reds"),                                # |dense-lidar|
                colorize_depth(pl_d, valid_mask=pl_m, vmax=5,
                               cmap="Reds"),                                # |pred-lidar|
                colorize_depth(np.abs(pn - gd), valid_mask=md, vmax=5,
                               cmap="Reds"),                                # |pred-dense|
            ])
            sid = batch.get("sample_id", [f"{i:04d}"])[0]
            safe = str(sid).replace("/", "__")
            Image.fromarray(panel).save(
                os.path.join(viz_dir, f"{i:04d}_{safe}.png"))

    # --- aggregate ---
    row_mean_dense_lidar = row_sum_dense_lidar / np.maximum(row_cnt, 1)
    row_mean_pred_lidar = row_sum_pred_lidar / np.maximum(row_cnt, 1)
    row_mean_pred_dense = row_sum_pred_dense / np.maximum(row_cnt, 1)
    row_mean_jump = row_sum_pred_jump / np.maximum(row_cnt, 1)

    # global Spearman over rows (across all samples averaged)
    if (row_cnt > 0).sum() > 30:
        global_rho, _ = spearmanr(row_mean_dense_lidar[row_cnt > 0],
                                  row_mean_jump[row_cnt > 0])
    else:
        global_rho = float("nan")
    per_sample_rho_mean = float(np.mean(rho_list)) if rho_list else float("nan")
    per_sample_rho_q90 = float(np.quantile(rho_list, 0.9)) if rho_list else float("nan")
    per_sample_rho_q10 = float(np.quantile(rho_list, 0.1)) if rho_list else float("nan")

    # CSV: per-row means
    rows = np.arange(H_canon)
    csv_p = os.path.join(out_dir, f"{args.split}_per_row.csv")
    with open(csv_p, "w") as f:
        f.write("y_row,mean_dense_minus_lidar,mean_pred_minus_lidar,"
                "mean_pred_minus_dense,mean_pred_vertical_jump\n")
        for y in rows:
            f.write(f"{y},{row_mean_dense_lidar[y]:.6f},"
                    f"{row_mean_pred_lidar[y]:.6f},"
                    f"{row_mean_pred_dense[y]:.6f},"
                    f"{row_mean_jump[y]:.6f}\n")
    # CSV: per-sample summary (for the scatter plot of per-sample ρ)
    ps_csv = os.path.join(out_dir, f"{args.split}_per_sample.csv")
    with open(ps_csv, "w") as f:
        f.write("sample_id,n_anchors,rho,mean_mismatch_at_lidar,"
                "mean_jump,mean_pred_lidar_err\n")
        for r in per_sample_rows:
            f.write(f"{r['sample_id']},{r['n_anchors']},{r['rho']:.6f},"
                    f"{r['mean_mismatch_at_lidar']:.6f},"
                    f"{r['mean_jump']:.6f},"
                    f"{r['mean_pred_lidar_err']:.6f}\n")

    # Summary
    print()
    print("=" * 72)
    print(f"checkpoint = {args.checkpoint}")
    print(f"split      = {args.split}   N = {N}   anchors/img avg = "
          f"{np.mean(n_anchors):.0f}")
    print("=" * 72)
    print()
    print("Per-row aggregated signals (averaged across samples):")
    # Find peak rows in dense-lidar mismatch
    valid_rows = row_cnt > 0
    if valid_rows.any():
        top5 = np.argsort(-row_mean_dense_lidar)[:5]
        print(f"  top 5 |dense-lidar| rows (y, mean |Δ| in m):")
        for y in top5:
            print(f"    y={y:4d}  dense-lidar={row_mean_dense_lidar[y]:.3f}m  "
                  f"pred-lidar={row_mean_pred_lidar[y]:.3f}m  "
                  f"pred_jump={row_mean_jump[y]:.4f}m")
    print()
    print("Hypothesis correlation tests "
          "(higher ρ = more stripe at high-mismatch rows):")
    print(f"  per-sample Spearman ρ(row |dense-lidar|, row pred-jump):")
    print(f"    n_samples = {len(rho_list)}")
    print(f"    mean      = {per_sample_rho_mean:.4f}")
    print(f"    q10/q90   = {per_sample_rho_q10:.4f} / {per_sample_rho_q90:.4f}")
    print(f"  global Spearman over per-row means = {global_rho:.4f}")
    print()
    print(f"viz       : {viz_dir}/*.png   (RGB | pred | |dense-lidar| | "
          f"|pred-lidar| | |pred-dense|, color=Reds vmax=5m)")
    print(f"per-row csv: {csv_p}")

    # Also save a small text summary
    with open(os.path.join(out_dir, f"{args.split}_summary.txt"), "w") as f:
        f.write(f"checkpoint = {args.checkpoint}\n")
        f.write(f"split      = {args.split}   N = {N}\n")
        f.write(f"avg LiDAR anchors per image = {np.mean(n_anchors):.0f}\n\n")
        f.write(f"per-sample Spearman ρ(row |dense-lidar|, row pred-jump):\n")
        f.write(f"  n samples = {len(rho_list)}\n")
        f.write(f"  mean      = {per_sample_rho_mean:.4f}\n")
        f.write(f"  q10/q90   = {per_sample_rho_q10:.4f} / {per_sample_rho_q90:.4f}\n")
        f.write(f"global Spearman over per-row means = {global_rho:.4f}\n\n")
        f.write(f"Interpretation:\n")
        f.write(f"  ρ > 0  → high |dense-lidar| at row R coincides with high\n")
        f.write(f"           vertical pred jump at row R → supports the\n")
        f.write(f"           hypothesis that stripes are caused by the model\n")
        f.write(f"           being forced toward LiDAR at mismatched rows.\n")
        f.write(f"  ρ ≈ 0  → no link, stripes come from another source.\n")
        f.write(f"  ρ < 0  → counter-evidence.\n")


if __name__ == "__main__":
    main()
