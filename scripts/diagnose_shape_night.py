#!/usr/bin/env python3
"""Per-sample day-vs-night diagnostic for shape losses, with distance-bin
decomposition.

For each sample in the chosen split, the script:
  1. Runs the model forward.
  2. Computes the two shape losses (same as training):
       - loss_shape_global = AffineInvariantL1Loss(pred, dense_gt, dense_mask)
       - loss_shape_grid   = AffineInvariantBlockGridL1Loss(pred, dense_gt,
                                                            dense_mask)
  3. Replicates the loss-time alignment to obtain the (scale, shift)-
     applied pred_aligned at full resolution.
  4. Bins the per-pixel residual |pred_aligned - dense_gt| by dense_gt
     range (0-10, 10-20, ..., 90-100m) and reports the mean per bin.

Outputs:
  - Console: DAY/NIGHT mean/median/q90/max, plus per-range tables for both
    global and grid, plus NIGHT/DAY ratios per range.
  - CSV (if --out_csv): one row per sample with all loss values and all
    per-range bins.

Usage:
    python scripts/diagnose_shape_night.py \\
      --checkpoint output/shape_lidar/best.pt \\
      --split val
"""
import argparse
import csv
import logging
import os
import sys

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from src.dataset.nuscenes import NuScenesRadarDepthDataset                # noqa: E402
from src.loss.losses import (                                              # noqa: E402
    AffineInvariantBlockGridL1Loss,
    AffineInvariantL1Loss,
    _align_depth_affine,
    _masked_avg_pool,
)
from src.model.radartaco import RadarTaco                                  # noqa: E402
from src.model.rgb_only import RGBOnlyDepth                                # noqa: E402

logging.basicConfig(format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                    level=logging.INFO)
logger = logging.getLogger("radartaco.diag")

# Distance bins for per-range decomposition. Mirrors RANGE_BINS_FINE from
# src/evaluation/metrics.py so it's directly comparable to the eval report.
RANGE_BINS = [(0, 10), (10, 20), (20, 30), (30, 40), (40, 50),
              (50, 60), (60, 70), (70, 80), (80, 90), (90, 100)]
RANGE_LABELS = [f"{a}-{b}m" for a, b in RANGE_BINS]


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


# -------------------------- alignment replication ------------------------ #
# These replicate the alignment step inside AffineInvariantL1Loss and
# AffineInvariantBlockGridL1Loss so we can extract pred_aligned without
# changing the loss API. Parameters are pulled from the (already
# constructed) loss module so we stay in sync.

@torch.no_grad()
def compute_pred_aligned_global(pred, target, mask, loss_module):
    if mask.dtype != torch.bool:
        mask = mask > 0.5
    if mask.float().sum() < 1.0:
        return pred.detach().clone()
    ar = loss_module.align_resolution
    pred_lr, mask_lr = _masked_avg_pool(pred, mask, (ar, ar))
    target_lr, _ = _masked_avg_pool(target, mask, (ar, ar))
    pred_flat = pred_lr.flatten(1)
    target_flat = target_lr.flatten(1)
    mask_lr_f = mask_lr.float().flatten(1)
    weight = mask_lr_f / target_flat.clamp_min(loss_module.eps)
    scale, shift = _align_depth_affine(pred_flat, target_flat, weight,
                                        trunc=loss_module.trunc)
    scale_ok = (scale >= loss_module.scale_min) & (scale <= loss_module.scale_max)
    shift_ok = shift.abs() <= loss_module.shift_max_abs
    valid = scale_ok & shift_ok
    scale = torch.where(valid, scale, torch.ones_like(scale))
    shift = torch.where(valid, shift, torch.zeros_like(shift))
    return scale[:, None, None, None] * pred + shift[:, None, None, None]


@torch.no_grad()
def compute_pred_aligned_grid(pred, target, mask, loss_module):
    if mask.dtype != torch.bool:
        mask = mask > 0.5
    if mask.float().sum() < 1.0:
        return pred.detach().clone()
    B, _, H, W = pred.shape
    K = loss_module.K
    bh, bw = H // K, W // K
    H_c, W_c = bh * K, bw * K
    pred_c = pred[..., :H_c, :W_c]
    target_c = target[..., :H_c, :W_c]
    mask_c = mask[..., :H_c, :W_c]

    def _blockify(t):
        t = t.view(B, 1, K, bh, K, bw)
        t = t.permute(0, 2, 4, 1, 3, 5).contiguous()
        return t.view(B * K * K, 1, bh, bw)

    pred_b = _blockify(pred_c)
    target_b = _blockify(target_c)
    mask_b = _blockify(mask_c)
    n_per_block = mask_b.view(B * K * K, -1).sum(dim=-1)

    ar = loss_module.align_resolution
    pred_lr, mask_lr = _masked_avg_pool(pred_b, mask_b, (ar, ar))
    target_lr, _ = _masked_avg_pool(target_b, mask_b, (ar, ar))
    N_lr = ar * ar
    pred_lr_flat = pred_lr.view(B * K * K, N_lr)
    target_lr_flat = target_lr.view(B * K * K, N_lr)
    mask_lr_f = mask_lr.float().view(B * K * K, N_lr)
    weight = mask_lr_f / target_lr_flat.clamp_min(loss_module.eps)
    scale, shift = _align_depth_affine(pred_lr_flat, target_lr_flat, weight,
                                        trunc=loss_module.trunc)
    scale_ok = (scale >= loss_module.scale_min) & (scale <= loss_module.scale_max)
    shift_ok = shift.abs() <= loss_module.shift_max_abs
    block_ok = (n_per_block >= loss_module.min_inliers) & scale_ok & shift_ok
    scale = torch.where(block_ok, scale, torch.ones_like(scale))
    shift = torch.where(block_ok, shift, torch.zeros_like(shift))

    pred_b_aligned = scale[:, None, None, None] * pred_b + shift[:, None, None, None]
    # Blocks that fell back to identity → keep unaligned pred (so per-bin
    # residuals see no spurious large error from a bad ROE fit).
    keep4 = block_ok[:, None, None, None].float()
    blocks_assembled = pred_b_aligned * keep4 + pred_b * (1.0 - keep4)
    blocks_assembled = blocks_assembled.view(B, K, K, 1, bh, bw)
    blocks_assembled = blocks_assembled.permute(0, 3, 1, 4, 2, 5).contiguous()
    assembled = blocks_assembled.view(B, 1, H_c, W_c)
    if H_c != H or W_c != W:
        out = pred.clone()
        out[..., :H_c, :W_c] = assembled
        return out
    return assembled


def per_range_residual(pred_aligned, target, mask, range_bins=RANGE_BINS):
    """Mean |pred_aligned - target| within mask, binned by target depth.

    pred_aligned, target, mask: (1, 1, H, W) tensors.
    Returns dict {"{a}-{b}m": mae_or_nan}.
    """
    pa = pred_aligned[0, 0].cpu().numpy()
    tg = target[0, 0].cpu().numpy()
    mk = mask[0, 0].cpu().numpy()
    if mk.dtype != bool:
        mk = mk > 0.5
    diff = np.abs(pa - tg)
    out = {}
    for d_min, d_max in range_bins:
        bin_mask = mk & (tg >= d_min) & (tg < d_max)
        n = int(bin_mask.sum())
        out[f"{d_min}-{d_max}m"] = float(diff[bin_mask].mean()) if n > 0 else float("nan")
    return out


# -------------------------- main routine --------------------------------- #

@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--split", default="val", choices=["train", "val", "test",
                                                       "val_day", "val_night",
                                                       "test_day", "test_night"])
    p.add_argument("--data_root", default="/data/public/nuScenes/derived")
    p.add_argument("--dense_gt_dir", default=None,
                   help="override; default = from ckpt's sibling config.yaml")
    p.add_argument("--out_dir", default=None,
                   help="output directory; defaults to "
                        "<ckpt_dir>/diagnose_shape. Both summary.txt and "
                        "per_sample.csv are saved here (named by split).")
    p.add_argument("--out_csv", default=None,
                   help="override path of the per-sample CSV.")
    p.add_argument("--out_txt", default=None,
                   help="override path of the human-readable summary.")
    p.add_argument("--no_save", action="store_true",
                   help="don't save anything, only print to terminal.")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    sibling = os.path.join(os.path.dirname(args.checkpoint), "config.yaml")
    if not os.path.exists(sibling):
        raise SystemExit(f"missing training config: {sibling}")
    cfg = OmegaConf.load(sibling)

    dense_gt_dir = args.dense_gt_dir or cfg.dataset.dense_gt_dir
    logger.info(f"dense_gt_dir = {dense_gt_dir}  "
                f"(from {'flag' if args.dense_gt_dir else 'ckpt config'})")

    # Resolve output paths. By default everything goes into
    # <ckpt_dir>/diagnose_shape/{split}_summary.txt + {split}.csv.
    if args.no_save:
        out_csv_path = None
        out_txt_path = None
    else:
        out_dir = args.out_dir or os.path.join(
            os.path.dirname(args.checkpoint), "diagnose_shape")
        os.makedirs(out_dir, exist_ok=True)
        out_csv_path = args.out_csv or os.path.join(
            out_dir, f"{args.split}.csv")
        out_txt_path = args.out_txt or os.path.join(
            out_dir, f"{args.split}_summary.txt")
        logger.info(f"out_dir = {out_dir}")
        logger.info(f"  csv → {out_csv_path}")
        logger.info(f"  txt → {out_txt_path}")

    split_file = f"{args.data_root}/splits/{args.split}.txt"
    if not os.path.exists(split_file):
        raise SystemExit(f"missing split file: {split_file}")

    ds = NuScenesRadarDepthDataset(
        data_root=args.data_root,
        split_file=split_file,
        dense_gt_dir=dense_gt_dir,
        radar_3d_dir=cfg.dataset.get("radar_3d_dir", "radar_3d"),
        night_ids_file=cfg.dataset.get("night_ids_file", None),
        rel_depth_dir=None,
        max_radar_points=int(cfg.dataset.max_radar_points),
        max_depth=float(cfg.dataset.max_depth),
        min_depth=float(cfg.dataset.min_depth),
        resize_to_hw=None, augmentation=False,
    )
    loader = DataLoader(ds, batch_size=1, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = build_model(cfg, device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"], strict=True)
    logger.info(f"loaded {args.checkpoint} (epoch={ckpt.get('epoch', '?')})")

    shape_global = AffineInvariantL1Loss(
        align_resolution=int(cfg.loss.get("shape_align_resolution", 32)),
        trunc=float(cfg.loss.get("shape_trunc", 1.0)),
        scale_min=float(cfg.loss.get("shape_scale_min", 0.1)),
        scale_max=float(cfg.loss.get("shape_scale_max", 10.0)),
        shift_max_abs=float(cfg.loss.get("shape_shift_max_abs", 100.0)),
    ).to(device)
    shape_grid = AffineInvariantBlockGridL1Loss(
        K=int(cfg.loss.get("shape_grid_K", 8)),
        trunc=float(cfg.loss.get("shape_trunc", 1.0)),
        min_inliers=int(cfg.loss.get("shape_grid_min_inliers", 30)),
        align_resolution=int(cfg.loss.get("shape_grid_align_resolution", 16)),
        scale_min=float(cfg.loss.get("shape_scale_min", 0.1)),
        scale_max=float(cfg.loss.get("shape_scale_max", 10.0)),
        shift_max_abs=float(cfg.loss.get("shape_shift_max_abs", 100.0)),
    ).to(device)

    rows = []
    for i, batch in enumerate(tqdm(loader)):
        if args.limit and i >= args.limit:
            break
        batch = {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
                 for k, v in batch.items()}
        pred = model(batch["rgb_norm"], batch["radar_points"], batch["radar_mask"],
                     batch.get("rel_depth"))
        dense_gt = batch["depth_gt_dense"]
        dense_mask = batch["valid_mask_dense"]

        l_g = shape_global(pred, dense_gt, dense_mask).item()
        l_b = shape_grid(pred, dense_gt, dense_mask).item()

        pa_g = compute_pred_aligned_global(pred, dense_gt, dense_mask, shape_global)
        pa_b = compute_pred_aligned_grid(pred, dense_gt, dense_mask, shape_grid)
        pr_g = per_range_residual(pa_g, dense_gt, dense_mask)
        pr_b = per_range_residual(pa_b, dense_gt, dense_mask)

        is_night = bool(batch["is_night"][0].item()) if torch.is_tensor(batch["is_night"]) else False
        row = {
            "sample_id": batch["sample_id"][0],
            "is_night": is_night,
            "loss_shape_global": l_g,
            "loss_shape_grid": l_b,
        }
        for k in RANGE_LABELS:
            row[f"global/{k}"] = pr_g[k]
            row[f"grid/{k}"] = pr_b[k]
        rows.append(row)

    # ----------------------------- reporting ----------------------------- #
    # Tee summary to both stdout and a buffer (later written to --out_txt).
    summary_lines: list[str] = []

    def emit(s: str = "") -> None:
        print(s)
        summary_lines.append(s)

    emit()
    emit("=" * 78)
    emit(f"checkpoint = {args.checkpoint}")
    emit(f"split = {args.split}   total = {len(rows)} samples")
    emit(f"dense_gt_dir = {dense_gt_dir}")
    emit("=" * 78)

    day = [r for r in rows if not r["is_night"]]
    night = [r for r in rows if r["is_night"]]

    def overall_stats(subset, label):
        if not subset:
            emit(f"  {label}: (no samples)")
            return None
        g = np.array([r["loss_shape_global"] for r in subset])
        b = np.array([r["loss_shape_grid"] for r in subset])
        emit("")
        emit(f"  {label} (n={len(subset)}):")
        emit(f"    overall  shape_global  mean={g.mean():.4f}  med={np.median(g):.4f}  "
             f"q90={np.quantile(g, 0.9):.4f}  max={g.max():.4f}")
        emit(f"    overall  shape_grid    mean={b.mean():.4f}  med={np.median(b):.4f}  "
             f"q90={np.quantile(b, 0.9):.4f}  max={b.max():.4f}")
        return g.mean(), b.mean()

    s_day_overall = overall_stats(day, "DAY")
    s_night_overall = overall_stats(night, "NIGHT")

    def per_range_means(subset):
        out_g, out_b = {}, {}
        for k in RANGE_LABELS:
            g_vals = [r[f"global/{k}"] for r in subset if not np.isnan(r[f"global/{k}"])]
            b_vals = [r[f"grid/{k}"]   for r in subset if not np.isnan(r[f"grid/{k}"])]
            out_g[k] = float(np.mean(g_vals)) if g_vals else float("nan")
            out_b[k] = float(np.mean(b_vals)) if b_vals else float("nan")
        return out_g, out_b

    def print_per_range_table(subset, label):
        if not subset:
            return None, None
        pg, pb = per_range_means(subset)
        emit("")
        emit(f"  Per-range shape MAE — {label}:")
        emit(f"    {'range':<10}  {'global':>9}  {'grid':>9}")
        for k in RANGE_LABELS:
            emit(f"    {k:<10}  {pg[k]:>9.3f}  {pb[k]:>9.3f}")
        return pg, pb

    pg_day, pb_day = print_per_range_table(day, "DAY")
    pg_night, pb_night = print_per_range_table(night, "NIGHT")

    if s_day_overall and s_night_overall:
        ratio_g = s_night_overall[0] / s_day_overall[0] if s_day_overall[0] > 0 else float("nan")
        ratio_b = s_night_overall[1] / s_day_overall[1] if s_day_overall[1] > 0 else float("nan")
        emit("")
        emit(f"  NIGHT/DAY overall ratio:   shape_global={ratio_g:.2f}x   "
             f"shape_grid={ratio_b:.2f}x")
        if ratio_g > 1.3 or ratio_b > 1.3:
            emit("  → night samples carry MORE overall shape error → DA-v2 / "
                 "refined_grid quality is the bottleneck at night.")
        else:
            emit("  → night/day overall shape loss is similar.")

        if pg_day and pg_night:
            emit("")
            emit("  NIGHT/DAY per-range ratio:")
            emit(f"    {'range':<10}  {'global':>8}  {'grid':>8}")
            for k in RANGE_LABELS:
                rg = (pg_night[k] / pg_day[k]) if pg_day[k] > 0 and not np.isnan(pg_day[k]) \
                    and not np.isnan(pg_night[k]) else float("nan")
                rb = (pb_night[k] / pb_day[k]) if pb_day[k] > 0 and not np.isnan(pb_day[k]) \
                    and not np.isnan(pb_night[k]) else float("nan")
                mark = "  ← night gap large" if (rg > 1.5 or rb > 1.5) else ""
                emit(f"    {k:<10}  {rg:>8.2f}x  {rb:>8.2f}x{mark}")

    if out_csv_path and rows:
        os.makedirs(os.path.dirname(out_csv_path) or ".", exist_ok=True)
        with open(out_csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            for r in rows:
                w.writerow(r)
        logger.info(f"wrote {out_csv_path}  ({len(rows)} rows)")

    if out_txt_path:
        os.makedirs(os.path.dirname(out_txt_path) or ".", exist_ok=True)
        with open(out_txt_path, "w") as f:
            f.write("\n".join(summary_lines) + "\n")
        logger.info(f"wrote {out_txt_path}")


if __name__ == "__main__":
    main()
