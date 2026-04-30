#!/usr/bin/env python3
"""Visualize per-radar-point Radar-centered Attention heatmaps.

For a single sample, this script:
  1. Toggles `record_attention=True` on every RadarCenteredAttention block.
  2. Runs a forward pass.
  3. For the chosen fusion layer/block, picks N radar points and renders
     a panel showing — for each radar point — which image pixels it attends
     to most strongly (heatmap overlay on the RGB).

Memory note: attention tensor is (1, heads, H, W, K). With batch=1, heads=4,
K=128 and H×W at the layer's feature resolution. To stay manageable the
script downsamples the input image to `--input-size H W` (default 224 320).

Usage:
    # Random val sample, layer 1 node-block, top 12 radar points
    python scripts/visualize_attention.py checkpoint=output/baseline_gnn/best.pt

    # Specific sample, layer 3 (deepest), edge block
    python scripts/visualize_attention.py checkpoint=output/baseline_gnn/best.pt \
        attn_layer=3 attn_block=edge sample_idx=42

    # Higher input resolution (uses more memory)
    python scripts/visualize_attention.py checkpoint=output/baseline_gnn/best.pt \
        input_size='[480, 640]'
"""
import logging
import os
import sys

import hydra
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from src.dataset.nuscenes import NuScenesRadarDepthDataset                   # noqa: E402
from src.evaluation.viz import (build_grid, colorize_depth,                  # noqa: E402
                                overlay_radar_points, rgb_to_uint8)
from src.model.radar_fusion import RadarCenteredAttention                    # noqa: E402
from src.model.radartaco import RadarTaco                                    # noqa: E402

logging.basicConfig(format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                    level=logging.INFO)
logger = logging.getLogger("visualize_attention")


def _get_attn_modules(model: RadarTaco):
    """Return list of (layer_idx, block_kind, module) triples in pyramid order.

    layer_idx ∈ {1, 2, 3} (1-indexed for human readability)
    block_kind ∈ {'node', 'edge'}
    """
    out = []
    fusion = model.radar_fusion
    for l in range(fusion.L):
        out.append((l + 1, "node", fusion.node_blocks[l].attn))
        out.append((l + 1, "edge", fusion.edge_blocks[l].attn))
    return out


def _select_radar_points(attn_per_pixel: np.ndarray,    # (H, W, K)
                         radar_mask: np.ndarray,         # (K,) bool
                         n: int,
                         strategy: str = "spread") -> list:
    """Pick `n` radar point indices to visualize.

    strategies:
      'first'   — first n valid points
      'energy'  — points whose attention map has the highest peak (top-n by max)
      'spread'  — first sort valid points by max attention then keep every k-th
                  to spread coverage across all attended regions (default)
    """
    valid = np.where(radar_mask)[0].tolist()
    if not valid:
        return []
    if strategy == "first":
        return valid[:n]
    peak = {k: float(attn_per_pixel[..., k].max()) for k in valid}
    if strategy == "energy":
        return sorted(valid, key=lambda k: -peak[k])[:n]
    sorted_by_peak = sorted(valid, key=lambda k: -peak[k])
    if len(sorted_by_peak) <= n:
        return sorted_by_peak
    step = max(1, len(sorted_by_peak) // n)
    return sorted_by_peak[::step][:n]


def _heatmap_overlay(rgb_uint8: np.ndarray,
                     attn_pixel: np.ndarray,            # (H_img, W_img) float [0,1]
                     radar_xy: tuple,
                     alpha: float = 0.55,
                     cmap_name: str = "inferno") -> np.ndarray:
    """RGB darkened underlay + colored attention heatmap + radar point marker."""
    import matplotlib.cm as cm
    import cv2

    cmap = cm.get_cmap(cmap_name)
    a = np.clip(attn_pixel, 0.0, 1.0)
    heat = (cmap(a)[..., :3] * 255).astype(np.uint8)

    # Darken the RGB so the heatmap pops, then alpha-blend.
    base = (rgb_uint8.astype(np.float32) * 0.5).clip(0, 255).astype(np.uint8)
    out = ((1 - alpha) * base.astype(np.float32)
           + alpha * heat.astype(np.float32)).clip(0, 255).astype(np.uint8)
    out = np.ascontiguousarray(out)                         # cv2 requires this

    # Mark the radar point.
    x, y = int(radar_xy[0]), int(radar_xy[1])
    H, W, _ = out.shape
    if 0 <= x < W and 0 <= y < H:
        cv2.drawMarker(out, (x, y), (0, 255, 255), markerType=cv2.MARKER_TILTED_CROSS,
                       markerSize=14, thickness=2, line_type=cv2.LINE_AA)
        cv2.circle(out, (x, y), 8, (0, 255, 255), 2, cv2.LINE_AA)
    return out


@hydra.main(version_base="1.3", config_path="../config", config_name="config")
def main(cfg: DictConfig) -> None:
    if not cfg.get("checkpoint"):
        raise SystemExit("checkpoint=<path/to/.pt> is required (Hydra override).")

    layer = int(cfg.get("attn_layer", 1))                # 1-indexed
    block = str(cfg.get("attn_block", "node"))           # 'node' or 'edge'
    assert block in ("node", "edge"), block
    n_top = int(cfg.get("attn_top_n", 12))
    strategy = str(cfg.get("attn_select", "spread"))     # 'first' / 'energy' / 'spread'
    sample_idx = int(cfg.get("sample_idx", 0))
    in_size = tuple(cfg.get("input_size", [224, 320]))   # (H, W)
    out_dir = cfg.get("attn_out_dir",
                      os.path.join(os.path.dirname(cfg.checkpoint),
                                   f"attn_l{layer}_{block}"))
    os.makedirs(out_dir, exist_ok=True)
    eval_split = cfg.get("eval_split", "val")
    split_file = getattr(cfg.dataset, f"split_{eval_split}", None) or \
        os.path.join(cfg.dataset.data_root, "splits", f"{eval_split}.txt")

    logger.info(f"layer={layer}, block={block}, top_n={n_top}, "
                f"input_size={in_size}, sample_idx={sample_idx}, "
                f"split={eval_split}, out_dir={out_dir}")

    # ------------------------------------------------------------ dataset --
    ds = NuScenesRadarDepthDataset(
        data_root=cfg.dataset.data_root,
        split_file=split_file,
        dense_gt_dir=cfg.dataset.get("dense_gt_dir", "depth_interp"),
        radar_3d_dir=cfg.dataset.get("radar_3d_dir", "radar_3d"),
        night_ids_file=cfg.dataset.get("night_ids_file", None),
        max_radar_points=int(cfg.dataset.max_radar_points),
        max_depth=float(cfg.dataset.max_depth),
        min_depth=float(cfg.dataset.min_depth),
        resize_to_hw=in_size,
        augmentation=False,
    )
    if sample_idx >= len(ds):
        raise SystemExit(f"sample_idx {sample_idx} >= dataset size {len(ds)}")
    sample = ds[sample_idx]
    logger.info(f"sample: {sample['sample_id']}  "
                f"radar_pts={int(sample['radar_mask'].sum())}  "
                f"is_night={bool(sample['is_night'])}")

    # ------------------------------------------------------------- model --
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
    ckpt = torch.load(cfg.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"], strict=True)
    logger.info(f"loaded {cfg.checkpoint} (epoch={ckpt.get('epoch', '?')})")

    # ----------------------------------------------------- record attention --
    for _, _, m in _get_attn_modules(model):
        m.record_attention = True

    with torch.inference_mode():
        rgb = sample["rgb_norm"].unsqueeze(0).to(device)
        radar_pts = sample["radar_points"].unsqueeze(0).to(device)
        radar_mask = sample["radar_mask"].unsqueeze(0).to(device)
        _ = model(rgb, radar_pts, radar_mask)

    # turn it off again (frees future references)
    for _, _, m in _get_attn_modules(model):
        m.record_attention = False

    # ------------------------------------------------------- pick the block --
    target = None
    for li, bk, m in _get_attn_modules(model):
        if li == layer and bk == block:
            target = m
            break
    if target is None or target.last_attn is None:
        raise SystemExit(f"no recorded attention for layer={layer} block={block}")

    attn = target.last_attn[0]                             # (heads, H_l, W_l, K)
    keep = target.last_keep[0]                             # (H_l, W_l, K)
    H_l, W_l = target.last_hw
    H_img, W_img = in_size
    logger.info(f"recorded attention: shape={tuple(attn.shape)}  feat_hw={(H_l, W_l)}")

    # mean over heads → (H_l, W_l, K)
    attn_mean = attn.mean(dim=0).numpy().astype(np.float32)

    # zero-out scores outside the keep mask (cosmetic; -inf scores already
    # got 0 weight after softmax, but rows fully masked may have residual)
    attn_mean = attn_mean * keep.numpy().astype(np.float32)

    # Pick radar points to visualize.
    radar_mask_np = sample["radar_mask"].numpy().astype(bool)
    radar_pts_np = sample["radar_points"].numpy()           # (K, 3): x, y, depth
    chosen = _select_radar_points(attn_mean, radar_mask_np, n_top, strategy=strategy)
    logger.info(f"selected {len(chosen)} radar points (strategy={strategy})")

    # ----------------------------------------------------------- render --
    rgb_img = rgb_to_uint8(sample["rgb_norm"].numpy())     # (H_img, W_img, 3)
    panels = []
    # Header panel: RGB + all radar points marked, depth-colored
    header = overlay_radar_points(rgb_img, radar_pts_np, radar_mask_np,
                                  vmax=float(cfg.dataset.max_depth))
    panels.append(header)

    for k in chosen:
        a = attn_mean[..., k]                               # (H_l, W_l)
        a_t = torch.from_numpy(a)[None, None].float()
        a_up = F.interpolate(a_t, size=(H_img, W_img),
                             mode="bilinear", align_corners=False)[0, 0].numpy()
        # normalize per-point so even small global scores show clearly
        if a_up.max() > 1e-8:
            a_up = a_up / a_up.max()
        # channels 3, 4 = (x_pix, y_pix) in the 6-channel hybrid layout
        x, y = float(radar_pts_np[k, 3]), float(radar_pts_np[k, 4])
        panel = _heatmap_overlay(rgb_img, a_up, (x, y))
        panels.append(panel)

    # Lay out: 4 columns wide grid (header in top-left).
    ncols = min(4, len(panels))
    rows_needed = (len(panels) + ncols - 1) // ncols
    while len(panels) < rows_needed * ncols:
        panels.append(np.zeros_like(panels[0]))            # pad with black
    grid = build_grid(panels, ncols=ncols)

    sid = sample["sample_id"].replace("/", "__")
    out_path = os.path.join(out_dir, f"{sid}_l{layer}_{block}.png")
    Image.fromarray(grid).save(out_path)
    logger.info(f"saved {out_path}  (panel shape {grid.shape})")


if __name__ == "__main__":
    main()
