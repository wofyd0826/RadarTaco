#!/usr/bin/env python3
"""Standalone evaluation: load a checkpoint, run on the test split (default),
dump metrics + viz.

Pipeline convention:
    train  → train.txt        (scripts/train.py)
    val    → val.txt           (per-epoch validation inside train.py)
    test   → test.txt          (this script — final evaluation)

Override `eval_split=val` (or `eval_split=val_day` / `val_night`) to
evaluate on a different split.

Usage:
    python scripts/eval.py checkpoint=output/<run>/best.pt
    python scripts/eval.py checkpoint=output/<run>/best.pt eval_split=val
    python scripts/eval.py checkpoint=output/<run>/best.pt eval_split=val_night
"""
import json
import logging
import os
import sys

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from src.dataset.nuscenes import NuScenesRadarDepthDataset                   # noqa: E402
from src.evaluation.metrics import DepthEvaluator                            # noqa: E402
from src.evaluation.viz import (build_grid, colorize_depth, colorize_error,  # noqa: E402
                                overlay_radar_points, rgb_to_uint8)
from src.model.radartaco import RadarTaco                                    # noqa: E402

logging.basicConfig(format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                    level=logging.INFO)
logger = logging.getLogger("radartaco.eval")


@hydra.main(version_base="1.3", config_path="../config", config_name="config")
def main(cfg: DictConfig) -> None:
    if not cfg.get("checkpoint"):
        raise SystemExit("checkpoint=<path/to/.pt> is required (Hydra override).")
    out_dir = cfg.get("eval_out_dir", os.path.join(os.path.dirname(cfg.checkpoint), "eval"))
    os.makedirs(out_dir, exist_ok=True)
    viz_dir = os.path.join(out_dir, "viz")
    os.makedirs(viz_dir, exist_ok=True)

    # Default to the held-out test split for final evaluation. Use
    #   eval_split=val           — same data the trainer validates on
    #   eval_split=val_day       — day-only subset of val
    #   eval_split=val_night     — night-only subset of val (Dark scenario eval)
    split = cfg.get("eval_split", "test")
    assert split in ("train", "val", "test",
                     "train_day", "train_night",
                     "val_day", "val_night",
                     "test_day", "test_night"), split
    split_file = getattr(cfg.dataset, f"split_{split}", None) or \
        os.path.join(cfg.dataset.data_root, "splits", f"{split}.txt")
    if not os.path.exists(split_file):
        raise SystemExit(f"split file not found: {split_file}")
    logger.info(f"eval split: {split} → {split_file}")

    ds = NuScenesRadarDepthDataset(
        data_root=cfg.dataset.data_root,
        split_file=split_file,
        dense_gt_dir=cfg.dataset.get("dense_gt_dir", "depth_interp"),
        night_ids_file=cfg.dataset.get("night_ids_file", None),
        max_radar_points=int(cfg.dataset.max_radar_points),
        max_depth=float(cfg.dataset.max_depth),
        min_depth=float(cfg.dataset.min_depth),
        resize_to_hw=None, augmentation=False,
    )
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
    ).to(device).eval()
    ckpt = torch.load(cfg.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"], strict=True)
    logger.info(f"loaded {cfg.checkpoint} (epoch={ckpt.get('epoch', '?')})")

    evaluator = DepthEvaluator(min_depth=float(cfg.dataset.min_depth),
                               max_depth=float(cfg.dataset.max_depth))
    all_metrics = []
    save_every = int(cfg.get("save_viz_every", 50))
    per_sample_rows = []

    for i, batch in enumerate(tqdm(loader)):
        batch = {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
                 for k, v in batch.items()}
        with torch.inference_mode():
            pred = model(batch["rgb_norm"], batch["radar_points"], batch["radar_mask"])
        pn = pred[0, 0].float().cpu().numpy()
        gn = batch["depth_gt_lidar"][0, 0].cpu().numpy()
        mn = batch["valid_mask_lidar"][0, 0].cpu().numpy().astype(bool)
        is_n = bool(batch["is_night"][0].item()) if torch.is_tensor(batch["is_night"]) else False
        m = evaluator.evaluate_sample(pn, gn, mn, is_night=is_n)
        all_metrics.append(m)
        per_sample_rows.append({
            "sample_id": batch["sample_id"][0],
            "is_night": is_n,
            "mae_0_80m": m["overall"]["0-80m"]["mae"],
            "rmse_0_80m": m["overall"]["0-80m"]["rmse"],
            "mae_far_50_80m": m["far"]["50-80m"]["mae"],
        })
        if save_every > 0 and i % save_every == 0:
            rgb = rgb_to_uint8(batch["rgb_norm"][0].cpu().numpy())
            radar_pts = batch["radar_points"][0].cpu().numpy()
            rmask = batch["radar_mask"][0].cpu().numpy().astype(bool)
            panel = build_grid([
                rgb,
                overlay_radar_points(rgb, radar_pts, rmask, vmax=cfg.dataset.max_depth),
                colorize_depth(pn, vmax=cfg.dataset.max_depth),
                colorize_error(pn, gn, mn, vmax=10.0),
                colorize_depth(gn, valid_mask=mn, vmax=cfg.dataset.max_depth),
            ])
            Image.fromarray(panel).save(os.path.join(viz_dir, f"{i:06d}.png"))

    agg = evaluator.aggregate_metrics(all_metrics)
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(agg, f, indent=2)
    with open(os.path.join(out_dir, "summary.txt"), "w") as f:
        for category, ranges in agg.items():
            f.write(f"=== {category} ===\n")
            for rname, metrics in ranges.items():
                line = "  " + rname + "  " + "  ".join(
                    f"{k}={v:.4f}" for k, v in metrics.items()
                )
                f.write(line + "\n")
            f.write("\n")

    if cfg.get("save_per_sample_csv", False):
        import csv
        path = os.path.join(out_dir, "per_sample.csv")
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(per_sample_rows[0].keys()))
            w.writeheader()
            for r in per_sample_rows:
                w.writerow(r)

    head = agg["overall"].get("0-80m", {})
    head100 = agg["overall"].get("0-100m", {})
    night = agg.get("night", {}).get("0-80m", {})
    far = agg.get("far", {}).get("50-80m", {})
    far100 = agg.get("far", {}).get("80-100m", {})
    logger.info(
        f"DONE  0-80m MAE={head.get('mae', float('nan')):.1f}/RMSE={head.get('rmse', float('nan')):.1f}  "
        f"|  0-100m MAE={head100.get('mae', float('nan')):.1f}/RMSE={head100.get('rmse', float('nan')):.1f}  "
        f"|  far 50-80m MAE={far.get('mae', float('nan')):.1f}  80-100m MAE={far100.get('mae', float('nan')):.1f}  "
        f"|  night MAE={night.get('mae', float('nan')):.1f}"
    )
    logger.info(f"wrote {os.path.join(out_dir, 'metrics.json')}")


if __name__ == "__main__":
    main()
