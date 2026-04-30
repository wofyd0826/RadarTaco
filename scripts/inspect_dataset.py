#!/usr/bin/env python3
"""Visualize a few samples from a dataset to verify the loader is correct."""
import logging
import os
import sys

import hydra
import numpy as np
from omegaconf import DictConfig
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from src.dataset.nuscenes import NuScenesRadarDepthDataset                   # noqa: E402
from src.dataset.sim import SimRadarDepthDataset                             # noqa: E402
from src.evaluation.viz import (build_grid, colorize_depth,                  # noqa: E402
                                overlay_radar_points, rgb_to_uint8)

logging.basicConfig(format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                    level=logging.INFO)
logger = logging.getLogger("inspect_dataset")


@hydra.main(version_base="1.3", config_path="../config", config_name="config")
def main(cfg: DictConfig) -> None:
    n = int(cfg.get("inspect_n", 5))
    out_dir = cfg.get("inspect_out_dir", "./output/inspect")
    os.makedirs(out_dir, exist_ok=True)

    if cfg.dataset.name in ("nuscenes", "mixed"):
        ds = NuScenesRadarDepthDataset(
            data_root=cfg.dataset.data_root,
            split_file=cfg.dataset.split_val,
            dense_gt_dir=cfg.dataset.get("dense_gt_dir", "depth_interp"),
            radar_3d_dir=cfg.dataset.get("radar_3d_dir", "radar_3d"),
            night_ids_file=cfg.dataset.get("night_ids_file", None),
            max_radar_points=int(cfg.dataset.max_radar_points),
            max_depth=float(cfg.dataset.max_depth),
            min_depth=float(cfg.dataset.min_depth),
            resize_to_hw=None, augmentation=False,
        )
    elif cfg.dataset.name in ("hypersim", "vkitti2"):
        ds = SimRadarDepthDataset(
            data_root=cfg.dataset.data_root,
            split_file=cfg.dataset.split_train,
            dataset_type=cfg.dataset.dataset_type,
            radar_simulation=cfg.dataset.radar_simulation,
            num_radar_points=(int(cfg.dataset.num_radar_points_min),
                              int(cfg.dataset.num_radar_points_max)),
            depth_noise_std=float(cfg.dataset.depth_noise_std),
            resize_to_hw=tuple(cfg.dataset.train_resize_hw)
                if cfg.dataset.get("train_resize_hw") else None,
            max_radar_points=int(cfg.dataset.max_radar_points),
            max_depth=float(cfg.dataset.max_depth),
            min_depth=float(cfg.dataset.min_depth),
            augmentation=False,
        )
    else:
        raise ValueError(f"unknown dataset name: {cfg.dataset.name}")

    logger.info(f"dataset {cfg.dataset.name}: {len(ds)} samples")
    for i in range(min(n, len(ds))):
        s = ds[i]
        rgb = rgb_to_uint8(s["rgb_norm"].numpy())
        radar_pts = s["radar_points"].numpy()
        rmask = s["radar_mask"].numpy().astype(bool)
        gt = s["depth_gt_lidar"][0].numpy()
        valid = s["valid_mask_lidar"][0].numpy().astype(bool)
        panel = build_grid([
            rgb,
            overlay_radar_points(rgb, radar_pts, rmask, vmax=float(cfg.dataset.max_depth)),
            colorize_depth(gt, valid_mask=valid, vmax=float(cfg.dataset.max_depth)),
        ])
        flag = " (night)" if bool(s.get("is_night", False)) else ""
        sid = s.get("sample_id", str(i)).replace("/", "__")
        path = os.path.join(out_dir, f"{i:03d}_{sid}.png")
        Image.fromarray(panel).save(path)
        logger.info(f"#{i}{flag}  radar pts={int(rmask.sum())}  saved → {path}")


if __name__ == "__main__":
    main()
