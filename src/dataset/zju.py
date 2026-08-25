"""ZJU-4DRadarCam dataset (Li et al., ICRA'24, RadarCam-Depth).

Layout (under `data_root`):
    image/{ts}.png             RGB, 1280×720
    radar/{ts}.npy             (N, 3) float64 — (x_pix, y_pix, depth)
                                already projected by dataset authors
    gt/{ts}.png                uint16 (×256) — single-frame sparse LiDAR
    gt_interp/{ts}.png         uint16 (×256) — Delaunay-interpolated dense
    train.txt / val.txt / test.txt   one timestamp per line

Paper §4.2 (ZJU): "since it contains denser LiDAR returns and depth maps,
we directly interpolate D_gt to obtain D_acc as the RadarCam-Depth [22]"
→ `depth_gt_dense` is read from `gt_interp/` directly (no accumulation step).

Camera intrinsic for inverse-projecting (x_pix, y_pix, depth) → ego 3D is
registered in `intrinsics.ZJU_4DRADARCAM_INTRINSIC` (values from the
dataset owner, github issue #7 comment 2146469866).
"""
import os
from typing import Dict, Optional

import numpy as np
import torch
from PIL import Image, ImageFile

from .base import BaseRadarDepthDataset
from .intrinsics import ZJU_4DRADARCAM_INTRINSIC, expand_to_6ch

ImageFile.LOAD_TRUNCATED_IMAGES = True


class ZjuRadarDepthDataset(BaseRadarDepthDataset):
    """ZJU-4DRadarCam loader producing the unified dataset dict."""

    def __init__(
        self,
        data_root: str,
        split_file: str,
        image_dir: str = "image",
        radar_dir: str = "radar",
        gt_dir: str = "gt",
        gt_interp_dir: str = "gt_interp",
        rel_depth_dir: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.data_root = data_root
        self.image_dir = image_dir
        self.radar_dir = radar_dir
        self.gt_dir = gt_dir
        self.gt_interp_dir = gt_interp_dir
        self.rel_depth_dir = rel_depth_dir
        self.samples = self._load_split(split_file)

    @staticmethod
    def _load_split(split_file: str):
        out = []
        with open(split_file) as f:
            for line in f:
                ts = line.strip()
                if ts:
                    out.append(ts)
        return out

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        ts = self.samples[idx]

        rgb_path = os.path.join(self.data_root, self.image_dir, f"{ts}.png")
        rgb = np.asarray(Image.open(rgb_path).convert("RGB"))
        rgb_norm = self._normalize_rgb(rgb)

        radar_path = os.path.join(self.data_root, self.radar_dir, f"{ts}.npy")
        radar_raw = np.load(radar_path).astype(np.float32)
        if radar_raw.shape[0] == 0:
            radar_points = np.zeros((0, 6), dtype=np.float32)
        else:
            radar_points = expand_to_6ch(radar_raw, ZJU_4DRADARCAM_INTRINSIC)
        radar_points_t, radar_mask = self._pad_radar_points(radar_points)

        d_lidar_path = os.path.join(self.data_root, self.gt_dir, f"{ts}.png")
        d_lidar = np.asarray(Image.open(d_lidar_path), dtype=np.float32) / 256.0
        depth_gt_lidar, valid_mask_lidar = self._make_depth_tensor(d_lidar)

        d_dense_path = os.path.join(self.data_root, self.gt_interp_dir, f"{ts}.png")
        if os.path.exists(d_dense_path):
            d_dense = np.asarray(Image.open(d_dense_path), dtype=np.float32) / 256.0
            depth_gt_dense, valid_mask_dense = self._make_depth_tensor(d_dense)
        else:
            depth_gt_dense = depth_gt_lidar.clone()
            valid_mask_dense = valid_mask_lidar.clone()

        H, W = depth_gt_lidar.shape[-2], depth_gt_lidar.shape[-1]
        rel_depth_path = None
        if self.rel_depth_dir is not None:
            rel_depth_path = os.path.join(self.data_root, self.rel_depth_dir, f"{ts}.png")
        rel_depth = self._load_rel_depth(rel_depth_path, (H, W))

        sample = {
            "rgb_norm": rgb_norm,
            "radar_points": radar_points_t,
            "radar_mask": radar_mask,
            "depth_gt_lidar": depth_gt_lidar,
            "depth_gt_dense": depth_gt_dense,
            "valid_mask_lidar": valid_mask_lidar,
            "valid_mask_dense": valid_mask_dense,
            "rel_depth": rel_depth,
            "is_night": torch.tensor(False, dtype=torch.bool),
            "is_sim": torch.tensor(False, dtype=torch.bool),
            "sample_id": ts,
        }
        sample = self._apply_resize(sample)
        sample = self._apply_random_crop(sample)
        sample = self._apply_augmentation(sample)
        return sample
