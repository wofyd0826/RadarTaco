"""Abstract base class for radar-depth datasets.

All concrete datasets emit a unified output dict:
    rgb_norm           [3, H, W] float32, [-1, 1]
    radar_points       [K_max, 6] float32 — hybrid layout:
                         (front, left, up, x_pix, y_pix, depth) padded
                          ↑      ↑    ↑    ↑      ↑      ↑
                            ego-frame 3D │    image plane   radar
                            (meters)     │    (pixels)      depth (m)
                                         │
                          ego 3D consumed by encoders (GNN/MLP) for
                          kNN and per-point features in metric units.
                          x_pix (channel 3) is the only channel consumed
                          by Radar-centered Attention's horizontal window.
    radar_mask         [K_max] bool
    depth_gt_lidar     [1, H, W] float32  (sparse single-frame LiDAR)
    depth_gt_dense     [1, H, W] float32  (dense or interp-densified)
    valid_mask_lidar   [1, H, W] bool
    valid_mask_dense   [1, H, W] bool
    is_night           bool (False for sim datasets)
    is_sim             bool (True for sim datasets)
    sample_id          str
"""
import random
from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import resize as tv_resize

from .intrinsics import (CH_DEPTH, CH_FRONT, CH_LEFT, CH_UP,           # noqa: F401
                         CH_XPIX, CH_YPIX, RADAR_CHANNELS)


class BaseRadarDepthDataset(Dataset, ABC):
    SPATIAL_KEYS = ("rgb_norm", "depth_gt_lidar", "depth_gt_dense",
                    "valid_mask_lidar", "valid_mask_dense")

    def __init__(
        self,
        max_radar_points: int = 128,
        resize_to_hw: Optional[Tuple[int, int]] = None,
        crop_to_hw: Optional[Tuple[int, int]] = None,
        max_depth: float = 100.0,
        min_depth: float = 1e-3,
        augmentation: bool = True,
        lr_flip_p: float = 0.5,
    ):
        super().__init__()
        self.max_radar_points = max_radar_points
        self.resize_to_hw = tuple(resize_to_hw) if resize_to_hw else None
        self.crop_to_hw = tuple(crop_to_hw) if crop_to_hw else None
        self.max_depth = max_depth
        self.min_depth = min_depth
        self.augmentation = augmentation
        self.lr_flip_p = lr_flip_p

    @abstractmethod
    def __len__(self) -> int: ...

    @abstractmethod
    def __getitem__(self, idx: int) -> Dict: ...

    # ------------------------------------------------------------ helpers --
    def _normalize_rgb(self, rgb: np.ndarray) -> torch.Tensor:
        rgb = rgb.astype(np.float32) / 255.0 * 2.0 - 1.0
        return torch.from_numpy(rgb).permute(2, 0, 1)

    def _pad_radar_points(self, points: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        """Pad / sub-sample radar points to (K_max, RADAR_CHANNELS) and emit a mask.

        `points` must be (N, RADAR_CHANNELS=6) — see module docstring for the
        channel layout. Use `intrinsics.expand_to_6ch` to convert from the
        legacy (N, 3) image-projected form.
        """
        if points.ndim != 2 or points.shape[1] != RADAR_CHANNELS:
            raise ValueError(f"radar points must be (N, {RADAR_CHANNELS}); got {points.shape}")
        N = min(points.shape[0], self.max_radar_points)
        padded = np.zeros((self.max_radar_points, RADAR_CHANNELS), dtype=np.float32)
        mask = np.zeros(self.max_radar_points, dtype=bool)
        if N > 0:
            if points.shape[0] > self.max_radar_points:
                idx = np.random.choice(points.shape[0], self.max_radar_points, replace=False)
                points = points[idx]
                N = self.max_radar_points
            padded[:N] = points[:N]
            mask[:N] = True
        return torch.from_numpy(padded), torch.from_numpy(mask)

    def _make_depth_tensor(self, depth: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        valid = (depth > self.min_depth) & (depth < self.max_depth) & np.isfinite(depth)
        depth = np.clip(depth, 0, self.max_depth)
        return (torch.from_numpy(depth).unsqueeze(0).float(),
                torch.from_numpy(valid).unsqueeze(0).bool())

    # ----------------------------------------------------------- transforms --
    def _apply_resize(self, sample: Dict) -> Dict:
        if self.resize_to_hw is None:
            return sample
        target_h, target_w = self.resize_to_hw
        _, orig_h, orig_w = sample["rgb_norm"].shape
        if orig_h == target_h and orig_w == target_w:
            return sample
        scale_y = target_h / orig_h
        scale_x = target_w / orig_w

        sample["rgb_norm"] = tv_resize(
            sample["rgb_norm"], [target_h, target_w],
            interpolation=InterpolationMode.BILINEAR, antialias=True,
        )
        for key in ("depth_gt_lidar", "depth_gt_dense"):
            sample[key] = tv_resize(sample[key], [target_h, target_w],
                                    interpolation=InterpolationMode.NEAREST_EXACT)
        for key in ("valid_mask_lidar", "valid_mask_dense"):
            sample[key] = tv_resize(sample[key].float(), [target_h, target_w],
                                    interpolation=InterpolationMode.NEAREST_EXACT).bool()

        radar = sample["radar_points"]
        m = sample["radar_mask"]
        # Only the image-pixel channels (CH_XPIX, CH_YPIX) scale with resize.
        # Camera-frame (xc, yc, zc) and depth are physical meter quantities
        # and are invariant to image resampling.
        radar[m, CH_XPIX] *= scale_x
        radar[m, CH_YPIX] *= scale_y
        sample["radar_points"] = radar
        return sample

    def _apply_random_crop(self, sample: Dict) -> Dict:
        if self.crop_to_hw is None:
            return sample
        target_h, target_w = self.crop_to_hw
        _, H, W = sample["rgb_norm"].shape
        if H < target_h or W < target_w or (H == target_h and W == target_w):
            return sample
        y0 = random.randint(0, H - target_h)
        x0 = random.randint(0, W - target_w)
        for key in self.SPATIAL_KEYS:
            sample[key] = sample[key][..., y0:y0 + target_h, x0:x0 + target_w]
        radar = sample["radar_points"]
        mask = sample["radar_mask"].clone()
        if mask.any():
            xs = radar[:, CH_XPIX] - x0
            ys = radar[:, CH_YPIX] - y0
            in_bounds = (mask
                         & (xs >= 0) & (xs < target_w)
                         & (ys >= 0) & (ys < target_h))
            # Shift image-pixel coords; metric channels (xc,yc,zc,depth) are
            # invariant to crop. Drop out-of-bounds points entirely.
            radar[:, CH_XPIX] = torch.where(in_bounds, xs, torch.zeros_like(xs))
            radar[:, CH_YPIX] = torch.where(in_bounds, ys, torch.zeros_like(ys))
            radar[~in_bounds] = 0.0
            sample["radar_points"] = radar
            sample["radar_mask"] = in_bounds
        return sample

    def _apply_augmentation(self, sample: Dict) -> Dict:
        if not self.augmentation:
            return sample
        if random.random() < self.lr_flip_p:
            W = sample["rgb_norm"].shape[2]
            for key in self.SPATIAL_KEYS:
                sample[key] = sample[key].flip(-1)
            radar = sample["radar_points"]
            m = sample["radar_mask"]
            # Flip image pixel x AND ego `left` (mirror across the vertical
            # plane through the camera optical axis). front, up, depth are
            # unchanged by a horizontal flip.
            radar[m, CH_XPIX] = W - 1 - radar[m, CH_XPIX]
            radar[m, CH_LEFT] = -radar[m, CH_LEFT]
            sample["radar_points"] = radar
        return sample
