"""Simulation dataset (Hypersim / vKITTI2) with synthetic radar points.

Sim datasets have no real radar, so we sample synthetic radar points from
the dense GT depth map. Two modes:
    simple    — uniform random sampling from valid GT pixels
    augmented — y-coords drawn from the empirical nuScenes distribution
                (mean=0.573 of H, std=0.027) + Gaussian depth noise σ=0.5m

Each emitted sample carries `is_sim=True` so loss/aux logic can route
sim-only objectives separately from real (nuScenes) batches.
"""
import os
from typing import Dict, Tuple

import numpy as np
import torch
from PIL import Image

from .base import BaseRadarDepthDataset
from .intrinsics import INTRINSICS_BY_NAME, expand_to_6ch


class SimRadarDepthDataset(BaseRadarDepthDataset):

    def __init__(
        self,
        data_root: str,
        split_file: str,
        dataset_type: str = "hypersim",
        radar_simulation: str = "augmented",
        num_radar_points: Tuple[int, int] = (30, 60),
        depth_noise_std: float = 0.5,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.data_root = data_root
        self.dataset_type = dataset_type
        self.radar_simulation = radar_simulation
        self.num_radar_points = num_radar_points
        self.depth_noise_std = depth_noise_std
        if dataset_type == "hypersim":
            self.depth_scale = 1000.0
            self.max_depth_dataset = 65.0
        elif dataset_type == "vkitti2":
            self.depth_scale = 100.0
            self.max_depth_dataset = 80.0
        else:
            raise ValueError(f"unknown dataset_type: {dataset_type}")
        # Synthetic camera intrinsics for inverse projection (used by radar
        # encoders' kNN to live in metric units). The placeholder values in
        # `intrinsics.py` are scaled at sim time to whatever resolution the
        # current sample is rendered at — see `_simulate_radar` below.
        self._intrinsic_template = INTRINSICS_BY_NAME[dataset_type]
        self.samples = self._load_split(split_file)

    @staticmethod
    def _load_split(split_file: str):
        with open(split_file) as f:
            lines = [l.strip() for l in f if l.strip()]
        out = []
        for line in lines:
            parts = line.split()
            if len(parts) >= 2:
                out.append({"rgb": parts[0], "depth": parts[1]})
            else:
                out.append({"id": parts[0]})
        return out

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        s = self.samples[idx]
        rgb = np.asarray(Image.open(os.path.join(self.data_root, s["rgb"])).convert("RGB"))
        rgb_norm = self._normalize_rgb(rgb)
        depth_raw = self._load_depth(os.path.join(self.data_root, s["depth"]))
        depth_gt, valid_mask = self._make_depth_tensor(depth_raw)

        sample = {
            "rgb_norm": rgb_norm,
            # placeholder; will overwrite after resize so coords match the
            # final resolution
            "radar_points": torch.zeros(self.max_radar_points, 6),
            "radar_mask": torch.zeros(self.max_radar_points, dtype=torch.bool),
            "depth_gt_lidar": depth_gt,
            "depth_gt_dense": depth_gt.clone(),
            "valid_mask_lidar": valid_mask,
            "valid_mask_dense": valid_mask.clone(),
            "is_night": torch.tensor(False, dtype=torch.bool),
            "is_sim": torch.tensor(True, dtype=torch.bool),
            "sample_id": f"{self.dataset_type}/{idx}",
        }
        sample = self._apply_resize(sample)

        depth_resized = sample["depth_gt_lidar"][0].numpy()
        radar = self._simulate_radar(depth_resized, depth_resized.shape)
        radar_pts, radar_mask = self._pad_radar_points(radar)
        sample["radar_points"] = radar_pts
        sample["radar_mask"] = radar_mask

        sample = self._apply_random_crop(sample)
        sample = self._apply_augmentation(sample)
        return sample

    # ----------------------------------------------------------- helpers --
    def _load_depth(self, path: str) -> np.ndarray:
        if path.endswith(".npy"):
            depth = np.load(path).astype(np.float32)
        elif path.endswith(".hdf5") or path.endswith(".h5"):
            import h5py
            with h5py.File(path, "r") as f:
                depth = np.array(f["dataset"], dtype=np.float32)
        else:
            raw = np.asarray(Image.open(path), dtype=np.float32)
            if self.dataset_type == "vkitti2":
                raw[raw >= 65535] = 0.0
            depth = raw / self.depth_scale
        depth[~np.isfinite(depth)] = 0.0
        return np.clip(depth, 0.0, self.max_depth_dataset)

    def _simulate_radar(self, depth_gt: np.ndarray, hw: Tuple[int, int]) -> np.ndarray:
        """Sample synthetic radar points from a dense GT depth map.

        Returns (N, 6) hybrid layout: (front, left, up, x_pix, y_pix,
        depth) in the *current* (possibly resized) image resolution.
        Intrinsics are rescaled to match `hw` so the ego-frame 3D coords
        stay in physically consistent meter units.
        """
        H, W = hw
        valid_mask = (depth_gt > self.min_depth) & (depth_gt < self.max_depth)
        ys, xs = np.where(valid_mask)
        if len(ys) == 0:
            return np.zeros((0, 6), dtype=np.float32)
        n_min, n_max = self.num_radar_points
        n = min(np.random.randint(n_min, n_max + 1), len(ys))
        if self.radar_simulation == "simple":
            idx = np.random.choice(len(ys), n, replace=False)
        else:
            y_norm = ys.astype(np.float64) / H
            w = np.exp(-0.5 * ((y_norm - 0.573) / 0.027) ** 2)
            w /= w.sum()
            idx = np.random.choice(len(ys), n, replace=False, p=w)
        sx = xs[idx].astype(np.float32)
        sy = ys[idx].astype(np.float32)
        sd = depth_gt[ys[idx], xs[idx]].astype(np.float32)
        if self.radar_simulation == "augmented":
            sd = np.maximum(sd + np.random.normal(0, self.depth_noise_std, n).astype(np.float32),
                            self.min_depth)
        pts3 = np.stack([sx, sy, sd], axis=-1)                  # (N, 3) image-projected
        # Rescale intrinsics from the dataset's reference resolution to the
        # current sampling resolution.
        sx_scale = W / float(self._intrinsic_template["W_orig"])
        sy_scale = H / float(self._intrinsic_template["H_orig"])
        intr = {
            "fx": self._intrinsic_template["fx"] * sx_scale,
            "fy": self._intrinsic_template["fy"] * sy_scale,
            "cx": self._intrinsic_template["cx"] * sx_scale,
            "cy": self._intrinsic_template["cy"] * sy_scale,
        }
        return expand_to_6ch(pts3, intr)
