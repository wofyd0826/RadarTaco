"""Simulation dataset (Hypersim / vKITTI2) with synthetic radar points.

Sim datasets have no real radar, so we sample synthetic radar points from
the dense GT depth map. Three modes:
    simple          uniform random sampling from valid GT pixels
    augmented       y-coords drawn from the empirical nuScenes distribution
                    (mean=0.573 of H, std=0.027) + Gaussian depth noise σ=0.5m
    semantic_aware  vKITTI2 only — multiplies the augmented y-bias weight
                    with a per-pixel radar reflectivity map derived from
                    the official classSegmentation PNG. Concentrates radar
                    points on vehicles / poles / guard-rails — the
                    surfaces real automotive mmWave radar actually sees
                    most often. See `vkitti2_classes.py` for the class →
                    reflectivity weights. Hypersim ships no equivalent
                    class map in our setup so semantic_aware silently
                    falls back to augmented for hypersim with a one-shot
                    warning at first sample access.

Each emitted sample carries `is_sim=True` so loss/aux logic can route
sim-only objectives separately from real (nuScenes) batches.
"""
import os
import warnings
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .base import BaseRadarDepthDataset
from .intrinsics import INTRINSICS_BY_NAME, expand_to_6ch
from .vkitti2_classes import (derive_classgt_path,
                              reflectivity_map_from_classgt)


class SimRadarDepthDataset(BaseRadarDepthDataset):

    def __init__(
        self,
        data_root: str,
        split_file: str,
        dataset_type: str = "hypersim",
        radar_simulation: str = "augmented",
        num_radar_points: Tuple[int, int] = (30, 60),
        depth_noise_std: float = 0.5,
        class_weights: Optional[Dict[str, float]] = None,    # vkitti2 semantic
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.data_root = data_root
        self.dataset_type = dataset_type
        self.radar_simulation = radar_simulation
        self.num_radar_points = num_radar_points
        self.depth_noise_std = depth_noise_std
        self.class_weights = dict(class_weights) if class_weights else None
        self._semantic_fallback_warned = False
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
        rgb_path = os.path.join(self.data_root, s["rgb"])
        rgb = np.asarray(Image.open(rgb_path).convert("RGB"))
        rgb_norm = self._normalize_rgb(rgb)
        depth_raw = self._load_depth(os.path.join(self.data_root, s["depth"]))
        depth_gt, valid_mask = self._make_depth_tensor(depth_raw)

        # Optionally load + compute the per-pixel reflectivity weight map
        # (semantic_aware mode, vKITTI2 only). Done at original resolution;
        # we resize the float weight map alongside depth below.
        reflect_map_orig = self._load_reflectivity_map(rgb_path)         # (H, W) or None

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
        # Resize reflectivity map to current resolution (same as depth).
        # NEAREST so discrete class boundaries are preserved without
        # introducing intermediate fractional weights.
        reflect_map = self._resize_reflect_map(reflect_map_orig, depth_resized.shape)
        radar = self._simulate_radar(depth_resized, depth_resized.shape, reflect_map)
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

    def _load_reflectivity_map(self, rgb_path: str) -> Optional[np.ndarray]:
        """Load vKITTI2 classSegmentation PNG → per-pixel reflectivity map.

        Returns None when:
          - sampling mode is not semantic_aware
          - dataset_type is not vkitti2 (one-shot warning then fall back)
          - the matching classSegmentation file does not exist
        """
        if self.radar_simulation != "semantic_aware":
            return None
        if self.dataset_type != "vkitti2":
            if not self._semantic_fallback_warned:
                warnings.warn(
                    f"semantic_aware mode requested but dataset_type="
                    f"{self.dataset_type!r} has no class map in this setup; "
                    f"falling back to augmented sampling.", stacklevel=2,
                )
                self._semantic_fallback_warned = True
            return None
        cls_path = derive_classgt_path(rgb_path)
        if not os.path.exists(cls_path):
            if not self._semantic_fallback_warned:
                warnings.warn(
                    f"classSegmentation missing at {cls_path}; falling back "
                    f"to augmented sampling for this and any other affected "
                    f"samples.", stacklevel=2,
                )
                self._semantic_fallback_warned = True
            return None
        cls_arr = np.array(Image.open(cls_path))                # (H, W, 3)
        return reflectivity_map_from_classgt(cls_arr, self.class_weights)

    @staticmethod
    def _resize_reflect_map(
        reflect_map: Optional[np.ndarray],
        target_hw: Tuple[int, int],
    ) -> Optional[np.ndarray]:
        if reflect_map is None:
            return None
        if reflect_map.shape == tuple(target_hw):
            return reflect_map
        # NEAREST so discrete class boundaries don't fracture into
        # intermediate fractional weights at sub-pixel scale.
        t = torch.from_numpy(reflect_map).unsqueeze(0).unsqueeze(0).float()
        t = F.interpolate(t, size=target_hw, mode="nearest")
        return t[0, 0].numpy()

    def _simulate_radar(
        self,
        depth_gt: np.ndarray,
        hw: Tuple[int, int],
        reflect_map: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Sample synthetic radar points from a dense GT depth map.

        Returns (N, 6) hybrid layout: (front, left, up, x_pix, y_pix,
        depth) in the *current* (possibly resized) image resolution.
        Intrinsics are rescaled to match `hw` so the ego-frame 3D coords
        stay in physically consistent meter units.

        When `reflect_map` is provided (semantic_aware mode on vKITTI2),
        the per-pixel sampling probability becomes
            p ∝ (y-bias Gaussian) × reflectivity_weight
        so points concentrate on radar-reflective surfaces (vehicles /
        guard rails / poles) while still respecting the horizon-centric
        spatial distribution of real radar.
        """
        H, W = hw
        valid_mask = (depth_gt > self.min_depth) & (depth_gt < self.max_depth)
        if reflect_map is not None:
            # Restrict to pixels with non-zero reflectivity weight too.
            valid_mask = valid_mask & (reflect_map > 0.0)
        ys, xs = np.where(valid_mask)
        if len(ys) == 0:
            # Defensive: a frame may have zero reflective pixels (rare).
            # Fall back to plain valid mask so we still emit some points.
            valid_mask = (depth_gt > self.min_depth) & (depth_gt < self.max_depth)
            ys, xs = np.where(valid_mask)
            reflect_map = None
            if len(ys) == 0:
                return np.zeros((0, 6), dtype=np.float32)
        n_min, n_max = self.num_radar_points
        n = min(np.random.randint(n_min, n_max + 1), len(ys))
        if self.radar_simulation == "simple":
            idx = np.random.choice(len(ys), n, replace=False)
        else:
            y_norm = ys.astype(np.float64) / H
            w = np.exp(-0.5 * ((y_norm - 0.573) / 0.027) ** 2)
            if reflect_map is not None:
                w = w * reflect_map[ys, xs].astype(np.float64)
            s = w.sum()
            if not np.isfinite(s) or s <= 0:
                w = np.ones_like(w)                              # degenerate fallback
                s = w.sum()
            w /= s
            idx = np.random.choice(len(ys), n, replace=False, p=w)
        sx = xs[idx].astype(np.float32)
        sy = ys[idx].astype(np.float32)
        sd = depth_gt[ys[idx], xs[idx]].astype(np.float32)
        if self.radar_simulation in ("augmented", "semantic_aware"):
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
