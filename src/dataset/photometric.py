"""Photometric augmentation that simulates low-light conditions on RGB.

Combines four common low-light degradations to turn a clean day-time
training image into a "fake night" sample with random severity:

    1. Gamma > 1     ── darkens midtones (γ=2 ≈ exposure under by ~1 stop)
    2. Brightness×k  ── multiplicative scale toward black
    3. Contrast×c    ── compression toward the mean (haze/glare effect)
    4. Gaussian noise ── sensor read noise that grows at high ISO / low light

Inputs and outputs are torch tensors in [-1, 1] (the same RGB normalization
used by `BaseRadarDepthDataset._normalize_rgb`).

Why this matters for our research:
nuScenes train has ~12% night frames, validation has ~10%. A model trained
without augmentation tends to ignore the under-represented night regime.
Photometrically darkening day frames during training forces the radar
encoder + fusion to remain useful when RGB features are degraded — directly
targeting the "dark scenario" focus area without changing the sampler.

By default we skip already-night samples (`apply_to_night=False`) — they
are already at the target distribution and further darkening just
introduces noise.
"""
import random
from typing import Tuple

import torch


class NightLikeAugmentation:
    """Low-light photometric augmentation. Stateless & cheap (CPU-friendly)."""

    def __init__(
        self,
        prob: float = 0.3,
        gamma_range: Tuple[float, float] = (1.5, 3.0),
        brightness_range: Tuple[float, float] = (0.3, 0.7),
        contrast_range: Tuple[float, float] = (0.5, 0.9),
        noise_std_range: Tuple[float, float] = (0.0, 0.05),
        apply_to_night: bool = False,
    ) -> None:
        self.prob = float(prob)
        self.gamma_range = tuple(gamma_range)
        self.brightness_range = tuple(brightness_range)
        self.contrast_range = tuple(contrast_range)
        self.noise_std_range = tuple(noise_std_range)
        self.apply_to_night = bool(apply_to_night)

    def __call__(
        self,
        rgb_norm: torch.Tensor,                              # (3, H, W) in [-1, 1]
        is_night: bool = False,
    ) -> torch.Tensor:
        # Skip already-dark samples by default — additional darkening just
        # adds entropy without expanding the input distribution.
        if is_night and not self.apply_to_night:
            return rgb_norm
        if random.random() >= self.prob:
            return rgb_norm

        gamma = random.uniform(*self.gamma_range)
        brightness = random.uniform(*self.brightness_range)
        contrast = random.uniform(*self.contrast_range)
        noise_std = random.uniform(*self.noise_std_range)

        # Operate in [0, 1] for gamma/brightness; pow() needs non-negative.
        x = ((rgb_norm + 1.0) * 0.5).clamp(0.0, 1.0)
        x = x.pow(gamma)
        x = x * brightness
        if contrast != 1.0:
            mean = x.mean(dim=(-2, -1), keepdim=True)
            x = (x - mean) * contrast + mean
        x = x.clamp(0.0, 1.0)

        out = x * 2.0 - 1.0
        if noise_std > 0:
            out = (out + torch.randn_like(out) * noise_std).clamp(-1.0, 1.0)
        return out


def build_from_cfg(cfg) -> "NightLikeAugmentation | None":
    """Build a NightLikeAugmentation from a dataset.photometric_aug config
    block. Returns None if the block is missing or `enabled: false`.
    """
    if cfg is None:
        return None
    if not cfg.get("enabled", False):
        return None
    return NightLikeAugmentation(
        prob=float(cfg.get("prob", 0.3)),
        gamma_range=tuple(cfg.get("gamma_range", (1.5, 3.0))),
        brightness_range=tuple(cfg.get("brightness_range", (0.3, 0.7))),
        contrast_range=tuple(cfg.get("contrast_range", (0.5, 0.9))),
        noise_std_range=tuple(cfg.get("noise_std_range", (0.0, 0.05))),
        apply_to_night=bool(cfg.get("apply_to_night", False)),
    )
