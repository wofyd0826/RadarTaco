"""Loss functions for RadarTaco.

Primary (paper Eq. 5):
    ℓ_L1 = (1/|Ω_lidar|) Σ |D − D_lidar| + λ · (1/|Ω_dense|) Σ |D − D_dense|
with λ = 1.0.

Optional sim-only auxiliary objectives (focus 2: simulation dataset):
    EdgeAwareSmoothness  — Monodepth2-style; sharpens depth where RGB is flat
    GradientMatching     — multi-scale L1 on depth gradients vs dense GT;
                            transfers structure / sharpness priors from sim's
                            clean dense depth.

Aux losses are routed by per-sample `is_sim` flag in the trainer.
"""
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskedL1Loss(nn.Module):
    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        diff = (pred - target).abs() * mask.float()
        n = mask.float().sum().clamp(min=1.0)
        return diff.sum() / n


class EdgeAwareSmoothnessLoss(nn.Module):
    """Monodepth2-style edge-aware smoothness."""

    def __init__(self, rgb_grad_weight: float = 1.0, normalize_by_mean: bool = True) -> None:
        super().__init__()
        self.rgb_grad_weight = rgb_grad_weight
        self.normalize_by_mean = normalize_by_mean

    def forward(self, depth: torch.Tensor, rgb: torch.Tensor) -> torch.Tensor:
        if self.normalize_by_mean:
            mean_d = depth.mean(dim=(2, 3), keepdim=True).clamp(min=1e-6)
            depth = depth / mean_d
        d_dx = (depth[..., :, 1:] - depth[..., :, :-1]).abs()
        d_dy = (depth[..., 1:, :] - depth[..., :-1, :]).abs()
        i_dx = (rgb[..., :, 1:] - rgb[..., :, :-1]).abs().mean(1, keepdim=True)
        i_dy = (rgb[..., 1:, :] - rgb[..., :-1, :]).abs().mean(1, keepdim=True)
        d_dx = d_dx * torch.exp(-self.rgb_grad_weight * i_dx)
        d_dy = d_dy * torch.exp(-self.rgb_grad_weight * i_dy)
        return d_dx.mean() + d_dy.mean()


class GradientMatchingLoss(nn.Module):
    """Multi-scale L1 on depth gradients vs dense GT (MiDaS-style).

    Encourages the predicted depth to share local structure (edges, corners)
    with the GT — particularly effective on sim's clean dense depth where
    every pixel carries a reliable gradient signal.
    """

    def __init__(self, scales: int = 4, eps: float = 1e-3) -> None:
        super().__init__()
        self.scales = scales
        self.eps = eps

    @staticmethod
    def _grad(x: torch.Tensor):
        gx = x[..., :, 1:] - x[..., :, :-1]
        gy = x[..., 1:, :] - x[..., :-1, :]
        return gx, gy

    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        loss = pred.new_zeros(())
        p, t, m = pred, target, mask.float()
        for s in range(self.scales):
            if s > 0:
                p = F.avg_pool2d(p, 2)
                t = F.avg_pool2d(t, 2)
                m = F.avg_pool2d(m, 2)
            gpx, gpy = self._grad(p)
            gtx, gty = self._grad(t)
            mx = (m[..., :, 1:] * m[..., :, :-1])
            my = (m[..., 1:, :] * m[..., :-1, :])
            n_x = mx.sum().clamp(min=1.0)
            n_y = my.sum().clamp(min=1.0)
            loss = loss + ((gpx - gtx).abs() * mx).sum() / n_x \
                        + ((gpy - gty).abs() * my).sum() / n_y
        return loss / max(self.scales, 1)


class RadarTacoLoss(nn.Module):
    """L1 against `D_lidar` + λ·L1 against `D_dense` + (optional) edge-aware smoothness.

    Sim-only auxiliary losses are NOT applied here — see `compose_losses` in
    `loss.factory` which routes per-sample sources.
    """

    def __init__(
        self,
        lam: float = 1.0,
        w_smooth: float = 0.0,
        smoothness_rgb_grad_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.lam = lam
        self.w_smooth = w_smooth
        self.l1 = MaskedL1Loss()
        self.smoothness = EdgeAwareSmoothnessLoss(rgb_grad_weight=smoothness_rgb_grad_weight)

    def forward(
        self,
        pred: torch.Tensor,
        depth_gt_lidar: torch.Tensor,
        depth_gt_dense: torch.Tensor,
        valid_mask_lidar: torch.Tensor,
        valid_mask_dense: torch.Tensor,
        rgb_norm: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        l_lidar = self.l1(pred, depth_gt_lidar, valid_mask_lidar)
        l_dense = self.l1(pred, depth_gt_dense, valid_mask_dense)
        if self.w_smooth > 0:
            if rgb_norm is None:
                raise ValueError("rgb_norm required when w_smooth > 0")
            l_smooth = self.smoothness(pred, rgb_norm)
        else:
            l_smooth = pred.new_zeros(())
        total = l_lidar + self.lam * l_dense + self.w_smooth * l_smooth
        return {
            "loss_total": total,
            "loss_lidar": l_lidar.detach(),
            "loss_dense": l_dense.detach(),
            "loss_smooth": l_smooth.detach(),
        }
