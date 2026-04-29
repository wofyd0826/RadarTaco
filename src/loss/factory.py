"""Composite loss with per-sample source routing.

Real (nuScenes) samples and synthetic (Hypersim/vKITTI2) samples can require
different objectives:
    - Real: Eq. 5 L1(D, D_lidar) + λ·L1(D, D_dense)
    - Sim:  L1 (with the dense GT as both targets) + GradientMatchingLoss

`ComposedLoss` accepts a per-sample `is_sim` flag and routes accordingly,
returning a single scalar loss for backprop and detached components for
logging.
"""
from typing import Dict

import torch
import torch.nn as nn

from .losses import (
    EdgeAwareSmoothnessLoss,
    GradientMatchingLoss,
    MaskedL1Loss,
    RadarTacoLoss,
)


class ComposedLoss(nn.Module):
    """Routes per-sample sources to the right combination of objectives."""

    def __init__(
        self,
        lam: float = 1.0,
        w_smooth: float = 0.0,
        w_sim_grad: float = 0.0,
        w_sim_smooth: float = 0.0,
        smoothness_rgb_grad_weight: float = 1.0,
        grad_scales: int = 4,
    ) -> None:
        super().__init__()
        self.lam = lam
        self.w_smooth = w_smooth
        self.w_sim_grad = w_sim_grad
        self.w_sim_smooth = w_sim_smooth
        self.real_loss = RadarTacoLoss(
            lam=lam, w_smooth=w_smooth,
            smoothness_rgb_grad_weight=smoothness_rgb_grad_weight,
        )
        self.l1 = MaskedL1Loss()
        self.grad = GradientMatchingLoss(scales=grad_scales)
        self.smoothness = EdgeAwareSmoothnessLoss(rgb_grad_weight=smoothness_rgb_grad_weight)

    def forward(
        self,
        pred: torch.Tensor,
        batch: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        is_sim = batch["is_sim"].view(-1).bool()
        sim_idx = torch.where(is_sim)[0]
        real_idx = torch.where(~is_sim)[0]
        zero = pred.new_zeros(())

        # ------ real loss (paper Eq. 5) ------
        if real_idx.numel() > 0:
            r = self.real_loss(
                pred[real_idx],
                batch["depth_gt_lidar"][real_idx],
                batch["depth_gt_dense"][real_idx],
                batch["valid_mask_lidar"][real_idx],
                batch["valid_mask_dense"][real_idx],
                rgb_norm=batch["rgb_norm"][real_idx],
            )
            l_real = r["loss_total"]
            l_lidar = r["loss_lidar"]
            l_dense = r["loss_dense"]
            l_smooth = r["loss_smooth"]
        else:
            l_real = zero; l_lidar = zero; l_dense = zero; l_smooth = zero

        # ------ sim loss (L1 + grad match + optional edge-aware smoothness) ------
        if sim_idx.numel() > 0:
            sim_pred = pred[sim_idx]
            sim_gt = batch["depth_gt_dense"][sim_idx]
            sim_mask = batch["valid_mask_dense"][sim_idx]
            l_sim_l1 = self.l1(sim_pred, sim_gt, sim_mask)
            l_sim_grad = (self.grad(sim_pred, sim_gt, sim_mask)
                          if self.w_sim_grad > 0 else zero)
            l_sim_smooth = (self.smoothness(sim_pred, batch["rgb_norm"][sim_idx])
                            if self.w_sim_smooth > 0 else zero)
            l_sim = l_sim_l1 + self.w_sim_grad * l_sim_grad + self.w_sim_smooth * l_sim_smooth
        else:
            l_sim = zero
            l_sim_l1 = zero
            l_sim_grad = zero
            l_sim_smooth = zero

        # weight by sample fraction so total ≈ mean across batch
        n_total = pred.shape[0]
        w_real = real_idx.numel() / max(n_total, 1)
        w_sim = sim_idx.numel() / max(n_total, 1)
        total = w_real * l_real + w_sim * l_sim

        return {
            "loss_total": total,
            "loss_lidar": l_lidar.detach(),
            "loss_dense": l_dense.detach(),
            "loss_smooth": l_smooth.detach(),
            "loss_sim_l1": l_sim_l1.detach() if torch.is_tensor(l_sim_l1) else torch.tensor(0.0),
            "loss_sim_grad": l_sim_grad.detach() if torch.is_tensor(l_sim_grad) else torch.tensor(0.0),
            "loss_sim_smooth": l_sim_smooth.detach() if torch.is_tensor(l_sim_smooth) else torch.tensor(0.0),
        }


def build_loss(cfg) -> nn.Module:
    """Build a `ComposedLoss` from a Hydra/OmegaConf loss config block."""
    return ComposedLoss(
        lam=float(cfg.get("lam", 1.0)),
        w_smooth=float(cfg.get("w_smooth", 0.0)),
        w_sim_grad=float(cfg.get("w_sim_grad", 0.0)),
        w_sim_smooth=float(cfg.get("w_sim_smooth", 0.0)),
        smoothness_rgb_grad_weight=float(cfg.get("smoothness_rgb_grad_weight", 1.0)),
        grad_scales=int(cfg.get("grad_scales", 4)),
    )
