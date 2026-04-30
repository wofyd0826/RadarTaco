"""Composite loss with per-sample source routing + multi-scale aux loss.

Real (nuScenes) samples and synthetic (Hypersim/vKITTI2) samples can require
different objectives:
    - Real: Eq. 5 L1(D, D_lidar) + λ·L1(D, D_dense)
    - Sim:  L1 (with the dense GT as both targets) + GradientMatchingLoss

`ComposedLoss` accepts a per-sample `is_sim` flag and routes accordingly,
returning a single scalar loss for backprop and detached components for
logging.

Multi-scale auxiliary loss:
    When the decoder is configured with `multi_scale=True` it returns a dict
        {"depth": d_full, "depth_s2": d_1_2, "depth_s4": d_1_4, ...}
    `ComposedLoss` detects the dict, applies the existing real/sim routing
    to `d_full`, and adds an L1 aux term per coarse scale (predictions are
    bilinear-upsampled to the GT resolution). Aux loss uses the dense GT
    (paper's Dacc) for both real and sim samples — it has dense per-pixel
    coverage at every scale, which is what multi-scale supervision needs.
"""
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

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
        w_multi_scale: float = 0.0,
        smoothness_rgb_grad_weight: float = 1.0,
        grad_scales: int = 4,
        multi_scale_per_level_weights: Optional[Dict[str, float]] = None,
    ) -> None:
        super().__init__()
        self.lam = lam
        self.w_smooth = w_smooth
        self.w_sim_grad = w_sim_grad
        self.w_sim_smooth = w_sim_smooth
        self.w_multi_scale = w_multi_scale
        # Optional per-scale weighting (key like "depth_s2"). Uniform if None.
        self.multi_scale_per_level_weights = (
            dict(multi_scale_per_level_weights) if multi_scale_per_level_weights else None
        )
        self.real_loss = RadarTacoLoss(
            lam=lam, w_smooth=w_smooth,
            smoothness_rgb_grad_weight=smoothness_rgb_grad_weight,
        )
        self.l1 = MaskedL1Loss()
        self.grad = GradientMatchingLoss(scales=grad_scales)
        self.smoothness = EdgeAwareSmoothnessLoss(rgb_grad_weight=smoothness_rgb_grad_weight)

    # ----------------------------------------------------- main forward --
    def forward(
        self,
        pred,                                                # Tensor or dict
        batch: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        # Multi-scale decoder returns dict; single-scale returns Tensor.
        if isinstance(pred, dict):
            depth_main = pred["depth"]
            aux_preds = {k: v for k, v in pred.items() if k != "depth"}
        else:
            depth_main = pred
            aux_preds = {}

        is_sim = batch["is_sim"].view(-1).bool()
        sim_idx = torch.where(is_sim)[0]
        real_idx = torch.where(~is_sim)[0]
        zero = depth_main.new_zeros(())

        # ------ real loss (paper Eq. 5) ------
        if real_idx.numel() > 0:
            r = self.real_loss(
                depth_main[real_idx],
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
            sim_pred = depth_main[sim_idx]
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
        n_total = depth_main.shape[0]
        w_real = real_idx.numel() / max(n_total, 1)
        w_sim = sim_idx.numel() / max(n_total, 1)
        total = w_real * l_real + w_sim * l_sim

        # ------ multi-scale aux loss (per-pixel L1 vs dense GT, all samples) ------
        if aux_preds and self.w_multi_scale > 0:
            l_ms = self._multi_scale_loss(aux_preds, batch)
            total = total + self.w_multi_scale * l_ms
        else:
            l_ms = zero

        return {
            "loss_total": total,
            "loss_lidar": l_lidar.detach(),
            "loss_dense": l_dense.detach(),
            "loss_smooth": l_smooth.detach(),
            "loss_sim_l1": l_sim_l1.detach() if torch.is_tensor(l_sim_l1) else torch.tensor(0.0),
            "loss_sim_grad": l_sim_grad.detach() if torch.is_tensor(l_sim_grad) else torch.tensor(0.0),
            "loss_sim_smooth": l_sim_smooth.detach() if torch.is_tensor(l_sim_smooth) else torch.tensor(0.0),
            "loss_multi_scale": l_ms.detach(),
        }

    # ----------------------------------------------- multi-scale helper --
    def _multi_scale_loss(
        self,
        aux_preds: Dict[str, torch.Tensor],
        batch: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """L1 of bilinear-upsampled aux predictions vs dense GT.

        Uses the dense GT (depth_gt_dense / valid_mask_dense) which is
        populated for both real (Delaunay-interpolated lidar) and sim
        (true dense GT) samples. This gives strong per-pixel signal at
        every scale. Per-scale weights default to uniform; override
        via `multi_scale_per_level_weights`.
        """
        gt = batch["depth_gt_dense"]                              # (B, 1, H, W)
        mask = batch["valid_mask_dense"]
        target_hw = gt.shape[-2:]

        losses = []
        weights = []
        for key, p in aux_preds.items():
            p_up = F.interpolate(p, size=target_hw, mode="bilinear", align_corners=False)
            losses.append(self.l1(p_up, gt, mask))
            if self.multi_scale_per_level_weights is not None:
                weights.append(float(self.multi_scale_per_level_weights.get(key, 1.0)))
            else:
                weights.append(1.0)

        # Weighted mean (so the magnitude is comparable to single-scale L1).
        loss_t = torch.stack(losses)
        w_t = loss_t.new_tensor(weights)
        return (loss_t * w_t).sum() / w_t.sum().clamp(min=1e-8)


def build_loss(cfg) -> nn.Module:
    """Build a `ComposedLoss` from a Hydra/OmegaConf loss config block."""
    per_level = cfg.get("multi_scale_per_level_weights", None)
    if per_level is not None:
        # OmegaConf dicts → plain dict
        per_level = dict(per_level)
    return ComposedLoss(
        lam=float(cfg.get("lam", 1.0)),
        w_smooth=float(cfg.get("w_smooth", 0.0)),
        w_sim_grad=float(cfg.get("w_sim_grad", 0.0)),
        w_sim_smooth=float(cfg.get("w_sim_smooth", 0.0)),
        w_multi_scale=float(cfg.get("w_multi_scale", 0.0)),
        smoothness_rgb_grad_weight=float(cfg.get("smoothness_rgb_grad_weight", 1.0)),
        grad_scales=int(cfg.get("grad_scales", 4)),
        multi_scale_per_level_weights=per_level,
    )
