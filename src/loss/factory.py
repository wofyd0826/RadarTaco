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
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .losses import (
    AffineInvariantBlockGridL1Loss,
    AffineInvariantL1Loss,
    AntiStripeLoss,
    EdgeAwareSmoothnessLoss,
    GradientCosineLoss,
    GradientMatchingLoss,
    GradientShapeLoss,
    GradientSignLoss,
    LidarAlignedBlockGradShapeLoss,
    LidarAnchoredBlockGridShapeLoss,
    LidarAnchoredGlobalShapeLoss,
    MaskedL1Loss,
    RadarTacoLoss,
    affine_anchor_penalty,
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
        w_lidar: float = 1.0,
        # MoGe-2-style affine-invariant supervision against the dense GT.
        # When `w_shape_global > 0` or `w_shape_grid > 0`, the model gets
        # shape-only supervision (scale + shift absorbed via ROE solver)
        # and the absolute scale should come from the LiDAR L1 term.
        w_shape_global: float = 0.0,
        w_shape_grid: float = 0.0,
        # LiDAR-anchored shape losses — same idea as the standard shape
        # losses above but the affine (scale, shift) is solved against
        # `lidar` at LiDAR pixels (not against dense_gt). The shape
        # residual is taken against dense_gt over its valid mask.
        w_anchored_global: float = 0.0,
        w_anchored_grid: float = 0.0,
        # Auxiliary penalty: pulls each per-image / per-block (a, b)
        # toward identity (1, 0). When applied to anchored shape losses,
        # this encourages the model to be metric-consistent with dense_gt
        # so the LiDAR L1 weight can be safely reduced.
        w_affine_anchor: float = 0.0,
        # Consistency mask on the sparse-LiDAR L1 term: skip LiDAR pixels
        # where |dense_gt − lidar| exceeds this threshold (meters). These
        # are exactly the pixels driving stripe artifacts (Spearman ρ ≈ 0.88
        # between per-row |dense-lidar| and per-row pred vertical jump).
        # Default 0 = disabled (no mask, fully backward-compatible).
        lidar_consistency_threshold: float = 0.0,
        # Anti-stripe loss: RGB-guided vertical-only smoothness localized
        # to (optionally) LiDAR-row neighborhoods. Directly suppresses the
        # ∂y jumps that the L1_lidar vs shape_loss conflict produces,
        # without touching either supervision signal.
        w_anti_stripe: float = 0.0,
        anti_stripe_gamma: float = 10.0,
        anti_stripe_row_radius: int = 4,
        anti_stripe_near_lidar_only: bool = True,
        anti_stripe_normalize_by_mean: bool = False,
        # Gradient-only shape loss: L1 on log-depth gradients vs dense_gt.
        # Completely scale-invariant — does NOT compete with L1_lidar over
        # absolute metric. Drop-in replacement for AffineInvariantL1Loss
        # when the dense_gt's absolute values are unreliable.
        w_grad_shape: float = 0.0,
        grad_shape_scales: int = 4,
        grad_shape_use_log: bool = True,
        grad_shape_per_scale_weights=None,
        # Soft truncation (MoGe ROE-style robust loss): per-pixel residuals
        # |∂log pred − ∂log target| are clamped to this value BEFORE the
        # average. Suppresses outlier pixels (= regions where dense_gt is
        # least reliable, typically mid-range on DAv2-derived GT). null
        # disables. trunc=1.0 means depth ratio differences > e (~2.7×)
        # are all treated equally — bigger differences contribute no more
        # than this.
        grad_shape_trunc: Optional[float] = None,
        # LiDAR-conditioned shape mask: only supervise grad_shape on
        # pixels within `dilate` of a LiDAR point that agrees with
        # dense_gt within `threshold` meters. Restricts supervision to
        # regions where DAv2 is verifiably correct (no conflict to learn
        # at the wrong-ratio pixels). 0 disables (full image).
        grad_shape_consistency_threshold: float = 0.0,
        grad_shape_consistency_dilate: int = 8,
        # Block-LiDAR-aligned grad_shape: per K×K block, ROE-align
        # dense_gt → lidar BEFORE computing grad supervision. Directly
        # corrects DAv2's block-local ratio error.
        w_grad_shape_aligned: float = 0.0,
        grad_shape_aligned_K: int = 8,
        grad_shape_aligned_scales: int = 4,
        grad_shape_aligned_use_log: bool = True,
        grad_shape_aligned_trunc: Optional[float] = None,
        grad_shape_aligned_roe_trunc: float = 1.0,
        grad_shape_aligned_min_lidar_inliers: int = 8,
        grad_shape_aligned_align_resolution: int = 16,
        # Gradient-SIGN loss: same as grad_shape but only enforces the
        # *direction* (sign) of dense_gt's gradients, weighted by |gt|.
        # Use when dense_gt's ordering is trustworthy but its absolute
        # ratios are not (e.g. DAv2-derived dense GT). Can be combined
        # with `w_grad_shape > 0` for a soft magnitude prior on top.
        w_grad_sign: float = 0.0,
        grad_sign_scales: int = 4,
        grad_sign_use_log: bool = True,
        grad_sign_per_scale_weights=None,
        # Gradient-COSINE loss: same intent as grad_sign (DAv2's ordering,
        # not its noisy ratios) but on the full 2D gradient vector
        # direction with natural raw magnitude in [0, 2].
        w_grad_cos: float = 0.0,
        grad_cos_scales: int = 4,
        grad_cos_use_log: bool = True,
        grad_cos_per_scale_weights=None,
        anchored_min_lidar_inliers_global: int = 30,
        anchored_min_lidar_inliers_block: int = 8,
        shape_align_resolution: int = 32,
        shape_grid_K: int = 8,
        shape_grid_align_resolution: int = 16,
        shape_trunc: float = 1.0,
        shape_grid_min_inliers: int = 30,
        shape_scale_min: float = 0.1,
        shape_scale_max: float = 10.0,
        shape_shift_max_abs: float = 100.0,
        # Depth-bin loss mask (loss-specialist experiment). When set,
        # all supervision terms are restricted to pixels whose GT depth
        # falls in [depth_bin_min, depth_bin_max) — L1 uses depth_gt_lidar,
        # shape/grad terms use depth_gt_dense. Non-bin pixels are dropped
        # from every mask so no gradient flows there. Both None (default)
        # disables — behaviour identical to the un-modified loss.
        depth_bin_min: Optional[float] = None,
        depth_bin_max: Optional[float] = None,
        # MoE router loss (depth-specialised MoE fusion). Active when the
        # model returns `router_logits` (list of (B, K)) and `router_gt`
        # ((B,) int64) in its prediction dict. Cross-entropy of each
        # block's router against the shared per-sample GT bin label;
        # averaged over blocks. Only enabled during stage-1 (teacher-
        # forcing) — the model does not emit `router_gt` in stage 2.
        w_router: float = 0.0,
        # Per-pixel MoE bin mask. When True and the model provides
        # `router_gts` in its prediction dict (stage-1 with GT-forced
        # routing), each pixel is included in supervision ONLY when its
        # own depth-bin equals the routed bin of the token containing it.
        # Reproduces the loss-mask specialist regime inside MoE: focused
        # gradient per expert, at the cost of leaving minority pixels in
        # mixed tokens unsupervised.
        moe_pixel_bin_mask: bool = False,
        moe_pixel_bin_mask_bins: Optional[Tuple[float, ...]] = None,
    ) -> None:
        super().__init__()
        self.lam = lam
        self.w_smooth = w_smooth
        self.w_sim_grad = w_sim_grad
        self.w_sim_smooth = w_sim_smooth
        self.w_multi_scale = w_multi_scale
        self.w_lidar = w_lidar
        self.lidar_consistency_threshold = float(lidar_consistency_threshold)
        # Optional per-scale weighting (key like "depth_s2"). Uniform if None.
        self.multi_scale_per_level_weights = (
            dict(multi_scale_per_level_weights) if multi_scale_per_level_weights else None
        )
        self.real_loss = RadarTacoLoss(
            lam=lam, w_smooth=w_smooth,
            smoothness_rgb_grad_weight=smoothness_rgb_grad_weight,
            w_lidar=w_lidar,
        )
        self.l1 = MaskedL1Loss()
        self.grad = GradientMatchingLoss(scales=grad_scales)
        self.smoothness = EdgeAwareSmoothnessLoss(rgb_grad_weight=smoothness_rgb_grad_weight)
        self.w_shape_global = float(w_shape_global)
        self.w_shape_grid = float(w_shape_grid)
        self.shape_global = (
            AffineInvariantL1Loss(align_resolution=shape_align_resolution,
                                  trunc=shape_trunc,
                                  scale_min=shape_scale_min,
                                  scale_max=shape_scale_max,
                                  shift_max_abs=shape_shift_max_abs)
            if self.w_shape_global > 0 else None
        )
        self.shape_grid = (
            AffineInvariantBlockGridL1Loss(K=shape_grid_K, trunc=shape_trunc,
                                           min_inliers=shape_grid_min_inliers,
                                           align_resolution=shape_grid_align_resolution,
                                           scale_min=shape_scale_min,
                                           scale_max=shape_scale_max,
                                           shift_max_abs=shape_shift_max_abs)
            if self.w_shape_grid > 0 else None
        )
        self.w_anchored_global = float(w_anchored_global)
        self.w_anchored_grid = float(w_anchored_grid)
        self.w_affine_anchor = float(w_affine_anchor)
        # Instantiate the LiDAR-anchored ROE modules whenever EITHER their
        # loss is used (`w_anchored_*`) OR the affine-anchor penalty is
        # active (`w_affine_anchor > 0`). In the latter case we only need
        # their (a, b) — solved from (pred, lidar) — to drive the penalty
        # that pulls metric scale toward LiDAR. The L1-vs-dense residual
        # they return is ignored unless `w_anchored_* > 0`.
        need_global = self.w_anchored_global > 0 or self.w_affine_anchor > 0
        need_grid = self.w_anchored_grid > 0 or self.w_affine_anchor > 0
        self.anchored_global = (
            LidarAnchoredGlobalShapeLoss(
                align_resolution=shape_align_resolution,
                trunc=shape_trunc,
                scale_min=shape_scale_min,
                scale_max=shape_scale_max,
                shift_max_abs=shape_shift_max_abs,
                min_lidar_inliers=anchored_min_lidar_inliers_global,
            )
            if need_global else None
        )
        self.anchored_grid = (
            LidarAnchoredBlockGridShapeLoss(
                K=shape_grid_K,
                trunc=shape_trunc,
                min_lidar_inliers=anchored_min_lidar_inliers_block,
                min_dense_inliers=shape_grid_min_inliers,
                align_resolution=shape_grid_align_resolution,
                scale_min=shape_scale_min,
                scale_max=shape_scale_max,
                shift_max_abs=shape_shift_max_abs,
            )
            if need_grid else None
        )
        # Anti-stripe (RGB-guided vertical smoothness) + gradient-only
        # shape (log-depth gradient L1). Both are stripe countermeasures
        # that DO NOT alter the metric supervision (L1_lidar) — see the
        # class docstrings for the rationale.
        self.w_anti_stripe = float(w_anti_stripe)
        self.anti_stripe = (
            AntiStripeLoss(
                gamma=anti_stripe_gamma,
                row_radius=anti_stripe_row_radius,
                near_lidar_only=anti_stripe_near_lidar_only,
                normalize_by_mean=anti_stripe_normalize_by_mean,
            )
            if self.w_anti_stripe > 0 else None
        )
        self.w_grad_shape = float(w_grad_shape)
        self.grad_shape = (
            GradientShapeLoss(scales=grad_shape_scales,
                              use_log=grad_shape_use_log,
                              per_scale_weights=grad_shape_per_scale_weights,
                              trunc=grad_shape_trunc)
            if self.w_grad_shape > 0 else None
        )
        self.grad_shape_consistency_threshold = float(grad_shape_consistency_threshold)
        self.grad_shape_consistency_dilate = int(grad_shape_consistency_dilate)
        self.w_grad_shape_aligned = float(w_grad_shape_aligned)
        self.grad_shape_aligned = (
            LidarAlignedBlockGradShapeLoss(
                K=grad_shape_aligned_K,
                scales=grad_shape_aligned_scales,
                use_log=grad_shape_aligned_use_log,
                trunc=grad_shape_aligned_trunc,
                roe_trunc=grad_shape_aligned_roe_trunc,
                min_lidar_inliers=grad_shape_aligned_min_lidar_inliers,
                align_resolution=grad_shape_aligned_align_resolution,
                scale_min=shape_scale_min,
                scale_max=shape_scale_max,
                shift_max_abs=shape_shift_max_abs,
            )
            if self.w_grad_shape_aligned > 0 else None
        )
        self.w_grad_sign = float(w_grad_sign)
        self.grad_sign = (
            GradientSignLoss(scales=grad_sign_scales,
                             use_log=grad_sign_use_log,
                             per_scale_weights=grad_sign_per_scale_weights)
            if self.w_grad_sign > 0 else None
        )
        self.w_grad_cos = float(w_grad_cos)
        self.grad_cos = (
            GradientCosineLoss(scales=grad_cos_scales,
                               use_log=grad_cos_use_log,
                               per_scale_weights=grad_cos_per_scale_weights)
            if self.w_grad_cos > 0 else None
        )
        self.depth_bin_min = (float(depth_bin_min)
                              if depth_bin_min is not None else None)
        self.depth_bin_max = (float(depth_bin_max)
                              if depth_bin_max is not None else None)
        self.w_router = float(w_router)
        self.moe_pixel_bin_mask = bool(moe_pixel_bin_mask)
        if self.moe_pixel_bin_mask:
            assert moe_pixel_bin_mask_bins is not None and len(moe_pixel_bin_mask_bins) >= 2, \
                "moe_pixel_bin_mask=True requires moe_pixel_bin_mask_bins (>=2 edges)"
            self.register_buffer(
                "_moe_pix_mask_bins",
                torch.tensor(list(moe_pixel_bin_mask_bins), dtype=torch.float32),
            )
        else:
            self._moe_pix_mask_bins = None

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

        # Depth-bin loss mask (loss-specialist experiment). Intersect the
        # LiDAR and dense valid masks with pixels whose GT depth is inside
        # [depth_bin_min, depth_bin_max). Downstream loss terms then see
        # zero gradient outside the bin — mirroring the effect of hard
        # partitioning the training data by depth range.
        if self.depth_bin_min is not None or self.depth_bin_max is not None:
            lo = self.depth_bin_min if self.depth_bin_min is not None else 0.0
            hi = self.depth_bin_max if self.depth_bin_max is not None else float("inf")
            bin_l = (batch["depth_gt_lidar"] >= lo) & (batch["depth_gt_lidar"] < hi)
            bin_d = (batch["depth_gt_dense"] >= lo) & (batch["depth_gt_dense"] < hi)
            batch = dict(batch)  # shallow copy — other tensors shared
            batch["valid_mask_lidar"] = batch["valid_mask_lidar"] & bin_l
            batch["valid_mask_dense"] = batch["valid_mask_dense"] & bin_d

        # MoE per-pixel bin mask. Only include pixels whose true depth-bin
        # matches the routed bin of the containing token (per node-grid
        # router_gt, which is the finest MoE grid). Non-matching pixels
        # get no gradient — reproduces the loss-mask specialist regime.
        rg_list = aux_preds.get("router_gts") if isinstance(pred, dict) else None
        if (self.moe_pixel_bin_mask
                and rg_list is not None
                and any(t is not None for t in rg_list)
                and "depth_gt_dense" in batch):
            # Pick the finest (largest H*W) router_gt for mask granularity.
            token_gt = None
            for t in rg_list:
                if t is None: continue
                if token_gt is None or (t.shape[-2] * t.shape[-1] >
                                         token_gt.shape[-2] * token_gt.shape[-1]):
                    token_gt = t
            with torch.no_grad():
                dgd = batch["depth_gt_dense"]                    # (B, 1, H, W)
                edges = self._moe_pix_mask_bins                  # (K+1,)
                K = int(edges.numel()) - 1
                ds = dgd.squeeze(1)                              # (B, H, W)
                pix_bin = torch.zeros_like(ds, dtype=torch.long)
                for i in range(K):
                    lo_i = float(edges[i]); hi_i = float(edges[i + 1])
                    pix_bin = torch.where((ds >= lo_i) & (ds < hi_i),
                                          pix_bin.new_tensor(i, dtype=torch.long),
                                          pix_bin)
                pix_bin = torch.where(ds >= float(edges[-1]),
                                      pix_bin.new_tensor(K - 1, dtype=torch.long),
                                      pix_bin)
                # Upsample token_gt (B, Ht, Wt) → (B, H, W) nearest.
                token_up = F.interpolate(
                    token_gt.unsqueeze(1).float(),
                    size=pix_bin.shape[-2:],
                    mode="nearest",
                ).squeeze(1).long()
                match = (pix_bin == token_up)                    # (B, H, W)
            batch = dict(batch)
            # Broadcast to (B, 1, H, W) to intersect with valid_mask_*.
            match_b = match.unsqueeze(1)
            batch["valid_mask_lidar"] = batch["valid_mask_lidar"] & match_b
            batch["valid_mask_dense"] = batch["valid_mask_dense"] & match_b

        is_sim = batch["is_sim"].view(-1).bool()
        sim_idx = torch.where(is_sim)[0]
        real_idx = torch.where(~is_sim)[0]
        zero = depth_main.new_zeros(())

        # ------ real loss (paper Eq. 5) ------
        if real_idx.numel() > 0:
            mask_lidar = batch["valid_mask_lidar"][real_idx]
            # Consistency mask — drop LiDAR pixels where dense_gt strongly
            # disagrees with lidar; these are the per-row hard-fit targets
            # that create stripes (diagnostic ρ ≈ 0.88 with pred jumps).
            if self.lidar_consistency_threshold > 0:
                lidar_t = batch["depth_gt_lidar"][real_idx]
                dense_t = batch["depth_gt_dense"][real_idx]
                consistent = (dense_t - lidar_t).abs() < self.lidar_consistency_threshold
                # Also require dense to be valid where we evaluate consistency.
                consistent = consistent & batch["valid_mask_dense"][real_idx]
                mask_lidar = mask_lidar & consistent
            r = self.real_loss(
                depth_main[real_idx],
                batch["depth_gt_lidar"][real_idx],
                batch["depth_gt_dense"][real_idx],
                mask_lidar,
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

        # ------ MoGe-2-style shape losses on the dense GT (all real samples) ------
        # Both terms operate on the dense GT (e.g. depth_refined_grid) but
        # leave per-image scale/shift free — only per-pixel shape is
        # supervised. Pair with `w_lidar > 0` to anchor metric from sparse
        # LiDAR (paper "scale-from-LiDAR, shape-from-dense" decomposition).
        l_shape_g = zero
        l_shape_b = zero
        if real_idx.numel() > 0 and (self.shape_global is not None or self.shape_grid is not None):
            shape_pred = depth_main[real_idx]
            shape_gt = batch["depth_gt_dense"][real_idx]
            shape_mask = batch["valid_mask_dense"][real_idx]
            if self.shape_global is not None:
                l_shape_g = self.shape_global(shape_pred, shape_gt, shape_mask)
                total = total + w_real * self.w_shape_global * l_shape_g
            if self.shape_grid is not None:
                l_shape_b = self.shape_grid(shape_pred, shape_gt, shape_mask)
                total = total + w_real * self.w_shape_grid * l_shape_b

        # ------ LiDAR-anchored shape losses (+ optional affine-anchor penalty) ------
        # The affine (a, b) is solved from pred vs LiDAR at LiDAR pixels;
        # the L1 residual is taken against dense_gt. Combined with
        # `w_affine_anchor` (pulls a→1, b→0), this gives the model a
        # soft metric anchor toward dense_gt while LiDAR keeps the affine
        # locked to its scale. Designed to let `w_lidar` be reduced
        # without losing metric accuracy.
        l_anchored_g = zero
        l_anchored_b = zero
        l_anchor_pen = zero
        if real_idx.numel() > 0 and (
            self.anchored_global is not None or self.anchored_grid is not None
        ):
            ap = depth_main[real_idx]
            adense = batch["depth_gt_dense"][real_idx]
            alidar = batch["depth_gt_lidar"][real_idx]
            amask_d = batch["valid_mask_dense"][real_idx]
            amask_l = batch["valid_mask_lidar"][real_idx]
            anchor_pens = []
            if self.anchored_global is not None:
                l_anchored_g, sg, fg = self.anchored_global(
                    ap, adense, alidar, amask_d, amask_l)
                # Loss only when w_anchored_global > 0
                if self.w_anchored_global > 0:
                    total = total + w_real * self.w_anchored_global * l_anchored_g
                # (a, b) penalty only when w_affine_anchor > 0
                # — uses the (pred, lidar)-solved affine for LiDAR-metric anchor
                if self.w_affine_anchor > 0:
                    anchor_pens.append(affine_anchor_penalty(sg, fg))
            if self.anchored_grid is not None:
                l_anchored_b, sb, fb, keep_b = self.anchored_grid(
                    ap, adense, alidar, amask_d, amask_l)
                if self.w_anchored_grid > 0:
                    total = total + w_real * self.w_anchored_grid * l_anchored_b
                if self.w_affine_anchor > 0:
                    anchor_pens.append(
                        affine_anchor_penalty(sb, fb, keep_b))
            if anchor_pens:
                l_anchor_pen = torch.stack(anchor_pens).mean()
                total = total + w_real * self.w_affine_anchor * l_anchor_pen

        # ------ anti-stripe + gradient-only shape (real samples) ------
        l_anti_stripe = zero
        l_grad_shape = zero
        l_grad_shape_aligned = zero
        l_grad_sign = zero
        l_grad_cos = zero
        if real_idx.numel() > 0:
            if self.anti_stripe is not None:
                # Anti-stripe is a self-consistency loss on (pred, rgb)
                # gradients — it never touches dense_gt values, so the
                # dense valid_mask is intentionally NOT applied. Invalid
                # regions (sky, ego, padding) still have meaningful
                # RGB structure and a stripe there is just as unphysical.
                l_anti_stripe = self.anti_stripe(
                    depth_main[real_idx],
                    batch["rgb_norm"][real_idx],
                    lidar_mask=batch["valid_mask_lidar"][real_idx],
                )
                total = total + w_real * self.w_anti_stripe * l_anti_stripe
            if self.grad_shape is not None:
                # Optional LiDAR-conditioned mask: trust only pixels near
                # a LiDAR point that agrees with dense_gt within τ.
                trust_mask = None
                if self.grad_shape_consistency_threshold > 0:
                    lidar_t = batch["depth_gt_lidar"][real_idx]
                    dense_t = batch["depth_gt_dense"][real_idx]
                    mask_l = batch["valid_mask_lidar"][real_idx]
                    consistent = (dense_t - lidar_t).abs() < self.grad_shape_consistency_threshold
                    consistent = consistent & mask_l
                    if self.grad_shape_consistency_dilate > 0:
                        r = self.grad_shape_consistency_dilate
                        k = 2 * r + 1
                        consistent = F.max_pool2d(
                            consistent.float(),
                            kernel_size=k, stride=1, padding=r,
                        ) > 0.5
                    trust_mask = consistent
                # Depth-bin restriction (loss-specialist experiment) OR
                # per-pixel MoE bin mask. grad_shape defaults to whole
                # image; force it to the bin-restricted dense mask so no
                # gradient flows outside the supervised region. Both modes
                # modify batch["valid_mask_dense"] earlier in forward().
                if (self.depth_bin_min is not None
                        or self.depth_bin_max is not None
                        or self.moe_pixel_bin_mask):
                    bin_mask = batch["valid_mask_dense"][real_idx]
                    trust_mask = bin_mask if trust_mask is None else (trust_mask & bin_mask)
                l_grad_shape = self.grad_shape(
                    depth_main[real_idx],
                    batch["depth_gt_dense"][real_idx],
                    mask=trust_mask,
                )
                total = total + w_real * self.w_grad_shape * l_grad_shape
            if self.grad_shape_aligned is not None:
                # Per-block ROE alignment of dense_gt to LiDAR, then
                # multi-scale log-grad L1 vs pred. Block-local DAv2
                # ratio errors are absorbed by per-block (s, b).
                l_grad_shape_aligned, _sg, _bg, _kg = self.grad_shape_aligned(
                    depth_main[real_idx],
                    batch["depth_gt_dense"][real_idx],
                    batch["depth_gt_lidar"][real_idx],
                    batch["valid_mask_lidar"][real_idx],
                )
                total = total + w_real * self.w_grad_shape_aligned * l_grad_shape_aligned
            if self.grad_sign is not None:
                # Same no-mask reasoning as grad_shape — the |gt| weight
                # already suppresses flat (== invalid-fill) regions to
                # near zero, so the loss is self-robust.
                l_grad_sign = self.grad_sign(
                    depth_main[real_idx],
                    batch["depth_gt_dense"][real_idx],
                )
                total = total + w_real * self.w_grad_sign * l_grad_sign
            if self.grad_cos is not None:
                # Same self-suppression via |g_target| weighting → no mask.
                l_grad_cos = self.grad_cos(
                    depth_main[real_idx],
                    batch["depth_gt_dense"][real_idx],
                )
                total = total + w_real * self.w_grad_cos * l_grad_cos

        # ------ multi-scale aux loss (per-pixel L1 vs dense GT, all samples) ------
        if aux_preds and self.w_multi_scale > 0:
            l_ms = self._multi_scale_loss(aux_preds, batch)
            total = total + self.w_multi_scale * l_ms
        else:
            l_ms = zero

        # ------ MoE router CE loss (per-token, teacher-forcing / stage 1) ---
        # Per-token routing means each MoE block emits logits of shape
        # (B, K, H_block, W_block) with matching int64 GT (B, H_block, W_block)
        # — one pair per block. `F.cross_entropy` handles 2D-class inputs
        # natively (spatial dims are treated as batch axes).
        l_router = zero
        if (self.w_router > 0
                and isinstance(pred, dict)
                and "router_logits" in pred
                and "router_gts" in pred):
            ce_list = []
            for lg, gt in zip(pred["router_logits"], pred["router_gts"]):
                if gt is None:                # stage-2 / eval — no GT for this block
                    continue
                ce_list.append(F.cross_entropy(lg, gt))
            if ce_list:
                l_router = torch.stack(ce_list).mean()
                total = total + self.w_router * l_router

        return {
            "loss_total": total,
            "loss_lidar": l_lidar.detach(),
            "loss_dense": l_dense.detach(),
            "loss_smooth": l_smooth.detach(),
            "loss_sim_l1": l_sim_l1.detach() if torch.is_tensor(l_sim_l1) else torch.tensor(0.0),
            "loss_sim_grad": l_sim_grad.detach() if torch.is_tensor(l_sim_grad) else torch.tensor(0.0),
            "loss_sim_smooth": l_sim_smooth.detach() if torch.is_tensor(l_sim_smooth) else torch.tensor(0.0),
            "loss_multi_scale": l_ms.detach(),
            "loss_shape_global": l_shape_g.detach() if torch.is_tensor(l_shape_g) else torch.tensor(0.0),
            "loss_anchored_global": l_anchored_g.detach() if torch.is_tensor(l_anchored_g) else torch.tensor(0.0),
            "loss_anchored_grid":   l_anchored_b.detach() if torch.is_tensor(l_anchored_b) else torch.tensor(0.0),
            "loss_affine_anchor":   l_anchor_pen.detach() if torch.is_tensor(l_anchor_pen) else torch.tensor(0.0),
            "loss_shape_grid":   l_shape_b.detach() if torch.is_tensor(l_shape_b) else torch.tensor(0.0),
            "loss_anti_stripe":  l_anti_stripe.detach() if torch.is_tensor(l_anti_stripe) else torch.tensor(0.0),
            "loss_grad_shape":   l_grad_shape.detach() if torch.is_tensor(l_grad_shape) else torch.tensor(0.0),
            "loss_grad_shape_aligned": l_grad_shape_aligned.detach() if torch.is_tensor(l_grad_shape_aligned) else torch.tensor(0.0),
            "loss_grad_sign":    l_grad_sign.detach()  if torch.is_tensor(l_grad_sign)  else torch.tensor(0.0),
            "loss_grad_cos":     l_grad_cos.detach()   if torch.is_tensor(l_grad_cos)   else torch.tensor(0.0),
            "loss_router":       l_router.detach() if torch.is_tensor(l_router) else torch.tensor(0.0),
        }

    # ----------------------------------------------- multi-scale helper --
    def _multi_scale_loss(
        self,
        aux_preds: Dict[str, torch.Tensor],
        batch: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Paper Eq.5 applied at each auxiliary scale (Singh decoder parity).

        Each aux prediction is bilinear-upsampled to GT resolution and
        scored against BOTH `depth_gt_lidar` (sparse) and `depth_gt_dense`
        (accumulated) — the same `L1_lidar + λ·L1_dense` structure as the
        main full-res loss. Per-scale weights default to uniform; override
        via `multi_scale_per_level_weights`.
        """
        gt_lidar = batch["depth_gt_lidar"]
        mask_lidar = batch["valid_mask_lidar"]
        gt_dense = batch["depth_gt_dense"]
        mask_dense = batch["valid_mask_dense"]
        target_hw = gt_dense.shape[-2:]

        losses = []
        weights = []
        for key, p in aux_preds.items():
            p_up = F.interpolate(p, size=target_hw, mode="bilinear", align_corners=False)
            if self.w_lidar > 0:
                l_lidar_s = self.l1(p_up, gt_lidar, mask_lidar)
            else:
                l_lidar_s = p_up.new_zeros(())
            l_dense_s = self.l1(p_up, gt_dense, mask_dense)
            losses.append(self.w_lidar * l_lidar_s + self.lam * l_dense_s)
            if self.multi_scale_per_level_weights is not None:
                weights.append(float(self.multi_scale_per_level_weights.get(key, 1.0)))
            else:
                weights.append(1.0)

        # Weighted mean (so the magnitude is comparable to single-scale L1).
        loss_t = torch.stack(losses)
        w_t = loss_t.new_tensor(weights)
        return (loss_t * w_t).sum() / w_t.sum().clamp(min=1e-8)


def _as_weights(v):
    """Convert OmegaConf list / None / Sequence to plain list of floats or None."""
    if v is None:
        return None
    return [float(x) for x in v]


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
        w_lidar=float(cfg.get("w_lidar", 1.0)),
        w_shape_global=float(cfg.get("w_shape_global", 0.0)),
        w_shape_grid=float(cfg.get("w_shape_grid", 0.0)),
        shape_align_resolution=int(cfg.get("shape_align_resolution", 32)),
        shape_grid_K=int(cfg.get("shape_grid_K", 8)),
        shape_grid_align_resolution=int(cfg.get("shape_grid_align_resolution", 16)),
        shape_trunc=float(cfg.get("shape_trunc", 1.0)),
        shape_grid_min_inliers=int(cfg.get("shape_grid_min_inliers", 30)),
        shape_scale_min=float(cfg.get("shape_scale_min", 0.1)),
        shape_scale_max=float(cfg.get("shape_scale_max", 10.0)),
        shape_shift_max_abs=float(cfg.get("shape_shift_max_abs", 100.0)),
        w_anchored_global=float(cfg.get("w_anchored_global", 0.0)),
        w_anchored_grid=float(cfg.get("w_anchored_grid", 0.0)),
        w_affine_anchor=float(cfg.get("w_affine_anchor", 0.0)),
        lidar_consistency_threshold=float(cfg.get("lidar_consistency_threshold", 0.0)),
        w_anti_stripe=float(cfg.get("w_anti_stripe", 0.0)),
        anti_stripe_gamma=float(cfg.get("anti_stripe_gamma", 10.0)),
        anti_stripe_row_radius=int(cfg.get("anti_stripe_row_radius", 4)),
        anti_stripe_near_lidar_only=bool(cfg.get("anti_stripe_near_lidar_only", True)),
        anti_stripe_normalize_by_mean=bool(cfg.get("anti_stripe_normalize_by_mean", False)),
        w_grad_shape=float(cfg.get("w_grad_shape", 0.0)),
        grad_shape_scales=int(cfg.get("grad_shape_scales", 4)),
        grad_shape_use_log=bool(cfg.get("grad_shape_use_log", True)),
        grad_shape_per_scale_weights=_as_weights(cfg.get("grad_shape_per_scale_weights", None)),
        grad_shape_trunc=(float(cfg.get("grad_shape_trunc"))
                          if cfg.get("grad_shape_trunc") is not None else None),
        grad_shape_consistency_threshold=float(cfg.get("grad_shape_consistency_threshold", 0.0)),
        grad_shape_consistency_dilate=int(cfg.get("grad_shape_consistency_dilate", 8)),
        w_grad_shape_aligned=float(cfg.get("w_grad_shape_aligned", 0.0)),
        grad_shape_aligned_K=int(cfg.get("grad_shape_aligned_K", 8)),
        grad_shape_aligned_scales=int(cfg.get("grad_shape_aligned_scales", 4)),
        grad_shape_aligned_use_log=bool(cfg.get("grad_shape_aligned_use_log", True)),
        grad_shape_aligned_trunc=(float(cfg.get("grad_shape_aligned_trunc"))
                                  if cfg.get("grad_shape_aligned_trunc") is not None else None),
        grad_shape_aligned_roe_trunc=float(cfg.get("grad_shape_aligned_roe_trunc", 1.0)),
        grad_shape_aligned_min_lidar_inliers=int(cfg.get("grad_shape_aligned_min_lidar_inliers", 8)),
        grad_shape_aligned_align_resolution=int(cfg.get("grad_shape_aligned_align_resolution", 16)),
        w_grad_sign=float(cfg.get("w_grad_sign", 0.0)),
        grad_sign_scales=int(cfg.get("grad_sign_scales", 4)),
        grad_sign_use_log=bool(cfg.get("grad_sign_use_log", True)),
        grad_sign_per_scale_weights=_as_weights(cfg.get("grad_sign_per_scale_weights", None)),
        w_grad_cos=float(cfg.get("w_grad_cos", 0.0)),
        grad_cos_scales=int(cfg.get("grad_cos_scales", 4)),
        grad_cos_use_log=bool(cfg.get("grad_cos_use_log", True)),
        grad_cos_per_scale_weights=_as_weights(cfg.get("grad_cos_per_scale_weights", None)),
        anchored_min_lidar_inliers_global=int(
            cfg.get("anchored_min_lidar_inliers_global", 30)),
        anchored_min_lidar_inliers_block=int(
            cfg.get("anchored_min_lidar_inliers_block", 8)),
        depth_bin_min=(float(cfg.get("depth_bin_min"))
                       if cfg.get("depth_bin_min") is not None else None),
        depth_bin_max=(float(cfg.get("depth_bin_max"))
                       if cfg.get("depth_bin_max") is not None else None),
        w_router=float(cfg.get("w_router", 0.0)),
        moe_pixel_bin_mask=bool(cfg.get("moe_pixel_bin_mask", False)),
        moe_pixel_bin_mask_bins=(tuple(float(x) for x in cfg.get("moe_pixel_bin_mask_bins"))
                                  if cfg.get("moe_pixel_bin_mask_bins") is not None else None),
    )
