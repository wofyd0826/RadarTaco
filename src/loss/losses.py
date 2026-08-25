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
from typing import Dict, Optional, Sequence


def _normalize_scale_weights(weights: Optional[Sequence[float]],
                             scales: int) -> Sequence[float]:
    """Return per-scale weights that sum to 1.

    - None  → uniform [1/scales, ..., 1/scales] (preserves prior behaviour
              of `loss / scales` averaging across scales).
    - list  → renormalised so sum == 1, preserving the *relative* per-scale
              ratios the user specified.
    """
    if weights is None:
        return [1.0 / max(scales, 1)] * scales
    if len(weights) != scales:
        raise ValueError(
            f"per-scale weights length {len(weights)} != scales {scales}")
    s = float(sum(weights))
    if s <= 0:
        raise ValueError(f"per-scale weights must sum to > 0, got {weights}")
    return [float(w) / s for w in weights]

import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

# MoGe-2 ROE solver (Robust, Optimal, Efficient — paper §3.3). Imported
# lazily to keep MoGe an optional dependency.
_MOGE_ROOT = "/workspace/MoGe"
if os.path.isdir(_MOGE_ROOT) and _MOGE_ROOT not in sys.path:
    sys.path.insert(0, _MOGE_ROOT)


def _align_depth_affine(*args, **kwargs):
    from moge.utils.alignment import align_depth_affine
    return align_depth_affine(*args, **kwargs)


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


def _masked_avg_pool(x: torch.Tensor, mask: torch.Tensor, out_hw,
                      valid_fraction: float = 0.1):
    """Mask-aware average pooling to `out_hw`. Returns (x_lr, mask_lr).

    x:    (B, 1, H, W) float
    mask: (B, 1, H, W) bool / 0-1 float
    Output mask is True where the corresponding low-res cell contains at
    least `valid_fraction` of its input pixels valid (default 10%).
    """
    m = mask.float()
    sum_xm = F.adaptive_avg_pool2d(x * m, out_hw)
    avg_m = F.adaptive_avg_pool2d(m, out_hw)            # in [0, 1]
    x_lr = sum_xm / avg_m.clamp_min(1e-6)
    mask_lr = avg_m >= valid_fraction
    return x_lr, mask_lr


class AffineInvariantL1Loss(nn.Module):
    """Per-image affine-invariant L1 (MoGe-2 §3.3, global variant).

    Solves a single per-image (scale, shift) on (`pred`, `target`) via the
    ROE truncated-L1 solver, applies it to `pred`, then computes masked L1
    against `target` at full resolution. The model's depth can drift in
    metric scale — only per-pixel shape is supervised here.

    Algorithm:
      1. Mask-aware downsample of pred/target/mask to `align_resolution`
         to keep the ROE solve cheap.
      2. ROE solve `(scale, shift) = argmin Σ_mask min(trunc, (1/gt) ·
         |scale·pred + shift − gt|)`. trunc=1.0 means residuals beyond
         100% relative depth-error are capped — robust to outlier GT.
      3. `pred_aligned = scale · pred + shift` (full resolution).
      4. `loss = mean_mask |pred_aligned − target|`.

    The (scale, shift) are differentiable in pred (gradient flows through
    the closed-form ROE selection), matching MoGe-2's training behavior.
    """

    def __init__(
        self,
        align_resolution: int = 32,
        trunc: float = 1.0,
        eps: float = 1e-3,
        scale_min: float = 0.1,
        scale_max: float = 10.0,
        shift_max_abs: float = 100.0,
    ) -> None:
        super().__init__()
        self.align_resolution = int(align_resolution)
        self.trunc = float(trunc)
        self.eps = float(eps)
        self.scale_min = float(scale_min)
        self.scale_max = float(scale_max)
        self.shift_max_abs = float(shift_max_abs)

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        if mask.dtype != torch.bool:
            mask = mask > 0.5
        if mask.float().sum() < 1.0:
            return pred.new_zeros(())
        B = pred.shape[0]
        ar = self.align_resolution
        pred_lr, mask_lr = _masked_avg_pool(pred, mask, (ar, ar))
        target_lr, _ = _masked_avg_pool(target, mask, (ar, ar))
        pred_flat = pred_lr.flatten(1)                                       # (B, ar*ar)
        target_flat = target_lr.flatten(1)
        mask_lr_f = mask_lr.float().flatten(1)
        # weight = mask · 1/gt → loss/weight equivalent to MoGe's relative
        # depth-error weighting; clamp gt to eps to avoid divide-by-zero in
        # cells with extreme close-range averages.
        weight = mask_lr_f / target_flat.clamp_min(self.eps)
        scale, shift = _align_depth_affine(
            pred_flat, target_flat, weight, trunc=self.trunc,
        )
        # Reject ill-posed alignments: extreme scale/shift come from
        # near-uniform inputs where ROE's denominator floors at 1e-7.
        scale_ok = (scale >= self.scale_min) & (scale <= self.scale_max)
        shift_ok = shift.abs() <= self.shift_max_abs
        valid = scale_ok & shift_ok
        scale = torch.where(valid, scale, torch.ones_like(scale))
        shift = torch.where(valid, shift, torch.zeros_like(shift))
        pred_aligned = scale[:, None, None, None] * pred + shift[:, None, None, None]
        diff = (pred_aligned - target).abs() * mask.float()
        n = mask.float().sum().clamp_min(1.0)
        return diff.sum() / n


class AffineInvariantBlockGridL1Loss(nn.Module):
    """K×K block-grid affine-invariant L1 (deterministic local variant).

    MoGe-2's `affine_invariant_local_loss` samples random patches with
    density-aware sampling; we use a fixed K×K block partition for
    deterministic training-time behavior and easier comparison with our
    `depth_refined_grid` inference. Each block gets its own ROE-fit
    (scale, shift), then per-block masked L1.

    Memory bound: ROE is solved on a mask-aware downsample of each block
    to `align_resolution × align_resolution` cells (default 16×16 = 256
    anchors per block). This keeps the (anchors_total, N) anchor-expansion
    inside `align_depth_affine` bounded by (B·K²·ar²) ≈ 200 MB even at
    B=12 / 900×1600. The recovered (scale, shift) are then applied to the
    FULL-RES block pred, and the L1 is computed on the full mask.

    Blocks with fewer than `min_inliers` valid pixels (counted on the
    full-res mask) are skipped (the block contributes zero to the loss,
    denominator excludes them).
    """

    def __init__(
        self,
        K: int = 8,
        trunc: float = 1.0,
        min_inliers: int = 30,
        eps: float = 1e-3,
        align_resolution: int = 16,
        # Reject degenerate ROE solutions. ROE's closed-form scale can
        # explode when a block has near-uniform pred/target (denominator
        # `src_2 - src_1 → 0` falls back to 1e-7, blowing scale up). These
        # bounds catch the bad blocks AND keep gradients clean.
        scale_min: float = 0.1,
        scale_max: float = 10.0,
        shift_max_abs: float = 100.0,
    ) -> None:
        super().__init__()
        self.K = int(K)
        self.trunc = float(trunc)
        self.min_inliers = int(min_inliers)
        self.eps = float(eps)
        self.align_resolution = int(align_resolution)
        self.scale_min = float(scale_min)
        self.scale_max = float(scale_max)
        self.shift_max_abs = float(shift_max_abs)

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        if mask.dtype != torch.bool:
            mask = mask > 0.5
        if mask.float().sum() < 1.0:
            return pred.new_zeros(())
        B, _, H, W = pred.shape
        K = self.K
        bh, bw = H // K, W // K
        H_c, W_c = bh * K, bw * K
        pred_c = pred[..., :H_c, :W_c]
        target_c = target[..., :H_c, :W_c]
        mask_c = mask[..., :H_c, :W_c]
        # (B, 1, H_c, W_c) → (B*K*K, 1, bh, bw) blockwise view, contiguous.
        def _blockify(t):
            t = t.view(B, 1, K, bh, K, bw)
            t = t.permute(0, 2, 4, 1, 3, 5).contiguous()  # (B, K, K, 1, bh, bw)
            return t.view(B * K * K, 1, bh, bw)
        pred_b = _blockify(pred_c)                          # (B*K*K, 1, bh, bw)
        target_b = _blockify(target_c)
        mask_b = _blockify(mask_c)
        # Per-block inlier count (use ORIGINAL full-res mask).
        n_per_block = mask_b.view(B * K * K, -1).sum(dim=-1)  # (B*K*K,)

        # Mask-aware downsample per block to ar×ar for the ROE solve.
        ar = self.align_resolution
        pred_lr, mask_lr = _masked_avg_pool(pred_b, mask_b, (ar, ar))
        target_lr, _ = _masked_avg_pool(target_b, mask_b, (ar, ar))
        N_lr = ar * ar
        pred_lr_flat = pred_lr.view(B * K * K, N_lr)
        target_lr_flat = target_lr.view(B * K * K, N_lr)
        mask_lr_f = mask_lr.float().view(B * K * K, N_lr)
        weight = mask_lr_f / target_lr_flat.clamp_min(self.eps)

        # Batched ROE — (B·K², ar²) anchors. ~200 MB peak at 12·64·256.
        scale, shift = _align_depth_affine(
            pred_lr_flat, target_lr_flat, weight, trunc=self.trunc,
        )

        # Reject blocks with too few inliers OR with degenerate (scale,
        # shift) — extreme values come from near-uniform pred/target where
        # ROE's denominator drops to its 1e-7 floor and explodes the affine.
        scale_ok = (scale >= self.scale_min) & (scale <= self.scale_max)
        shift_ok = shift.abs() <= self.shift_max_abs
        block_ok = (n_per_block >= self.min_inliers) & scale_ok & shift_ok
        scale = torch.where(block_ok, scale, torch.ones_like(scale))
        shift = torch.where(block_ok, shift, torch.zeros_like(shift))

        # Apply per-block (scale, shift) to FULL-RES block pred.
        pred_b_aligned = scale[:, None, None, None] * pred_b + shift[:, None, None, None]
        # L1 on full-res mask, zeroing blocks that fell back to identity.
        block_keep = block_ok[:, None, None, None].float()
        diff = (pred_b_aligned - target_b).abs() * mask_b.float() * block_keep
        n = (mask_b.float() * block_keep).sum().clamp_min(1.0)
        return diff.sum() / n


class LidarAnchoredGlobalShapeLoss(nn.Module):
    """Per-image (scale, shift) is solved from LiDAR (not dense GT).

    Differences from `AffineInvariantL1Loss`:
      • ROE is solved on `(pred, lidar)` over `mask_lidar` pixels
        (the LiDAR-anchored variant: affine is locked to LiDAR's metric).
      • The resulting affine is applied to pred, then the L1 residual is
        computed against `dense_gt` over `mask_dense` (the shape signal).

    Conceptually:
        aligned_pred = a · pred + b   where (a, b) ≈ best fit pred→lidar
        loss         = mean_{mask_dense} |aligned_pred − dense_gt|

    Returns (loss, scale, shift). The caller (factory) can apply an
    auxiliary `(scale-1)² + shift²` regularizer to pull the affine
    toward identity — i.e. encourage the model to be metric-consistent
    with dense_gt as well, not only LiDAR.
    """

    def __init__(
        self,
        align_resolution: int = 32,
        trunc: float = 1.0,
        eps: float = 1e-3,
        scale_min: float = 0.1,
        scale_max: float = 10.0,
        shift_max_abs: float = 100.0,
        min_lidar_inliers: int = 30,
    ) -> None:
        super().__init__()
        self.align_resolution = int(align_resolution)
        self.trunc = float(trunc)
        self.eps = float(eps)
        self.scale_min = float(scale_min)
        self.scale_max = float(scale_max)
        self.shift_max_abs = float(shift_max_abs)
        self.min_lidar_inliers = int(min_lidar_inliers)

    def forward(
        self,
        pred: torch.Tensor,
        dense_gt: torch.Tensor,
        lidar: torch.Tensor,
        mask_dense: torch.Tensor,
        mask_lidar: torch.Tensor,
    ):
        if mask_dense.dtype != torch.bool:
            mask_dense = mask_dense > 0.5
        if mask_lidar.dtype != torch.bool:
            mask_lidar = mask_lidar > 0.5
        B = pred.shape[0]
        zero_loss = pred.new_zeros(())
        identity_scale = pred.new_ones(B)
        identity_shift = pred.new_zeros(B)
        if mask_lidar.float().sum() < self.min_lidar_inliers \
                or mask_dense.float().sum() < 1.0:
            return zero_loss, identity_scale, identity_shift

        ar = self.align_resolution
        # ROE on (pred, lidar) over LiDAR pixels — the anchor.
        pred_lr, mask_lr = _masked_avg_pool(pred, mask_lidar, (ar, ar))
        lidar_lr, _ = _masked_avg_pool(lidar, mask_lidar, (ar, ar))
        pred_flat = pred_lr.flatten(1)
        lidar_flat = lidar_lr.flatten(1)
        mask_lr_f = mask_lr.float().flatten(1)
        weight = mask_lr_f / lidar_flat.clamp_min(self.eps)
        scale, shift = _align_depth_affine(
            pred_flat, lidar_flat, weight, trunc=self.trunc,
        )
        scale_ok = (scale >= self.scale_min) & (scale <= self.scale_max)
        shift_ok = shift.abs() <= self.shift_max_abs
        valid = scale_ok & shift_ok
        scale = torch.where(valid, scale, torch.ones_like(scale))
        shift = torch.where(valid, shift, torch.zeros_like(shift))

        pred_aligned = scale[:, None, None, None] * pred + shift[:, None, None, None]
        # L1 residual is on the DENSE mask (shape supervision).
        diff = (pred_aligned - dense_gt).abs() * mask_dense.float()
        n = mask_dense.float().sum().clamp_min(1.0)
        loss = diff.sum() / n
        return loss, scale, shift


class LidarAnchoredBlockGridShapeLoss(nn.Module):
    """K×K block-grid version of `LidarAnchoredGlobalShapeLoss`.

    Each block solves its own (scale, shift) from the LiDAR pixels inside
    that block, then the L1 residual is taken against `dense_gt` on the
    dense mask within that block. Blocks with fewer than
    `min_lidar_inliers` LiDAR pixels fall back to identity (no penalty,
    no anchor contribution).

    Returns (loss, scale, shift, block_keep) where (scale, shift) have
    shape (B·K·K,) and block_keep is a bool tensor of the same shape
    marking which block-level affines were actually fit. The caller
    uses block_keep to compute the affine-anchor penalty only on fit
    blocks.
    """

    def __init__(
        self,
        K: int = 8,
        trunc: float = 1.0,
        min_lidar_inliers: int = 8,
        min_dense_inliers: int = 30,
        eps: float = 1e-3,
        align_resolution: int = 16,
        scale_min: float = 0.1,
        scale_max: float = 10.0,
        shift_max_abs: float = 100.0,
    ) -> None:
        super().__init__()
        self.K = int(K)
        self.trunc = float(trunc)
        self.min_lidar_inliers = int(min_lidar_inliers)
        self.min_dense_inliers = int(min_dense_inliers)
        self.eps = float(eps)
        self.align_resolution = int(align_resolution)
        self.scale_min = float(scale_min)
        self.scale_max = float(scale_max)
        self.shift_max_abs = float(shift_max_abs)

    def forward(
        self,
        pred: torch.Tensor,
        dense_gt: torch.Tensor,
        lidar: torch.Tensor,
        mask_dense: torch.Tensor,
        mask_lidar: torch.Tensor,
    ):
        if mask_dense.dtype != torch.bool:
            mask_dense = mask_dense > 0.5
        if mask_lidar.dtype != torch.bool:
            mask_lidar = mask_lidar > 0.5
        B, _, H, W = pred.shape
        K = self.K
        bh, bw = H // K, W // K
        H_c, W_c = bh * K, bw * K
        pred_c = pred[..., :H_c, :W_c]
        dense_c = dense_gt[..., :H_c, :W_c]
        lidar_c = lidar[..., :H_c, :W_c]
        mask_d_c = mask_dense[..., :H_c, :W_c]
        mask_l_c = mask_lidar[..., :H_c, :W_c]

        def _blockify(t):
            t = t.view(B, 1, K, bh, K, bw)
            t = t.permute(0, 2, 4, 1, 3, 5).contiguous()  # (B, K, K, 1, bh, bw)
            return t.view(B * K * K, 1, bh, bw)

        pred_b = _blockify(pred_c)
        dense_b = _blockify(dense_c)
        lidar_b = _blockify(lidar_c)
        mask_d_b = _blockify(mask_d_c)
        mask_l_b = _blockify(mask_l_c)

        n_lidar_per_block = mask_l_b.view(B * K * K, -1).sum(dim=-1)
        n_dense_per_block = mask_d_b.view(B * K * K, -1).sum(dim=-1)

        # Mask-aware downsample using LIDAR mask — that's what the
        # ROE solve will weigh by.
        ar = self.align_resolution
        pred_lr, mask_lr = _masked_avg_pool(pred_b, mask_l_b, (ar, ar))
        lidar_lr, _ = _masked_avg_pool(lidar_b, mask_l_b, (ar, ar))
        N_lr = ar * ar
        pred_lr_flat = pred_lr.view(B * K * K, N_lr)
        lidar_lr_flat = lidar_lr.view(B * K * K, N_lr)
        mask_lr_f = mask_lr.float().view(B * K * K, N_lr)
        weight = mask_lr_f / lidar_lr_flat.clamp_min(self.eps)

        scale, shift = _align_depth_affine(
            pred_lr_flat, lidar_lr_flat, weight, trunc=self.trunc,
        )

        scale_ok = (scale >= self.scale_min) & (scale <= self.scale_max)
        shift_ok = shift.abs() <= self.shift_max_abs
        block_ok = (n_lidar_per_block >= self.min_lidar_inliers) \
            & (n_dense_per_block >= self.min_dense_inliers) \
            & scale_ok & shift_ok
        scale = torch.where(block_ok, scale, torch.ones_like(scale))
        shift = torch.where(block_ok, shift, torch.zeros_like(shift))

        pred_b_aligned = scale[:, None, None, None] * pred_b + shift[:, None, None, None]
        block_keep = block_ok[:, None, None, None].float()
        # L1 residual against DENSE GT (shape supervision).
        diff = (pred_b_aligned - dense_b).abs() * mask_d_b.float() * block_keep
        n = (mask_d_b.float() * block_keep).sum().clamp_min(1.0)
        loss = diff.sum() / n
        return loss, scale, shift, block_ok


class AntiStripeLoss(nn.Module):
    """RGB-guided vertical-only smoothness targeting LiDAR-row stripes.

    Stripes are unphysical vertical (∂y) depth jumps where the underlying
    RGB has no matching edge. The original conflict happens at LiDAR rows
    (L1_lidar vs shape loss), but the *definition* of a stripe — "depth
    ∂y large where RGB ∂y is small" — is general, so we penalise it
    everywhere by default. The RGB-guided weight `exp(-γ·|∂y rgb|)`
    automatically vanishes at real edges, so real depth discontinuities
    are protected.

    Loss:
        w_y(p) = exp(-γ · |∂y rgb_gray(p)|)
        L = mean_{p ∈ R} ( |∂y pred(p)| · w_y(p) )

    R defaults to the WHOLE image. This is a self-consistency loss
    on (pred, rgb) gradients — it never inspects depth GT values, so
    `valid_mask_dense` is *not* applied: invalid regions (sky, ego,
    crop padding) still have meaningful RGB structure, and a stripe
    there is just as unphysical as anywhere else. In particular, LiDAR
    beams can reach the sky and create stripes in dense-invalid rows
    that a valid_mask would silently protect.

    Optional restrictions:
      • near_lidar_only=True → only pixels within `row_radius` vertical
                      dilation of any LiDAR pixel (legacy behaviour).
      • valid_mask          → AND-ed in if explicitly passed by the caller.
    """

    def __init__(self, gamma: float = 10.0, row_radius: int = 4,
                 near_lidar_only: bool = False,
                 normalize_by_mean: bool = False) -> None:
        super().__init__()
        self.gamma = float(gamma)
        self.row_radius = int(row_radius)
        self.near_lidar_only = bool(near_lidar_only)
        self.normalize_by_mean = bool(normalize_by_mean)

    def forward(self, pred: torch.Tensor, rgb: torch.Tensor,
                lidar_mask: Optional[torch.Tensor] = None,
                valid_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.normalize_by_mean:
            mean_d = pred.mean(dim=(2, 3), keepdim=True).clamp_min(1e-6)
            p = pred / mean_d
        else:
            p = pred

        dy_pred = (p[..., 1:, :] - p[..., :-1, :]).abs()
        rgb_gray = rgb.mean(1, keepdim=True)
        dy_rgb = (rgb_gray[..., 1:, :] - rgb_gray[..., :-1, :]).abs()
        w_y = torch.exp(-self.gamma * dy_rgb)
        term = dy_pred * w_y

        # Build the per-gradient-pixel mask by AND-ing every optional source.
        mg = None
        if valid_mask is not None:
            vm = valid_mask.float()
            v_pair = vm[..., 1:, :] * vm[..., :-1, :]   # AND of the two rows
            mg = v_pair if mg is None else mg * v_pair
        if self.near_lidar_only and lidar_mask is not None:
            k = 2 * self.row_radius + 1
            dilated = F.max_pool2d(lidar_mask.float(),
                                   kernel_size=(k, 1), stride=1,
                                   padding=(self.row_radius, 0))
            l_pair = torch.maximum(dilated[..., 1:, :], dilated[..., :-1, :])
            mg = l_pair if mg is None else mg * l_pair

        if mg is None:
            return term.mean()
        n = mg.sum().clamp_min(1.0)
        return (term * mg).sum() / n


class LidarAlignedBlockGradShapeLoss(nn.Module):
    """Block-local LiDAR-aligned dense_gt → multi-scale log-grad L1 vs pred.

    Motivation:
        Pure GradientShapeLoss matches ∂log pred to ∂log dense_gt, but
        depth_refined_grid's absolute ratios are noisy (DAv2 + global
        ROE only roughly metric-aligned). This loss FIRST aligns
        dense_gt to LiDAR's metric per K×K block via a robust ROE solve,
        then computes log-grad L1 against the *aligned* target. Each
        block carries its own (s, b) that absorbs DAv2's block-local
        ratio error; the resulting gradient field is LiDAR-consistent
        within each block, while block-to-block scale freedom remains.

    Algorithm:
        1. Partition image into K×K blocks (block size ~H/K × W/K).
        2. For each block, ROE-solve (dense_gt → lidar) on its LiDAR
           pixels: (s_b, b_b) = argmin Σ trunc(...), exactly as
           AffineInvariantL1Loss does — but the affine is fit on
           (dense_gt, lidar) instead of (pred, target).
        3. dense_gt_aligned[block] = s_b · dense_gt[block] + b_b. Blocks
           lacking LiDAR or producing out-of-bound (s, b) fall back to
           identity (no correction) — they still contribute supervision
           via the original dense_gt, just unaligned.
        4. Compute the standard multi-scale log-grad L1 between pred
           and dense_gt_aligned (delegates to GradientShapeLoss).

    Returns (loss, scale, shift, block_keep) like the sibling
    LidarAnchored class so the caller can optionally hook them for
    diagnostics / affine-anchor penalty.
    """

    def __init__(
        self,
        K: int = 8,
        scales: int = 4,
        use_log: bool = True,
        eps: float = 1e-3,
        trunc: Optional[float] = None,
        # ROE solver settings (mirror LidarAnchoredBlockGridShapeLoss)
        roe_trunc: float = 1.0,
        min_lidar_inliers: int = 8,
        align_resolution: int = 16,
        scale_min: float = 0.1,
        scale_max: float = 10.0,
        shift_max_abs: float = 100.0,
    ) -> None:
        super().__init__()
        self.K = int(K)
        self.eps = float(eps)
        self.roe_trunc = float(roe_trunc)
        self.min_lidar_inliers = int(min_lidar_inliers)
        self.align_resolution = int(align_resolution)
        self.scale_min = float(scale_min)
        self.scale_max = float(scale_max)
        self.shift_max_abs = float(shift_max_abs)
        # Multi-scale log-grad L1 against the aligned dense_gt.
        self.grad_shape = GradientShapeLoss(
            scales=scales, use_log=use_log, eps=eps, trunc=trunc,
        )

    def _block_align_dense(self, dense_gt, lidar, mask_lidar):
        """Per-block ROE solve (dense_gt → lidar), apply to dense_gt."""
        B, _, H, W = dense_gt.shape
        K = self.K
        bh, bw = H // K, W // K
        H_c, W_c = bh * K, bw * K
        dense_c = dense_gt[..., :H_c, :W_c]
        lidar_c = lidar[..., :H_c, :W_c]
        mask_l_c = mask_lidar[..., :H_c, :W_c]

        def _blockify(t):
            t = t.view(B, 1, K, bh, K, bw)
            t = t.permute(0, 2, 4, 1, 3, 5).contiguous()
            return t.view(B * K * K, 1, bh, bw)

        def _unblockify(t):
            t = t.view(B, K, K, 1, bh, bw)
            t = t.permute(0, 3, 1, 4, 2, 5).contiguous()
            return t.view(B, 1, H_c, W_c)

        dense_b = _blockify(dense_c)
        lidar_b = _blockify(lidar_c)
        mask_l_b = _blockify(mask_l_c)
        n_lidar_per_block = mask_l_b.view(B * K * K, -1).sum(dim=-1)

        ar = self.align_resolution
        dense_lr, mask_lr = _masked_avg_pool(dense_b, mask_l_b, (ar, ar))
        lidar_lr, _ = _masked_avg_pool(lidar_b, mask_l_b, (ar, ar))
        N_lr = ar * ar
        dense_lr_flat = dense_lr.view(B * K * K, N_lr)
        lidar_lr_flat = lidar_lr.view(B * K * K, N_lr)
        mask_lr_f = mask_lr.float().view(B * K * K, N_lr)
        weight = mask_lr_f / lidar_lr_flat.clamp_min(self.eps)

        # ROE solver requires at least one non-zero weight per row.
        # Filter on both inlier count AND post-downsample weight sum;
        # blocks failing either are kept at identity (no correction).
        has_lidar = n_lidar_per_block >= self.min_lidar_inliers
        weight_sum = weight.sum(dim=-1)
        has_weight = weight_sum > 1e-9
        proceed = has_lidar & has_weight
        scale = torch.ones(B * K * K, device=dense_gt.device, dtype=dense_b.dtype)
        shift = torch.zeros(B * K * K, device=dense_gt.device, dtype=dense_b.dtype)
        if proceed.any():
            idx = torch.where(proceed)[0]
            s_v, b_v = _align_depth_affine(
                dense_lr_flat[idx], lidar_lr_flat[idx], weight[idx],
                trunc=self.roe_trunc,
            )
            scale[idx] = s_v
            shift[idx] = b_v
        scale_ok = (scale >= self.scale_min) & (scale <= self.scale_max)
        shift_ok = shift.abs() <= self.shift_max_abs
        block_ok = has_lidar & scale_ok & shift_ok
        # Identity for invalid blocks → dense_gt left unchanged there.
        scale = torch.where(block_ok, scale, torch.ones_like(scale))
        shift = torch.where(block_ok, shift, torch.zeros_like(shift))

        dense_b_aligned = scale[:, None, None, None] * dense_b + shift[:, None, None, None]
        dense_aligned_inner = _unblockify(dense_b_aligned)

        # Pad-back if H_c != H (right/bottom margins): keep original dense_gt.
        if H_c != H or W_c != W:
            out = dense_gt.clone()
            out[..., :H_c, :W_c] = dense_aligned_inner
        else:
            out = dense_aligned_inner
        return out, scale, shift, block_ok

    def forward(
        self,
        pred: torch.Tensor,
        dense_gt: torch.Tensor,
        lidar: torch.Tensor,
        mask_lidar: torch.Tensor,
    ):
        if mask_lidar.dtype != torch.bool:
            mask_lidar = mask_lidar > 0.5
        # 1) per-block alignment of dense_gt to LiDAR
        dense_aligned, scale, shift, block_ok = self._block_align_dense(
            dense_gt, lidar, mask_lidar
        )
        # 2) supervise pred's log-gradient field against the aligned target.
        # detach the alignment so the gradient flows ONLY through pred
        # (matching the spirit of the anchor: dense_gt_aligned acts as
        #  a fixed target derived from LiDAR + dense_gt).
        loss = self.grad_shape(pred, dense_aligned.detach())
        return loss, scale, shift, block_ok


class GradientShapeLoss(nn.Module):
    """Multi-scale L1 on *log-depth* gradients vs dense GT.

    Supervises only the *shape* (relative depth gradients) of the dense
    GT, NOT its absolute values — log differentiation makes the loss
    completely invariant to per-image metric scale:

        ∂_x log(c · D) = ∂_x log(D),  ∂_y log(c · D) = ∂_y log(D)

    So the L1_lidar term can freely set the absolute metric scale
    without fighting this loss — exactly the conflict that drives
    stripes when `AffineInvariantL1Loss` or raw L1 is used on dense_gt.

    Multi-scale (MiDaS / Singh) — gradients at 1/1, 1/2, 1/4, 1/8 to
    capture both fine edges and coarse structure.

    By default the loss runs on the WHOLE image with no valid-mask. On
    nuScenes (depth_refined_grid) the dense_invalid pixels are filled
    with max_depth (100 m), so log(target) is bounded and the
    invalid→valid boundary contributes a small, non-pathological signal
    (measured |∂y log_pred − ∂y log_target| ≈ 0.02 vs ≈ 0.15 in valid
    regions). Including them lets the model learn the natural prior
    that sky / ego / far-field is locally flat. Pass `mask` explicitly
    to restrict the loss if your dense_gt fill convention differs.
    """

    def __init__(self, scales: int = 4, use_log: bool = True,
                 eps: float = 1e-3,
                 per_scale_weights: Optional[Sequence[float]] = None,
                 trunc: Optional[float] = None) -> None:
        super().__init__()
        self.scales = int(scales)
        self.use_log = bool(use_log)
        self.eps = float(eps)
        self.per_scale_weights = _normalize_scale_weights(per_scale_weights, self.scales)
        # Soft truncation (MoGe ROE-style robust loss): per-pixel residuals
        # |∂log pred − ∂log target| are clamped to `trunc` BEFORE averaging.
        # This caps the influence of pixels where dense_gt's ratio is far
        # from pred's — typically the regions where DAv2-derived dense_gt
        # is least reliable (mid-range 20–50 m on nuScenes). None disables.
        self.trunc = float(trunc) if trunc is not None else None

    @staticmethod
    def _grad(x: torch.Tensor):
        gx = x[..., :, 1:] - x[..., :, :-1]
        gy = x[..., 1:, :] - x[..., :-1, :]
        return gx, gy

    def forward(self, pred: torch.Tensor, target: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.use_log:
            p = torch.log(pred.clamp_min(self.eps))
            t = torch.log(target.clamp_min(self.eps))
        else:
            p, t = pred, target
        m = mask.float() if mask is not None else None
        loss = pred.new_zeros(())
        for s in range(self.scales):
            if s > 0:
                p = F.avg_pool2d(p, 2)
                t = F.avg_pool2d(t, 2)
                if m is not None:
                    m = F.avg_pool2d(m, 2)
            gpx, gpy = self._grad(p)
            gtx, gty = self._grad(t)
            res_x = (gpx - gtx).abs()
            res_y = (gpy - gty).abs()
            if self.trunc is not None:
                # Caps gradient flow per pixel to `trunc` — outlier pixels
                # contribute exactly `trunc` (not their actual residual).
                res_x = res_x.clamp_max(self.trunc)
                res_y = res_y.clamp_max(self.trunc)
            if m is None:
                scale_loss = res_x.mean() + res_y.mean()
            else:
                mx = m[..., :, 1:] * m[..., :, :-1]
                my = m[..., 1:, :] * m[..., :-1, :]
                n_x = mx.sum().clamp_min(1.0)
                n_y = my.sum().clamp_min(1.0)
                scale_loss = (res_x * mx).sum() / n_x \
                           + (res_y * my).sum() / n_y
            loss = loss + self.per_scale_weights[s] * scale_loss
        return loss


class GradientCosineLoss(nn.Module):
    """Multi-scale cosine-similarity loss on gradient VECTORS.

    A natural-magnitude alternative to GradientSignLoss: enforces same
    gradient VECTOR DIRECTION between pred and target instead of just
    per-axis sign agreement. Captures the joint (∂x, ∂y) angle, so
    (1, 5) and (5, 1) — both all-positive but different angles — are
    distinguishable. Magnitude of pred is FREE → no fight with DAv2's
    (possibly inaccurate) absolute gradient ratios.

    Loss per scale (vectors cropped to common (H-1, W-1)):
        cos = (g_pred · g_target) / (|g_pred| · |g_target|)
        L = mean( |g_target| · (1 - cos) )

    Properties:
      • 1 - cos ∈ [0, 2]: natural raw magnitude → no need for weights
        in the hundreds/thousands range like with the hinge sign loss.
      • |g_target| weight: flat regions (|g_t| ≈ 0, target direction
        meaningless) contribute ~0 — same self-suppression as sign loss.
      • Pred gradient magnitude unconstrained → only direction matters,
        which is precisely DAv2's reliable signal.
      • Log-domain → invariant to per-image metric scale → no fight
        with L1_lidar over absolute depth.
    """

    def __init__(self, scales: int = 4, use_log: bool = True,
                 eps: float = 1e-3,
                 per_scale_weights: Optional[Sequence[float]] = None) -> None:
        super().__init__()
        self.scales = int(scales)
        self.use_log = bool(use_log)
        self.eps = float(eps)
        self.per_scale_weights = _normalize_scale_weights(per_scale_weights, self.scales)

    @staticmethod
    def _grad_vectors(x: torch.Tensor):
        """Return (gx, gy) both cropped to common (H-1, W-1) shape so
        they form per-pixel 2D gradient vectors."""
        gx = x[..., :, 1:] - x[..., :, :-1]        # (B, 1, H,   W-1)
        gy = x[..., 1:, :] - x[..., :-1, :]        # (B, 1, H-1, W)
        gx_c = gx[..., :-1, :]                     # (B, 1, H-1, W-1)
        gy_c = gy[..., :, :-1]                     # (B, 1, H-1, W-1)
        return gx_c, gy_c

    def forward(self, pred: torch.Tensor, target: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.use_log:
            p = torch.log(pred.clamp_min(self.eps))
            t = torch.log(target.clamp_min(self.eps))
        else:
            p, t = pred, target
        m = mask.float() if mask is not None else None
        loss = pred.new_zeros(())
        for s in range(self.scales):
            if s > 0:
                p = F.avg_pool2d(p, 2)
                t = F.avg_pool2d(t, 2)
                if m is not None:
                    m = F.avg_pool2d(m, 2)
            gpx, gpy = self._grad_vectors(p)
            gtx, gty = self._grad_vectors(t)
            dot = gpx * gtx + gpy * gty
            norm_p = (gpx * gpx + gpy * gpy + self.eps).sqrt()
            norm_t = (gtx * gtx + gty * gty + self.eps).sqrt()
            cos = dot / (norm_p * norm_t)
            penalty = (1.0 - cos) * norm_t       # |g_target|-weighted
            if m is None:
                scale_loss = penalty.mean()
            else:
                # AND of all four corners of the 2x2 cell defining the vector pixel
                mc = m[..., :-1, :-1] * m[..., :-1, 1:] \
                   * m[..., 1:,  :-1] * m[..., 1:,  1:]
                n = mc.sum().clamp_min(1.0)
                scale_loss = (penalty * mc).sum() / n
            loss = loss + self.per_scale_weights[s] * scale_loss
        return loss


class GradientSignLoss(nn.Module):
    """Multi-scale gradient *direction* matching — sign-only hinge.

    Enforces same-direction gradients between pred and dense_gt (i.e.
    same neighbor ordering) WITHOUT pinning their magnitudes. Use this
    when dense_gt's ordering is reliable but its absolute depth ratios
    are not — e.g. DAv2-derived dense GT, where "A is farther than B"
    is trustworthy but "A is 2.3× farther than B" is not.

    Loss (per scale s, per axis):
        penalty = |∂target_s| · relu( -∂pred_s · sign(∂target_s) )

    Key properties:
      • sign(∂target) — direction we want pred to follow.
      • relu(−∂pred · sign(∂target)) — hinge: zero when pred has the same
        sign, |∂pred| when opposite. Pred is FREE to pick any positive
        magnitude as long as it points the same way.
      • |∂target| weight — pixels where target gradient is small (flat
        regions, unreliable ordering) contribute almost nothing; sharp
        edges (clear ordering) dominate. Self-down-weights target noise.
      • Multi-scale (1/1, 1/2, 1/4, 1/8) — fine scale carries neighbor
        ordering (preserves DAv2's edge / boundary detail), coarse scale
        carries large-region ordering.
      • Log-domain by default → metric-scale-invariant, never fights
        L1_lidar over absolute depth.
    """

    def __init__(self, scales: int = 4, use_log: bool = True,
                 eps: float = 1e-3,
                 per_scale_weights: Optional[Sequence[float]] = None) -> None:
        super().__init__()
        self.scales = int(scales)
        self.use_log = bool(use_log)
        self.eps = float(eps)
        self.per_scale_weights = _normalize_scale_weights(per_scale_weights, self.scales)

    @staticmethod
    def _grad(x: torch.Tensor):
        gx = x[..., :, 1:] - x[..., :, :-1]
        gy = x[..., 1:, :] - x[..., :-1, :]
        return gx, gy

    def forward(self, pred: torch.Tensor, target: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.use_log:
            p = torch.log(pred.clamp_min(self.eps))
            t = torch.log(target.clamp_min(self.eps))
        else:
            p, t = pred, target
        m = mask.float() if mask is not None else None
        loss = pred.new_zeros(())
        for s in range(self.scales):
            if s > 0:
                p = F.avg_pool2d(p, 2)
                t = F.avg_pool2d(t, 2)
                if m is not None:
                    m = F.avg_pool2d(m, 2)
            gpx, gpy = self._grad(p)
            gtx, gty = self._grad(t)
            # |gt|-weighted hinge on sign disagreement. sign(0) == 0,
            # so flat regions contribute nothing (no noise amplification).
            pen_x = gtx.abs() * F.relu(-gpx * gtx.sign())
            pen_y = gty.abs() * F.relu(-gpy * gty.sign())
            if m is None:
                scale_loss = pen_x.mean() + pen_y.mean()
            else:
                mx = m[..., :, 1:] * m[..., :, :-1]
                my = m[..., 1:, :] * m[..., :-1, :]
                n_x = mx.sum().clamp_min(1.0)
                n_y = my.sum().clamp_min(1.0)
                scale_loss = (pen_x * mx).sum() / n_x \
                           + (pen_y * my).sum() / n_y
            loss = loss + self.per_scale_weights[s] * scale_loss
        return loss


def affine_anchor_penalty(scale: torch.Tensor, shift: torch.Tensor,
                          block_keep: torch.Tensor = None) -> torch.Tensor:
    """(scale-1)² + shift² mean penalty, optionally restricted to
    blocks that were actually fit. Used to pull the affine toward
    identity (a=1, b=0) — encourages pred to be metric-consistent
    with dense_gt instead of relying solely on the affine to absorb
    any scale/shift difference between the two.
    """
    pen = (scale - 1.0).pow(2) + shift.pow(2)
    if block_keep is not None:
        keep = block_keep.float()
        n = keep.sum().clamp_min(1.0)
        return (pen * keep).sum() / n
    return pen.mean()


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
        w_lidar: float = 1.0,
    ) -> None:
        super().__init__()
        self.lam = lam
        self.w_smooth = w_smooth
        self.w_lidar = w_lidar
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
        # `w_lidar=0` skips the sparse LiDAR term entirely — used when the
        # dense GT is itself ROE-aligned to LiDAR (e.g. depth_refined_grid)
        # so the sparse-vs-dense decomposition collapses.
        if self.w_lidar > 0:
            l_lidar = self.l1(pred, depth_gt_lidar, valid_mask_lidar)
        else:
            l_lidar = pred.new_zeros(())
        l_dense = self.l1(pred, depth_gt_dense, valid_mask_dense)
        if self.w_smooth > 0:
            if rgb_norm is None:
                raise ValueError("rgb_norm required when w_smooth > 0")
            l_smooth = self.smoothness(pred, rgb_norm)
        else:
            l_smooth = pred.new_zeros(())
        total = self.w_lidar * l_lidar + self.lam * l_dense + self.w_smooth * l_smooth
        return {
            "loss_total": total,
            "loss_lidar": l_lidar.detach(),
            "loss_dense": l_dense.detach(),
            "loss_smooth": l_smooth.detach(),
        }
