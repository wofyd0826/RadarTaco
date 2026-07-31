"""Poisson reconstruction prototype — fuse dense_gt + LiDAR into a new GT.

Solves
    Δ(dense_new) = Δ(dense_gt)               on non-LiDAR pixels (Neumann boundary)
    dense_new(p) = LiDAR(p)                  on LiDAR pixels    (Dirichlet)

so dense_new follows the LOCAL SHAPE (Laplacian) of dense_gt while passing
EXACTLY through the LiDAR samples — no Voronoi seams, no stripes.

Outputs a 2×3 panel per sample:
    [ RGB         | dense_gt          | dense_new (Poisson)  ]
    [ LiDAR overlay | dense_new − dense_gt | residual at LiDAR ]
"""
import os
import sys
import time
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pyamg
import scipy.ndimage as ndi
import scipy.sparse as sp
import torch

sys.path.insert(0, os.path.abspath("."))
from hydra import compose, initialize_config_dir
from src.util.loaders import build_loaders


def build_laplacian_2d(H: int, W: int) -> sp.csr_matrix:
    """5-point stencil Laplacian on H×W grid with Neumann boundaries.

    L = I_y ⊗ Dx + Dy ⊗ I_x  (Kronecker decomposition)
    Dx, Dy are 1-D Laplacians with -1 (instead of -2) on the boundary
    diagonal entries → encodes zero-gradient (Neumann) at the image edge.
    """
    dx_main = -2.0 * np.ones(W); dx_main[0] = -1.0; dx_main[-1] = -1.0
    Dx = sp.diags([dx_main, np.ones(W - 1), np.ones(W - 1)], [0, -1, 1])
    dy_main = -2.0 * np.ones(H); dy_main[0] = -1.0; dy_main[-1] = -1.0
    Dy = sp.diags([dy_main, np.ones(H - 1), np.ones(H - 1)], [0, -1, 1])
    L = sp.kron(sp.eye(H), Dx) + sp.kron(Dy, sp.eye(W))
    return L.tocsr()


def _poisson_hard(dense_gt: np.ndarray, lidar: np.ndarray,
                  mask_lidar: np.ndarray, L: sp.csr_matrix) -> np.ndarray:
    """Hard Dirichlet — LiDAR pixels strictly pinned to LiDAR value."""
    H, W = dense_gt.shape
    N = H * W
    mask_flat = mask_lidar.astype(bool).flatten()
    non_idx = np.where(~mask_flat)[0]
    lidar_idx = np.where(mask_flat)[0]

    L_nn = L[non_idx][:, non_idx]
    L_nl = L[non_idx][:, lidar_idx]
    dense_flat = dense_gt.astype(np.float64).flatten()
    lidar_flat = lidar.astype(np.float64).flatten()

    b_non = L.dot(dense_flat)[non_idx] - L_nl.dot(lidar_flat[lidar_idx])
    A = (-L_nn).tocsr()
    b = -b_non

    x0 = dense_flat[non_idx].copy()
    ml = pyamg.ruge_stuben_solver(A)
    residuals: list = []
    x_non = ml.solve(b, x0=x0, tol=1e-8, maxiter=200, residuals=residuals)
    print(f"  AMG (hard): {len(residuals)-1} V-cycles, "
          f"residual {residuals[0]:.3e} → {residuals[-1]:.3e}")

    x_full = np.empty(N, dtype=np.float64)
    x_full[non_idx] = x_non
    x_full[lidar_idx] = lidar_flat[lidar_idx]
    return x_full.reshape(H, W).astype(np.float32)


def _poisson_soft(dense_gt: np.ndarray, lidar: np.ndarray,
                  mask_lidar: np.ndarray, L: sp.csr_matrix,
                  lam: float) -> np.ndarray:
    """Soft Dirichlet — LiDAR pixels enter as a weighted L2 penalty
    rather than a hard constraint.

    Minimises:
        E(x) = ½ ‖∇x − ∇dense_gt‖² + (λ/2) Σ_{p∈LiDAR} (x_p − lidar_p)²

    Setting ∂E/∂x = 0 yields the symmetric, positive-definite system
        (−L + λ · diag(mask)) · x = −L · dense_gt + λ · mask · lidar

    Properties vs hard:
      • λ → ∞ : reduces to hard Dirichlet (LiDAR exact)
      • λ → 0 : LiDAR ignored — pure dense_gt smoothing
      • Intermediate λ : LiDAR pulls but its influence DECAYS over distance
        (no isolated spike — penalty is spread by the Laplacian).
    """
    H, W = dense_gt.shape
    mask_flat = mask_lidar.astype(np.float64).flatten()
    dense_flat = dense_gt.astype(np.float64).flatten()
    lidar_flat = lidar.astype(np.float64).flatten()

    A = (-L + lam * sp.diags(mask_flat)).tocsr()
    b = -L.dot(dense_flat) + lam * mask_flat * lidar_flat

    x0 = dense_flat.copy()
    x0[mask_flat > 0] = lidar_flat[mask_flat > 0]
    ml = pyamg.ruge_stuben_solver(A)
    residuals: list = []
    x = ml.solve(b, x0=x0, tol=1e-8, maxiter=200, residuals=residuals)
    print(f"  AMG (soft λ={lam:g}): {len(residuals)-1} V-cycles, "
          f"residual {residuals[0]:.3e} → {residuals[-1]:.3e}")
    return x.reshape(H, W).astype(np.float32)


def cross_bilateral_reconstruct(rgb_norm: np.ndarray,
                                lidar: np.ndarray,
                                mask_lidar: np.ndarray,
                                sigma_s: float = 10.0,
                                sigma_r: float = 0.10,
                                K: int = 20) -> np.ndarray:
    """RGB-guided cross-bilateral interpolation of sparse LiDAR.

    For each pixel p compute a weighted average of the K nearest LiDAR
    points q, with weights coming from two Gaussian kernels:
        w_spatial(p, q) = exp(-‖p − q‖² / 2σ_s²)            ← pixels close in space
        w_range(p, q)   = exp(-‖RGB(p) − RGB(q)‖² / 2σ_r²)  ← pixels of similar colour
        w(p, q) = w_spatial · w_range

        dense_new(p) = Σ w(p, q) · lidar(q) / Σ w(p, q)

    Properties (vs Poisson):
      • Spike-free by construction — output is a *weighted average* of
        nearby LiDAR values; can't concentrate a mismatch at a single
        pixel.
      • No DAv2 / dense_gt dependence at all — so DAv2's biased absolute
        ratios cannot leak into the GT.
      • RGB-guided edge preservation: pixels in the same coloured region
        share LiDAR values; at colour boundaries the kernel cuts off.
      • Truly local: a pixel far from any LiDAR row only blends LiDAR
        values inside its colour-similar neighbourhood.
    """
    from scipy.spatial import cKDTree

    H, W = lidar.shape
    # De-normalise ImageNet RGB to [0, 1] for stable σ_r interpretation.
    mean = np.array([0.485, 0.456, 0.406])[:, None, None]
    std  = np.array([0.229, 0.224, 0.225])[:, None, None]
    rgb_chw = rgb_norm if rgb_norm.shape[0] == 3 else rgb_norm.transpose(2, 0, 1)
    rgb_img = np.clip(rgb_chw * std + mean, 0.0, 1.0).transpose(1, 2, 0)  # (H,W,3)

    ys, xs = np.where(mask_lidar)
    if len(ys) == 0:
        return np.zeros_like(lidar)

    lidar_pts = np.stack([ys, xs], axis=1).astype(np.float64)
    tree = cKDTree(lidar_pts)

    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    all_pix = np.stack([yy.ravel(), xx.ravel()], axis=1).astype(np.float64)

    k = min(K, len(ys))
    dists, idx = tree.query(all_pix, k=k)
    if k == 1:
        dists = dists[:, None]
        idx = idx[:, None]

    ws_spatial = np.exp(-dists**2 / (2.0 * sigma_s**2))            # (HW, k)

    rgb_flat = rgb_img.reshape(H * W, 3)                            # (HW, 3)
    rgb_at_lidar = rgb_img[ys, xs]                                  # (N_lidar, 3)
    rgb_nb = rgb_at_lidar[idx]                                      # (HW, k, 3)
    dr2 = ((rgb_flat[:, None, :] - rgb_nb)**2).sum(axis=-1)         # (HW, k)
    ws_range = np.exp(-dr2 / (2.0 * sigma_r**2))

    w = ws_spatial * ws_range
    w_sum = w.sum(axis=-1)
    lidar_nb = lidar[ys[idx], xs[idx]]                              # (HW, k)
    val_sum = (w * lidar_nb).sum(axis=-1)

    dense_new = (val_sum / (w_sum + 1e-8)).reshape(H, W).astype(np.float32)
    print(f"  cross-bilateral: σ_s={sigma_s} σ_r={sigma_r} K={k}, "
          f"avg ‖w_sum‖ = {w_sum.mean():.3f}")
    return dense_new


def piecewise_polyfit_reconstruct(dense_gt: np.ndarray,
                                  lidar: np.ndarray,
                                  mask_lidar: np.ndarray,
                                  rgb_norm: np.ndarray,
                                  degree: int = 2,
                                  n_clusters: int = 16,
                                  min_segment_size: int = 200) -> np.ndarray:
    """Segmentation-based piecewise polynomial fit (POLAR's LiDAR
    adaptation, per-segment instead of per-image).

    Pipeline:
      1. K-means clustering in a 6-D feature space:
            (R, G, B, x/W·2, y/H·2, log(dense_gt)/5)
         — RGB defines colour-similar regions, xy weakly enforces spatial
         coherence, log(depth) separates near/far surfaces that share
         colour. Mixed RGB+spatial+depth produces semantically meaningful
         clusters that approximate object regions.
      2. Per-cluster connected-components → individual segments
         (split disjoint colour regions into separate fits).
      3. Each segment with ≥ (degree+1) LiDAR points → its own polynomial
         d ≈ Σ c_i z^i fitted by closed-form least squares.
      4. Small / LiDAR-less segments → fall back to a GLOBAL polynomial
         (fit from all valid LiDAR), preserving plausible metric scale.
      5. Apply each segment's polynomial to its dense_gt pixels.

    The resulting "blotches" (per-segment corrections) then align with
    object boundaries instead of with LiDAR scan-line geometry.
    """
    import cv2

    H, W = dense_gt.shape
    rgb = rgb_norm.transpose(1, 2, 0).astype(np.float32)         # (H, W, 3)
    xs, ys = np.meshgrid(np.arange(W), np.arange(H))
    log_d = np.log(np.clip(dense_gt, 1e-3, None)).astype(np.float32)
    features = np.stack([
        rgb[..., 0], rgb[..., 1], rgb[..., 2],
        (xs / W).astype(np.float32) * 2.0,
        (ys / H).astype(np.float32) * 2.0,
        log_d / 5.0,
    ], axis=-1).reshape(-1, 6)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.5)
    _, labels, _ = cv2.kmeans(features, n_clusters, None, criteria,
                              3, cv2.KMEANS_PP_CENTERS)
    labels = labels.reshape(H, W).astype(np.int32)

    # Split each colour cluster into spatially-connected components.
    segments = np.zeros((H, W), dtype=np.int32)
    next_id = 1
    for k in range(n_clusters):
        mask_k = (labels == k).astype(np.uint8)
        n_cc, cc = cv2.connectedComponents(mask_k, connectivity=8)
        for c in range(1, n_cc):
            ss = (cc == c)
            if ss.sum() >= min_segment_size:
                segments[ss] = next_id
                next_id += 1
    n_segments = next_id - 1

    # GLOBAL polynomial (fallback for small / LiDAR-less segments).
    z_all = dense_gt[mask_lidar].astype(np.float64)
    d_all = lidar[mask_lidar].astype(np.float64)
    V_all = np.vstack([z_all ** i for i in range(degree + 1)]).T
    global_coeffs, *_ = np.linalg.lstsq(V_all, d_all, rcond=None)

    dense_new = dense_gt.astype(np.float64).copy()
    n_local = 0
    n_global = 0
    for sid in range(1, n_segments + 1):
        seg_mask = (segments == sid)
        seg_lidar_mask = seg_mask & mask_lidar
        n_l = int(seg_lidar_mask.sum())
        if n_l >= degree + 1:
            z = dense_gt[seg_lidar_mask].astype(np.float64)
            d = lidar[seg_lidar_mask].astype(np.float64)
            V = np.vstack([z ** i for i in range(degree + 1)]).T
            coeffs, *_ = np.linalg.lstsq(V, d, rcond=None)
            n_local += 1
        else:
            coeffs = global_coeffs
            n_global += 1
        z_seg = dense_gt[seg_mask].astype(np.float64)
        z_powers = np.stack([z_seg ** i for i in range(degree + 1)], axis=-1)
        dense_new[seg_mask] = z_powers @ coeffs

    # Unsegmented pixels (small islands dropped by min_size) → GLOBAL fit.
    unseg = (segments == 0)
    n_unseg = int(unseg.sum())
    if n_unseg > 0:
        z_us = dense_gt[unseg].astype(np.float64)
        z_powers = np.stack([z_us ** i for i in range(degree + 1)], axis=-1)
        dense_new[unseg] = z_powers @ global_coeffs

    dense_new = np.clip(dense_new, 0.1, 120.0)
    err = np.abs(dense_new[mask_lidar] - lidar[mask_lidar]).mean()
    print(f"  piecewise-polyfit K={n_clusters} deg={degree} min={min_segment_size}: "
          f"{n_segments} segments ({n_local} local-fit, {n_global} global-fallback), "
          f"{n_unseg} unseg px, err@LiDAR={err:.3f} m")
    return dense_new.astype(np.float32)


def edge_aware_residual_reconstruct(dense_gt: np.ndarray,
                                    lidar: np.ndarray,
                                    mask_lidar: np.ndarray,
                                    rgb_norm: np.ndarray,
                                    sigma_s: float = 30.0,
                                    sigma_r: float = 0.15,
                                    K: int = 200,
                                    guide: str = "rgb") -> np.ndarray:
    """Edge-aware residual interp — joint-bilateral spread of LiDAR
    residuals, guided by similarity in a chosen *guide* channel.

    `guide` selects the edge channel:
      • "rgb"        — RGB difference (default, raw ImageNet-normalised)
                       σ_r in normalised-RGB units (typ. 0.10–0.30)
      • "depth"      — dense_gt depth difference (METERS)
                       σ_r typ. 1–5 m
      • "depth_log"  — log-depth (compresses ratio mismatches)
                       σ_r typ. 0.05–0.20 (≈ log ratio)

    Depth-based guidance keys spread to *geometric* boundaries (object
    contours visible in the dense_gt depth itself) — independent of
    texture / shadow / colour, which can be noisy edge cues. log-depth
    further makes the guide scale-invariant (far-away depth differences
    of the same *ratio* are treated like close-up ones).

    Weight per (query p, LiDAR neighbour q):
        w_spatial(p, q) = exp(-‖p - q‖² / 2σ_s²)
        w_range(p, q)   = exp(-‖G(p) - G(q)‖² / 2σ_r²)   (G = guide channel)
        w(p, q)         = w_spatial(p, q) · w_range(p, q)
    """
    from scipy.spatial import cKDTree

    H, W = dense_gt.shape
    ys, xs = np.where(mask_lidar)                    # (M,)
    if len(ys) == 0:
        return dense_gt.astype(np.float32)
    residuals = lidar[ys, xs] - dense_gt[ys, xs]     # (M,)

    # Build the guide channel — shape (C, H, W) flattened to (HW, C).
    if guide == "rgb":
        if rgb_norm is None:
            raise ValueError("guide='rgb' requires rgb_norm")
        guide_img = rgb_norm                          # (3, H, W)
    elif guide == "depth":
        guide_img = dense_gt[None, ...]               # (1, H, W) meters
    elif guide == "depth_log":
        guide_img = np.log(np.clip(dense_gt, 1e-3, None))[None, ...]
    else:
        raise ValueError(f"unknown guide '{guide}' (rgb / depth / depth_log)")

    C = guide_img.shape[0]
    guide_at_lidar = guide_img[:, ys, xs].T          # (M, C)
    coords_l = np.stack([ys, xs], axis=-1).astype(np.float32)
    tree = cKDTree(coords_l)

    k = int(min(K, len(ys)))
    yq, xq = np.indices((H, W))
    coords_q = np.stack([yq.ravel(), xq.ravel()], axis=-1).astype(np.float32)
    dists, idx = tree.query(coords_q, k=k, workers=-1)   # (HW, K)

    w_spatial = np.exp(-dists**2 / (2.0 * sigma_s**2))   # (HW, K)
    guide_q = guide_img.reshape(C, -1).T                  # (HW, C)
    guide_neighbors = guide_at_lidar[idx]                 # (HW, K, C)
    guide_diff2 = np.sum((guide_q[:, None, :] - guide_neighbors) ** 2, axis=-1)
    w_range = np.exp(-guide_diff2 / (2.0 * sigma_r**2))   # (HW, K)
    w = w_spatial * w_range                               # (HW, K)

    r_neighbors = residuals[idx]                          # (HW, K)
    w_sum = w.sum(axis=-1)
    r_blend = (w * r_neighbors).sum(axis=-1) / np.maximum(w_sum, 1e-8)
    r_blend = r_blend.reshape(H, W).astype(np.float64)

    dense_new = dense_gt.astype(np.float64) + r_blend
    n_clip_lo = int((dense_new < 0.0).sum())
    n_clip_hi = int((dense_new > 120.0).sum())
    dense_new = np.clip(dense_new, 0.1, 120.0)

    print(f"  edge-aware residual K={k} σ_s={sigma_s} σ_r={sigma_r} "
          f"guide={guide}: r̃ abs.mean={np.abs(r_blend).mean():.3f}, "
          f"max|r̃|={np.abs(r_blend).max():.2f}, "
          f"clipped (<0 / >120) = {n_clip_lo} / {n_clip_hi}")
    return dense_new.astype(np.float32)


def edge_aware_residual_reconstruct_gpu(dense_gt: np.ndarray,
                                        lidar: np.ndarray,
                                        mask_lidar: np.ndarray,
                                        rgb_norm: np.ndarray,
                                        sigma_s: float = 30.0,
                                        sigma_r: float = 0.15,
                                        K: int = 200,
                                        guide: str = "rgb",
                                        chunk: int = 65536,
                                        device: str = "cuda") -> np.ndarray:
    """GPU port of `edge_aware_residual_reconstruct` — uses torch.cdist
    + topk in chunks instead of scipy.cKDTree. ~25× faster than CPU
    on a single GPU (sample-level: ~25 s → ~1 s).
    """
    H, W = dense_gt.shape
    ys, xs = np.where(mask_lidar)
    if len(ys) == 0:
        return dense_gt.astype(np.float32)
    residuals_np = lidar[ys, xs].astype(np.float32) - dense_gt[ys, xs].astype(np.float32)

    if guide == "rgb":
        if rgb_norm is None:
            raise ValueError("guide='rgb' requires rgb_norm")
        guide_img = rgb_norm.astype(np.float32)              # (3, H, W)
    elif guide == "depth":
        guide_img = dense_gt.astype(np.float32)[None, ...]
    elif guide == "depth_log":
        guide_img = np.log(np.clip(dense_gt, 1e-3, None)).astype(np.float32)[None, ...]
    else:
        raise ValueError(f"unknown guide '{guide}' (rgb / depth / depth_log)")
    C = guide_img.shape[0]

    dev = torch.device(device)
    coords_l = torch.from_numpy(np.stack([ys, xs], axis=-1)).float().to(dev)        # (M, 2)
    guide_l = torch.from_numpy(guide_img[:, ys, xs].T).float().to(dev)              # (M, C)
    residuals = torch.from_numpy(residuals_np).to(dev)                              # (M,)

    yq, xq = np.indices((H, W))
    coords_q_full = np.stack([yq.ravel(), xq.ravel()], axis=-1).astype(np.float32)  # (HW, 2)
    guide_q_full = guide_img.reshape(C, -1).T.astype(np.float32)                    # (HW, C)

    k = int(min(K, len(ys)))
    inv_2ss = 1.0 / (2.0 * sigma_s ** 2)
    inv_2sr = 1.0 / (2.0 * sigma_r ** 2)

    r_blend_out = torch.zeros(H * W, dtype=torch.float32, device=dev)
    n_total = H * W
    for start in range(0, n_total, chunk):
        end = min(start + chunk, n_total)
        coords_q = torch.from_numpy(coords_q_full[start:end]).to(dev)
        guide_q = torch.from_numpy(guide_q_full[start:end]).to(dev)
        d2 = torch.cdist(coords_q, coords_l).pow_(2)                                # (Q, M)
        dvals, idx = d2.topk(k, dim=1, largest=False)                                # (Q, K)
        w_sp = torch.exp_(dvals.mul_(-inv_2ss))                                      # in-place
        guide_nn = guide_l[idx]                                                      # (Q, K, C)
        gd2 = ((guide_q[:, None, :] - guide_nn) ** 2).sum(dim=-1)                    # (Q, K)
        w_rn = torch.exp_(gd2.mul_(-inv_2sr))
        w = w_sp * w_rn
        r_nn = residuals[idx]                                                        # (Q, K)
        r_blend_out[start:end] = (w * r_nn).sum(dim=-1) / w.sum(dim=-1).clamp_min(1e-8)

    r_blend = r_blend_out.cpu().numpy().reshape(H, W).astype(np.float64)
    dense_new = dense_gt.astype(np.float64) + r_blend
    n_clip_lo = int((dense_new < 0.0).sum())
    n_clip_hi = int((dense_new > 120.0).sum())
    dense_new = np.clip(dense_new, 0.1, 120.0)

    print(f"  GPU edge-aware K={k} σ_s={sigma_s} σ_r={sigma_r} "
          f"guide={guide}: r̃ abs.mean={np.abs(r_blend).mean():.3f}, "
          f"max|r̃|={np.abs(r_blend).max():.2f}, "
          f"clipped (<0 / >120) = {n_clip_lo} / {n_clip_hi}")
    return dense_new.astype(np.float32)


def _grad_2d(u: np.ndarray):
    """Forward-difference gradient with zero Neumann at far boundary."""
    gx = np.zeros_like(u)
    gy = np.zeros_like(u)
    gx[:, :-1] = u[:, 1:] - u[:, :-1]
    gy[:-1, :] = u[1:, :] - u[:-1, :]
    return gx, gy


def _div_2d(px: np.ndarray, py: np.ndarray) -> np.ndarray:
    """Negative adjoint of _grad_2d — divergence of (px, py)."""
    dx = np.zeros_like(px)
    dy = np.zeros_like(py)
    dx[:, 0] = px[:, 0]
    dx[:, 1:-1] = px[:, 1:-1] - px[:, :-2]
    dx[:, -1] = -px[:, -2]
    dy[0, :] = py[0, :]
    dy[1:-1, :] = py[1:-1, :] - py[:-2, :]
    dy[-1, :] = -py[-2, :]
    return dx + dy


def tv_l2_reconstruct(dense_gt: np.ndarray,
                      lidar: np.ndarray,
                      mask_lidar: np.ndarray,
                      w_lidar: float = 100.0,
                      w_dense: float = 1.0,
                      alpha: float = 0.5,
                      n_iter: int = 100) -> np.ndarray:
    """TV-L2 variational depth refinement (Chambolle–Pock primal-dual).

    Solves:
        min_d  (w_lidar / 2) Σ_{p∈L} (d_p − d_lidar,p)²
             + (w_dense / 2) Σ_p     (d_p − d_dense,p)²
             + α · TV(d)

    Piecewise-flat regulariser — bridges LiDAR sparsity smoothly without
    inheriting the scan-line geometry as a low-frequency pattern.

    LiDAR pixels get a much higher per-pixel data weight (w_lidar) than
    non-LiDAR pixels (w_dense). When w_lidar → ∞ this becomes a hard
    constraint; finite w_lidar lets very noisy LiDAR points be smoothed
    slightly by the TV term.
    """
    H, W = dense_gt.shape
    W_mat = np.full((H, W), w_dense, dtype=np.float64)
    W_mat[mask_lidar] = w_lidar
    d_ref = dense_gt.astype(np.float64).copy()
    d_ref[mask_lidar] = lidar[mask_lidar].astype(np.float64)

    # Chambolle–Pock step sizes (operator norm of grad ≤ √8 in 2D)
    L = np.sqrt(8.0)
    tau = 1.0 / L
    sigma = 1.0 / L

    u = d_ref.copy()
    u_bar = u.copy()
    px = np.zeros_like(u)
    py = np.zeros_like(u)

    for _ in range(n_iter):
        gx, gy = _grad_2d(u_bar)
        px = px + sigma * gx
        py = py + sigma * gy
        # Project onto L²-ball of radius α (per pixel)
        norm = np.sqrt(px * px + py * py) / alpha
        norm = np.maximum(1.0, norm)
        px /= norm
        py /= norm

        div_p = _div_2d(px, py)
        u_new = (u + tau * (div_p + W_mat * d_ref)) / (1.0 + tau * W_mat)
        u_bar = 2.0 * u_new - u
        u = u_new

    u = np.clip(u, 0.1, 120.0)
    err_at_lidar = np.abs(u[mask_lidar] - lidar[mask_lidar]).mean()
    print(f"  TV-L2 α={alpha} w_lidar={w_lidar} w_dense={w_dense} "
          f"n_iter={n_iter}: err@LiDAR={err_at_lidar:.3f} m")
    return u.astype(np.float32)


def locally_weighted_polyfit_reconstruct(dense_gt: np.ndarray,
                                         lidar: np.ndarray,
                                         mask_lidar: np.ndarray,
                                         degree: int = 2,
                                         sigma_s=50.0,
                                         ridge: float = 1e-4) -> np.ndarray:
    """Locally-Weighted Polynomial Fit (LWPF) — POLAR-style depth correction
    with LiDAR in place of radar, applied LOCALLY via spatial Gaussian
    weighting.

    Model:
        d(x, y) ≈ Σ_{i=0..N} c_i(x, y) · z(x, y)^i
    where z = dense_gt (scaleless / MDE-refined depth) and
          c_i(x, y) are spatially-varying polynomial coefficients
          fitted from LiDAR points weighted by a spatial Gaussian
          (σ = sigma_s pixels) around each query pixel.

    Properties:
      • degree=1 → per-pixel affine (scale + shift) — classical
                   LiDAR-aware scale recovery, but with a smooth
                   spatial transition between local fits.
      • degree≥2 → can correct cross-region misalignments (POLAR
                   insight), still LiDAR-anchored.
      • σ_s small → very local fit (high fidelity, may oscillate
                    in low-LiDAR regions).
      • σ_s large → near-global fit (smooth, less expressive).
      • σ_s may be (σ_y, σ_x) tuple for ANISOTROPIC weighting —
        useful with the 32-beam horizontal-line LiDAR of nuScenes:
        σ_y > σ_x helps bridge the empty rows between LiDAR lines.
      • No cell boundaries by construction — coefficient fields are
        smooth functions of (x, y).

    Efficient implementation:
        Build "moment" images via separable Gaussian filtering:
            M_p(x,y) = Σ_j w_j(x,y) · z_j^p                  p = 0..2N
            R_p(x,y) = Σ_j w_j(x,y) · z_j^p · d_j            p = 0..N
        At each pixel solve the (N+1)×(N+1) normal equation
            A c = b,  A_{ik} = M_{i+k},  b_i = R_i
        Total cost: O((3N+2) · H · W · σ_s) for filtering
                  + O((N+1)^3 · H · W)      for batched solve.

    Ridge regularisation (`ridge`) handles near-singular A in regions
    far from any LiDAR (small effective sample count).
    """
    from scipy.ndimage import gaussian_filter

    H, W = dense_gt.shape
    N = int(degree)
    z = dense_gt.astype(np.float64)
    d = lidar.astype(np.float64)

    # 0-out everything outside the LiDAR mask: only those pixels
    # contribute to the moment sums.
    mask_f = mask_lidar.astype(np.float64)
    z_anchor = np.where(mask_lidar, z, 0.0)
    d_anchor = np.where(mask_lidar, d, 0.0)

    # Pre-compute z_anchor^p · mask    for p = 0..2N        → for A
    # and       z_anchor^p · d_anchor for p = 0..N         → for b
    # Each becomes a moment image after Gaussian filtering.
    M = []  # M[p] = G_σ * Σ_j w_j z_j^p   (length 2N+1)
    R = []  # R[p] = G_σ * Σ_j w_j z_j^p d_j (length N+1)
    z_powers_at_anchor = [mask_f.copy()]  # z^0 at anchor = mask
    for p in range(1, 2 * N + 1):
        z_powers_at_anchor.append(z_powers_at_anchor[-1] * z_anchor)
    for p in range(2 * N + 1):
        M.append(gaussian_filter(z_powers_at_anchor[p],
                                 sigma=sigma_s, mode="constant", cval=0.0))
    for p in range(N + 1):
        R.append(gaussian_filter(z_powers_at_anchor[p] * d_anchor,
                                 sigma=sigma_s, mode="constant", cval=0.0))

    # Build per-pixel (N+1)×(N+1) Gram matrix A and (N+1) RHS b.
    # Stack moment images into tensors of shape (H, W, N+1, N+1) / (H, W, N+1).
    A = np.empty((H, W, N + 1, N + 1), dtype=np.float64)
    for i in range(N + 1):
        for k in range(N + 1):
            A[..., i, k] = M[i + k]
    b = np.stack(R, axis=-1)  # (H, W, N+1)

    # Tikhonov regulariser — scaled by M[0] (effective sample count) to
    # remain meaningful across regions with different LiDAR density.
    eye = np.eye(N + 1)[None, None]
    reg = ridge * np.maximum(M[0], 1e-12)[..., None, None] * eye
    A_reg = A + reg

    # Mark pixels where the local LiDAR neighbourhood is too thin — these
    # will be overwritten with the dense_gt fallback below regardless of
    # what `c` ends up being.
    support = M[0]
    weak = support < 1e-6 * max(1, (N + 1) ** 2)

    # Per-pixel solve. `solve` would fail for the WHOLE image if any single
    # Gram matrix is singular (common at deg≥4 in low-support regions);
    # `pinv` returns the Moore-Penrose pseudoinverse, which is well-defined
    # for singular matrices and degenerates gracefully. Cost: one batched
    # SVD on (N+1)² matrices — ~1–2 s for N=4 on 1.44M pixels.
    c = (np.linalg.pinv(A_reg) @ b[..., None])[..., 0]  # (H, W, N+1)

    # Evaluate polynomial Σ c_i · z^i at each pixel.
    z_powers = [np.ones_like(z)]
    for p in range(1, N + 1):
        z_powers.append(z_powers[-1] * z)
    dense_new = np.zeros_like(z)
    for p in range(N + 1):
        dense_new += c[..., p] * z_powers[p]

    # Where the local fit had essentially no LiDAR support, fall back
    # to dense_gt — `weak` was determined above (matches the identity
    # injection threshold for batched-solve robustness).
    dense_new[weak] = z[weak]

    # Polynomial extrapolation can explode when z(x,y) lies outside the
    # range of LiDAR-sampled z values in that local neighbourhood. Clip
    # the result to the valid metric depth range — match the dataset's
    # max-depth convention (nuScenes / ZJU / VoD all evaluate up to 80 m,
    # with dense_gt capped at 100 m). Values < 0 are nonphysical.
    n_pre_clip_low = int((dense_new < 0.0).sum())
    n_pre_clip_high = int((dense_new > 120.0).sum())
    dense_new = np.clip(dense_new, 0.1, 120.0)

    # Diagnostics.
    err_at_lidar = np.abs(dense_new[mask_lidar] - lidar[mask_lidar]).mean()
    c_mean_abs = [float(np.abs(c[..., p]).mean()) for p in range(N + 1)]
    print(f"  LWPF deg={N} σ_s={sigma_s}: "
          f"|c_i|.mean = {['%.3g' % v for v in c_mean_abs]}, "
          f"weak-support px = {int(weak.sum())} / {H*W}, "
          f"extrap clipped (<0 / >120) = {n_pre_clip_low} / {n_pre_clip_high}, "
          f"err@LiDAR = {err_at_lidar:.3f} m")
    return dense_new.astype(np.float32)


def residual_interp_reconstruct(dense_gt: np.ndarray,
                                lidar: np.ndarray,
                                mask_lidar: np.ndarray,
                                sigma_s: float = 50.0,
                                K: int = 20) -> np.ndarray:
    """Residual interpolation with GLOBAL Gaussian-weighted spread.

    Like `depth_lidar_filled` (Voronoi-based residual fill) but the
    LiDAR-point residuals are *blended* by a Gaussian kernel applied
    over the WHOLE image — i.e. every pixel is the Gaussian-weighted
    average of EVERY LiDAR point (close points get more weight, far
    points still contribute). Implemented via separable Gaussian
    filtering — O(N·σ) for any σ, no k-NN truncation, no cell
    boundaries.

    Algorithm:
        residual_field(p) = lidar(p) − dense_gt(p)   if p ∈ LiDAR
                          = 0                        otherwise
        mask_field(p)     = 1 if p ∈ LiDAR else 0
        r̃(p) = G_σ * residual_field  /  G_σ * mask_field    (NW-style smoothing)
        dense_new(p) = dense_gt(p) + r̃(p)

    Properties:
      • dense_gt's shape (DAv2 fine detail) is preserved exactly.
      • ALL LiDAR points contribute — k-NN truncation removed, so the
        soft transition between rows is global instead of locally
        nearest-only (no implicit cell boundary).
      • Larger σ → smoother spread, weaker per-point fidelity.
      • Smaller σ → tighter spread, sharper boundaries.

    The `K` argument is kept for backward-compat but ignored.
    """
    from scipy.ndimage import gaussian_filter

    residual_field = np.zeros_like(dense_gt, dtype=np.float64)
    residual_field[mask_lidar] = (lidar[mask_lidar] - dense_gt[mask_lidar])
    mask_field = mask_lidar.astype(np.float64)

    num = gaussian_filter(residual_field, sigma=sigma_s, mode="constant", cval=0.0)
    den = gaussian_filter(mask_field,     sigma=sigma_s, mode="constant", cval=0.0)

    # NW-style ratio. Far from any LiDAR, denominator is tiny — fall back to 0.
    r_blend = np.where(den > 1e-8, num / np.maximum(den, 1e-8), 0.0)

    print(f"  residual-interp GLOBAL σ_s={sigma_s}: "
          f"r̃ mean={r_blend.mean():+.3f}, abs.mean={np.abs(r_blend).mean():.3f}, "
          f"max|r̃|={np.abs(r_blend).max():.2f}")
    return (dense_gt.astype(np.float64) + r_blend).astype(np.float32)


def cross_bilateral_depth_reconstruct(dense_gt: np.ndarray,
                                       lidar: np.ndarray,
                                       mask_lidar: np.ndarray,
                                       sigma_s: float = 10.0,
                                       sigma_r: float = 2.0,
                                       K: int = 20,
                                       use_log: bool = False,
                                       eps_log: float = 1e-3) -> np.ndarray:
    """DAv2-depth-guided cross-bilateral interpolation of sparse LiDAR.

    Same structure as `cross_bilateral_reconstruct` but the *range*
    kernel uses dense_gt (DAv2) depth similarity instead of RGB
    similarity:

        w_range(p, q) = exp(-‖dense_gt(p) − dense_gt(q)‖² / 2σ_r²)

    Rationale: DAv2's *absolute ratios* are unreliable, but its
    *relative ordering* is good — pixels with similar DAv2 depth almost
    always belong to the same surface, even when RGB fails to segment
    them (e.g. uniformly-shaded road, fine texture).

    σ_r is in METERS (or log-depth units if use_log=True), unlike the
    RGB version where σ_r is in [0, 1] colour units.

    use_log=True uses log-depth → σ_r becomes a *relative* depth
    similarity (e.g. σ_r=0.1 ≈ 10 % ratio difference).
    """
    from scipy.spatial import cKDTree

    H, W = lidar.shape
    ys, xs = np.where(mask_lidar)
    if len(ys) == 0:
        return np.zeros_like(lidar)

    lidar_pts = np.stack([ys, xs], axis=1).astype(np.float64)
    tree = cKDTree(lidar_pts)

    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    all_pix = np.stack([yy.ravel(), xx.ravel()], axis=1).astype(np.float64)

    k = min(K, len(ys))
    dists, idx = tree.query(all_pix, k=k)
    if k == 1:
        dists = dists[:, None]
        idx = idx[:, None]

    ws_spatial = np.exp(-dists**2 / (2.0 * sigma_s**2))

    if use_log:
        depth_field = np.log(np.clip(dense_gt, eps_log, None))
    else:
        depth_field = dense_gt.astype(np.float64)

    depth_at_lidar = depth_field[ys, xs]            # (N_lidar,)
    depth_flat = depth_field.flatten()              # (HW,)
    depth_nb = depth_at_lidar[idx]                  # (HW, k)
    dr2 = (depth_flat[:, None] - depth_nb)**2       # (HW, k)
    ws_range = np.exp(-dr2 / (2.0 * sigma_r**2))

    w = ws_spatial * ws_range
    w_sum = w.sum(axis=-1)
    lidar_nb = lidar[ys[idx], xs[idx]]
    val_sum = (w * lidar_nb).sum(axis=-1)

    dense_new = (val_sum / (w_sum + 1e-8)).reshape(H, W).astype(np.float32)
    print(f"  cross-bilateral-DEPTH: σ_s={sigma_s} σ_r={sigma_r} K={k} "
          f"{'(log)' if use_log else '(raw m)'}, avg w_sum = {w_sum.mean():.3f}")
    return dense_new


def poisson_reconstruct(dense_gt: np.ndarray, lidar: np.ndarray,
                        mask_lidar: np.ndarray, L: sp.csr_matrix,
                        mode: str = "hard", lam: float = 1.0,
                        eps_log: float = 1e-3,
                        rgb_norm: Optional[np.ndarray] = None,
                        sigma_s: float = 10.0,
                        sigma_r: float = 0.10,
                        bilat_K: int = 20) -> np.ndarray:
    """Solve Poisson system: shape from dense_gt, values pinned to LiDAR.

    Modes:
      • hard      : exact LiDAR Dirichlet (default; spike-prone)
      • soft      : LiDAR as L2 penalty with strength λ (spike spread)
      • log       : `hard` on log-depth (compresses ratio mismatches)
      • bilateral : NOT Poisson — RGB-guided cross-bilateral of LiDAR.
                    Spike-free by construction, no dense_gt dependence.
                    Requires rgb_norm (3, H, W) ImageNet-normalised RGB.
      • residual  : LiDAR residual interp via GLOBAL Gaussian spread.
      • lwpf      : Locally-Weighted Polynomial Fit (POLAR-style, LiDAR-
                    anchored, local). Reads LWPF_DEGREE env var (default 2).
    """
    if mode == "bilateral":
        if rgb_norm is None:
            raise ValueError("mode='bilateral' requires rgb_norm")
        return cross_bilateral_reconstruct(
            rgb_norm, lidar, mask_lidar,
            sigma_s=sigma_s, sigma_r=sigma_r, K=bilat_K,
        )
    if mode == "bilateral_depth":
        return cross_bilateral_depth_reconstruct(
            dense_gt, lidar, mask_lidar,
            sigma_s=sigma_s, sigma_r=sigma_r, K=bilat_K,
            use_log=False,
        )
    if mode == "bilateral_depth_log":
        return cross_bilateral_depth_reconstruct(
            dense_gt, lidar, mask_lidar,
            sigma_s=sigma_s, sigma_r=sigma_r, K=bilat_K,
            use_log=True,
        )
    if mode == "residual":
        # sigma_s passed via the σ_r arg slot for convenience.
        return residual_interp_reconstruct(
            dense_gt, lidar, mask_lidar,
            sigma_s=sigma_r, K=bilat_K,
        )
    if mode == "segpoly":
        # Segmentation-based piecewise polynomial fit. sigma_r slot is
        # repurposed (ignored — pass any value). Configs via env vars:
        #   SEG_K   — number of K-means clusters (default 16)
        #   SEG_DEG — polynomial degree (default 2)
        #   SEG_MIN — minimum segment size in pixels (default 200)
        if rgb_norm is None:
            raise ValueError("mode='segpoly' requires rgb_norm")
        seg_k = int(os.environ.get("SEG_K", "16"))
        seg_deg = int(os.environ.get("SEG_DEG", "2"))
        seg_min = int(os.environ.get("SEG_MIN", "200"))
        return piecewise_polyfit_reconstruct(
            dense_gt, lidar, mask_lidar, rgb_norm,
            degree=seg_deg, n_clusters=seg_k, min_segment_size=seg_min,
        )
    if mode == "edge_residual":
        # Edge-aware residual interp (joint-bilateral, guide-channel-aware).
        # σ_s = sigma_r (px), guide selected by EDGE_GUIDE env var:
        #   "rgb"       (default) — σ_r typ 0.10–0.30 (normalised RGB units)
        #   "depth"               — σ_r typ 1–5 (meters)
        #   "depth_log"           — σ_r typ 0.05–0.20 (log-ratio)
        # EDGE_SIGMA_R sets σ_r (default depends on guide), EDGE_K = K.
        edge_guide = os.environ.get("EDGE_GUIDE", "rgb")
        _default_sigma_r_by_guide = {"rgb": 0.15, "depth": 2.0, "depth_log": 0.10}
        edge_sigma_r = float(os.environ.get(
            "EDGE_SIGMA_R", _default_sigma_r_by_guide.get(edge_guide, 0.15)))
        edge_k = int(os.environ.get("EDGE_K", "200"))
        use_gpu = os.environ.get("EDGE_USE_GPU", "0") == "1" and torch.cuda.is_available()
        fn = (edge_aware_residual_reconstruct_gpu if use_gpu
              else edge_aware_residual_reconstruct)
        return fn(
            dense_gt, lidar, mask_lidar, rgb_norm,
            sigma_s=sigma_r, sigma_r=edge_sigma_r, K=edge_k, guide=edge_guide,
        )
    if mode == "tv":
        # TV-L2 variational refinement. sigma_r slot carries α (TV strength).
        # w_lidar / w_dense / n_iter from env vars (sane defaults).
        w_lidar = float(os.environ.get("TV_W_LIDAR", "100.0"))
        w_dense = float(os.environ.get("TV_W_DENSE", "1.0"))
        n_iter = int(os.environ.get("TV_N_ITER", "150"))
        return tv_l2_reconstruct(
            dense_gt, lidar, mask_lidar,
            w_lidar=w_lidar, w_dense=w_dense, alpha=sigma_r, n_iter=n_iter,
        )
    if mode == "lwpf":
        # Locally-Weighted Polynomial Fit (LiDAR-anchored, POLAR-style local).
        # sigma_r slot carries σ_x (pixels); polynomial degree from env var
        # LWPF_DEGREE (default 2); anisotropy ratio σ_y/σ_x from env var
        # LWPF_ANISO (default 1.0 = isotropic). For nuScenes 32-beam LiDAR
        # ratio > 1 bridges the empty rows between scan lines.
        deg = int(os.environ.get("LWPF_DEGREE", "2"))
        aniso = float(os.environ.get("LWPF_ANISO", "1.0"))
        sigma_kw = (sigma_r * aniso, sigma_r) if aniso != 1.0 else sigma_r
        return locally_weighted_polyfit_reconstruct(
            dense_gt, lidar, mask_lidar,
            degree=deg, sigma_s=sigma_kw,
        )
    if mode == "log":
        log_dg = np.log(np.clip(dense_gt, eps_log, None))
        log_li = np.log(np.clip(lidar,    eps_log, None))
        log_new = _poisson_hard(log_dg, log_li, mask_lidar, L)
        return np.exp(log_new).astype(np.float32)
    if mode == "soft":
        return _poisson_soft(dense_gt, lidar, mask_lidar, L, lam=lam)
    if mode == "hard":
        return _poisson_hard(dense_gt, lidar, mask_lidar, L)
    raise ValueError(f"unknown poisson mode: {mode}")


def denorm_rgb(rgb_norm: np.ndarray) -> np.ndarray:
    """De-normalise the dataset's [-1, 1] RGB tensor back to [0, 1] for display.

    Matches base.py:_normalize_rgb which uses rgb/255*2 − 1, NOT ImageNet
    stats.
    """
    img = (rgb_norm + 1.0) / 2.0
    return np.clip(img.transpose(1, 2, 0), 0, 1)


def load_depth_png(data_root: str, dir_name: str, sample_id: str) -> np.ndarray:
    """Load nuScenes-style depth png (uint16 × 256 → meters; 0 = invalid)."""
    from PIL import Image
    path = os.path.join(data_root, dir_name, f"{sample_id}.png")
    return np.asarray(Image.open(path), dtype=np.float32) / 256.0


def main():
    # Positional args:
    #   sys.argv[1]: shape source (Laplacian)            default depth_refined
    #   sys.argv[2]: anchor / Dirichlet target           default depth_lidar
    #   sys.argv[3]: dynamic-object threshold (m)        default 1.0
    #                  — only used when anchor ≠ depth_lidar (acc filtering)
    #                  set 0 to disable
    #   sys.argv[4]: patch size K for Dirichlet dilation default 5
    #                  K=1 means no dilation (single-pixel Dirichlet)
    #                  K=5 → each anchor expands to 5×5 patch with the
    #                  same value, replacing spike with smoother bump
    shape_dir = sys.argv[1] if len(sys.argv) > 1 else "depth_refined"
    anchor_dir = sys.argv[2] if len(sys.argv) > 2 else "depth_lidar"
    dynamic_th = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
    patch_K = int(sys.argv[4]) if len(sys.argv) > 4 else 1
    mode = sys.argv[5] if len(sys.argv) > 5 else "hard"
    # sys.argv[6] is mode-dependent:
    #   soft               → λ (penalty strength)              default 1.0
    #   bilateral          → σ_r in RGB [0, 1] units            default 0.10
    #   bilateral_depth    → σ_r in METERS                      default 2.0
    #   bilateral_depth_log→ σ_r in log-depth units (~ratio)    default 0.10
    _default_sigma_r = {
        "bilateral": 0.10,
        "bilateral_depth": 2.0,
        "bilateral_depth_log": 0.10,
        "residual": 50.0,        # σ_s in pixels for GLOBAL Gaussian residual spread
        "lwpf": 50.0,            # σ_s in pixels for Locally-Weighted Polynomial Fit
                                 # (also reads LWPF_DEGREE env var, default 2)
        "edge_residual": 30.0,   # σ_s in pixels (joint-bilateral RGB-guided)
        "tv": 0.5,               # α (TV strength) for TV-L2 refinement
        "segpoly": 0.0,          # unused (segpoly reads SEG_K / SEG_DEG / SEG_MIN env)
    }
    lam = float(sys.argv[6]) if len(sys.argv) > 6 else _default_sigma_r.get(mode, 1.0)

    # Fix all RNGs so successive runs (varying patch_K / shape_dir) operate
    # on the IDENTICAL 4 samples — enables direct visual comparison.
    import random
    torch.manual_seed(0); np.random.seed(0); random.seed(0)

    with initialize_config_dir(config_dir=os.path.abspath("config"), version_base=None):
        cfg = compose(config_name="config", overrides=[
            "+experiment=shape_lidar_grad_cos",
            f"dataset.dense_gt_dir={shape_dir}",
        ])
    data_root = cfg.dataset.data_root
    print(f"shape_dir    = {shape_dir}")
    print(f"anchor_dir   = {anchor_dir}")
    if anchor_dir != "depth_lidar":
        print(f"dynamic_th   = {dynamic_th} m (single-frame LiDAR vs anchor agreement)")
    print(f"patch_K      = {patch_K} (1 = pointwise; 5 = 5×5 patch Dirichlet)")
    print(f"mode         = {mode}{' (λ='+str(lam)+')' if mode=='soft' else ''}")
    train_loader, _, _ = build_loaders(cfg)
    # Disable train-time augmentation (photometric "night-like" gamma) so the
    # visualised RGB matches the original camera output — prototype is for
    # inspection, not training. Mutate BEFORE iter() so DataLoader workers
    # inherit the disabled state at fork time.
    _ds = train_loader.dataset
    if hasattr(_ds, "dataset"):
        _ds = _ds.dataset
    _ds.augmentation = False
    _ds.photometric_aug = None
    batch = next(iter(train_loader))

    _lwpf_deg_env = int(os.environ.get("LWPF_DEGREE", "2"))
    _lwpf_aniso_env = float(os.environ.get("LWPF_ANISO", "1.0"))
    _aniso_tag = f"_aniso{_lwpf_aniso_env:g}" if _lwpf_aniso_env != 1.0 else ""
    _fill_tag = f"_fill{os.environ.get('INVALID_FILL', 'dense_gt')}" if os.environ.get('INVALID_FILL') == 'shift' else ""
    _edge_guide = os.environ.get("EDGE_GUIDE", "rgb")
    _edge_sigma_r_default = {"rgb": 0.15, "depth": 2.0, "depth_log": 0.10}.get(
        _edge_guide, 0.15)
    _edge_sigma_r = float(os.environ.get("EDGE_SIGMA_R", str(_edge_sigma_r_default)))
    _tv_w_lidar = float(os.environ.get("TV_W_LIDAR", "100.0"))
    mode_suffix = (
        f"_{mode}"
        + (f"_lam{lam:g}" if mode == "soft" else "")
        + (f"_sigma{lam:g}" if mode == "residual" else "")
        + (f"_deg{_lwpf_deg_env}_sigma{lam:g}{_aniso_tag}" if mode == "lwpf" else "")
        + (f"_{_edge_guide}_sigma{lam:g}_r{_edge_sigma_r:g}" if mode == "edge_residual" else "")
        + (f"_a{lam:g}_wl{_tv_w_lidar:g}" if mode == "tv" else "")
        + (f"_K{int(os.environ.get('SEG_K','16'))}"
           f"_deg{int(os.environ.get('SEG_DEG','2'))}"
           f"_min{int(os.environ.get('SEG_MIN','200'))}" if mode == "segpoly" else "")
        + _fill_tag
    )
    tag = (f"{shape_dir.replace('depth_', '')}__"
           f"{anchor_dir.replace('depth_', '')}_K{patch_K}{mode_suffix}")
    out_dir = f"/workspace/RadarTaco/output/poisson_prototype_{tag}"
    os.makedirs(out_dir, exist_ok=True)

    # Build Laplacian once (image size is fixed within a batch).
    H, W = batch["depth_gt_dense"].shape[-2:]
    print(f"image size: {H}×{W}")
    t0 = time.time()
    L = build_laplacian_2d(H, W)
    print(f"Laplacian built ({L.shape}, nnz={L.nnz}) in {time.time()-t0:.2f}s")

    n_samples = min(int(os.environ.get("N_SAMPLES", "12")),
                    batch["depth_gt_dense"].shape[0])
    for s in range(n_samples):
        dense_gt = batch["depth_gt_dense"][s, 0].cpu().numpy()
        lidar = batch["depth_gt_lidar"][s, 0].cpu().numpy()
        mask_l_sf = batch["valid_mask_lidar"][s, 0].cpu().numpy().astype(bool)
        # valid_mask_dense: dense_gt 의 (depth > min_depth) & (depth < max_depth)
        # & finite. 주로 sky / max-depth-cap (100m) 픽셀이 invalid.
        mask_dense = batch["valid_mask_dense"][s, 0].cpu().numpy().astype(bool)
        rgb = batch["rgb_norm"][s].cpu().numpy()
        sample_id = batch["sample_id"][s]

        # (A) LiDAR pixel 중 dense_gt invalid (e.g. sky / 100m cap) 한 위치는
        # fit 에서 제외 — 잘못된 (z, d) 쌍이 polynomial / bilateral guide 를
        #오염시키는 것을 방지.
        n_lidar_before = int(mask_l_sf.sum())
        mask_l_sf &= mask_dense
        n_dropped_lidar = n_lidar_before - int(mask_l_sf.sum())

        # ---- Load anchor (depth_acc) directly ------------------------------
        if anchor_dir == "depth_lidar":
            # Use single-frame LiDAR as-is (backward-compatible setup)
            anchor = lidar.copy()
            mask_anchor = mask_l_sf.copy()
        else:
            anchor_img = load_depth_png(data_root, anchor_dir, sample_id)
            mask_anchor = anchor_img > 0
            # Dynamic-object filtering: where both single-frame LiDAR AND anchor
            # exist, they should agree (static scene). Drop disagreeing anchor
            # pixels (= moving objects whose accumulated location is wrong).
            if dynamic_th > 0:
                both = mask_anchor & mask_l_sf
                disagree = both & (np.abs(anchor_img - lidar) > dynamic_th)
                mask_anchor = mask_anchor & ~disagree
                n_drop = int(disagree.sum())
            else:
                n_drop = 0
            # (A) anchor 픽셀도 dense_gt invalid 영역 제외
            mask_anchor = mask_anchor & mask_dense
            anchor = anchor_img

        # ---- Patch-wise Dirichlet (K×K dilation) ---------------------------
        # Replace each anchor point with a K×K patch carrying the SAME
        # value (= value of the nearest original anchor point). The
        # Poisson solver then sees a small *region* of fixed values
        # instead of a single isolated point — the resulting solution
        # transitions smoothly from the patch out to dense_gt's shape
        # instead of producing a spike.
        if patch_K > 1:
            radius = patch_K // 2
            mask_dilated = ndi.binary_dilation(mask_anchor, iterations=radius)
            # distance_transform_edt(~mask, return_indices=True) returns,
            # for every pixel, the (y, x) of the nearest mask=True pixel.
            inds = ndi.distance_transform_edt(~mask_anchor,
                                              return_distances=False,
                                              return_indices=True)
            anchor_spread = anchor[inds[0], inds[1]]
            n_added = int(mask_dilated.sum() - mask_anchor.sum())
            mask_anchor = mask_dilated
            anchor = anchor_spread

        print(f"\n--- sample {s} ({sample_id}) ---")
        print(f"  dense_gt valid pixels: {int(mask_dense.sum())} "
              f"({mask_dense.sum()/(H*W)*100:.2f}%) — dropped {n_dropped_lidar} "
              f"LiDAR px outside valid_mask_dense")
        print(f"  single-frame LiDAR pixels: {mask_l_sf.sum()} ({mask_l_sf.sum()/(H*W)*100:.2f}%)")
        print(f"  anchor ({anchor_dir}) pixels: {mask_anchor.sum()} "
              f"({mask_anchor.sum()/(H*W)*100:.2f}%)")
        if anchor_dir != "depth_lidar" and dynamic_th > 0:
            print(f"  dynamic-object filter dropped {n_drop} anchor pixels (|acc - lidar| > {dynamic_th}m)")
        if patch_K > 1:
            print(f"  patch dilation K={patch_K} added {n_added} pixels "
                  f"({n_added/(H*W)*100:.2f}% extra Dirichlet coverage)")
        print(f"  dense_gt range: [{dense_gt.min():.2f}, {dense_gt.max():.2f}]")
        print(f"  anchor (at anchor pix) mean: {anchor[mask_anchor].mean():.2f}")

        # Treat anchor as the Dirichlet boundary for the Poisson solver.
        # (Keep variable names mask_l / lidar to minimize downstream changes.)
        mask_l = mask_anchor
        lidar = anchor

        t0 = time.time()
        # `lam` is repurposed as σ_r (bilateral) or σ_s (residual).
        bilat_kw = {"sigma_r": lam} if (
            mode.startswith("bilateral")
            or mode in ("residual", "lwpf", "edge_residual", "tv", "segpoly")
        ) else {}
        dense_new = poisson_reconstruct(dense_gt, lidar, mask_l, L,
                                        mode=mode, lam=lam,
                                        rgb_norm=rgb, **bilat_kw)
        dt = time.time() - t0
        print(f"  Poisson solve: {dt:.2f}s")

        # (C) dense_gt invalid 영역 (sky / 100m cap 등) 의 처리.
        # INVALID_FILL env var:
        #   "dense_gt" (default): dense_gt 값 그대로 — hard boundary
        #   "shift": invalid 영역을 dense_gt + (nearest valid 픽셀에서의
        #            dense_new − dense_gt) 로 채움. 경계 jump = 0,
        #            invalid 영역 안에서도 dense_gt 의 상대 변화 보존.
        n_invalid = int((~mask_dense).sum())
        fill_mode = os.environ.get("INVALID_FILL", "dense_gt")
        if n_invalid > 0:
            if fill_mode == "shift":
                inds = ndi.distance_transform_edt(
                    ~mask_dense, return_indices=True, return_distances=False)
                delta = (dense_new - dense_gt)[inds[0], inds[1]]
                dense_new = np.where(mask_dense, dense_new, dense_gt + delta)
            else:
                dense_new = np.where(mask_dense, dense_new, dense_gt)
            print(f"  filled invalid (mode={fill_mode}) at {n_invalid} px "
                  f"({n_invalid/(H*W)*100:.2f}%)")

        lidar_err = np.abs(dense_new[mask_l] - lidar[mask_l])
        diff = dense_new - dense_gt
        print(f"  ‖dense_new − lidar‖ at LiDAR : mean={lidar_err.mean():.4f}m, "
              f"max={lidar_err.max():.4f}m")
        print(f"  Δ to dense_gt           : mean={np.abs(diff).mean():.3f}m, "
              f"max={np.abs(diff).max():.3f}m")

        # ---- visualise: RGB + LiDAR overlay | dense_dav2 | dense_new -------
        fig, axes = plt.subplots(1, 3, figsize=(30, 6))
        vmin, vmax = 0, 80
        cmap = "turbo"

        rgb_img = denorm_rgb(rgb)
        axes[0].imshow(rgb_img)
        ys_l, xs_l = np.where(mask_l)
        if len(ys_l) > 0:
            lidar_vals = lidar[ys_l, xs_l]
            sc = axes[0].scatter(xs_l, ys_l, c=lidar_vals,
                                 vmin=vmin, vmax=vmax, cmap=cmap, s=3)
            plt.colorbar(sc, ax=axes[0], fraction=0.04)
        axes[0].set_title(f"sample {s}: RGB + LiDAR overlay "
                          f"({len(ys_l)} pts)")
        axes[0].axis("off")

        im = axes[1].imshow(dense_gt, vmin=vmin, vmax=vmax, cmap=cmap)
        axes[1].set_title(f"sample {s}: dense_dav2 ({shape_dir})")
        axes[1].axis("off")
        plt.colorbar(im, ax=axes[1], fraction=0.04)

        im = axes[2].imshow(dense_new, vmin=vmin, vmax=vmax, cmap=cmap)
        axes[2].set_title(f"sample {s}: dense_new ({mode}"
                          + (f" σ_s={lam:g}" if mode in ("edge_residual", "residual", "lwpf") else "")
                          + (f" λ={lam:g}" if mode == "soft" else "")
                          + ")")
        axes[2].axis("off")
        plt.colorbar(im, ax=axes[2], fraction=0.04)

        out_path = f"{out_dir}/sample_{s:02d}.png"
        plt.tight_layout()
        plt.savefig(out_path, dpi=100, bbox_inches="tight")
        plt.close()
        print(f"  saved → {out_path}")

    print(f"\nDone. {n_samples} sample(s) at {out_dir}/")


if __name__ == "__main__":
    main()
