"""Polynomial GT refinement on top of depth_refined (POLAR-style, offline LS).

Pipeline (per image):
  1. Read affine-aligned depth_refined (uint16 PNG ×256, metric).
  2. Read sparse LiDAR depth_lidar (same format).
  3. Per-image, fit a polynomial in depth space:
         d_gt(x) ≈ Σ_{i=0..N} c_i · z_n(x)^i,
     where z_n = clip(z / max_depth, 0, 1) is the normalized refined depth.
     Uses IRLS-truncated weighted ridge regression (ROE-style robustness).
  4. Sanity check monotonicity on [0, 1]; if violated and `--mono_fallback`,
     halve the degree and retry.
  5. Apply poly to the full refined depth, clip to [min_depth, max_depth], save.

The polynomial sits on top of the already affine-aligned refined depth, so
it is a residual correction (small coefficients except c_1 ≈ 1, c_0 ≈ 0).
This mirrors POLAR's depth-space polynomial fit, but solved in closed form
per image (no learning) using LiDAR as anchor.

Usage:
    CUDA_VISIBLE_DEVICES=3 python scripts/refine_polynomial.py \\
      --split /data/public/nuScenes/derived/splits/test.txt \\
      --data_root /data/public/nuScenes/derived \\
      --refined_dir depth_refined \\
      --gt_dir depth_lidar \\
      --out_root_template /data/public/nuScenes/derived/depth_refined_poly{N} \\
      --degrees 2,4,6
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm


def iter_split(split_file):
    with open(split_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            sid = line.split("\t")[0]
            yield sid


def fit_poly_irls(z_src: np.ndarray, d_tgt: np.ndarray,
                  degree: int, ridge_lambda: float,
                  trunc: float, n_iters: int,
                  eps: float = 1e-8) -> tuple[np.ndarray, dict]:
    """Robust polynomial least-squares via IRLS + ridge.

    Fits d_tgt ≈ Σ c_i · z_src^i on a 1D set of anchor pairs.

    Robust weight scheme:
      w_i = w0_i · min(1, trunc / |w0_i · residual_i|)
    with w0_i = 1/d_tgt_i (so weighted residual is the relative depth error).
    This mirrors MoGe-2's truncated weighted L1 (ROE).

    Returns (coefficients of shape (degree+1,), info dict).
    """
    n = z_src.shape[0]
    z = z_src.astype(np.float64)
    t = d_tgt.astype(np.float64)
    w0 = 1.0 / np.clip(t, eps, None)         # relative-error weighting
    V = np.vander(z, N=degree + 1, increasing=True)   # (n, degree+1)

    c = None
    inlier_frac = 1.0
    for it in range(n_iters):
        if c is None:
            w = w0
        else:
            r = V @ c - t
            r_w = w0 * np.abs(r)
            cap = float(trunc)
            mask = (r_w <= cap).astype(np.float64)
            inlier_frac = float(mask.mean())
            w = w0 * np.minimum(1.0, cap / np.maximum(r_w, eps))
        WV = w[:, None] * V
        A = V.T @ WV + ridge_lambda * np.eye(degree + 1)
        b_v = V.T @ (w * t)
        try:
            c = np.linalg.solve(A, b_v)
        except np.linalg.LinAlgError:
            c = np.linalg.lstsq(A, b_v, rcond=None)[0]
    return c.astype(np.float64), {"inlier_frac": inlier_frac, "n_anchors": int(n)}


def check_monotonic(coeffs: np.ndarray, n_grid: int = 200) -> bool:
    """Check df/dz > 0 on [0, 1] using a uniform grid."""
    deg = coeffs.shape[0] - 1
    if deg < 1:
        return True
    # derivative coefficients: d c_i z^i / dz = i c_i z^{i-1}
    d_coeffs = coeffs[1:] * np.arange(1, deg + 1, dtype=np.float64)
    zs = np.linspace(0.0, 1.0, n_grid)
    V = np.vander(zs, N=deg, increasing=True)  # (n_grid, deg)
    deriv = V @ d_coeffs
    return bool(np.all(deriv > 0))


def apply_poly(z_full: np.ndarray, coeffs: np.ndarray,
               z_norm_clip: tuple[float, float]) -> np.ndarray:
    """Evaluate Σ c_i · z^i on full image, with extrapolation clamp."""
    deg = coeffs.shape[0] - 1
    z = np.clip(z_full.astype(np.float64),
                z_norm_clip[0], z_norm_clip[1])
    # Horner-like accumulation to avoid Vandermonde memory for large H×W
    out = np.full_like(z, coeffs[-1])
    for k in range(deg - 1, -1, -1):
        out = out * z + coeffs[k]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--data_root", required=True,
                    help="nuScenes derived root.")
    ap.add_argument("--refined_dir", default="depth_refined",
                    help="Source affine-refined depth dir (under data_root).")
    ap.add_argument("--gt_dir", default="depth_lidar",
                    help="Anchor sparse LiDAR dir (under data_root).")
    ap.add_argument("--out_root_template", required=True,
                    help="Output dir template with {N} for degree, e.g. "
                         ".../depth_refined_poly{N}")
    ap.add_argument("--degrees", default="2,4,6",
                    help="Comma-separated polynomial degrees to produce.")

    ap.add_argument("--min_depth", type=float, default=0.5)
    ap.add_argument("--max_depth", type=float, default=100.0,
                    help="Used both for clipping anchors and for z normalization.")
    ap.add_argument("--scale", type=float, default=256.0,
                    help="PNG uint16 quantization (depth_m × scale).")

    ap.add_argument("--ridge_lambda", type=float, default=1e-3,
                    help="L2 ridge on polynomial coefficients (normalized z).")
    ap.add_argument("--trunc", type=float, default=1.0,
                    help="IRLS truncation in weighted-residual (relative) space.")
    ap.add_argument("--n_iters", type=int, default=3)
    ap.add_argument("--min_inliers", type=int, default=100)

    ap.add_argument("--mono_check", action="store_true", default=True,
                    help="Check df/dz>0 on [0,1] and degrade degree if violated.")
    ap.add_argument("--no_mono_check", dest="mono_check", action="store_false")
    ap.add_argument("--mono_min_degree", type=int, default=1,
                    help="Lowest degree to degrade to (1 = affine).")

    ap.add_argument("--save_coeffs", action="store_true",
                    help="Dump per-sample coefficients as JSONL.")
    ap.add_argument("--skip_existing", action="store_true", default=True)
    ap.add_argument("--no_skip_existing", dest="skip_existing", action="store_false")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    degrees = [int(x) for x in args.degrees.split(",") if x.strip()]
    out_dirs = {N: args.out_root_template.format(N=N) for N in degrees}
    for d in out_dirs.values():
        os.makedirs(d, exist_ok=True)

    coeffs_log = {N: None for N in degrees}
    if args.save_coeffs:
        for N in degrees:
            coeffs_log[N] = open(os.path.join(out_dirs[N], "coeffs.jsonl"), "a")

    items = list(iter_split(args.split))
    if args.limit:
        items = items[: args.limit]
    print(f"[poly_refine] {len(items)} samples  degrees={degrees}")
    for N, d in out_dirs.items():
        print(f"  poly{N} -> {d}")

    z_clip = (0.0, 1.0)   # z is normalized depth ∈ [0,1]
    saved = {N: 0 for N in degrees}
    skipped = {N: 0 for N in degrees}
    fallback_count = {N: 0 for N in degrees}
    missing = 0
    low_inliers = 0

    for sid in tqdm(items):
        refined_path = os.path.join(args.data_root, args.refined_dir, f"{sid}.png")
        gt_path = os.path.join(args.data_root, args.gt_dir, f"{sid}.png")

        # Skip-existing check (all degrees done?)
        if args.skip_existing:
            todo = [N for N in degrees
                    if not os.path.exists(os.path.join(out_dirs[N], f"{sid}.png"))]
            for N in degrees:
                if N not in todo:
                    skipped[N] += 1
            if not todo:
                continue
        else:
            todo = list(degrees)

        if not os.path.exists(refined_path) or not os.path.exists(gt_path):
            missing += 1
            continue

        z_full = np.asarray(Image.open(refined_path), dtype=np.float32) / args.scale
        gt = np.asarray(Image.open(gt_path), dtype=np.float32) / args.scale
        valid = (gt > args.min_depth) & (gt < args.max_depth) \
            & (z_full > 0) & np.isfinite(z_full) & np.isfinite(gt)

        n_anch = int(valid.sum())
        if n_anch < args.min_inliers:
            low_inliers += 1
            # Fallback: copy refined as-is to all degree outputs.
            q = (z_full * args.scale).round().clip(0, 65535).astype(np.uint16)
            for N in todo:
                out_p = os.path.join(out_dirs[N], f"{sid}.png")
                os.makedirs(os.path.dirname(out_p), exist_ok=True)
                Image.fromarray(q, mode="I;16").save(out_p)
                fallback_count[N] += 1
            continue

        # Normalize z to [0, 1] using fixed max_depth (consistent across images).
        z_anch = np.clip(z_full[valid], 0.0, args.max_depth) / args.max_depth
        d_anch = gt[valid]
        z_full_n = z_full / args.max_depth

        # Fit each requested degree.
        for N in todo:
            cur_deg = N
            coeffs = None
            info = None
            while cur_deg >= args.mono_min_degree:
                coeffs, info = fit_poly_irls(
                    z_anch, d_anch,
                    degree=cur_deg,
                    ridge_lambda=args.ridge_lambda,
                    trunc=args.trunc,
                    n_iters=args.n_iters,
                )
                if not args.mono_check or check_monotonic(coeffs):
                    break
                cur_deg -= 1
            if cur_deg < args.mono_min_degree:
                # Fallback to identity.
                refined = z_full
                fallback_count[N] += 1
            else:
                refined = apply_poly(z_full_n, coeffs, z_clip).astype(np.float32)
                if cur_deg < N:
                    fallback_count[N] += 1

            refined = np.clip(refined, 0.0, args.max_depth)
            refined = np.where(np.isfinite(refined), refined, 0.0)
            q = (refined * args.scale).round().clip(0, 65535).astype(np.uint16)
            out_p = os.path.join(out_dirs[N], f"{sid}.png")
            os.makedirs(os.path.dirname(out_p), exist_ok=True)
            Image.fromarray(q, mode="I;16").save(out_p)
            saved[N] += 1

            if args.save_coeffs and coeffs_log[N] is not None:
                coeffs_log[N].write(json.dumps({
                    "sid": sid,
                    "degree_used": cur_deg,
                    "coeffs": coeffs.tolist() if coeffs is not None else None,
                    "inlier_frac": info["inlier_frac"] if info else None,
                    "n_anchors": n_anch,
                }) + "\n")

    if args.save_coeffs:
        for f in coeffs_log.values():
            if f is not None:
                f.close()

    print(f"[done] missing={missing} low_inliers={low_inliers}")
    for N in degrees:
        print(f"  poly{N}: saved={saved[N]} skipped={skipped[N]} "
              f"fallback={fallback_count[N]}")


if __name__ == "__main__":
    main()
