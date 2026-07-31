"""Upgrade sparse LiDAR GT by filling between-stripe gaps with
dense_gt-shape anchored to LiDAR values via residual interpolation.

For each sample:
    r(y, x)  = lidar_gt(y, x) − dense_gt(y, x)        at LiDAR pixels
    r_full   = interpolate(r) over the whole image     (Delaunay-linear)
    new_gt   = dense_gt + r_full

Properties:
  • At LiDAR pixels: new_gt = lidar_gt (exact metric anchor preserved).
  • At empty pixels: new_gt follows dense_gt's shape but is shifted by
    a smooth blend of nearby LiDAR-residuals → metric scale tracks LiDAR
    while local shape stays dense_gt-clean (no horizontal stripes).
  • Outside the LiDAR convex hull (sky, image corners), r_full → 0
    (nearest-neighbor extrap) so new_gt = dense_gt there.

Usage:
    python3 scripts/build_lidar_filled_gt.py \\
      --split /data/public/nuScenes/derived/splits/test.txt \\
      --data_root /data/public/nuScenes/derived \\
      --dense_dir depth_refined_grid \\
      --lidar_dir depth_lidar \\
      --out_dir  /data/public/nuScenes/derived/depth_lidar_filled
"""
import argparse
import os
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.interpolate import griddata
from scipy.ndimage import distance_transform_edt
from tqdm import tqdm


def iter_split(split_file):
    with open(split_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield line.split("\t")[0]


def fill_residual(residual: np.ndarray, valid: np.ndarray,
                  method: str = "linear",
                  max_fill_dist: float = 10.0):
    """Interpolate sparse residual over the (H, W) grid, restricted to a
    band around LiDAR anchors.

    Returns (filled_residual, fill_mask) where fill_mask is True where the
    residual is trustworthy (inside the LiDAR-vicinity band).

    method:
      • "linear"  : Delaunay-triangulation linear interpolation inside hull,
                    NN outside.
      • "nearest" : nearest-LiDAR-residual everywhere (cheap, blocky).
    """
    H, W = residual.shape
    ys, xs = np.where(valid)
    if ys.size < 3:
        return np.zeros_like(residual), np.zeros_like(valid)

    if method == "nearest":
        out = _nearest_fill(residual, valid)
    else:
        grid_y, grid_x = np.mgrid[0:H, 0:W]
        points = np.column_stack([ys, xs])
        vals = residual[ys, xs]
        lin = griddata(points, vals, (grid_y, grid_x),
                       method="linear", fill_value=np.nan)
        nan_mask = ~np.isfinite(lin)
        if nan_mask.any():
            nn = _nearest_fill(residual, valid)
            lin[nan_mask] = nn[nan_mask]
        out = lin.astype(np.float32)

    if max_fill_dist <= 0:
        return out, np.ones_like(valid)

    # Restrict fill to pixels within `max_fill_dist` of any LiDAR point.
    dist = distance_transform_edt(~valid)
    fill_mask = dist <= max_fill_dist
    out = np.where(fill_mask, out, 0.0).astype(np.float32)
    return out, fill_mask


def _nearest_fill(residual: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Each empty pixel inherits the residual of the nearest LiDAR pixel."""
    _, nn_idx = distance_transform_edt(~valid, return_indices=True)
    return residual[nn_idx[0], nn_idx[1]].astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--dense_dir", default="depth_refined_grid")
    ap.add_argument("--lidar_dir", default="depth_lidar")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--method", default="nearest",
                    choices=["nearest", "linear"],
                    help="`nearest`: each LiDAR point shifts only its own "
                         "Voronoi cell (within max_fill_dist) by its residual "
                         "— no interpolation between distinct points. "
                         "`linear`: Delaunay-linear interp across LiDAR points "
                         "(can mix residuals from non-adjacent points).")
    ap.add_argument("--max_fill_dist", type=float, default=20.0,
                    help="Pixels: only fill within this distance of a LiDAR "
                         "point. Outside, output = 0 (invalid → no supervision). "
                         "Set <=0 to fill the entire image with dense_gt.")
    ap.add_argument("--residual_clip", type=float, default=3.0,
                    help="Clip residual to ±this (m) before propagation. "
                         "Safety net for pathological K×K affine misfits — "
                         "a single LiDAR point with residual=-20m would otherwise "
                         "shift its entire neighborhood by -20m.")
    ap.add_argument("--scale", type=float, default=256.0)
    ap.add_argument("--min_depth", type=float, default=0.5)
    ap.add_argument("--max_depth", type=float, default=100.0)
    ap.add_argument("--min_anchors", type=int, default=50)
    ap.add_argument("--skip_existing", action="store_true", default=True)
    ap.add_argument("--no_skip_existing", dest="skip_existing", action="store_false")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    sids = list(iter_split(args.split))
    if args.limit:
        sids = sids[: args.limit]
    print(f"[fill] {len(sids)} samples  method={args.method}")
    print(f"  dense_dir = {args.dense_dir}")
    print(f"  lidar_dir = {args.lidar_dir}")
    print(f"  out_dir   = {args.out_dir}")

    saved = skipped = missing = low_anchor = 0
    for sid in tqdm(sids):
        out_p = os.path.join(args.out_dir, f"{sid}.png")
        os.makedirs(os.path.dirname(out_p), exist_ok=True)
        if args.skip_existing and os.path.exists(out_p):
            skipped += 1
            continue
        dense_p = os.path.join(args.data_root, args.dense_dir, f"{sid}.png")
        lidar_p = os.path.join(args.data_root, args.lidar_dir, f"{sid}.png")
        if not (os.path.exists(dense_p) and os.path.exists(lidar_p)):
            missing += 1
            continue
        dense = np.asarray(Image.open(dense_p), dtype=np.float32) / args.scale
        lidar = np.asarray(Image.open(lidar_p), dtype=np.float32) / args.scale
        valid = (lidar > args.min_depth) & (lidar < args.max_depth) \
            & (dense > args.min_depth) & (dense < args.max_depth)
        n = int(valid.sum())
        if n < args.min_anchors:
            low_anchor += 1
            # Not enough anchors — emit all-zeros (invalid) to skip this
            # sample in any downstream L1 supervision.
            out = np.zeros_like(dense)
        else:
            # Residual for propagation: clip to keep bad far-range fits
            # from polluting their LiDAR's Voronoi cell.
            residual = np.zeros_like(dense)
            residual[valid] = np.clip(lidar[valid] - dense[valid],
                                      -args.residual_clip, args.residual_clip)
            r_full, fill_mask = fill_residual(residual, valid,
                                              method=args.method,
                                              max_fill_dist=args.max_fill_dist)
            # Inside fill band: dense + nearest-LiDAR-residual
            # Outside fill band: 0 (no supervision)
            out = np.zeros_like(dense)
            out[fill_mask] = (dense[fill_mask] + r_full[fill_mask]).clip(0.0, args.max_depth)
            # Restore EXACT lidar at EVERY LiDAR pixel — these are always
            # inside the fill band by construction.
            lidar_present = (lidar > args.min_depth) & (lidar < args.max_depth)
            out[lidar_present] = lidar[lidar_present]

        q = (np.clip(out, 0.0, 65535.0 / args.scale) * args.scale)\
            .round().clip(0, 65535).astype(np.uint16)
        Image.fromarray(q, mode="I;16").save(out_p)
        saved += 1

    print(f"[done] saved={saved} skipped={skipped} missing={missing} "
          f"low_anchor={low_anchor}")


if __name__ == "__main__":
    main()
