"""Depth-Anything-V2 + MoGe-style ROE refinement on the nuScenes test split.

Pipeline (per image):
  1. Run DA-v2 to get a relative inverse-depth-like map (higher = closer).
  2. Read sparse LiDAR GT (depth_lidar uint16 PNG ×256).
  3. Solve a robust affine alignment in DISPARITY space using MoGe-2's
     ROE solver (truncated weighted L1):
         min_{a, b}  Σ_i min(trunc, w_i · |a · pred_disp_i + b − 1/gt_z_i|)
     with w_i = 1/(1/gt_z_i) = gt_z_i  → weighted residual is the relative
     disp error, mirroring MoGe's training-time choice (trunc=1.0).
  4. refined_disp = a · pred_disp + b   (full image)
  5. refined_depth = 1 / clamp(refined_disp, 1/max_depth, inf)
  6. Save as uint16 PNG ×256 under <out_root>/<sample_id>.png.

Usage:
    CUDA_VISIBLE_DEVICES=3 python scripts/infer_da_v2_refined.py \\
      --split /data/public/nuScenes/derived/splits/test.txt \\
      --data_root /data/public/nuScenes/derived \\
      --out_root /data/public/nuScenes/derived/depth_refined \\
      --encoder vitl
"""
import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

# Allow importing both DA-v2 and MoGe utils
_DA_ROOT = "/workspace/Depth-Anything-V2"
_MOGE_ROOT = "/workspace/MoGe"
for p in (_DA_ROOT, _MOGE_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

from depth_anything_v2.dpt import DepthAnythingV2          # noqa: E402
from moge.utils.alignment import align_depth_affine        # noqa: E402

DA_V2_CONFIGS = {
    "vits": dict(encoder="vits", features=64,  out_channels=[48, 96, 192, 384]),
    "vitb": dict(encoder="vitb", features=128, out_channels=[96, 192, 384, 768]),
    "vitl": dict(encoder="vitl", features=256, out_channels=[256, 512, 1024, 1024]),
}


def load_da_v2(encoder: str, device: torch.device) -> DepthAnythingV2:
    from huggingface_hub import hf_hub_download
    repo = {
        "vits": "depth-anything/Depth-Anything-V2-Small",
        "vitb": "depth-anything/Depth-Anything-V2-Base",
        "vitl": "depth-anything/Depth-Anything-V2-Large",
    }[encoder]
    ckpt_file = hf_hub_download(repo_id=repo, filename=f"depth_anything_v2_{encoder}.pth")
    model = DepthAnythingV2(**DA_V2_CONFIGS[encoder])
    model.load_state_dict(torch.load(ckpt_file, map_location="cpu"))
    return model.to(device).eval()


def iter_split(split_file):
    with open(split_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            sid = parts[0]
            img = parts[1] if len(parts) > 1 else None
            yield sid, img


def refine_with_roe(pred_disp: np.ndarray, gt_depth: np.ndarray,
                    valid: np.ndarray, trunc: float, eps: float,
                    min_inliers: int) -> tuple[np.ndarray, dict]:
    """Solve scale*pred_disp + shift ≈ 1/gt_depth via MoGe ROE.

    Returns refined_depth (H, W) and a stats dict.
    """
    H, W = pred_disp.shape
    if int(valid.sum()) < min_inliers:
        # Not enough anchor points — fall back to identity (saved as 0s).
        return np.zeros_like(pred_disp, dtype=np.float32), {
            "ok": False, "n_anchors": int(valid.sum()),
        }

    src = pred_disp[valid].astype(np.float32)                  # (N,)
    tgt = (1.0 / gt_depth[valid].astype(np.float32))           # (N,) true disparity
    weight = (1.0 / np.clip(tgt, eps, None)).astype(np.float32)  # = gt_depth_clipped

    src_t = torch.from_numpy(src)[None]                        # (1, N)
    tgt_t = torch.from_numpy(tgt)[None]
    w_t = torch.from_numpy(weight)[None]

    with torch.no_grad():
        scale, shift = align_depth_affine(src_t, tgt_t, w_t, trunc=float(trunc))
    a = float(scale[0].item())
    b = float(shift[0].item())

    refined_disp = a * pred_disp + b                           # (H, W)
    # Clamp to a positive disparity floor so 1/refined_disp ≤ max_depth.
    return refined_disp, {"ok": True, "scale": a, "shift": b,
                          "n_anchors": int(valid.sum())}


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split", required=True)
    p.add_argument("--data_root", required=True,
                   help="nuScenes derived root (with depth_lidar/).")
    p.add_argument("--out_root", required=True)
    p.add_argument("--gt_dir", default="depth_lidar")
    p.add_argument("--encoder", default="vitl", choices=list(DA_V2_CONFIGS))
    p.add_argument("--input_size", type=int, default=518)
    p.add_argument("--min_depth", type=float, default=0.5,
                   help="lower clip for LiDAR GT (m) when building target_disp.")
    p.add_argument("--max_depth_m", type=float, default=None,
                   help="depth clip before encoding (default 65535/256 ≈ 256m).")
    p.add_argument("--trunc", type=float, default=1.0,
                   help="ROE truncation in weighted-residual (relative) space.")
    p.add_argument("--min_inliers", type=int, default=50)
    p.add_argument("--eps", type=float, default=1e-6)
    p.add_argument("--scale", type=float, default=256.0)
    p.add_argument("--fp16", action="store_true", default=True)
    p.add_argument("--no_fp16", dest="fp16", action="store_false")
    p.add_argument("--skip_existing", action="store_true", default=True)
    p.add_argument("--no_skip_existing", dest="skip_existing", action="store_false")
    p.add_argument("--limit", type=int, default=0)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_da_v2(args.encoder, device)
    if args.fp16:
        model.half()
    max_m = args.max_depth_m if args.max_depth_m is not None else 65535.0 / args.scale
    disp_floor = 1.0 / max_m   # any predicted disp below this maps to max_m

    items = list(iter_split(args.split))
    if args.limit:
        items = items[:args.limit]
    print(f"[infer_da_v2_refined] {len(items)} samples → {args.out_root}  "
          f"(encoder={args.encoder} trunc={args.trunc})")

    bad_count = saved = skipped = missing = 0
    for sid, image_path in tqdm(items):
        out_path = os.path.join(args.out_root, f"{sid}.png")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        if args.skip_existing and os.path.exists(out_path):
            skipped += 1
            continue
        if not image_path or not os.path.exists(image_path):
            print(f"[warn] missing image: {image_path}", file=sys.stderr)
            missing += 1
            continue
        gt_path = os.path.join(args.data_root, args.gt_dir, f"{sid}.png")
        if not os.path.exists(gt_path):
            print(f"[warn] missing GT lidar: {gt_path}", file=sys.stderr)
            missing += 1
            continue

        # --- 1) DA-v2 inference ----------------------------------------------
        bgr = cv2.imread(image_path)
        if bgr is None:
            print(f"[warn] cv2 failed to read {image_path}", file=sys.stderr)
            missing += 1
            continue
        H, W = bgr.shape[:2]
        with torch.amp.autocast("cuda", enabled=args.fp16):
            pred_disp = model.infer_image(bgr, input_size=args.input_size)  # (H, W) float32
        pred_disp = pred_disp.astype(np.float32)
        if pred_disp.shape != (H, W):
            raise RuntimeError(f"DA-v2 output shape {pred_disp.shape} ≠ image {(H,W)}")

        # --- 2) Load LiDAR GT ------------------------------------------------
        gt = np.asarray(Image.open(gt_path), dtype=np.float32) / args.scale  # (H, W) m
        valid = (gt > args.min_depth) & (gt < max_m) & np.isfinite(gt) & np.isfinite(pred_disp)

        # --- 3) ROE alignment in disparity space -----------------------------
        refined_disp, info = refine_with_roe(
            pred_disp, gt, valid,
            trunc=args.trunc, eps=args.eps, min_inliers=args.min_inliers,
        )
        if not info["ok"]:
            bad_count += 1

        # --- 4) Disp → metric depth ------------------------------------------
        refined_depth = 1.0 / np.clip(refined_disp, disp_floor, None)
        refined_depth = np.where(np.isfinite(refined_depth), refined_depth, 0.0)
        refined_depth = np.clip(refined_depth, 0.0, max_m)
        q = (refined_depth * args.scale).round().clip(0, 65535).astype(np.uint16)
        Image.fromarray(q, mode="I;16").save(out_path)
        saved += 1

    print(f"[done] saved={saved} skipped={skipped} missing={missing} "
          f"bad_align={bad_count}")


if __name__ == "__main__":
    main()
