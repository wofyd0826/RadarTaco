#!/usr/bin/env python3
"""Precompute initial relative depth `D*` for the plug-in branch.

For every image in a split, runs a frozen mono depth predictor (default:
DPT-Hybrid via HuggingFace transformers) and saves the prediction as a
uint16 PNG under `<data_root>/<rel_depth_dir>/<sample_id>.png` with the
convention `pixel_value = clip(rel_depth, 0, 65.535) * 1000`.

The dataset loader reads these PNGs at training time (see
`BaseRadarDepthDataset._load_rel_depth`).

Usage:
    # nuScenes
    python scripts/precompute_relative_depth.py \\
        dataset=nuscenes \\
        +split_file=/data/public/nuScenes/derived/splits/train.txt \\
        +out_dir=/data/public/nuScenes/derived/relative_depth

    # ZJU-4DRadarCam
    python scripts/precompute_relative_depth.py \\
        +dataset_kind=zju \\
        +split_file=/data/public/ZJU-4DRadarCam/data/train.txt \\
        +image_root=/data/public/ZJU-4DRadarCam/data/image \\
        +out_dir=/data/public/ZJU-4DRadarCam/data/relative_depth

    # Pick a different mono model
    python scripts/precompute_relative_depth.py ... +model=Intel/dpt-large

Notes:
- DPT-Hybrid is paper Table 4's default plug-in predictor and is the
  closest mainstream proxy available via HF Hub. Depth-Anything-v2
  ("LiheYoung/depth-anything-v2-small-hf") is a stronger modern option;
  paper Table 4 shows it yields the lowest plug-in MAE.
- The model is run on whole input images at native resolution (no tiling).
  The HF DPTImageProcessor resizes internally and the output is upscaled
  back to image native HxW before saving.
- Output range: DPT-Hybrid outputs an UNCALIBRATED inverse-depth-like
  signal — magnitudes vary across datasets. We do NOT rescale to metric
  here because the model's auxiliary branch will learn the mapping
  during training. The PNG just preserves the raw output's float range
  (clamped to [0, 65.535] so it fits in uint16 / 1000).
"""
import argparse
import logging
import os
import sys
from typing import Iterable

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

logging.basicConfig(format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                    level=logging.INFO)
logger = logging.getLogger("radartaco.precompute_relative_depth")


def _load_predictor(model_name: str, device: str):
    """Load a HuggingFace depth estimation pipeline."""
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation
    logger.info(f"loading {model_name} via HuggingFace transformers...")
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModelForDepthEstimation.from_pretrained(model_name).to(device).eval()
    return model, processor


def _iter_nuscenes(split_file: str, data_root: str) -> Iterable[tuple]:
    """Yield (sample_id, image_path) for nuScenes splits (tab-separated)."""
    with open(split_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            sample_id = parts[0]
            image_path = parts[1] if len(parts) > 1 else None
            if image_path is None:
                # Fall back to a derived path under data_root
                image_path = os.path.join(data_root, "samples", "CAM_FRONT", f"{sample_id}.jpg")
            yield sample_id, image_path


def _iter_zju(split_file: str, image_root: str) -> Iterable[tuple]:
    """Yield (sample_id, image_path) for ZJU splits (one timestamp per line)."""
    with open(split_file) as f:
        for line in f:
            ts = line.strip()
            if not ts:
                continue
            yield ts, os.path.join(image_root, f"{ts}.png")


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_kind", choices=["nuscenes", "zju"], required=True)
    p.add_argument("--split_file", required=True,
                   help="path to splits/{train,val,test}.txt")
    p.add_argument("--out_dir", required=True,
                   help="directory to write `<sample_id>.png` (will be created)")
    p.add_argument("--data_root", default=None,
                   help="(nuScenes only) data root for resolving image paths "
                        "when split_file has no path column")
    p.add_argument("--image_root", default=None,
                   help="(ZJU only) directory of `<ts>.png` images")
    p.add_argument("--model", default="Intel/dpt-hybrid-midas",
                   help="HuggingFace depth predictor (default: DPT-Hybrid)")
    p.add_argument("--device", default="cuda")
    p.add_argument("--scale", type=float, default=1000.0,
                   help="uint16 encoding multiplier; saved_pixel = float × scale")
    p.add_argument("--skip_existing", action="store_true",
                   help="don't re-predict files that already exist in out_dir")
    args = p.parse_args()

    if args.dataset_kind == "nuscenes" and args.data_root is None:
        raise SystemExit("--data_root required for nuscenes")
    if args.dataset_kind == "zju" and args.image_root is None:
        raise SystemExit("--image_root required for zju")

    os.makedirs(args.out_dir, exist_ok=True)

    device = args.device if torch.cuda.is_available() else "cpu"
    model, processor = _load_predictor(args.model, device)

    if args.dataset_kind == "nuscenes":
        items = list(_iter_nuscenes(args.split_file, args.data_root))
    else:
        items = list(_iter_zju(args.split_file, args.image_root))
    logger.info(f"{args.dataset_kind}: {len(items)} samples → {args.out_dir}")

    skipped = 0
    for sample_id, image_path in tqdm(items):
        out_path = os.path.join(args.out_dir, f"{sample_id}.png")
        # `sample_id` may contain a "/" (e.g. `scene_NNNN/<filename>` on
        # nuScenes) — make the parent directory before writing.
        parent = os.path.dirname(out_path)
        if parent and not os.path.isdir(parent):
            os.makedirs(parent, exist_ok=True)
        if args.skip_existing and os.path.exists(out_path):
            skipped += 1
            continue
        if not os.path.exists(image_path):
            logger.warning(f"missing image: {image_path}")
            continue

        image = Image.open(image_path).convert("RGB")
        W, H = image.size

        inputs = processor(images=image, return_tensors="pt").to(device)
        out = model(**inputs)
        # HF returns `predicted_depth` shape (1, h, w). Upsample to (H, W).
        depth = torch.nn.functional.interpolate(
            out.predicted_depth.unsqueeze(1),
            size=(H, W), mode="bicubic", align_corners=False,
        )[0, 0].float().cpu().numpy()

        # Normalize per-image to [0, 1] for storage stability; the aux
        # branch will learn its own scale anyway.
        d_min, d_max = float(depth.min()), float(depth.max())
        if d_max - d_min > 1e-6:
            depth = (depth - d_min) / (d_max - d_min)
        else:
            depth = np.zeros_like(depth)
        # Encode into uint16 PNG.
        q = np.clip(depth * args.scale, 0, 65535).astype(np.uint16)
        Image.fromarray(q, mode="I;16").save(out_path)

    logger.info(f"done. wrote {len(items) - skipped} new files, "
                f"{skipped} skipped (already existed).")


if __name__ == "__main__":
    main()
