#!/usr/bin/env python3
"""Per-pixel POOLED depth metrics on ZJU-4DRadarCam test split.

Same checkpoint / model / valid-mask logic as `eval_zju.py`, but uses the
streaming pooled aggregator from `eval_pooled.py` so numbers are directly
comparable to RadarCam-Depth (Li et al., ICRA'24) Table 2.

Usage:
    python scripts/eval_zju_pooled.py +checkpoint=output/<run>/best.pt
"""
import json
import logging
import os
import sys

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from src.dataset.zju import ZjuRadarDepthDataset                              # noqa: E402
from src.evaluation.metrics import (RANGE_BINS_OVERALL, RANGE_BINS_FINE,      # noqa: E402
                                    RANGE_BINS_FAR)
from src.model.radartaco import RadarTaco                                     # noqa: E402
from src.model.rgb_only import RGBOnlyDepth                                   # noqa: E402

# Reuse PooledAccumulator + the model auto-override helper from
# scripts/eval_pooled.py by importing them directly. They are
# dataset-agnostic.
from scripts.eval_pooled import (PooledAccumulator,                           # noqa: E402
                                 _maybe_override_model_from_ckpt,
                                 _build_eval_model)

logging.basicConfig(format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                    level=logging.INFO)
logger = logging.getLogger("radartaco.eval_zju_pooled")


@hydra.main(version_base="1.3", config_path="../config", config_name="config")
def main(cfg: DictConfig) -> None:
    if not cfg.get("checkpoint"):
        raise SystemExit("checkpoint=<path/to/.pt> is required (Hydra override).")
    _maybe_override_model_from_ckpt(cfg)
    eval_mode = str(cfg.get("eval_mode", "independent")).lower()
    out_dir = cfg.get("eval_out_dir",
                      os.path.join(os.path.dirname(cfg.checkpoint), f"eval_zju_pooled_{eval_mode}"))
    os.makedirs(out_dir, exist_ok=True)

    eval_split = cfg.get("eval_split", "test")
    zju_cfg = cfg.dataset.get("zju", None)
    if zju_cfg is not None:
        zju_root = zju_cfg.get("data_root", "/data/public/ZJU-4DRadarCam/data")
        split_file = zju_cfg.get(f"split_{eval_split}", None)
    else:
        zju_root = cfg.get("zju_data_root", "/data/public/ZJU-4DRadarCam/data")
        split_file = None
    if not split_file:
        split_file = os.path.join(zju_root, f"{eval_split}.txt")
    if not os.path.exists(split_file):
        raise SystemExit(f"ZJU split file not found: {split_file}")
    logger.info(f"ZJU eval split: {eval_split} → {split_file}")

    use_rel_depth = (eval_mode == "plugin")
    ds = ZjuRadarDepthDataset(
        data_root=zju_root,
        split_file=split_file,
        image_dir=(zju_cfg.get("image_dir", "image") if zju_cfg else "image"),
        radar_dir=(zju_cfg.get("radar_dir", "radar") if zju_cfg else "radar"),
        gt_dir=(zju_cfg.get("gt_dir", "gt") if zju_cfg else "gt"),
        gt_interp_dir=(zju_cfg.get("gt_interp_dir", "gt_interp") if zju_cfg else "gt_interp"),
        rel_depth_dir=(zju_cfg.get("rel_depth_dir", None) if zju_cfg and use_rel_depth else None),
        rel_depth_dropout_prob=0.0,
        max_radar_points=int(cfg.dataset.get("max_radar_points", 128)),
        max_depth=float(cfg.dataset.get("max_depth", 100.0)),
        min_depth=float(cfg.dataset.get("min_depth", 1e-3)),
        resize_to_hw=None, augmentation=False,
    )
    loader = DataLoader(ds, batch_size=1, shuffle=False,
                        num_workers=int(cfg.dataset.get("num_workers", 4)),
                        pin_memory=True)
    logger.info(f"eval mode: {eval_mode}  (rel_depth_dir used: {use_rel_depth})")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = _build_eval_model(cfg, device)
    ckpt = torch.load(cfg.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"], strict=True)
    logger.info(f"loaded {cfg.checkpoint} (epoch={ckpt.get('epoch', '?')})")

    min_depth = float(cfg.dataset.get("min_depth", 1e-3))
    max_depth = float(cfg.dataset.get("max_depth", 100.0))

    overall = PooledAccumulator(RANGE_BINS_OVERALL)
    fine = PooledAccumulator(RANGE_BINS_FINE)
    far = PooledAccumulator(RANGE_BINS_FAR)

    n_samples = 0
    for batch in tqdm(loader):
        batch_dev = {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
                     for k, v in batch.items()}
        with torch.inference_mode():
            pred = model(batch_dev["rgb_norm"], batch_dev["radar_points"], batch_dev["radar_mask"],
                         batch_dev.get("rel_depth"))
        pn = pred[0, 0].float().cpu().numpy()
        gn = batch["depth_gt_lidar"][0, 0].cpu().numpy()
        mn = batch["valid_mask_lidar"][0, 0].cpu().numpy().astype(bool)
        pn = np.clip(pn, min_depth, max_depth)
        pn[np.isinf(pn)] = max_depth
        pn[np.isnan(pn)] = min_depth
        gn_safe = np.clip(gn, 1e-3, None)
        overall.add(pn, gn_safe, mn)
        fine.add(pn, gn_safe, mn)
        far.add(pn, gn_safe, mn)
        n_samples += 1

    agg = {
        "overall": overall.finalize(),
        "per_range": fine.finalize(),
        "far": far.finalize(),
        "n_samples": n_samples,
    }
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(agg, f, indent=2)
    with open(os.path.join(out_dir, "summary.txt"), "w") as f:
        for category in ("overall", "per_range", "far"):
            f.write(f"=== {category} (POOLED, ZJU) ===\n")
            for rname, m in agg[category].items():
                line = "  " + rname + "  " + "  ".join(
                    f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                    for k, v in m.items()
                )
                f.write(line + "\n")
            f.write("\n")
        f.write(f"n_samples={n_samples}\n")

    head = agg["overall"]["0-80m"]
    logger.info(
        f"POOLED ZJU 0-80m  MAE={head['mae']:.4f}  RMSE={head['rmse']:.4f}  "
        f"rel={head['rel']:.4f}  d1={head['delta1']:.4f}  n_pix={head['n_pixels']}"
    )
    logger.info(f"wrote {os.path.join(out_dir, 'summary.txt')}")


if __name__ == "__main__":
    main()
