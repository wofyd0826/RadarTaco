#!/usr/bin/env python3
"""Per-pixel POOLED depth metrics for direct comparison with paper-style numbers.

Same checkpoint / dataset / model / valid-mask logic as scripts/eval.py, but
the aggregator accumulates raw `|pred - gt|` and pixel counts across the
whole split and reports

    MAE_pooled  = sum_all_pixels(|p - g|) / sum_all_pixels(1)
    RMSE_pooled = sqrt(sum_all_pixels((p - g)^2) / sum_all_pixels(1))
    Rel_pooled  = sum_all_pixels(|p - g| / g) / sum_all_pixels(1)
    delta1      = sum_all_pixels(thresh < 1.25) / sum_all_pixels(1)

Range bins, pred clip range [1e-3, max_depth], and `valid_mask & (gt in [d_min, d_max))`
are identical to scripts/eval.py so the only difference vs. existing
`eval_test/summary.txt` is per-frame-mean → per-pixel-pool aggregation.

Usage:
    python scripts/eval_pooled.py checkpoint=output/baseline/best.pt
"""
import json
import logging
import os
import sys

import hydra
import numpy as np
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from tqdm import tqdm

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from src.dataset.nuscenes import NuScenesRadarDepthDataset                   # noqa: E402
from src.evaluation.metrics import (RANGE_BINS_OVERALL, RANGE_BINS_FINE,     # noqa: E402
                                    RANGE_BINS_FAR)
from src.model.radartaco import RadarTaco                                    # noqa: E402
from src.model.rgb_only import RGBOnlyDepth                                  # noqa: E402


def _maybe_override_model_from_ckpt(cfg) -> None:
    """If the checkpoint sits next to a saved `config.yaml`, override
    `cfg.model` with the one used to train the checkpoint."""
    sibling = os.path.join(os.path.dirname(cfg.checkpoint), "config.yaml")
    if not os.path.exists(sibling):
        return
    saved = OmegaConf.load(sibling)
    saved_model = saved.get("model", None)
    if saved_model is None:
        return
    cur_name = str(cfg.model.get("name", "radartaco")).lower()
    saved_name = str(saved_model.get("name", "radartaco")).lower()
    if cur_name == saved_name:
        return
    logger.info(f"auto-overriding cfg.model from {sibling}: "
                f"{cur_name} -> {saved_name}")
    OmegaConf.set_struct(cfg, False)
    cfg.model = saved_model
    OmegaConf.set_struct(cfg, True)


def _build_eval_model(cfg, device):
    name = str(cfg.model.get("name", "radartaco")).lower()
    if name == "rgb_only":
        m = RGBOnlyDepth(
            max_depth=float(cfg.dataset.max_depth),
            pretrained_image_encoder=False,
            output_mode=str(cfg.model.get("output_mode", "metric")),
            min_depth_clip=float(cfg.model.get("min_depth_clip", 0.5)),
            multi_scale=bool(cfg.model.get("multi_scale", False)),
            multi_scale_levels=tuple(cfg.model.get("multi_scale_levels", (2, 4, 8, 16))),
        )
    else:
        m = RadarTaco(
            radar_encoder_name=cfg.model.radar_encoder,
            max_depth=float(cfg.dataset.max_depth),
            max_radar_points=int(cfg.dataset.max_radar_points),
            k_neighbors=int(cfg.model.k_neighbors),
            a_l=tuple(cfg.model.a_l),
            radar_channels=tuple(cfg.model.radar_channels),
            attn_heads=int(cfg.model.attn_heads),
            mlp_hidden=int(cfg.model.get("mlp_hidden", 128)),
            pretrained_image_encoder=False,
            output_mode=str(cfg.model.get("output_mode", "metric")),
            min_depth_clip=float(cfg.model.get("min_depth_clip", 0.5)),
            multi_scale=bool(cfg.model.get("multi_scale", False)),
            multi_scale_levels=tuple(cfg.model.get("multi_scale_levels", (2, 4, 8, 16))),
        )
    return m.to(device).eval()

logging.basicConfig(format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                    level=logging.INFO)
logger = logging.getLogger("radartaco.eval_pooled")


class PooledAccumulator:
    """Streaming accumulator: keeps Σ|err|, Σerr², Σ|err|/g, Σ(thresh<τ), N per bin."""

    def __init__(self, range_bins):
        self.range_bins = list(range_bins)
        self.sum_abs = {f"{a}-{b}m": 0.0 for a, b in self.range_bins}
        self.sum_sq = {f"{a}-{b}m": 0.0 for a, b in self.range_bins}
        self.sum_rel = {f"{a}-{b}m": 0.0 for a, b in self.range_bins}
        self.sum_d1 = {f"{a}-{b}m": 0 for a, b in self.range_bins}
        self.sum_d2 = {f"{a}-{b}m": 0 for a, b in self.range_bins}
        self.sum_d3 = {f"{a}-{b}m": 0 for a, b in self.range_bins}
        self.count = {f"{a}-{b}m": 0 for a, b in self.range_bins}

    def add(self, pred, gt, valid_mask):
        # `pred`, `gt`, `valid_mask` are already-aligned 1D arrays of valid pixels.
        for a, b in self.range_bins:
            key = f"{a}-{b}m"
            sel = valid_mask & (gt >= a) & (gt < b)
            if not sel.any():
                continue
            p = pred[sel]
            g = gt[sel]
            err = p - g
            abs_err = np.abs(err)
            self.sum_abs[key] += abs_err.sum()
            self.sum_sq[key] += (err ** 2).sum()
            self.sum_rel[key] += (abs_err / g).sum()
            thresh = np.maximum(p / g, g / p)
            self.sum_d1[key] += int((thresh < 1.25).sum())
            self.sum_d2[key] += int((thresh < 1.25 ** 2).sum())
            self.sum_d3[key] += int((thresh < 1.25 ** 3).sum())
            self.count[key] += int(sel.sum())

    def finalize(self):
        out = {}
        for a, b in self.range_bins:
            key = f"{a}-{b}m"
            n = self.count[key]
            if n == 0:
                out[key] = {"mae": float("nan"), "rmse": float("nan"),
                            "rel": float("nan"), "delta1": float("nan"),
                            "delta2": float("nan"), "delta3": float("nan"),
                            "n_pixels": 0}
                continue
            out[key] = {
                "mae": self.sum_abs[key] / n,
                "rmse": float(np.sqrt(self.sum_sq[key] / n)),
                "rel": self.sum_rel[key] / n,
                "delta1": self.sum_d1[key] / n,
                "delta2": self.sum_d2[key] / n,
                "delta3": self.sum_d3[key] / n,
                "n_pixels": n,
            }
        return out


@hydra.main(version_base="1.3", config_path="../config", config_name="config")
def main(cfg: DictConfig) -> None:
    if not cfg.get("checkpoint"):
        raise SystemExit("checkpoint=<path/to/.pt> is required (Hydra override).")
    _maybe_override_model_from_ckpt(cfg)
    out_dir = cfg.get("eval_out_dir", os.path.join(os.path.dirname(cfg.checkpoint), "eval_pooled"))
    os.makedirs(out_dir, exist_ok=True)

    split = cfg.get("eval_split", "test")
    split_file = getattr(cfg.dataset, f"split_{split}", None) or \
        os.path.join(cfg.dataset.data_root, "splits", f"{split}.txt")
    if not os.path.exists(split_file):
        raise SystemExit(f"split file not found: {split_file}")
    logger.info(f"eval split: {split} → {split_file}")

    ds = NuScenesRadarDepthDataset(
        data_root=cfg.dataset.data_root,
        split_file=split_file,
        dense_gt_dir=cfg.dataset.get("dense_gt_dir", "depth_acc"),
        radar_3d_dir=cfg.dataset.get("radar_3d_dir", "radar_3d"),
        night_ids_file=cfg.dataset.get("night_ids_file", None),
        max_radar_points=int(cfg.dataset.max_radar_points),
        max_depth=float(cfg.dataset.max_depth),
        min_depth=float(cfg.dataset.min_depth),
        resize_to_hw=None, augmentation=False,
    )
    loader = DataLoader(ds, batch_size=1, shuffle=False,
                        num_workers=int(cfg.dataset.num_workers), pin_memory=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = _build_eval_model(cfg, device)
    ckpt = torch.load(cfg.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"], strict=True)
    logger.info(f"loaded {cfg.checkpoint} (epoch={ckpt.get('epoch', '?')})")

    min_depth = float(cfg.dataset.min_depth)
    max_depth = float(cfg.dataset.max_depth)

    overall = PooledAccumulator(RANGE_BINS_OVERALL)
    fine = PooledAccumulator(RANGE_BINS_FINE)
    far = PooledAccumulator(RANGE_BINS_FAR)
    day = PooledAccumulator(RANGE_BINS_OVERALL)
    night = PooledAccumulator(RANGE_BINS_OVERALL)

    n_samples = 0
    n_night = 0

    for batch in tqdm(loader):
        batch_dev = {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
                     for k, v in batch.items()}
        with torch.inference_mode():
            pred = model(batch_dev["rgb_norm"], batch_dev["radar_points"], batch_dev["radar_mask"])
        pn = pred[0, 0].float().cpu().numpy()
        gn = batch["depth_gt_lidar"][0, 0].cpu().numpy()
        mn = batch["valid_mask_lidar"][0, 0].cpu().numpy().astype(bool)
        is_n = bool(batch["is_night"][0].item()) if torch.is_tensor(batch["is_night"]) else False

        # Same clipping convention as scripts/evaluation/metrics.py:compute_range_metrics.
        pn = np.clip(pn, min_depth, max_depth)
        pn[np.isinf(pn)] = max_depth
        pn[np.isnan(pn)] = min_depth
        gn_safe = np.clip(gn, 1e-3, None)            # avoid div-by-zero in rel/thresh

        overall.add(pn, gn_safe, mn)
        fine.add(pn, gn_safe, mn)
        far.add(pn, gn_safe, mn)
        if is_n:
            night.add(pn, gn_safe, mn)
            n_night += 1
        else:
            day.add(pn, gn_safe, mn)
        n_samples += 1

    agg = {
        "overall": overall.finalize(),
        "per_range": fine.finalize(),
        "far": far.finalize(),
        "day": day.finalize(),
        "night": night.finalize(),
        "n_samples": n_samples,
        "n_night": n_night,
    }
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(agg, f, indent=2)
    with open(os.path.join(out_dir, "summary.txt"), "w") as f:
        for category in ("overall", "per_range", "far", "day", "night"):
            f.write(f"=== {category} (POOLED) ===\n")
            for rname, m in agg[category].items():
                line = "  " + rname + "  " + "  ".join(
                    f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                    for k, v in m.items()
                )
                f.write(line + "\n")
            f.write("\n")
        f.write(f"n_samples={n_samples}  n_night={n_night}\n")

    head = agg["overall"]["0-80m"]
    logger.info(
        f"POOLED 0-80m  MAE={head['mae']:.4f}  RMSE={head['rmse']:.4f}  "
        f"rel={head['rel']:.4f}  d1={head['delta1']:.4f}  n_pix={head['n_pixels']}"
    )
    logger.info(f"wrote {os.path.join(out_dir, 'summary.txt')}")


if __name__ == "__main__":
    main()
