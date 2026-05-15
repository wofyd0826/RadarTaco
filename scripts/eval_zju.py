#!/usr/bin/env python3
"""Per-frame depth evaluation on the ZJU-4DRadarCam test split.

Mirrors `scripts/eval.py` but loads `ZjuRadarDepthDataset` instead of
`NuScenesRadarDepthDataset`. Same DepthEvaluator + per-frame aggregation
so numbers are directly comparable to the nuScenes-side `eval.py` output.

Usage:
    python scripts/eval_zju.py +checkpoint=output/<run>/best.pt
    python scripts/eval_zju.py +checkpoint=... eval_split=val
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
from src.evaluation.metrics import DepthEvaluator                              # noqa: E402
from src.model.radartaco import RadarTaco                                      # noqa: E402
from src.model.rgb_only import RGBOnlyDepth                                    # noqa: E402

logging.basicConfig(format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                    level=logging.INFO)
logger = logging.getLogger("radartaco.eval_zju")


def _maybe_override_model_from_ckpt(cfg) -> None:
    """Restore `cfg.model` from the `config.yaml` saved beside the
    checkpoint. ZJU plug-in eval still needs `cfg.dataset.zju.rel_depth_dir`
    set explicitly (D* must be precomputed on ZJU images first)."""
    sibling = os.path.join(os.path.dirname(cfg.checkpoint), "config.yaml")
    if not os.path.exists(sibling):
        return
    saved = OmegaConf.load(sibling)
    saved_model = saved.get("model", None)
    if saved_model is None:
        return
    logger.info(f"restoring cfg.model from {sibling}")
    OmegaConf.set_struct(cfg, False)
    cfg.model = saved_model
    OmegaConf.set_struct(cfg, True)


def _build_eval_model(cfg, device):
    name = str(cfg.model.get("name", "radartaco")).lower()
    max_depth = float(cfg.dataset.get("max_depth", 100.0))
    if name == "rgb_only":
        m = RGBOnlyDepth(
            max_depth=max_depth,
            pretrained_image_encoder=False,
            output_mode=str(cfg.model.get("output_mode", "metric")),
            min_depth_clip=float(cfg.model.get("min_depth_clip", 0.5)),
            multi_scale=bool(cfg.model.get("multi_scale", False)),
            multi_scale_levels=tuple(cfg.model.get("multi_scale_levels", (2, 4, 8, 16))),
        )
    else:
        m = RadarTaco(
            radar_encoder_name=cfg.model.radar_encoder,
            max_depth=max_depth,
            max_radar_points=int(cfg.dataset.get("max_radar_points", 128)),
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
            use_aux_branch=bool(cfg.model.get("use_aux_branch", False)),
        )
    return m.to(device).eval()


@hydra.main(version_base="1.3", config_path="../config", config_name="config")
def main(cfg: DictConfig) -> None:
    if not cfg.get("checkpoint"):
        raise SystemExit("checkpoint=<path/to/.pt> is required (Hydra override).")
    _maybe_override_model_from_ckpt(cfg)
    eval_mode = str(cfg.get("eval_mode", "independent")).lower()
    out_dir = cfg.get("eval_out_dir",
                      os.path.join(os.path.dirname(cfg.checkpoint), f"eval_zju_{eval_mode}"))
    os.makedirs(out_dir, exist_ok=True)

    # Resolve ZJU split file. Prefer cfg.dataset.zju.split_<eval_split>;
    # fall back to a sensible default path.
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

    evaluator = DepthEvaluator(min_depth=float(cfg.dataset.get("min_depth", 1e-3)),
                               max_depth=float(cfg.dataset.get("max_depth", 100.0)))
    all_metrics = []
    per_sample_rows = []

    for batch in tqdm(loader):
        batch = {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
                 for k, v in batch.items()}
        with torch.inference_mode():
            pred = model(batch["rgb_norm"], batch["radar_points"], batch["radar_mask"],
                         batch.get("rel_depth"))
        pn = pred[0, 0].float().cpu().numpy()
        gn = batch["depth_gt_lidar"][0, 0].cpu().numpy()
        mn = batch["valid_mask_lidar"][0, 0].cpu().numpy().astype(bool)
        m = evaluator.evaluate_sample(pn, gn, mn, is_night=False)
        all_metrics.append(m)
        row = {"sample_id": batch["sample_id"][0], "is_night": False}
        for cat, ranges in m.items():
            if not isinstance(ranges, dict):
                continue
            for rname, mets in ranges.items():
                if not isinstance(mets, dict):
                    continue
                for k, v in mets.items():
                    row[f"{cat}/{rname}/{k}"] = v
        per_sample_rows.append(row)

    agg = evaluator.aggregate_metrics(all_metrics)
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(agg, f, indent=2)
    with open(os.path.join(out_dir, "summary.txt"), "w") as f:
        for category, ranges in agg.items():
            f.write(f"=== {category} ===\n")
            for rname, metrics in ranges.items():
                line = "  " + rname + "  " + "  ".join(
                    f"{k}={v:.4f}" for k, v in metrics.items()
                )
                f.write(line + "\n")
            f.write("\n")

    if per_sample_rows:
        import csv
        fieldnames = []
        seen = set()
        for r in per_sample_rows:
            for k in r:
                if k not in seen:
                    seen.add(k); fieldnames.append(k)
        path = os.path.join(out_dir, "per_sample.csv")
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            w.writeheader()
            for r in per_sample_rows:
                w.writerow(r)
        logger.info(f"wrote {path} ({len(per_sample_rows)} rows)")

    head = agg["overall"].get("0-80m", {})
    head50 = agg["overall"].get("0-50m", {})
    head70 = agg["overall"].get("0-70m", {})
    logger.info(
        f"DONE ZJU [{eval_split}]  "
        f"0-50m MAE={head50.get('mae', float('nan')):.4f}  "
        f"0-70m MAE={head70.get('mae', float('nan')):.4f}  "
        f"0-80m MAE={head.get('mae', float('nan')):.4f}"
    )
    logger.info(f"wrote {os.path.join(out_dir, 'metrics.json')}")


if __name__ == "__main__":
    main()
