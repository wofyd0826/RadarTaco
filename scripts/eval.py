#!/usr/bin/env python3
"""Standalone evaluation: load a checkpoint, run on the test split (default),
dump metrics + viz.

Pipeline convention:
    train  → train.txt        (scripts/train.py)
    val    → val.txt           (per-epoch validation inside train.py)
    test   → test.txt          (this script — final evaluation)

Override `eval_split=val` (or `eval_split=val_day` / `val_night`) to
evaluate on a different split.

Usage:
    python scripts/eval.py checkpoint=output/<run>/best.pt
    python scripts/eval.py checkpoint=output/<run>/best.pt eval_split=val
    python scripts/eval.py checkpoint=output/<run>/best.pt eval_split=val_night
"""
import json
import logging
import os
import sys

# Cap BLAS/OpenMP threadpools BEFORE importing torch — see scripts/train.py
# for the rationale (container cpu.max << host nproc).
_DEFAULT_INTRA_THREADS = "4"
for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
             "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, _DEFAULT_INTRA_THREADS)

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

torch.set_num_threads(int(os.environ["OMP_NUM_THREADS"]))
torch.set_num_interop_threads(2)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from src.dataset.nuscenes import NuScenesRadarDepthDataset                   # noqa: E402
from src.evaluation.metrics import DepthEvaluator                            # noqa: E402
from src.evaluation.viz import (build_grid, colorize_depth, colorize_error,  # noqa: E402
                                overlay_radar_points, rgb_to_uint8)
from src.model.radartaco import RadarTaco                                    # noqa: E402
from src.model.rgb_only import RGBOnlyDepth                                  # noqa: E402


def _maybe_override_model_from_ckpt(cfg) -> None:
    """If the checkpoint sits next to a saved `config.yaml`, override
    `cfg.model` with the one used to train the checkpoint. Also restore
    `dataset.rel_depth_dir` from the saved config so plug-in eval can
    locate the same `D*` directory that was used during training."""
    sibling = os.path.join(os.path.dirname(cfg.checkpoint), "config.yaml")
    if not os.path.exists(sibling):
        return
    saved = OmegaConf.load(sibling)
    OmegaConf.set_struct(cfg, False)
    saved_model = saved.get("model", None)
    if saved_model is not None:
        logger.info(f"restoring cfg.model from {sibling}")
        cfg.model = saved_model
    saved_ds = saved.get("dataset", None)
    if saved_ds is not None:
        rdd = saved_ds.get("rel_depth_dir", None)
        if rdd is not None and cfg.dataset.get("rel_depth_dir", None) is None:
            logger.info(f"restoring cfg.dataset.rel_depth_dir={rdd} from {sibling}")
            cfg.dataset.rel_depth_dir = rdd
        # Restore dense_gt_dir from training config too — otherwise the
        # dataset loader uses the default (depth_acc) and silently falls
        # back to LiDAR-sparse GT on test (where depth_acc isn't built).
        # That makes both viz and the evaluator's `depth_gt_dense` show the
        # sparse signal, even though training used a dense pseudo-GT.
        ddir = saved_ds.get("dense_gt_dir", None)
        default_ddir = "depth_acc"
        if ddir is not None and ddir != default_ddir \
                and cfg.dataset.get("dense_gt_dir", default_ddir) == default_ddir:
            logger.info(f"restoring cfg.dataset.dense_gt_dir={ddir} from {sibling}")
            cfg.dataset.dense_gt_dir = ddir
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
            use_aux_branch=bool(cfg.model.get("use_aux_branch", False)),
            moe_at_l=tuple(cfg.model.get("moe_at_l") or ()),
            moe_n_experts=int(cfg.model.get("moe_n_experts", 3)),
            moe_use_shared=bool(cfg.model.get("moe_use_shared", True)),
            moe_top_k=(int(cfg.model.get("moe_top_k"))
                       if cfg.model.get("moe_top_k") is not None else None),
            moe_stage=int(cfg.model.get("moe_stage", 2)),
            moe_bins=tuple(cfg.model.get("moe_bins", (0.0, 20.0, 50.0, 100.0))),
            moe_router_arch=str(cfg.model.get("moe_router_arch", "conv1x1")),
            moe_expert_ch_ratio=float(cfg.model.get("moe_expert_ch_ratio", 1.0)),
            moe_router_gt_type=str(cfg.model.get("moe_router_gt_type", "hard")),
            moe_overlap_bins=cfg.model.get("moe_overlap_bins", None),
            moe_shared_gate_mode=str(cfg.model.get("moe_shared_gate_mode", "always_on")),
            moe_shared_aux=bool(cfg.model.get("moe_shared_aux", False)),
            moe_per_spec_aux=bool(cfg.model.get("moe_per_spec_aux", False)),
            moe_router_dpt_source_layers=(
                tuple(cfg.model.get("moe_router_dpt_source_layers"))
                if cfg.model.get("moe_router_dpt_source_layers") is not None else None),
            moe_router_dpt_fusion_ch=int(cfg.model.get("moe_router_dpt_fusion_ch", 128)),
            moe_pre_fusion_enabled=bool(cfg.model.get("moe_pre_fusion_enabled", False)),
            moe_pre_fusion_feed_experts=bool(cfg.model.get("moe_pre_fusion_feed_experts", False)),
            moe_pre_fusion_ch_ratio=float(cfg.model.get("moe_pre_fusion_ch_ratio", 1.0)),
            moe_pre_fusion_router_detach=bool(cfg.model.get("moe_pre_fusion_router_detach", False)),
        )
    return m.to(device).eval()

logging.basicConfig(format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                    level=logging.INFO)
logger = logging.getLogger("radartaco.eval")


@hydra.main(version_base="1.3", config_path="../config", config_name="config")
def main(cfg: DictConfig) -> None:
    if not cfg.get("checkpoint"):
        raise SystemExit("checkpoint=<path/to/.pt> is required (Hydra override).")
    _maybe_override_model_from_ckpt(cfg)
    eval_mode = str(cfg.get("eval_mode", "independent")).lower()
    out_dir = cfg.get("eval_out_dir",
                      os.path.join(os.path.dirname(cfg.checkpoint), f"eval_{eval_mode}"))
    os.makedirs(out_dir, exist_ok=True)
    viz_dir = os.path.join(out_dir, "viz")
    os.makedirs(viz_dir, exist_ok=True)

    # Default to the held-out test split for final evaluation. Use
    #   eval_split=val           — same data the trainer validates on
    #   eval_split=val_day       — day-only subset of val
    #   eval_split=val_night     — night-only subset of val (Dark scenario eval)
    split = cfg.get("eval_split", "test")
    _TAG_SUFFIXES = ("day_clear", "day_rain", "night_clear", "night_rain")
    _ALLOWED = ({"train", "val", "test",
                 "train_day", "train_night",
                 "val_day", "val_night",
                 "test_day", "test_night"}
                | {f"{s}_{t}" for s in ("train", "val", "test") for t in _TAG_SUFFIXES})
    assert split in _ALLOWED, split
    split_file = getattr(cfg.dataset, f"split_{split}", None) or \
        os.path.join(cfg.dataset.data_root, "splits", f"{split}.txt")
    if not os.path.exists(split_file):
        raise SystemExit(f"split file not found: {split_file}")
    logger.info(f"eval split: {split} → {split_file}")

    # `eval_mode` selects how the plug-in branch (if present) is fed at
    # eval time. "independent" = pass zeros (paper Table 1 Indep row).
    # "plugin"     = use the precomputed D* from rel_depth_dir.
    # Default "independent" so legacy eval calls keep working.
    use_rel_depth = (eval_mode == "plugin")
    # Always evaluate metrics against raw single-frame depth_lidar so that
    # different training-time lidar_gt_dir choices (e.g. depth_lidar_filled)
    # produce comparable numbers. Override with `+eval_lidar_gt_dir=<dir>`.
    ds = NuScenesRadarDepthDataset(
        data_root=cfg.dataset.data_root,
        split_file=split_file,
        dense_gt_dir=cfg.dataset.get("dense_gt_dir", "depth_acc"),
        lidar_gt_dir=cfg.get("eval_lidar_gt_dir", "depth_lidar"),
        radar_3d_dir=cfg.dataset.get("radar_3d_dir", "radar_3d"),
        night_ids_file=cfg.dataset.get("night_ids_file", None),
        rel_depth_dir=cfg.dataset.get("rel_depth_dir", None) if use_rel_depth else None,
        rel_depth_dropout_prob=0.0,
        max_radar_points=int(cfg.dataset.max_radar_points),
        max_depth=float(cfg.dataset.max_depth),
        min_depth=float(cfg.dataset.min_depth),
        resize_to_hw=None, augmentation=False,
    )
    logger.info(f"eval mode: {eval_mode}  (rel_depth_dir used: {use_rel_depth})")
    loader = DataLoader(ds, batch_size=1, shuffle=False,
                        num_workers=int(cfg.dataset.num_workers), pin_memory=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = _build_eval_model(cfg, device)
    ckpt = torch.load(cfg.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"], strict=True)
    logger.info(f"loaded {cfg.checkpoint} (epoch={ckpt.get('epoch', '?')})")

    evaluator = DepthEvaluator(min_depth=float(cfg.dataset.min_depth),
                               max_depth=float(cfg.dataset.max_depth))
    all_metrics = []
    save_every = int(cfg.get("save_viz_every", 50))
    per_sample_rows = []

    for i, batch in enumerate(tqdm(loader)):
        batch = {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
                 for k, v in batch.items()}
        with torch.inference_mode():
            pred = model(batch["rgb_norm"], batch["radar_points"], batch["radar_mask"],
                         batch.get("rel_depth"))
        if isinstance(pred, dict):
            pred = pred["depth"]
        pn = pred[0, 0].float().cpu().numpy()
        gn = batch["depth_gt_lidar"][0, 0].cpu().numpy()
        mn = batch["valid_mask_lidar"][0, 0].cpu().numpy().astype(bool)
        is_n = bool(batch["is_night"][0].item()) if torch.is_tensor(batch["is_night"]) else False
        m = evaluator.evaluate_sample(pn, gn, mn, is_night=is_n)
        all_metrics.append(m)
        row = {"sample_id": batch["sample_id"][0], "is_night": is_n}
        # flatten all category × range × metric combos. evaluate_sample
        # returns {"overall": {...}, "far": {...}, ..., "is_night": bool},
        # so skip non-dict scalar fields.
        for cat, ranges in m.items():
            if not isinstance(ranges, dict):
                continue
            for rname, mets in ranges.items():
                if not isinstance(mets, dict):
                    continue
                for k, v in mets.items():
                    row[f"{cat}/{rname}/{k}"] = v
        per_sample_rows.append(row)
        if save_every > 0 and i % save_every == 0:
            rgb = rgb_to_uint8(batch["rgb_norm"][0].cpu().numpy())
            gn_dense = batch["depth_gt_dense"][0, 0].cpu().numpy()
            mn_dense = batch["valid_mask_dense"][0, 0].cpu().numpy().astype(bool)
            panel = build_grid([
                rgb,
                colorize_depth(gn_dense, valid_mask=mn_dense,
                               vmax=cfg.dataset.max_depth),    # refined dense GT
                colorize_depth(pn, vmax=cfg.dataset.max_depth),
                colorize_error(pn, gn, mn, vmax=10.0, point_radius=3),
                colorize_depth(gn, valid_mask=mn, vmax=cfg.dataset.max_depth, point_radius=3),
            ])
            Image.fromarray(panel).save(os.path.join(viz_dir, f"{i:06d}.png"))

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

    # Always save per-sample CSV — useful for downstream analysis
    # (failure-mode digging, day/night gap, far-range tail).
    if per_sample_rows:
        import csv
        # union of keys across rows (per_range bins differ per sample if a
        # sample has no points in a far bin, that key is absent).
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
        logger.info(f"wrote {path} ({len(per_sample_rows)} rows × {len(fieldnames)} cols)")

    head = agg["overall"].get("0-80m", {})
    head100 = agg["overall"].get("0-100m", {})
    night = agg.get("night", {}).get("0-80m", {})
    far = agg.get("far", {}).get("50-80m", {})
    far100 = agg.get("far", {}).get("80-100m", {})
    logger.info(
        f"DONE  0-80m MAE={head.get('mae', float('nan')):.1f}/RMSE={head.get('rmse', float('nan')):.1f}  "
        f"|  0-100m MAE={head100.get('mae', float('nan')):.1f}/RMSE={head100.get('rmse', float('nan')):.1f}  "
        f"|  far 50-80m MAE={far.get('mae', float('nan')):.1f}  80-100m MAE={far100.get('mae', float('nan')):.1f}  "
        f"|  night MAE={night.get('mae', float('nan')):.1f}"
    )
    logger.info(f"wrote {os.path.join(out_dir, 'metrics.json')}")


if __name__ == "__main__":
    main()
