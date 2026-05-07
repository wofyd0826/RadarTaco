#!/usr/bin/env python3
"""RadarTaco training entry-point (Hydra).

Examples:
    # Paper-faithful baseline (GNN encoder, nuScenes only, single-stage)
    python scripts/train.py +experiment=baseline_gnn

    # MLP encoder ablation
    python scripts/train.py +experiment=baseline_mlp

    # Mixed batch + sim-only edge/grad aux loss
    python scripts/train.py +experiment=mixed_with_aux

    # Two-stage:
    #   stage 1 — sim pretrain (nuScenes ratio = 0)
    python scripts/train.py +experiment=sim_pretrain_finetune \
        training=sim_pretrain dataset=mixed \
        dataset.sim.ratio_real=0.0 dataset.sim.ratio_sim=1.0
    #   stage 2 — nuScenes finetune from the sim-pretrained checkpoint
    python scripts/train.py +experiment=sim_pretrain_finetune \
        training=nuscenes_finetune \
        training.load_weights_from=output/sim_pretrain/<run>/best.pt
"""
import datetime as _dt
import logging
import os
import sys

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from src.evaluation.metrics import DepthEvaluator                            # noqa: E402
from src.loss.factory import build_loss                                      # noqa: E402
from src.model.radartaco import RadarTaco                                    # noqa: E402
from src.model.rgb_only import RGBOnlyDepth                                  # noqa: E402
from src.trainer.trainer import RadarTacoTrainer                             # noqa: E402


def _build_model(cfg, max_depth: float, max_radar_points: int):
    name = str(cfg.model.get("name", "radartaco")).lower()
    if name == "rgb_only":
        return RGBOnlyDepth(
            max_depth=max_depth,
            pretrained_image_encoder=bool(cfg.model.pretrained_image_encoder),
            output_mode=str(cfg.model.get("output_mode", "metric")),
            min_depth_clip=float(cfg.model.get("min_depth_clip", 0.5)),
            multi_scale=bool(cfg.model.get("multi_scale", False)),
            multi_scale_levels=tuple(cfg.model.get("multi_scale_levels", (2, 4, 8, 16))),
        )
    return RadarTaco(
        radar_encoder_name=cfg.model.radar_encoder,
        max_depth=max_depth,
        max_radar_points=max_radar_points,
        k_neighbors=int(cfg.model.k_neighbors),
        a_l=tuple(cfg.model.a_l),
        radar_channels=tuple(cfg.model.radar_channels),
        attn_heads=int(cfg.model.attn_heads),
        mlp_hidden=int(cfg.model.get("mlp_hidden", 128)),
        pretrained_image_encoder=bool(cfg.model.pretrained_image_encoder),
        output_mode=str(cfg.model.get("output_mode", "metric")),
        min_depth_clip=float(cfg.model.get("min_depth_clip", 0.5)),
        multi_scale=bool(cfg.model.get("multi_scale", False)),
        multi_scale_levels=tuple(cfg.model.get("multi_scale_levels", (2, 4, 8, 16))),
    )
from src.util.loaders import build_loaders, build_viz_sampler                # noqa: E402
from src.util.seeds import set_seed                                          # noqa: E402
from src.util.wandb_logger import WandbLogger                                # noqa: E402

logging.basicConfig(format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                    level=logging.INFO)
logger = logging.getLogger("radartaco.train")


def _resolve_output_dir(cfg) -> str:
    base = cfg.training.output_dir
    name = (cfg.wandb.get("name") if cfg.get("wandb") else None) \
        or _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    candidate = os.path.join(base, name)
    out = candidate
    i = 2
    while os.path.exists(out):
        out = f"{candidate}_{i}"
        i += 1
    os.makedirs(out, exist_ok=True)
    return out


@hydra.main(version_base="1.3", config_path="../config", config_name="config")
def main(cfg: DictConfig) -> None:
    cfg.training.output_dir = _resolve_output_dir(cfg)
    logger.info(f"output_dir: {cfg.training.output_dir}")
    with open(os.path.join(cfg.training.output_dir, "config.yaml"), "w") as f:
        f.write(OmegaConf.to_yaml(cfg))
    logger.info("config:\n" + OmegaConf.to_yaml(cfg))

    set_seed(int(cfg.training.seed))
    train_loader, val_loader = build_loaders(cfg)

    model = _build_model(cfg,
                         max_depth=float(cfg.dataset.max_depth),
                         max_radar_points=int(cfg.dataset.max_radar_points))
    n_p = sum(p.numel() for p in model.parameters()) / 1e6
    logger.info(f"{type(model).__name__} ({cfg.model.get('name', 'radartaco')}): {n_p:.2f}M params")

    loss_fn = build_loss(cfg.loss)
    evaluator = DepthEvaluator(min_depth=float(cfg.dataset.min_depth),
                               max_depth=float(cfg.dataset.max_depth))
    wandb_logger = WandbLogger(cfg, enabled=bool(cfg.wandb.get("enabled", False)))

    viz_n = int(cfg.training.get("viz_n_samples", 0))
    viz_sampler = build_viz_sampler(val_loader, viz_n)

    trainer = RadarTacoTrainer(cfg, model, loss_fn, evaluator,
                               train_loader, val_loader,
                               wandb_logger=wandb_logger,
                               viz_sampler=viz_sampler)
    if cfg.training.get("load_weights_from"):
        trainer.load_checkpoint(cfg.training.load_weights_from, weights_only=True)
    if cfg.training.get("resume_from"):
        trainer.load_checkpoint(cfg.training.resume_from, weights_only=False)

    try:
        trainer.train()
    finally:
        wandb_logger.finish()


if __name__ == "__main__":
    main()
