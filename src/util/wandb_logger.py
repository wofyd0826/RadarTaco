"""W&B logger — flat-key per-range metrics + auto-cleanup of old local runs."""
import logging
import os
from typing import Dict, Optional

from omegaconf import OmegaConf

logger = logging.getLogger(__name__)


class WandbLogger:
    def __init__(self, cfg, enabled: bool = True):
        self.enabled = enabled
        if not enabled:
            self.run = None
            return
        try:
            import wandb
            self.wandb = wandb
        except ImportError:
            logger.warning("wandb not installed, disabling")
            self.enabled = False
            return
        wandb_cfg = cfg.get("wandb", {})
        self._cleanup_local_runs(max_keep=4)
        self.run = wandb.init(
            project=wandb_cfg.get("project", "radartaco"),
            entity=wandb_cfg.get("entity", None),
            name=wandb_cfg.get("name", None),
            group=wandb_cfg.get("group", None),
            tags=list(wandb_cfg.get("tags", [])),
            notes=wandb_cfg.get("notes", None),
            config=OmegaConf.to_container(cfg, resolve=True),
            reinit=True,
        )
        wandb.define_metric("train/*", step_metric="train/step")
        wandb.define_metric("eval/*", step_metric="eval/step")

    def log_training_step(self, step: int, loss_dict: Dict[str, float], lr: float,
                          extras: Optional[Dict[str, float]] = None):
        if not self.enabled:
            return
        d = {"train/step": step, "train/lr": lr}
        for k, v in loss_dict.items():
            d[f"train/{k}"] = v
        if extras:
            for k, v in extras.items():
                if v is None:
                    continue
                d[f"train/{k}"] = v
        self.wandb.log(d)

    def log_validation(self, step: int, eval_results: Dict, images: Optional[Dict] = None):
        """Flatten nested dict and log in a single call so wandb doesn't drop later steps."""
        if not self.enabled:
            return
        flat = {"eval/step": step}
        for category, ranges in eval_results.items():
            if not isinstance(ranges, dict):
                continue
            for range_name, metrics in ranges.items():
                if not isinstance(metrics, dict):
                    continue
                for metric_name, value in metrics.items():
                    flat[f"eval/{category}/{range_name}/{metric_name}"] = value
        if images:
            for name, img in images.items():
                flat[name] = self.wandb.Image(img)
        self.wandb.log(flat)

    def _cleanup_local_runs(self, max_keep: int = 4):
        import shutil
        wd = os.path.join(os.getcwd(), "wandb")
        if not os.path.exists(wd):
            return
        run_dirs = sorted([
            d for d in os.listdir(wd)
            if d.startswith("run-") and os.path.isdir(os.path.join(wd, d))
        ])
        if len(run_dirs) <= max_keep:
            return
        for d in run_dirs[:-max_keep]:
            shutil.rmtree(os.path.join(wd, d))
            logger.info(f"cleaned up old wandb run: {d}")

    def finish(self):
        if self.enabled and self.run:
            self.wandb.finish()
