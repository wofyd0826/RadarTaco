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
        def _s(v):
            return None if v is None else str(v)
        self.run = wandb.init(
            project=_s(wandb_cfg.get("project", "radartaco")),
            entity=_s(wandb_cfg.get("entity", None)),
            name=_s(wandb_cfg.get("name", None)),
            group=_s(wandb_cfg.get("group", None)),
            tags=[str(t) for t in wandb_cfg.get("tags", [])],
            notes=_s(wandb_cfg.get("notes", None)),
            config=OmegaConf.to_container(cfg, resolve=True),
            reinit=True,
        )
        wandb.define_metric("train/*", step_metric="train/step")
        wandb.define_metric("eval/*", step_metric="eval/step")
        # Extra val domains share the primary eval/step axis so they line
        # up epoch-by-epoch with nuScenes. wandb only accepts suffix glob
        # patterns (`prefix/*`), so each extra prefix must be registered
        # individually — add new lines here if more domains are added.
        wandb.define_metric("eval_zju/*", step_metric="eval/step")

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

    def log_validation(self, step: int, eval_results: Dict,
                       images: Optional[Dict] = None, prefix: str = "eval"):
        """Flatten nested dict and log in a single call so wandb doesn't drop later steps.

        `prefix` controls the wandb key namespace, e.g.
            prefix="eval"      → eval/overall/0-80m/mae
            prefix="eval_zju"  → eval_zju/overall/0-80m/mae
        Different prefixes can be logged at the same step for side-by-side
        domain comparison (nuScenes vs ZJU val) without overwriting.
        """
        if not self.enabled:
            return
        flat = {f"{prefix}/step": step}
        for category, ranges in eval_results.items():
            if not isinstance(ranges, dict):
                continue
            for range_name, metrics in ranges.items():
                if not isinstance(metrics, dict):
                    continue
                for metric_name, value in metrics.items():
                    flat[f"{prefix}/{category}/{range_name}/{metric_name}"] = value
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
