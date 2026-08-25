"""RadarTaco trainer: AMP, gradient clipping, periodic snapshots, day/night-aware eval."""
from __future__ import annotations

import logging
import os
import time
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


class RadarTacoTrainer:
    def __init__(
        self,
        cfg,
        model: nn.Module,
        loss_fn: nn.Module,
        evaluator,
        train_loader: DataLoader,
        val_loader: DataLoader,
        wandb_logger=None,
        device: str = "cuda",
        viz_sampler: Optional[List[Dict]] = None,
        val_loaders_extra: Optional[Dict[str, DataLoader]] = None,
    ) -> None:
        self.cfg = cfg
        self.device = device
        self.model = model.to(device)
        self.loss_fn = loss_fn.to(device)
        self.evaluator = evaluator
        self.train_loader = train_loader
        self.val_loader = val_loader
        # Extra val loaders are evaluated each epoch for logging only —
        # best.pt selection always uses the primary `val_loader` (typically
        # nuScenes) to keep the metric target stable across experiments.
        self.val_loaders_extra: Dict[str, DataLoader] = dict(val_loaders_extra or {})
        self.wandb = wandb_logger
        self.viz_sampler = viz_sampler

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=cfg.training.lr,
            weight_decay=float(cfg.training.get("weight_decay", 0.0)),
        )
        self.amp = bool(cfg.training.amp)
        # fp16 needs a loss scaler and this loss overflows it often: with
        # w_grad_shape=100 the pre-clip grad norm sits around 10-40, the scaler
        # halves on every overflow and only doubles after 2000 clean steps, so
        # it can spiral down until 1/scale is inf and the gradients turn NaN.
        # bf16 carries fp32's exponent range, so no scaling is involved.
        self.amp_dtype = torch.bfloat16 if (
            self.amp and str(cfg.training.get("amp_dtype", "float16")) == "bfloat16"
        ) else torch.float16
        use_scaler = self.amp and self.amp_dtype is torch.float16
        self.scaler = GradScaler(enabled=use_scaler)
        # Scale this low means the scaler is losing ground rather than settling.
        self._min_scale = float(cfg.training.get("min_grad_scale", 1.0))
        os.makedirs(cfg.training.output_dir, exist_ok=True)
        self.global_step = 0
        self.best_metric = float("inf")
        self.start_epoch = 0
        # LR schedule shift for warm-start-with-carried-schedule. Epoch used
        # in the LR formula is (epoch + _lr_epoch_offset), so stage 2 picks
        # the schedule up where stage 1 left off instead of resetting to the
        # initial LR. Kept 0 by default; set by load_checkpoint(carry_schedule=True).
        self._lr_epoch_offset = 0
        # Consecutive non-finite training losses tolerated before giving up.
        self._nonfinite_steps = 0
        self._nonfinite_patience = int(
            cfg.training.get("nonfinite_patience", 25))

    # ---------------------------------------------------------- helpers --
    def _to_device(self, batch: Dict) -> Dict:
        return {k: (v.to(self.device, non_blocking=True) if torch.is_tensor(v) else v)
                for k, v in batch.items()}

    def _adjust_lr(self, epoch: int) -> None:
        effective_epoch = epoch + self._lr_epoch_offset
        n_decays = effective_epoch // int(self.cfg.training.lr_decay_step)
        new_lr = max(1e-7,
                     float(self.cfg.training.lr)
                     - n_decays * float(self.cfg.training.lr_decay_amount))
        for pg in self.optimizer.param_groups:
            pg["lr"] = new_lr

    # ----------------------------------------------------------- train --
    def train(self) -> None:
        # MoE + stage 1 → also run an oracle (teacher-forced) val each epoch
        # for monitoring the ceiling. Log-only; does not affect best.pt.
        run_oracle_val = (
            bool(getattr(self.cfg.model, "moe_at_l", None))
            and int(getattr(self.cfg.model, "moe_stage", 2)) == 1
        )
        for epoch in range(self.start_epoch, self.cfg.training.epochs):
            self._adjust_lr(epoch)
            self._train_one_epoch(epoch)
            metrics, agg = self.validate()
            extras = {}
            for name, loader in self.val_loaders_extra.items():
                _, agg_extra = self._validate_on(loader, log_tag=f"VAL[{name}]")
                extras[name] = agg_extra
            agg_oracle = None
            if run_oracle_val:
                _, agg_oracle = self._validate_on(
                    self.val_loader, log_tag="VAL[oracle]", oracle=True)
            viz = self._build_viz_panel() if (self.wandb is not None and self.viz_sampler) else None
            if self.wandb is not None:
                # wandb val step continues across stages: `epoch` restarts at
                # 0 in stage 2 (weights_only=True), but `_lr_epoch_offset` is
                # the number of epochs already run in the prior stage, so
                # `epoch + _lr_epoch_offset` is the true wandb x-value that
                # lines up with stage-1's val curves.
                wb_step = epoch + self._lr_epoch_offset
                self.wandb.log_validation(wb_step, agg, images=viz)
                for name, agg_extra in extras.items():
                    self.wandb.log_validation(wb_step, agg_extra, prefix=f"eval_{name}")
                if agg_oracle is not None:
                    self.wandb.log_validation(wb_step, agg_oracle, prefix="eval_oracle")
            self._save_checkpoint(metrics, epoch, agg)

    def _train_one_epoch(self, epoch: int) -> None:
        self.model.train()
        t0 = time.time()
        recent = []
        for it, batch in enumerate(self.train_loader):
            batch = self._to_device(batch)
            self.optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=self.amp, dtype=self.amp_dtype):
                pred = self.model(batch["rgb_norm"],
                                  batch["radar_points"],
                                  batch["radar_mask"],
                                  batch.get("rel_depth"),
                                  depth_gt_dense=batch.get("depth_gt_dense"))
                losses = self.loss_fn(pred, batch)
                loss = losses["loss_total"]
            # A non-finite loss poisons BatchNorm running stats on the spot
            # (they update in the forward, whatever the optimizer does), so the
            # run never recovers. Skip the step, and abort once it is clearly
            # not a one-off — otherwise training burns epochs on NaN, which is
            # exactly what happened to shape_lidar_grad_shape_edge_res_fix
            # (diverged at step 8284, ran ~48k more steps producing nothing).
            if not torch.isfinite(loss):
                self._nonfinite_steps += 1
                logger.warning(
                    f"ep {epoch} it {it+1}: non-finite loss ({float(loss)}) — "
                    f"skipping step ({self._nonfinite_steps}/{self._nonfinite_patience})"
                )
                if self._nonfinite_steps >= self._nonfinite_patience:
                    raise RuntimeError(
                        f"training diverged: {self._nonfinite_steps} non-finite "
                        f"losses in a row at ep {epoch} it {it+1}. Last finite "
                        f"checkpoint is in {self.cfg.training.output_dir}."
                    )
                self.optimizer.zero_grad(set_to_none=True)
                continue
            self._nonfinite_steps = 0
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            if self.scaler.is_enabled() and self.scaler.get_scale() < self._min_scale:
                raise RuntimeError(
                    f"grad scaler collapsed to {self.scaler.get_scale():g} at ep "
                    f"{epoch} it {it+1}: fp16 gradients keep overflowing and the "
                    f"scaler cannot recover (it only grows after 2000 clean "
                    f"steps). Switch to training.amp_dtype=bfloat16, or lower "
                    f"training.lr."
                )
            recent.append(loss.item())
            self.global_step += 1
            cur_lr = self.optimizer.param_groups[0]["lr"]
            if self.wandb is not None:
                ld = {k: float(v.detach()) for k, v in losses.items() if torch.is_tensor(v)}
                self.wandb.log_training_step(self.global_step, ld, cur_lr)
            if (it + 1) % int(self.cfg.training.log_interval) == 0:
                logger.info(
                    f"ep {epoch} it {it+1}/{len(self.train_loader)} "
                    f"loss={np.mean(recent[-int(self.cfg.training.log_interval):]):.4f} "
                    f"lr={cur_lr:.2e}  ({(time.time()-t0)/(it+1):.2f}s/it)"
                )

    # ------------------------------------------------------- validation --
    @torch.no_grad()
    def _validate_on(self, loader: DataLoader, log_tag: str = "VAL",
                     oracle: bool = False):
        self.model.eval()
        # Oracle mode: temporarily flip the MoE gate to use dense-GT (teacher
        # forcing) inside .forward. Restored in `finally` so a crash mid-epoch
        # can't leave the model stuck in oracle mode for downstream calls.
        if oracle:
            self.model._oracle_eval = True
        all_metrics = []
        try:
            for batch in loader:
                batch = self._to_device(batch)
                with autocast(enabled=self.amp, dtype=self.amp_dtype):
                    pred = self.model(batch["rgb_norm"],
                                      batch["radar_points"],
                                      batch["radar_mask"],
                                      batch.get("rel_depth"),
                                      depth_gt_dense=(batch.get("depth_gt_dense")
                                                      if oracle else None))
                if isinstance(pred, dict):
                    pred = pred["depth"]
                B = pred.shape[0]
                for b in range(B):
                    pn = pred[b, 0].float().cpu().numpy()
                    gn = batch["depth_gt_lidar"][b, 0].cpu().numpy()
                    mn = batch["valid_mask_lidar"][b, 0].cpu().numpy().astype(bool)
                    is_n = bool(batch["is_night"][b].item()) if torch.is_tensor(batch["is_night"]) else False
                    all_metrics.append(self.evaluator.evaluate_sample(pn, gn, mn, is_night=is_n))
        finally:
            if oracle:
                self.model._oracle_eval = False
        agg = self.evaluator.aggregate_metrics(all_metrics)
        head = agg.get("overall", {}).get("0-80m", {})
        head100 = agg.get("overall", {}).get("0-100m", {})
        far = agg.get("far", {}).get("50-80m", {})
        far100 = agg.get("far", {}).get("80-100m", {})
        night = agg.get("night", {}).get("0-80m", {})
        logger.info(
            f"{log_tag}  0-80m MAE={head.get('mae', float('nan')):.1f}/RMSE={head.get('rmse', float('nan')):.1f}  "
            f"|  0-100m MAE={head100.get('mae', float('nan')):.1f}/RMSE={head100.get('rmse', float('nan')):.1f}  "
            f"|  far 50-80m MAE={far.get('mae', float('nan')):.1f}  80-100m MAE={far100.get('mae', float('nan')):.1f}  "
            f"|  night MAE={night.get('mae', float('nan')):.1f}"
        )
        return head, agg

    def validate(self):
        return self._validate_on(self.val_loader, log_tag="VAL")

    @torch.no_grad()
    def _build_viz_panel(self):
        """Build a small dict of {key: HxWx3 uint8} for wandb image logging."""
        from src.evaluation.viz import (build_grid, colorize_depth, colorize_error,
                                        rgb_to_uint8)
        if not self.viz_sampler:
            return None
        self.model.eval()
        out: Dict[str, np.ndarray] = {}
        for i, sample in enumerate(self.viz_sampler):
            batch = {k: (v.unsqueeze(0).to(self.device) if torch.is_tensor(v) else v)
                     for k, v in sample.items()}
            with autocast(enabled=self.amp, dtype=self.amp_dtype):
                pred = self.model(batch["rgb_norm"], batch["radar_points"], batch["radar_mask"],
                                  batch.get("rel_depth"))
            if isinstance(pred, dict):
                pred = pred["depth"]
            rgb = rgb_to_uint8(batch["rgb_norm"][0].cpu().numpy())
            gt = batch["depth_gt_lidar"][0, 0].cpu().numpy()
            valid = batch["valid_mask_lidar"][0, 0].cpu().numpy().astype(bool)
            gt_dense = batch["depth_gt_dense"][0, 0].cpu().numpy()
            valid_dense = batch["valid_mask_dense"][0, 0].cpu().numpy().astype(bool)
            pn = pred[0, 0].float().cpu().numpy()
            panel = build_grid([
                rgb,
                # refined dense GT — show the WHOLE map (no valid_mask)
                # so sky / ego / padding fill (=max_depth) is visible too,
                # matching what the loss actually sees.
                colorize_depth(gt_dense,
                               vmax=self.cfg.dataset.max_depth),
                colorize_depth(pn, vmax=self.cfg.dataset.max_depth),
                colorize_error(pn, gt, valid, vmax=10.0, point_radius=3),
                colorize_depth(gt, valid_mask=valid, vmax=self.cfg.dataset.max_depth, point_radius=3),
            ])
            out[f"viz/sample_{i}"] = panel
        return out

    # --------------------------------------------------------- checkpoint --
    def _save_checkpoint(self, metrics: Dict, epoch: int, agg: Dict) -> None:
        out = self.cfg.training.output_dir
        torch.save({
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epoch": epoch,
            "global_step": self.global_step,
        }, os.path.join(out, "last.pt"))
        cur = metrics.get("mae", float("inf"))
        if cur < self.best_metric:
            self.best_metric = cur
            # global_step in best.pt lets a following stage warm-start with
            # `carry_schedule=True` (LR schedule + wandb step axis continue
            # from stage 1). Older best.pt files without this key still work.
            torch.save({"model": self.model.state_dict(),
                        "epoch": epoch,
                        "global_step": self.global_step},
                       os.path.join(out, "best.pt"))
            logger.info(f"saved best.pt at epoch {epoch} (MAE={cur:.1f})")
        snapshots = self.cfg.training.get("save_epoch_snapshots", None) or []
        target = epoch + 1
        if target in list(snapshots):
            snap = os.path.join(out, f"epoch_{target}.pt")
            torch.save({"model": self.model.state_dict(),
                        "optimizer": self.optimizer.state_dict(),
                        "epoch": epoch,
                        "global_step": self.global_step,
                        "metrics": agg}, snap)
            logger.info(f"saved snapshot {snap}")

    def load_checkpoint(self, path: str, weights_only: bool = False,
                        carry_schedule: bool = False) -> None:
        """Load a checkpoint.

        Modes:
          • weights_only=False   → full resume: model + optimizer + start_epoch
                                    + global_step. Continues the SAME run.
          • weights_only=True    → cross-run warm start: model only.
                                    Training restarts at epoch 0, LR at
                                    initial value, wandb step at 0.
          • weights_only=True + carry_schedule=True  → same as weights_only
                                    but also carries `global_step` (wandb
                                    x-axis) and `_lr_epoch_offset` (so the
                                    LR schedule continues where the loaded
                                    checkpoint ended). `start_epoch` still
                                    resets to 0 so the caller trains a
                                    fresh `cfg.training.epochs` epochs.
                                    Use for stage 1 → stage 2 chains.
        """
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model"], strict=not weights_only)
        if not weights_only and "optimizer" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer"])
        if not weights_only:
            self.global_step = ckpt.get("global_step", 0)
            # Resume from the *next* epoch — `epoch` in the ckpt is the
            # last completed one (saved after _train_one_epoch + validate).
            self.start_epoch = int(ckpt.get("epoch", -1)) + 1
        elif carry_schedule:
            # Cross-run warm start that keeps the schedule + step axis intact.
            # Missing keys (older checkpoints) fall back to 0 with a warning.
            gs = ckpt.get("global_step")
            ep = ckpt.get("epoch")
            if gs is None or ep is None:
                logger.warning(
                    f"carry_schedule=True but checkpoint lacks "
                    f"{'global_step ' if gs is None else ''}"
                    f"{'epoch' if ep is None else ''} — treating missing as 0"
                )
            self.global_step = int(gs if gs is not None else 0)
            self._lr_epoch_offset = int(ep) + 1 if ep is not None else 0
        logger.info(f"loaded checkpoint from {path} "
                    f"(weights_only={weights_only}, carry_schedule={carry_schedule}, "
                    f"start_epoch={self.start_epoch}, lr_epoch_offset={self._lr_epoch_offset}, "
                    f"global_step={self.global_step})")
