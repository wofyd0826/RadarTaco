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
    ) -> None:
        self.cfg = cfg
        self.device = device
        self.model = model.to(device)
        self.loss_fn = loss_fn.to(device)
        self.evaluator = evaluator
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.wandb = wandb_logger
        self.viz_sampler = viz_sampler

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=cfg.training.lr,
            weight_decay=float(cfg.training.get("weight_decay", 0.0)),
        )
        self.amp = bool(cfg.training.amp)
        self.scaler = GradScaler(enabled=self.amp)
        os.makedirs(cfg.training.output_dir, exist_ok=True)
        self.global_step = 0
        self.best_metric = float("inf")

    # ---------------------------------------------------------- helpers --
    def _to_device(self, batch: Dict) -> Dict:
        return {k: (v.to(self.device, non_blocking=True) if torch.is_tensor(v) else v)
                for k, v in batch.items()}

    def _adjust_lr(self, epoch: int) -> None:
        n_decays = epoch // int(self.cfg.training.lr_decay_step)
        new_lr = max(1e-7,
                     float(self.cfg.training.lr)
                     - n_decays * float(self.cfg.training.lr_decay_amount))
        for pg in self.optimizer.param_groups:
            pg["lr"] = new_lr

    # ----------------------------------------------------------- train --
    def train(self) -> None:
        for epoch in range(self.cfg.training.epochs):
            self._adjust_lr(epoch)
            self._train_one_epoch(epoch)
            metrics, agg = self.validate()
            viz = self._build_viz_panel() if (self.wandb is not None and self.viz_sampler) else None
            if self.wandb is not None:
                self.wandb.log_validation(epoch, agg, images=viz)
            self._save_checkpoint(metrics, epoch, agg)

    def _train_one_epoch(self, epoch: int) -> None:
        self.model.train()
        t0 = time.time()
        recent = []
        for it, batch in enumerate(self.train_loader):
            batch = self._to_device(batch)
            self.optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=self.amp):
                pred = self.model(batch["rgb_norm"],
                                  batch["radar_points"],
                                  batch["radar_mask"])
                losses = self.loss_fn(pred, batch)
                loss = losses["loss_total"]
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
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
    def validate(self):
        self.model.eval()
        all_metrics = []
        for batch in self.val_loader:
            batch = self._to_device(batch)
            with autocast(enabled=self.amp):
                pred = self.model(batch["rgb_norm"],
                                  batch["radar_points"],
                                  batch["radar_mask"])
            B = pred.shape[0]
            for b in range(B):
                pn = pred[b, 0].float().cpu().numpy()
                gn = batch["depth_gt_lidar"][b, 0].cpu().numpy()
                mn = batch["valid_mask_lidar"][b, 0].cpu().numpy().astype(bool)
                is_n = bool(batch["is_night"][b].item()) if torch.is_tensor(batch["is_night"]) else False
                all_metrics.append(self.evaluator.evaluate_sample(pn, gn, mn, is_night=is_n))
        agg = self.evaluator.aggregate_metrics(all_metrics)
        head = agg.get("overall", {}).get("0-80m", {})
        far = agg.get("far", {}).get("50-80m", {})
        night = agg.get("night", {}).get("0-80m", {})
        logger.info(
            f"VAL  0-80m  MAE={head.get('mae', float('nan')):.1f}  RMSE={head.get('rmse', float('nan')):.1f}  d1={head.get('delta1', float('nan')):.3f}  "
            f"|  far(50-80m) MAE={far.get('mae', float('nan')):.1f}  "
            f"|  night MAE={night.get('mae', float('nan')):.1f}"
        )
        return head, agg

    @torch.no_grad()
    def _build_viz_panel(self):
        """Build a small dict of {key: HxWx3 uint8} for wandb image logging."""
        from src.evaluation.viz import (build_grid, colorize_depth, colorize_error,
                                        overlay_radar_points, rgb_to_uint8)
        if not self.viz_sampler:
            return None
        self.model.eval()
        out: Dict[str, np.ndarray] = {}
        for i, sample in enumerate(self.viz_sampler):
            batch = {k: (v.unsqueeze(0).to(self.device) if torch.is_tensor(v) else v)
                     for k, v in sample.items()}
            with autocast(enabled=self.amp):
                pred = self.model(batch["rgb_norm"], batch["radar_points"], batch["radar_mask"])
            rgb = rgb_to_uint8(batch["rgb_norm"][0].cpu().numpy())
            radar_pts = batch["radar_points"][0].cpu().numpy()
            radar_mask = batch["radar_mask"][0].cpu().numpy().astype(bool)
            gt = batch["depth_gt_lidar"][0, 0].cpu().numpy()
            valid = batch["valid_mask_lidar"][0, 0].cpu().numpy().astype(bool)
            pn = pred[0, 0].float().cpu().numpy()
            panel = build_grid([
                rgb,
                overlay_radar_points(rgb, radar_pts, radar_mask, vmax=self.cfg.dataset.max_depth),
                colorize_depth(pn, vmax=self.cfg.dataset.max_depth),
                colorize_error(pn, gt, valid, vmax=10.0),
                colorize_depth(gt, valid_mask=valid, vmax=self.cfg.dataset.max_depth),
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
            torch.save({"model": self.model.state_dict(), "epoch": epoch},
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

    def load_checkpoint(self, path: str, weights_only: bool = False) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model"], strict=not weights_only)
        if not weights_only and "optimizer" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer"])
        if not weights_only:
            self.global_step = ckpt.get("global_step", 0)
        logger.info(f"loaded checkpoint from {path} (weights_only={weights_only})")
