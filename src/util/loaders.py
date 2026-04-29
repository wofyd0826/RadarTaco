"""Dataloader factory shared by train.py and eval.py."""
import logging
import os
from typing import Optional, Tuple

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Subset

from src.dataset.multi_loader import MultiDatasetLoader
from src.dataset.nuscenes import NuScenesRadarDepthDataset
from src.dataset.sim import SimRadarDepthDataset

logger = logging.getLogger(__name__)


def _common_dataset_kwargs(ds_cfg) -> dict:
    return dict(
        max_radar_points=int(ds_cfg.max_radar_points),
        max_depth=float(ds_cfg.max_depth),
        min_depth=float(ds_cfg.min_depth),
    )


def _build_nuscenes_loaders(cfg) -> Tuple[DataLoader, DataLoader]:
    ds_cfg = cfg.dataset
    common = _common_dataset_kwargs(ds_cfg)
    train_resize = tuple(ds_cfg.train_resize_hw) if ds_cfg.get("train_resize_hw") else None
    train_crop = tuple(ds_cfg.train_crop_hw) if ds_cfg.get("train_crop_hw") else None
    night_ids_file = ds_cfg.get("night_ids_file", None)

    train_ds = NuScenesRadarDepthDataset(
        data_root=ds_cfg.data_root,
        split_file=ds_cfg.split_train,
        dense_gt_dir=ds_cfg.get("dense_gt_dir", "depth_interp"),
        night_ids_file=night_ids_file,
        resize_to_hw=train_resize, crop_to_hw=train_crop,
        augmentation=True, **common,
    )
    val_ds = NuScenesRadarDepthDataset(
        data_root=ds_cfg.data_root,
        split_file=ds_cfg.split_val,
        dense_gt_dir=ds_cfg.get("dense_gt_dir", "depth_interp"),
        night_ids_file=night_ids_file,
        resize_to_hw=None, augmentation=False, **common,
    )
    n_val = cfg.training.get("val_subset_size", 1000)
    if n_val and n_val < len(val_ds):
        gen = torch.Generator().manual_seed(int(cfg.training.seed))
        idx = torch.randperm(len(val_ds), generator=gen)[:int(n_val)].tolist()
        val_ds = Subset(val_ds, idx)
    logger.info(f"nuScenes train={len(train_ds)}  val={len(val_ds)}")

    bsz = int(cfg.training.batch_size)
    train_loader = DataLoader(train_ds, batch_size=bsz, shuffle=True,
                              num_workers=int(ds_cfg.num_workers),
                              pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False,
                            num_workers=int(ds_cfg.num_workers), pin_memory=True)
    return train_loader, val_loader


def _build_sim_loaders(cfg):
    ds_cfg = cfg.dataset
    sim_cfg = ds_cfg.get("sim", None)
    if sim_cfg is None or not sim_cfg.get("enabled", False):
        return []
    common = _common_dataset_kwargs(ds_cfg)
    bsz = int(cfg.training.batch_size)
    sim_kw = dict(
        radar_simulation=sim_cfg.radar_simulation,
        num_radar_points=(int(sim_cfg.num_radar_points_min),
                          int(sim_cfg.num_radar_points_max)),
        depth_noise_std=float(sim_cfg.depth_noise_std),
    )
    resize = tuple(sim_cfg.train_resize_hw) if sim_cfg.get("train_resize_hw") else None
    out = []
    for name in ("hypersim", "vkitti2"):
        sub = sim_cfg.get(name, None)
        if sub is None or not sub.get("enabled", False):
            continue
        if not os.path.isdir(sub.data_root):
            logger.warning(f"{name}: data_root missing → skipping")
            continue
        ds = SimRadarDepthDataset(
            data_root=sub.data_root,
            split_file=sub.split_train,
            dataset_type=sub.dataset_type,
            resize_to_hw=resize, augmentation=True,
            **common, **sim_kw,
        )
        loader = DataLoader(ds, batch_size=bsz, shuffle=True,
                            num_workers=int(ds_cfg.num_workers),
                            pin_memory=True, drop_last=True)
        logger.info(f"{name}: {len(ds)} samples")
        out.append((name, loader))
    return out


def build_loaders(cfg) -> Tuple[object, DataLoader]:
    """Build (train_loader, val_loader). train_loader may be a MultiDatasetLoader."""
    nusc_train, val_loader = _build_nuscenes_loaders(cfg)
    sim_loaders = _build_sim_loaders(cfg)
    sim_cfg = cfg.dataset.get("sim", None)
    use_sim = bool(sim_cfg and sim_cfg.get("enabled", False) and len(sim_loaders) > 0)
    if not use_sim:
        return nusc_train, val_loader
    ratio_real = float(sim_cfg.ratio_real)
    ratio_sim = float(sim_cfg.ratio_sim)
    sim_each = ratio_sim / max(1, len(sim_loaders))
    loaders = [nusc_train] + [l for _, l in sim_loaders]
    probs = [ratio_real] + [sim_each] * len(sim_loaders)
    logger.info("mixed loader: nuScenes=%.2f + %s",
                ratio_real, ", ".join(f"{n}={sim_each:.2f}" for n, _ in sim_loaders))
    return MultiDatasetLoader(loaders, probs), val_loader


def build_viz_sampler(val_loader: DataLoader, n: int) -> Optional[list]:
    """Snapshot the first `n` validation samples for periodic image logging."""
    if n <= 0:
        return None
    out = []
    ds = val_loader.dataset
    for i in range(min(n, len(ds))):
        s = ds[i]
        # drop string keys; trainer adds batch dim via .unsqueeze(0)
        out.append({k: v for k, v in s.items() if not isinstance(v, str)})
    return out
