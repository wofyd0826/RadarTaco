"""Dataloader factory shared by train.py and eval.py."""
import logging
import os
from typing import Optional, Tuple

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Subset

from src.dataset.multi_loader import MixedBatchLoader, MultiDatasetLoader
from src.dataset.nuscenes import NuScenesRadarDepthDataset
from src.dataset.photometric import build_from_cfg as build_photometric_aug
from src.dataset.sim import SimRadarDepthDataset
from src.dataset.zju import ZjuRadarDepthDataset

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
    photometric_aug = build_photometric_aug(ds_cfg.get("photometric_aug", None))
    if photometric_aug is not None:
        logger.info("photometric night-like augmentation: enabled")

    rel_depth_dir = ds_cfg.get("rel_depth_dir", None)
    rel_drop = float(ds_cfg.get("rel_depth_dropout_prob", 0.0))
    train_ds = NuScenesRadarDepthDataset(
        data_root=ds_cfg.data_root,
        split_file=ds_cfg.split_train,
        dense_gt_dir=ds_cfg.get("dense_gt_dir", "depth_acc"),
        radar_3d_dir=ds_cfg.get("radar_3d_dir", "radar_3d"),
        night_ids_file=night_ids_file,
        rel_depth_dir=rel_depth_dir,
        rel_depth_dropout_prob=rel_drop,
        resize_to_hw=train_resize, crop_to_hw=train_crop,
        augmentation=True, photometric_aug=photometric_aug, **common,
    )
    val_ds = NuScenesRadarDepthDataset(
        data_root=ds_cfg.data_root,
        split_file=ds_cfg.split_val,
        dense_gt_dir=ds_cfg.get("dense_gt_dir", "depth_acc"),
        radar_3d_dir=ds_cfg.get("radar_3d_dir", "radar_3d"),
        night_ids_file=night_ids_file,
        rel_depth_dir=rel_depth_dir,
        rel_depth_dropout_prob=0.0,                   # never drop during val
        resize_to_hw=None, augmentation=False, photometric_aug=None, **common,
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


def _build_zju_val_loader(cfg) -> Optional[DataLoader]:
    """Build a ZJU-4DRadarCam *val* loader (parallel-logging, not used for best.pt).

    best.pt selection still uses the nuScenes val loader, but if ZJU
    training is enabled we also evaluate on the ZJU val split each epoch
    and log the metrics under a separate `eval_zju/...` prefix so we can
    track both domains' generalization without affecting the model
    selection target.
    """
    ds_cfg = cfg.dataset
    zju_cfg = ds_cfg.get("zju", None)
    if zju_cfg is None or not zju_cfg.get("enabled", False):
        return None
    split_val = zju_cfg.get("split_val", None)
    if not split_val or not os.path.exists(split_val):
        logger.warning(f"zju: split_val missing → no extra val logging")
        return None
    common = _common_dataset_kwargs(ds_cfg)
    ds = ZjuRadarDepthDataset(
        data_root=zju_cfg.data_root,
        split_file=split_val,
        image_dir=zju_cfg.get("image_dir", "image"),
        radar_dir=zju_cfg.get("radar_dir", "radar"),
        gt_dir=zju_cfg.get("gt_dir", "gt"),
        gt_interp_dir=zju_cfg.get("gt_interp_dir", "gt_interp"),
        resize_to_hw=None, augmentation=False, **common,
    )
    n_val = cfg.training.get("val_subset_size", 1000)
    if n_val and n_val < len(ds):
        gen = torch.Generator().manual_seed(int(cfg.training.seed))
        idx = torch.randperm(len(ds), generator=gen)[:int(n_val)].tolist()
        ds = Subset(ds, idx)
    logger.info(f"ZJU-4DRadarCam val (extra logging)={len(ds)}")
    return DataLoader(ds, batch_size=1, shuffle=False,
                      num_workers=int(ds_cfg.num_workers), pin_memory=True)


def _build_zju_loader(cfg) -> Optional[DataLoader]:
    """Build a ZJU-4DRadarCam *train* loader if configured.

    nuScenes remains the validation source so best.pt selection still
    tracks the comparison metric used in the paper's Table 1. Mixing in
    ZJU only affects the training stream.
    """
    ds_cfg = cfg.dataset
    zju_cfg = ds_cfg.get("zju", None)
    if zju_cfg is None or not zju_cfg.get("enabled", False):
        return None
    if not os.path.isdir(zju_cfg.data_root):
        logger.warning(f"zju: data_root missing → skipping")
        return None
    common = _common_dataset_kwargs(ds_cfg)
    resize = tuple(zju_cfg.train_resize_hw) if zju_cfg.get("train_resize_hw") else None
    crop = tuple(zju_cfg.train_crop_hw) if zju_cfg.get("train_crop_hw") else None
    ds = ZjuRadarDepthDataset(
        data_root=zju_cfg.data_root,
        split_file=zju_cfg.split_train,
        image_dir=zju_cfg.get("image_dir", "image"),
        radar_dir=zju_cfg.get("radar_dir", "radar"),
        gt_dir=zju_cfg.get("gt_dir", "gt"),
        gt_interp_dir=zju_cfg.get("gt_interp_dir", "gt_interp"),
        resize_to_hw=resize, crop_to_hw=crop, augmentation=True, **common,
    )
    logger.info(f"ZJU-4DRadarCam train={len(ds)}")
    bsz = int(cfg.training.batch_size)
    return DataLoader(ds, batch_size=bsz, shuffle=True,
                      num_workers=int(ds_cfg.num_workers),
                      pin_memory=True, drop_last=True)


def _build_sim_loaders(cfg):
    ds_cfg = cfg.dataset
    sim_cfg = ds_cfg.get("sim", None)
    if sim_cfg is None or not sim_cfg.get("enabled", False):
        return []
    common = _common_dataset_kwargs(ds_cfg)
    bsz = int(cfg.training.batch_size)
    photometric_aug = build_photometric_aug(ds_cfg.get("photometric_aug", None))
    base_sim_kw = dict(
        num_radar_points=(int(sim_cfg.num_radar_points_min),
                          int(sim_cfg.num_radar_points_max)),
        depth_noise_std=float(sim_cfg.depth_noise_std),
    )
    real_noise = sim_cfg.get("real_noise", None)
    if real_noise is not None:
        base_sim_kw["real_noise"] = OmegaConf.to_container(real_noise, resolve=True)
    resize = tuple(sim_cfg.train_resize_hw) if sim_cfg.get("train_resize_hw") else None
    out = []
    for name in ("hypersim", "vkitti2"):
        sub = sim_cfg.get(name, None)
        if sub is None or not sub.get("enabled", False):
            continue
        if not os.path.isdir(sub.data_root):
            logger.warning(f"{name}: data_root missing → skipping")
            continue
        # per-source override > sim-level default
        sim_mode = sub.get("radar_simulation", None) or sim_cfg.radar_simulation
        cls_w = sub.get("class_weights", None)
        cls_w = dict(cls_w) if cls_w else None
        ds = SimRadarDepthDataset(
            data_root=sub.data_root,
            split_file=sub.split_train,
            dataset_type=sub.dataset_type,
            radar_simulation=sim_mode,
            class_weights=cls_w,
            resize_to_hw=resize, augmentation=True,
            photometric_aug=photometric_aug,
            **common, **base_sim_kw,
        )
        logger.info(f"{name}: {len(ds)} samples (mode={sim_mode})")
        loader = DataLoader(ds, batch_size=bsz, shuffle=True,
                            num_workers=int(ds_cfg.num_workers),
                            pin_memory=True, drop_last=True)
        out.append((name, loader))
    return out


def _build_sim_val_loader(cfg) -> Optional[DataLoader]:
    """Build a val loader from the first enabled sim source (vKITTI2/Hypersim).

    Used when `cfg.training.stage == 'sim_pretrain'` so the pretrain stage
    selects best.pt by sim-domain val performance — i.e. "is the model
    learning a good sim representation" — instead of conflating pretrain
    quality with direct sim→real transfer (which is what the finetune
    stage is for).
    """
    ds_cfg = cfg.dataset
    sim_cfg = ds_cfg.get("sim", None)
    if sim_cfg is None:
        return None
    common = _common_dataset_kwargs(ds_cfg)
    for name in ("vkitti2", "hypersim"):              # vKITTI2 first (closer to nuScenes scenes)
        sub = sim_cfg.get(name, None)
        if sub is None or not sub.get("enabled", False):
            continue
        if not os.path.isdir(sub.data_root):
            continue
        split_val = sub.get("split_val", None)
        if not split_val:
            logger.warning(f"{name}: no split_val configured — skipping for sim val")
            continue
        sim_mode = sub.get("radar_simulation", None) or sim_cfg.radar_simulation
        cls_w = sub.get("class_weights", None)
        cls_w = dict(cls_w) if cls_w else None
        ds = SimRadarDepthDataset(
            data_root=sub.data_root,
            split_file=split_val,
            dataset_type=sub.dataset_type,
            radar_simulation=sim_mode,
            class_weights=cls_w,
            num_radar_points=(int(sim_cfg.num_radar_points_min),
                              int(sim_cfg.num_radar_points_max)),
            depth_noise_std=float(sim_cfg.depth_noise_std),
            resize_to_hw=None, augmentation=False,
            **common,
        )
        n_val = cfg.training.get("val_subset_size", 500)
        if n_val and n_val < len(ds):
            gen = torch.Generator().manual_seed(int(cfg.training.seed))
            idx = torch.randperm(len(ds), generator=gen)[:int(n_val)].tolist()
            ds = Subset(ds, idx)
        logger.info(f"sim val loader: {name} (n={len(ds)})")
        return DataLoader(ds, batch_size=1, shuffle=False,
                          num_workers=int(ds_cfg.num_workers), pin_memory=True)
    logger.warning("sim_pretrain stage requested but no enabled sim val source — falling back to nuScenes val")
    return None


def build_loaders(cfg):
    """Build (train_loader, val_loader, val_loaders_extra).

    - `train_loader` may be a MultiDatasetLoader (nuScenes + optional ZJU/sim).
    - `val_loader` is the primary validation source used for best.pt
      selection — normally nuScenes, except in `sim_pretrain` stage.
    - `val_loaders_extra: Dict[str, DataLoader]` holds optional secondary
      validation loaders (e.g. ZJU) that are evaluated per-epoch for
      logging only — they do not influence best.pt selection.
    """
    nusc_train, nusc_val_loader = _build_nuscenes_loaders(cfg)
    zju_train = _build_zju_loader(cfg)
    zju_val_loader = _build_zju_val_loader(cfg)
    sim_loaders = _build_sim_loaders(cfg)
    sim_cfg = cfg.dataset.get("sim", None)
    use_sim = bool(sim_cfg and sim_cfg.get("enabled", False) and len(sim_loaders) > 0)
    use_zju = zju_train is not None

    stage = str(cfg.training.get("stage", "single"))
    if stage == "sim_pretrain":
        sim_val_loader = _build_sim_val_loader(cfg)
        val_loader = sim_val_loader if sim_val_loader is not None else nusc_val_loader
    else:
        val_loader = nusc_val_loader

    val_loaders_extra: dict = {}
    if zju_val_loader is not None:
        val_loaders_extra["zju"] = zju_val_loader

    if not use_sim and not use_zju:
        return nusc_train, val_loader, val_loaders_extra

    # batch_mix=true short-circuits to per-batch mixing. ZJU is not yet
    # supported in this path (it stays per-batch alternating below).
    batch_mix = bool(sim_cfg.get("batch_mix", False)) if sim_cfg else False
    if batch_mix and use_sim and not use_zju:
        return _build_mixed_batch_loader(cfg, sim_loaders, val_loader, val_loaders_extra)

    # Compose mixed loader. ZJU is treated as another real-data source
    # (paper §3.4 trains on `[4, 22]` = nuScenes + ZJU jointly), separate
    # from sim mixing which is purely about pretraining/aux supervision.
    loaders = [nusc_train]
    probs: list = []
    parts = []
    ratio_real = float(sim_cfg.ratio_real) if sim_cfg is not None else 1.0
    ratio_sim = float(sim_cfg.ratio_sim) if sim_cfg is not None else 0.0
    if use_zju:
        zju_block = cfg.dataset.get("zju", None)
        ratio_zju = float(zju_block.get("ratio", 1.0)) if zju_block is not None else 1.0
    else:
        ratio_zju = 0.0
    probs.append(ratio_real)
    parts.append(f"nuScenes={ratio_real:.2f}")
    if use_zju:
        loaders.append(zju_train)
        probs.append(ratio_zju)
        parts.append(f"zju={ratio_zju:.2f}")
    if use_sim:
        sim_each = ratio_sim / max(1, len(sim_loaders))
        for name, sl in sim_loaders:
            loaders.append(sl)
            probs.append(sim_each)
            parts.append(f"{name}={sim_each:.2f}")
    logger.info("mixed loader: " + " + ".join(parts))
    # primary_idx=0 → epoch length anchored to nuScenes pass-through, so
    # ZJU / sim mixing does not inflate epoch time (see multi_loader.py).
    return (MultiDatasetLoader(loaders, probs, primary_idx=0),
            val_loader, val_loaders_extra)


def _build_mixed_batch_loader(cfg, sim_loaders_unused, val_loader, val_loaders_extra):
    """Re-build per-source loaders with quota-sized batches and wrap in
    MixedBatchLoader so each yielded batch contains a fixed share from
    each source.

    Resolution rule: all sources are forced to nuScenes' training
    resolution (cfg.dataset.train_resize_hw, defaulting to native
    900×1600) so torch.cat works across the batch dim.
    """
    ds_cfg = cfg.dataset
    sim_cfg = ds_cfg.sim
    bsz = int(cfg.training.batch_size)
    ratio_real = float(sim_cfg.ratio_real)
    ratio_sim = float(sim_cfg.ratio_sim)
    s = max(ratio_real + ratio_sim, 1e-8)
    n_real = int(round(bsz * ratio_real / s))
    n_real = max(0, min(bsz, n_real))
    n_sim_total = bsz - n_real

    # Determine sim sources actually enabled (re-discover; sim_loaders_unused
    # was built with the wrong batch size).
    sim_sources = []
    for name in ("hypersim", "vkitti2"):
        sub = sim_cfg.get(name, None)
        if sub is None or not sub.get("enabled", False):
            continue
        if not os.path.isdir(sub.data_root):
            continue
        sim_sources.append((name, sub))
    if not sim_sources or n_sim_total == 0:
        raise ValueError(
            f"batch_mix=true but resolved n_sim_total={n_sim_total}, "
            f"sim_sources={[n for n, _ in sim_sources]} — "
            f"check ratio_real/ratio_sim and sim source enabled flags"
        )
    n_sim_each = max(1, n_sim_total // len(sim_sources))
    # Distribute remainder so quotas exactly sum to bsz
    quotas_sim = [n_sim_each] * len(sim_sources)
    remainder = n_sim_total - sum(quotas_sim)
    for i in range(min(remainder, len(quotas_sim))):
        quotas_sim[i] += 1

    # Resolution to force on every source
    nusc_resize = (tuple(ds_cfg.train_resize_hw)
                   if ds_cfg.get("train_resize_hw") else (900, 1600))
    train_crop = tuple(ds_cfg.train_crop_hw) if ds_cfg.get("train_crop_hw") else None
    common = _common_dataset_kwargs(ds_cfg)

    # ----- nuScenes train ds rebuilt with batch=n_real -----
    photometric_aug = build_photometric_aug(ds_cfg.get("photometric_aug", None))
    rel_depth_dir = ds_cfg.get("rel_depth_dir", None)
    rel_drop = float(ds_cfg.get("rel_depth_dropout_prob", 0.0))
    nusc_train_ds = NuScenesRadarDepthDataset(
        data_root=ds_cfg.data_root,
        split_file=ds_cfg.split_train,
        dense_gt_dir=ds_cfg.get("dense_gt_dir", "depth_acc"),
        radar_3d_dir=ds_cfg.get("radar_3d_dir", "radar_3d"),
        night_ids_file=ds_cfg.get("night_ids_file", None),
        rel_depth_dir=rel_depth_dir,
        rel_depth_dropout_prob=rel_drop,
        resize_to_hw=tuple(ds_cfg.train_resize_hw) if ds_cfg.get("train_resize_hw") else None,
        crop_to_hw=train_crop,
        augmentation=True, photometric_aug=photometric_aug, **common,
    )
    nusc_loader_q = DataLoader(
        nusc_train_ds, batch_size=n_real, shuffle=True,
        num_workers=int(ds_cfg.num_workers), pin_memory=True, drop_last=True,
    ) if n_real > 0 else None

    # ----- sim sources rebuilt with their quotas at nuScenes resolution -----
    base_sim_kw = dict(
        num_radar_points=(int(sim_cfg.num_radar_points_min),
                          int(sim_cfg.num_radar_points_max)),
        depth_noise_std=float(sim_cfg.depth_noise_std),
    )
    real_noise = sim_cfg.get("real_noise", None)
    if real_noise is not None:
        base_sim_kw["real_noise"] = OmegaConf.to_container(real_noise, resolve=True)
    sim_loaders_q = []
    for (name, sub), quota in zip(sim_sources, quotas_sim):
        sim_mode = sub.get("radar_simulation", None) or sim_cfg.radar_simulation
        cls_w = sub.get("class_weights", None)
        cls_w = dict(cls_w) if cls_w else None
        ds = SimRadarDepthDataset(
            data_root=sub.data_root,
            split_file=sub.split_train,
            dataset_type=sub.dataset_type,
            radar_simulation=sim_mode,
            class_weights=cls_w,
            resize_to_hw=nusc_resize,                  # ← force nuScenes shape
            augmentation=True,
            photometric_aug=photometric_aug,
            **common, **base_sim_kw,
        )
        loader = DataLoader(
            ds, batch_size=quota, shuffle=True,
            num_workers=int(ds_cfg.num_workers), pin_memory=True, drop_last=True,
        )
        sim_loaders_q.append(loader)
        logger.info(f"  batch_mix sim source: {name} (quota={quota}, n={len(ds)}, resize={nusc_resize})")

    sources = ([nusc_loader_q] if nusc_loader_q is not None else []) + sim_loaders_q
    quotas = ([n_real] if n_real > 0 else []) + quotas_sim
    parts = []
    if n_real > 0:
        parts.append(f"nuScenes={n_real}")
    for (name, _), q in zip(sim_sources, quotas_sim):
        parts.append(f"{name}={q}")
    logger.info(f"MixedBatchLoader: per-batch quotas → {' + '.join(parts)} (total={sum(quotas)})")
    train_loader = MixedBatchLoader(sources, quotas, primary_idx=0)
    return train_loader, val_loader, val_loaders_extra


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
