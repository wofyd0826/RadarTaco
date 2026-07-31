"""Oracle depth-routing ensemble evaluation.

Forward the 3 depth-specialists (near/mid/far) on the test split, then
for each pixel select the specialist whose training range contains the
GT depth at that pixel. Evaluate the combined prediction against LiDAR
GT using the same DepthEvaluator as scripts/eval.py so numbers are
directly comparable to eval_independent/metrics.json.

Bins used for routing:
    near   → gt ∈ [0, 20)   m
    mid    → gt ∈ [20, 50)  m
    far    → gt ∈ [50, 100) m

Usage:
    python scripts/eval_depth_oracle_combine.py \
        [+near=output/shape_edge_res_bin_near/best.pt] \
        [+mid=output/shape_edge_res_bin_mid/best.pt] \
        [+far=output/shape_edge_res_bin_far/best.pt] \
        [+eval_split=test] [+eval_out_dir=...]
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

from src.dataset.nuscenes import NuScenesRadarDepthDataset  # noqa: E402
from src.evaluation.metrics import DepthEvaluator            # noqa: E402
from src.model.radartaco import RadarTaco                    # noqa: E402


logger = logging.getLogger("radartaco.eval_oracle")


def _load_ckpt(cfg_path: str, ckpt_path: str, device):
    saved = OmegaConf.load(cfg_path)
    m = RadarTaco(
        radar_encoder_name=saved.model.radar_encoder,
        max_depth=float(saved.dataset.max_depth),
        max_radar_points=int(saved.dataset.max_radar_points),
        k_neighbors=int(saved.model.k_neighbors),
        a_l=tuple(saved.model.a_l),
        radar_channels=tuple(saved.model.radar_channels),
        attn_heads=int(saved.model.attn_heads),
        mlp_hidden=int(saved.model.get("mlp_hidden", 128)),
        pretrained_image_encoder=False,
        output_mode=str(saved.model.get("output_mode", "metric")),
        min_depth_clip=float(saved.model.get("min_depth_clip", 0.5)),
        multi_scale=bool(saved.model.get("multi_scale", False)),
        multi_scale_levels=tuple(saved.model.get("multi_scale_levels", (2, 4, 8, 16))),
        use_aux_branch=bool(saved.model.get("use_aux_branch", False)),
    )
    sd = torch.load(ckpt_path, map_location=device, weights_only=False)
    m.load_state_dict(sd["model"])
    return m.eval().to(device)


@hydra.main(version_base="1.2", config_path="../config", config_name="config")
def main(cfg: DictConfig):
    logging.basicConfig(level=logging.INFO,
                        format="[%(asctime)s][%(name)s][%(levelname)s] - %(message)s")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    ck_near = cfg.get("near", "output/shape_edge_res_bin_near/best.pt")
    ck_mid  = cfg.get("mid",  "output/shape_edge_res_bin_mid/best.pt")
    ck_far  = cfg.get("far",  "output/shape_edge_res_bin_far/best.pt")

    ck_near = os.path.join(ROOT, ck_near) if not os.path.isabs(ck_near) else ck_near
    ck_mid  = os.path.join(ROOT, ck_mid)  if not os.path.isabs(ck_mid)  else ck_mid
    ck_far  = os.path.join(ROOT, ck_far)  if not os.path.isabs(ck_far)  else ck_far

    for p in (ck_near, ck_mid, ck_far):
        if not os.path.exists(p):
            raise SystemExit(f"checkpoint missing: {p}")

    logger.info(f"loading NEAR from {ck_near}")
    m_near = _load_ckpt(os.path.join(os.path.dirname(ck_near), "config.yaml"), ck_near, device)
    logger.info(f"loading MID  from {ck_mid}")
    m_mid  = _load_ckpt(os.path.join(os.path.dirname(ck_mid),  "config.yaml"), ck_mid,  device)
    logger.info(f"loading FAR  from {ck_far}")
    m_far  = _load_ckpt(os.path.join(os.path.dirname(ck_far),  "config.yaml"), ck_far,  device)

    split = cfg.get("eval_split", "test")
    # route_by: 'lidar' → route by sparse LiDAR GT (only affects metric-relevant
    # pixels — non-LiDAR pixels don't reach the metric anyway).
    # 'dense' → route by dense_gt (depth_edge_res) — full-image routing,
    # matches how specialists were trained (loss mask used dense_gt).
    route_by = cfg.get("route_by", "dense")
    assert route_by in ("lidar", "dense")
    out_dir = cfg.get("eval_out_dir",
                      os.path.join(ROOT, "output/depth_oracle_combine",
                                   f"eval_independent_{split}_route_{route_by}"))
    os.makedirs(out_dir, exist_ok=True)

    split_file = getattr(cfg.dataset, f"split_{split}", None) or \
        os.path.join(cfg.dataset.data_root, "splits", f"{split}.txt")
    logger.info(f"eval split: {split} → {split_file}")

    ds = NuScenesRadarDepthDataset(
        data_root=cfg.dataset.data_root,
        split_file=split_file,
        dense_gt_dir=cfg.dataset.dense_gt_dir,
        radar_3d_dir=cfg.dataset.radar_3d_dir,
        night_ids_file=cfg.dataset.night_ids_file,
        max_radar_points=int(cfg.dataset.max_radar_points),
        max_depth=float(cfg.dataset.max_depth),
        min_depth=float(cfg.dataset.min_depth),
        augmentation=False,
    )
    logger.info(f"dataset size: {len(ds)}")

    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)
    evaluator = DepthEvaluator()

    NEAR_HI, MID_HI = 20.0, 50.0  # bin boundaries (matches training)

    all_metrics = []
    with torch.no_grad():
        for i, batch in enumerate(tqdm(loader, total=len(loader))):
            rgb = batch["rgb_norm"].to(device, non_blocking=True)
            rp  = batch["radar_points"].to(device, non_blocking=True)
            rm  = batch["radar_mask"].to(device, non_blocking=True)

            p_near = m_near(rgb, rp, rm)
            p_mid  = m_mid (rgb, rp, rm)
            p_far  = m_far (rgb, rp, rm)
            # Models may return dict {"depth": ...} for multi-scale or Tensor
            def _flat(p):
                return p["depth"] if isinstance(p, dict) else p
            p_near, p_mid, p_far = _flat(p_near), _flat(p_mid), _flat(p_far)

            gt_lidar = batch["depth_gt_lidar"].to(device)   # (1,1,H,W)
            mask_l   = batch["valid_mask_lidar"].to(device) # (1,1,H,W)

            # Oracle routing key — choose which GT decides the bin per pixel.
            #   'lidar': sparse; non-LiDAR pixels default to `near` but are
            #            outside the metric mask (mask_l), so they don't
            #            affect reported mae.
            #   'dense': depth_edge_res; matches the training bin definition
            #            (loss mask used dense_gt) and gives coherent
            #            routing across the whole image.
            if route_by == "lidar":
                key = gt_lidar
            else:
                key = batch["depth_gt_dense"].to(device)

            near_m = (key > 0) & (key < NEAR_HI)
            mid_m  = (key >= NEAR_HI) & (key < MID_HI)
            far_m  = (key >= MID_HI)

            combined = torch.where(near_m, p_near,
                        torch.where(mid_m, p_mid,
                            torch.where(far_m, p_far, p_near)))

            pred_np = combined[0, 0].cpu().numpy()
            gt_np   = gt_lidar[0, 0].cpu().numpy()
            mask_np = mask_l[0, 0].cpu().numpy().astype(bool)
            is_night = bool(batch["is_night"].item())

            m = evaluator.evaluate_sample(pred_np, gt_np, mask_np, is_night=is_night)
            all_metrics.append(m)

    agg = evaluator.aggregate_metrics(all_metrics)
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(agg, f, indent=2)
    logger.info(f"wrote {os.path.join(out_dir, 'metrics.json')}")

    print("\n=== DEPTH-ORACLE COMBINE metrics ===")
    for cat in ("overall", "per_range", "far", "day", "night"):
        if cat not in agg: continue
        print(f"\n{cat}:")
        for r, v in agg[cat].items():
            mae = v.get("mae", float("nan"))
            print(f"  {r:>10}  mae={mae:.4f}")


if __name__ == "__main__":
    main()
