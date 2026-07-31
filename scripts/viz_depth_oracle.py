"""Visualise a few test samples with baseline + 3 depth-specialists + oracle.

Layout per sample (single PNG):
    Row 1 (predictions colored 0-80m):
        RGB+radar | GT_lidar | Baseline | S_near | S_mid | S_far | Oracle
    Row 2 (absolute error 0-10m clipped, LiDAR-mask only):
        (RGB)     | (GT)     | err      | err     | err    | err   | err

Usage:
    python scripts/viz_depth_oracle.py [+n=5] [+eval_split=test]
"""
import os
import sys

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from src.dataset.nuscenes import NuScenesRadarDepthDataset  # noqa: E402
from src.evaluation.viz import (build_grid, colorize_depth, colorize_error,  # noqa: E402
                                overlay_radar_points, rgb_to_uint8)
from src.model.radartaco import RadarTaco                    # noqa: E402


def _load_ckpt(cfg_path, ckpt_path, device):
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


def _label_bar(text, W, height=24, bg=(30, 30, 30), fg=(240, 240, 240)):
    """Return an (H, W, 3) uint8 label banner."""
    from PIL import Image, ImageDraw, ImageFont
    img = Image.new("RGB", (W, height), bg)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    draw.text((6, 4), text, fill=fg, font=font)
    return np.asarray(img)


@hydra.main(version_base="1.2", config_path="../config", config_name="config")
def main(cfg: DictConfig):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    n_samples = int(cfg.get("n", 5))
    split = cfg.get("eval_split", "test")
    out_dir = cfg.get("viz_out_dir",
                      os.path.join(ROOT, "output/depth_oracle_combine/viz"))
    os.makedirs(out_dir, exist_ok=True)

    ckpts = {
        "baseline": "output/shape_lidar_grad_shape_edge_res/best.pt",
        "S_near":   "output/shape_edge_res_bin_near/best.pt",
        "S_mid":    "output/shape_edge_res_bin_mid/best.pt",
        "S_far":    "output/shape_edge_res_bin_far/best.pt",
    }
    models = {}
    for k, p in ckpts.items():
        pabs = os.path.join(ROOT, p)
        cfgp = os.path.join(os.path.dirname(pabs), "config.yaml")
        print(f"[load] {k}: {pabs}")
        models[k] = _load_ckpt(cfgp, pabs, device)

    split_file = getattr(cfg.dataset, f"split_{split}", None) or \
        os.path.join(cfg.dataset.data_root, "splits", f"{split}.txt")
    # Use depth_edge_res (matches training) as the routing authority — it has
    # far higher coverage than depth_acc (90%+ vs ~10%). Sample loading needs
    # to point there explicitly since the default hydra config sets
    # dense_gt_dir=depth_acc.
    dense_gt_dir = cfg.get("dense_gt_dir_for_routing", "depth_edge_res")
    print(f"[routing] using dense_gt_dir={dense_gt_dir} as bin authority")
    ds = NuScenesRadarDepthDataset(
        data_root=cfg.dataset.data_root, split_file=split_file,
        dense_gt_dir=dense_gt_dir,
        radar_3d_dir=cfg.dataset.radar_3d_dir,
        night_ids_file=cfg.dataset.night_ids_file,
        max_radar_points=int(cfg.dataset.max_radar_points),
        max_depth=float(cfg.dataset.max_depth),
        min_depth=float(cfg.dataset.min_depth),
        augmentation=False,
    )

    # Pick a diverse handful: uniformly spaced through the split.
    idxs = np.linspace(0, len(ds) - 1, n_samples).astype(int)
    print(f"[pick] {n_samples} samples at indices: {list(idxs)}")

    NEAR_HI, MID_HI = 20.0, 50.0
    for k, i in enumerate(idxs):
        s = ds[int(i)]
        rgb = s["rgb_norm"].unsqueeze(0).to(device)
        rp  = s["radar_points"].unsqueeze(0).to(device)
        rm  = s["radar_mask"].unsqueeze(0).to(device)
        gt_l = s["depth_gt_lidar"].to(device).unsqueeze(0)
        mask_l = s["valid_mask_lidar"].to(device).unsqueeze(0)

        preds = {}
        with torch.no_grad():
            for name, m in models.items():
                p = m(rgb, rp, rm)
                p = p["depth"] if isinstance(p, dict) else p
                preds[name] = p

        # Oracle combine — route by dense_gt (depth_edge_res, 100% dense).
        key = s["depth_gt_dense"].to(device).unsqueeze(0)
        near_m = (key > 0) & (key < NEAR_HI)
        mid_m  = (key >= NEAR_HI) & (key < MID_HI)
        far_m  = (key >= MID_HI)
        oracle = torch.where(near_m, preds["S_near"],
                    torch.where(mid_m, preds["S_mid"],
                        torch.where(far_m, preds["S_far"], preds["S_near"])))

        n_tot = near_m.numel()
        print(f"  [route] near={100*near_m.sum().item()/n_tot:.1f}%  "
              f"mid={100*mid_m.sum().item()/n_tot:.1f}%  "
              f"far={100*far_m.sum().item()/n_tot:.1f}%")

        # to numpy
        rgb_np  = rgb_to_uint8(s["rgb_norm"].numpy())
        gt_np   = gt_l[0, 0].cpu().numpy()
        mask_np = mask_l[0, 0].cpu().numpy().astype(bool)

        img_rgb_radar = overlay_radar_points(
            rgb_np, s["radar_points"].numpy(), s["radar_mask"].numpy(), radius=4)
        img_gt = colorize_depth(gt_np, mask_np, vmax=80.0, point_radius=3)

        pred_imgs = []
        err_imgs = []
        for name in ("baseline", "S_near", "S_mid", "S_far"):
            pp = preds[name][0, 0].cpu().numpy()
            pred_imgs.append(colorize_depth(pp, vmax=80.0))
            err_imgs.append(colorize_error(pp, gt_np, mask_np, vmax=10.0, point_radius=3))
        pp = oracle[0, 0].cpu().numpy()
        pred_imgs.append(colorize_depth(pp, vmax=80.0))
        err_imgs.append(colorize_error(pp, gt_np, mask_np, vmax=10.0, point_radius=3))

        row1 = [img_rgb_radar, img_gt] + pred_imgs         # 7 cols
        H, W = img_rgb_radar.shape[:2]
        # dark spacer with label
        def _spacer(text):
            spacer = np.zeros((H, W, 3), dtype=np.uint8)
            bar = _label_bar(text, W, height=24, bg=(30, 30, 30))
            spacer[:24] = bar
            return spacer
        row2 = [_spacer("(input RGB)"), _spacer("(sparse LiDAR GT)")] + err_imgs
        grid = build_grid(row1 + row2, ncols=7)

        # Column labels
        col_names = ["RGB+radar", "GT_lidar",
                     "Baseline", "S_near", "S_mid", "S_far", "Oracle"]
        header = np.concatenate(
            [_label_bar(c, W, height=28) for c in col_names], axis=1)
        # sample metadata banner
        is_night = bool(s["is_night"].item())
        title = _label_bar(
            f"idx={int(i)}  sample={s['sample_id']}  night={is_night}",
            grid.shape[1], height=28, bg=(15, 15, 15))
        final = np.concatenate([title, header, grid], axis=0)

        out_path = os.path.join(out_dir, f"viz_{k:02d}_idx{int(i)}.png")
        Image.fromarray(final).save(out_path)
        print(f"[saved] {out_path}  ({final.shape[1]}×{final.shape[0]})")


if __name__ == "__main__":
    main()
