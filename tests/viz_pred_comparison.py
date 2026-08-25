"""Per-sample visualization: RGB + 5 preds in row 1, 5 error maps in row 2.

Models compared:
  1. Baseline
  2. no-shared v2 stage 1 — Normal (self-routing)
  3. no-shared v2 stage 1 — Oracle (per-token GT routing)
  4. no-shared v2 stage 2 — Normal
  5. no-shared v2 stage 2 — Oracle

Layout (per sample, 2 rows × 6 cols):
  Row 1: [RGB, pred_baseline, pred_s1_norm, pred_s1_oracle, pred_s2_norm, pred_s2_oracle]
  Row 2: [blank, err_baseline, err_s1_norm, err_s1_oracle, err_s2_norm, err_s2_oracle]
"""
import os, sys
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt
from omegaconf import OmegaConf

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, ROOT)

from src.dataset.nuscenes import NuScenesRadarDepthDataset
from src.model.radar_fusion import MoEFusionBlock
from scripts.train import _build_model


device = "cuda"
BINS = [0.0, 20.0, 50.0, 100.0]
DEPTH_VMAX = 80.0        # depth colormap upper bound
ERR_VMAX = 10.0          # error colormap upper bound
SAMPLE_IDXS = [0, 500, 1500]   # diverse val samples


def load_model(run_dir, ckpt="best.pt"):
    cfg = OmegaConf.load(os.path.join(run_dir, "config.yaml"))
    m = _build_model(cfg, float(cfg.dataset.max_depth), int(cfg.dataset.max_radar_points))
    sd = torch.load(os.path.join(run_dir, ckpt), map_location=device, weights_only=False)
    m.load_state_dict(sd["model"])
    print(f"  loaded {run_dir}/{ckpt}  epoch={sd.get('epoch','?')}")
    return m.eval().to(device)


def compute_token_gt(depth_gt_dense, tok_hw, bins):
    K = len(bins) - 1
    edges = torch.tensor(bins, device=depth_gt_dense.device, dtype=torch.float32)
    ds = depth_gt_dense.squeeze(1)
    pix_bin = torch.zeros_like(ds, dtype=torch.long)
    for i in range(K):
        pix_bin = torch.where((ds >= float(edges[i])) & (ds < float(edges[i+1])),
                              pix_bin.new_tensor(i), pix_bin)
    pix_bin = torch.where(ds >= float(edges[-1]),
                          pix_bin.new_tensor(K-1), pix_bin)
    onehot = F.one_hot(pix_bin, num_classes=K).permute(0, 3, 1, 2).float()
    frac = F.adaptive_avg_pool2d(onehot, tok_hw)
    return frac.argmax(dim=1)


def predict_normal(model, rgb, rp, rm):
    with torch.no_grad():
        out = model(rgb, rp, rm)
    return (out["depth"] if isinstance(out, dict) else out)[0, 0].cpu().numpy()


def predict_oracle(model, rgb, rp, rm, dgd):
    moe_blocks = [blk for blk in
                  (model.radar_fusion.node_blocks[2], model.radar_fusion.edge_blocks[2])
                  if isinstance(blk, MoEFusionBlock)]
    current_gt = {}
    def make_hook(blk):
        def _h(_m, _i, out):
            gt = current_gt.get(id(blk))
            if gt is None: return out
            forced = torch.full_like(out, -1e4)
            forced.scatter_(1, gt.unsqueeze(1), 1e4)
            return forced
        return _h
    hooks = [blk.router.register_forward_hook(make_hook(blk)) for blk in moe_blocks]
    current_gt[id(moe_blocks[0])] = compute_token_gt(dgd, (29, 50), BINS)
    current_gt[id(moe_blocks[1])] = compute_token_gt(dgd, (15, 25), BINS)
    with torch.no_grad():
        out = model(rgb, rp, rm)
    pred_np = (out["depth"] if isinstance(out, dict) else out)[0, 0].cpu().numpy()
    for h in hooks: h.remove()
    return pred_np


def denormalize_rgb(rgb_norm_tensor):
    """Dataset uses rgb/255*2-1 (see BaseRadarDepthDataset._normalize_rgb).
    Reverse: (x+1)/2 → [0, 1]."""
    x = rgb_norm_tensor.cpu().numpy()
    x = (x + 1.0) / 2.0
    x = np.clip(x, 0, 1).transpose(1, 2, 0)
    return x


def main():
    OUT = "output"
    baseline_dir = f"{OUT}/shape_lidar_grad_shape_edge_res"
    s1_dir = f"{OUT}/radartaco_moe_v2/radartaco_moe_stage1_v2"
    s2_dir = f"{OUT}/radartaco_moe_v2/radartaco_moe_stage2_v2"

    print("Loading models...")
    m_base = load_model(baseline_dir)
    m_s1 = load_model(s1_dir)
    m_s2 = load_model(s2_dir)

    base_cfg = OmegaConf.load(f"{baseline_dir}/config.yaml")
    split_file = os.path.join(base_cfg.dataset.data_root, "splits", "val.txt")
    ds = NuScenesRadarDepthDataset(
        data_root=base_cfg.dataset.data_root, split_file=split_file,
        dense_gt_dir=base_cfg.dataset.dense_gt_dir,
        radar_3d_dir=base_cfg.dataset.radar_3d_dir,
        night_ids_file=base_cfg.dataset.night_ids_file,
        max_radar_points=int(base_cfg.dataset.max_radar_points),
        max_depth=float(base_cfg.dataset.max_depth),
        min_depth=float(base_cfg.dataset.min_depth),
        augmentation=False,
    )
    print(f"Dataset: {len(ds):,} samples\n")

    N_SAMPLES = len(SAMPLE_IDXS)
    N_COLS = 6
    fig, axes = plt.subplots(N_SAMPLES * 2, N_COLS,
                              figsize=(N_COLS * 4.2, N_SAMPLES * 2 * 2.5))

    for si, idx in enumerate(SAMPLE_IDXS):
        print(f"Sample {idx}...")
        s = ds[idx]
        rgb = s["rgb_norm"].unsqueeze(0).to(device)
        rp = s["radar_points"].unsqueeze(0).to(device)
        rm = s["radar_mask"].unsqueeze(0).to(device)
        dgd = s["depth_gt_dense"].unsqueeze(0).to(device)
        # Use sparse LiDAR GT for error map (matches metric); dilate the
        # point cloud below so individual points are visible in the plot.
        gt_np = s["depth_gt_lidar"][0].cpu().numpy()
        mask_np = s["valid_mask_lidar"][0].cpu().numpy().astype(bool)

        preds = {
            "baseline":  predict_normal(m_base, rgb, rp, rm),
            "s1_normal": predict_normal(m_s1, rgb, rp, rm),
            "s1_oracle": predict_oracle(m_s1, rgb, rp, rm, dgd),
            "s2_normal": predict_normal(m_s2, rgb, rp, rm),
            "s2_oracle": predict_oracle(m_s2, rgb, rp, rm, dgd),
        }

        rgb_img = denormalize_rgb(s["rgb_norm"])

        row_pred = si * 2
        row_err = si * 2 + 1

        # Row 1: RGB + 5 preds
        axes[row_pred, 0].imshow(rgb_img)
        for c, key in enumerate(["baseline", "s1_normal", "s1_oracle", "s2_normal", "s2_oracle"]):
            axes[row_pred, c + 1].imshow(preds[key], cmap="viridis", vmin=0, vmax=DEPTH_VMAX)

        # Row 2: blank + 5 error maps. Sparse LiDAR points are dilated to a
        # small radius so they're visible; pixels outside the dilated radius
        # are NaN (rendered black via cmap.set_bad).
        # `dist` = pixel-wise distance to nearest LiDAR point.
        # `nn_indices` = indices of nearest LiDAR pixel — used to broadcast
        #   its error to nearby pixels within `POINT_RADIUS`.
        POINT_RADIUS = 4  # pixels
        dist, nn_indices = distance_transform_edt(~mask_np, return_indices=True)
        show_mask = dist <= POINT_RADIUS   # (H, W) where to paint

        axes[row_err, 0].imshow(np.zeros_like(rgb_img))
        for c, key in enumerate(["baseline", "s1_normal", "s1_oracle", "s2_normal", "s2_oracle"]):
            err_sparse = np.abs(preds[key] - gt_np) * mask_np.astype(float)
            err_dilated = err_sparse[tuple(nn_indices)]              # NN fill
            err_display = np.where(show_mask, err_dilated, np.nan)   # NaN outside radius
            cmap = plt.get_cmap("magma").copy()
            cmap.set_bad(color="black")
            axes[row_err, c + 1].imshow(err_display, cmap=cmap, vmin=0, vmax=ERR_VMAX)

        # Turn off ticks + labels for all subplots in this sample
        for r in [row_pred, row_err]:
            for c in range(N_COLS):
                axes[r, c].axis("off")

    plt.subplots_adjust(wspace=0.02, hspace=0.02, left=0.005, right=0.995, top=0.995, bottom=0.005)
    out_path = "output/analysis/viz_pred_comparison.png"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=120, bbox_inches="tight", pad_inches=0.02)
    plt.close()
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
