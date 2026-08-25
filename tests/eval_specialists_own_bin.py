"""Full-model specialists (shape_edge_res_bin_{near,mid,far}) —
   each evaluated on ITS OWN bin only. Full test split, no clipping.
"""
import argparse, os, sys, time
from collections import defaultdict

import numpy as np
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, ROOT)

from src.dataset.nuscenes import NuScenesRadarDepthDataset
from scripts.train import _build_model


device = "cuda" if torch.cuda.is_available() else "cpu"


def load_model(run_dir, ckpt="best.pt"):
    cfg = OmegaConf.load(os.path.join(run_dir, "config.yaml"))
    m = _build_model(cfg, float(cfg.dataset.max_depth), int(cfg.dataset.max_radar_points))
    sd = torch.load(os.path.join(run_dir, ckpt), map_location=device, weights_only=False)
    m.load_state_dict(sd["model"])
    print(f"  loaded {run_dir}/{ckpt}  epoch={sd.get('epoch','?')}")
    return m.eval().to(device)


def eval_one_bin(model, ds, indices, lo, hi):
    err_sum, n_sum = 0.0, 0
    with torch.no_grad():
        for i in tqdm(indices, desc=f"[{lo},{hi})m", leave=False):
            s = ds[int(i)]
            rgb = s["rgb_norm"].unsqueeze(0).to(device)
            rp = s["radar_points"].unsqueeze(0).to(device)
            rm = s["radar_mask"].unsqueeze(0).to(device)
            out = model(rgb, rp, rm)
            pred = out["depth"] if isinstance(out, dict) else out
            pred_np = pred[0, 0].cpu().numpy()
            gt_np = s["depth_gt_lidar"][0].cpu().numpy()
            mask_np = s["valid_mask_lidar"][0].cpu().numpy().astype(bool)
            m = mask_np & (gt_np >= lo) & (gt_np < hi) & (gt_np > 0)
            if m.any():
                err_sum += float(np.abs(pred_np[m] - gt_np[m]).sum())
                n_sum += int(m.sum())
    return err_sum / max(n_sum, 1), n_sum


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--split", default="test")
    args = ap.parse_args()

    OUT = "output"
    runs = [
        ("Near spec",  f"{OUT}/shape_edge_res_bin_near", (0.0, 20.0)),
        ("Mid spec",   f"{OUT}/shape_edge_res_bin_mid",  (20.0, 50.0)),
        ("Far spec",   f"{OUT}/shape_edge_res_bin_far",  (50.0, 100.0)),
    ]

    base_cfg = OmegaConf.load(f"{OUT}/shape_lidar_grad_shape_edge_res/config.yaml")
    split_file = os.path.join(base_cfg.dataset.data_root, "splits", f"{args.split}.txt")
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
    n = args.n or len(ds)
    idxs = list(range(n)) if n == len(ds) else np.linspace(0, len(ds)-1, n).astype(int).tolist()
    print(f"Eval on {args.split} — {len(idxs):,} samples\n")

    results = []
    for name, run_dir, (lo, hi) in runs:
        print(f"=== {name} — own bin [{lo},{hi})m ===")
        t0 = time.time()
        m = load_model(run_dir)
        mae, n = eval_one_bin(m, ds, idxs, lo, hi)
        results.append((name, lo, hi, mae, n))
        print(f"  MAE = {mae:.4f}  (pixels={n:,}, {time.time()-t0:.0f}s)\n")
        del m; torch.cuda.empty_cache()

    print("\n" + "=" * 68)
    print(f"Specialist own-bin MAE  ({args.split} split, {len(idxs):,} samples)")
    print("=" * 68)
    print(f"{'Specialist':<12} {'Range':<14} {'MAE':>10}   {'Pixels':>14}")
    print("-" * 68)
    for name, lo, hi, mae, n in results:
        print(f"{name:<12} [{lo:>4.0f},{hi:>4.0f})m  {mae:>10.4f}   {n:>14,}")


if __name__ == "__main__":
    main()
