"""Evaluate warm25 MoE experiments — best.pt only. Compare to cached baseline."""
import argparse, os, sys
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

# Cached baseline (best.pt @ e46, val 200 samples) from earlier runs.
BASELINE_MAE = {
    "[0.0,20.0)m": 0.7612,
    "[20.0,50.0)m": 3.6500,
    "[50.0,100.0)m": 8.7229,
}


def load_model(run_dir, ckpt):
    cfg = OmegaConf.load(os.path.join(run_dir, "config.yaml"))
    m = _build_model(cfg, float(cfg.dataset.max_depth), int(cfg.dataset.max_radar_points))
    sd = torch.load(os.path.join(run_dir, ckpt), map_location=device, weights_only=False)
    m.load_state_dict(sd["model"])
    print(f"  loaded {run_dir}/{ckpt}  epoch={sd.get('epoch','?')}")
    return m.eval().to(device), cfg


def per_range(pred, gt, mask, ranges):
    out = {}
    for (lo, hi) in ranges:
        m = mask & (gt >= lo) & (gt < hi) & (gt > 0)
        if m.any():
            err = np.abs(pred[m] - gt[m])
            out[f"[{lo},{hi})m"] = (float(err.sum()), int(m.sum()))
        else:
            out[f"[{lo},{hi})m"] = (0.0, 0)
    return out


def evaluate(model, ds, indices, ranges):
    acc = defaultdict(lambda: [0.0, 0])
    with torch.no_grad():
        for i in tqdm(indices, desc="eval", leave=False):
            s = ds[int(i)]
            rgb = s["rgb_norm"].unsqueeze(0).to(device)
            rp = s["radar_points"].unsqueeze(0).to(device)
            rm = s["radar_mask"].unsqueeze(0).to(device)
            out = model(rgb, rp, rm)
            pred = out["depth"] if isinstance(out, dict) else out
            pred_np = pred[0, 0].cpu().numpy()
            gt_np = s["depth_gt_lidar"][0].cpu().numpy()
            mask_np = s["valid_mask_lidar"][0].cpu().numpy().astype(bool)
            r = per_range(pred_np, gt_np, mask_np, ranges)
            for k, (se, n) in r.items():
                acc[k][0] += se; acc[k][1] += n
    return {k: (v[0]/v[1] if v[1]>0 else float("nan")) for k, v in acc.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    args = ap.parse_args()

    ranges = [(0.0, 20.0), (20.0, 50.0), (50.0, 100.0)]
    range_names = [f"[{lo},{hi})m" for (lo, hi) in ranges]
    OUT = "output"

    runs = [
        ("warm25 stage1 best",   f"{OUT}/radartaco_moe_stage1_warm25",           "best.pt"),
        ("warm25 stage2 best",   f"{OUT}/radartaco_moe_stage2_warm25",           "best.pt"),
        ("pixmask v1 stage1 best", f"{OUT}/radartaco_moe_stage1_warm25_pixmask", "best.pt"),
        ("pixmask v1 stage2 best", f"{OUT}/radartaco_moe_stage2_warm25_pixmask", "best.pt"),
    ]

    # Dataset from baseline config
    cfg = OmegaConf.load(f"{OUT}/shape_lidar_grad_shape_edge_res/config.yaml")
    split_file = os.path.join(cfg.dataset.data_root, "splits", "val.txt")
    ds = NuScenesRadarDepthDataset(
        data_root=cfg.dataset.data_root, split_file=split_file,
        dense_gt_dir=cfg.dataset.dense_gt_dir,
        radar_3d_dir=cfg.dataset.radar_3d_dir,
        night_ids_file=cfg.dataset.night_ids_file,
        max_radar_points=int(cfg.dataset.max_radar_points),
        max_depth=float(cfg.dataset.max_depth),
        min_depth=float(cfg.dataset.min_depth),
        augmentation=False,
    )
    idxs = np.linspace(0, len(ds)-1, args.n).astype(int).tolist()

    results = {"Baseline @ e46 (cached)": BASELINE_MAE}
    for name, run_dir, ckpt in runs:
        print(f"\n=== {name} ===")
        m, _ = load_model(run_dir, ckpt)
        results[name] = evaluate(m, ds, idxs, ranges)
        del m; torch.cuda.empty_cache()

    print("\n" + "=" * 90)
    print("MAE table (val 200 samples, per-bin LiDAR GT)")
    print("=" * 90)
    print(f"{'Setting':<30}" + "  ".join(f"{r:>14}" for r in range_names))
    print("-" * 90)
    for name in results:
        row = f"{name:<30}"
        for r in range_names:
            row += f"  {results[name][r]:>14.4f}"
        print(row)

    print("\n" + "=" * 90)
    print("Δ vs Baseline (negative = better)")
    print("=" * 90)
    base = BASELINE_MAE
    print(f"{'Setting':<30}" + "  ".join(f"{r:>18}" for r in range_names))
    print("-" * 90)
    for name in results:
        if name.startswith("Baseline"): continue
        row = f"{name:<30}"
        for r in range_names:
            d = results[name][r] - base[r]
            pct = 100 * d / base[r]
            row += f"  {d:>+8.4f} ({pct:+5.1f}%)"
        print(row)


if __name__ == "__main__":
    main()
