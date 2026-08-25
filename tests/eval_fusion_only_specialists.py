"""Evaluate fusion-only specialists vs baseline on same val subset.

For each of {near, mid, far}, load the specialist and compute per-range MAE.
Compare against baseline. The key head-to-head is per-bin: does specialist X
beat baseline in bin X?
"""
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


def load_model(run_dir, ckpt_name):
    cfg = OmegaConf.load(os.path.join(run_dir, "config.yaml"))
    m = _build_model(cfg, float(cfg.dataset.max_depth), int(cfg.dataset.max_radar_points))
    sd = torch.load(os.path.join(run_dir, ckpt_name), map_location=device, weights_only=False)
    m.load_state_dict(sd["model"])
    print(f"  loaded {run_dir}/{ckpt_name}  epoch={sd.get('epoch','?')}")
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
        for i in tqdm(indices, desc="eval"):
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
                acc[k][0] += se
                acc[k][1] += n
    return {k: (v[0] / v[1] if v[1] > 0 else float("nan")) for k, v in acc.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--split", default="val")
    ap.add_argument("--baseline", default="output/shape_lidar_grad_shape_edge_res")
    ap.add_argument("--baseline_ckpt", default="best.pt")
    ap.add_argument("--specialist_root", default="output")
    ap.add_argument("--specialist_ckpts", nargs="+",
                    default=["best.pt", "epoch_25.pt", "last.pt"])
    args = ap.parse_args()

    ranges = [(0.0, 20.0), (20.0, 50.0), (50.0, 100.0)]
    range_names = [f"[{lo},{hi})m" for (lo, hi) in ranges]
    labels = ["near", "mid", "far"]

    # ── Baseline ──
    print("=== Baseline ===")
    m_base, cfg_base = load_model(args.baseline, args.baseline_ckpt)

    split_file = getattr(cfg_base.dataset, f"split_{args.split}", None) or \
        os.path.join(cfg_base.dataset.data_root, "splits", f"{args.split}.txt")
    ds = NuScenesRadarDepthDataset(
        data_root=cfg_base.dataset.data_root, split_file=split_file,
        dense_gt_dir=cfg_base.dataset.dense_gt_dir,
        radar_3d_dir=cfg_base.dataset.radar_3d_dir,
        night_ids_file=cfg_base.dataset.night_ids_file,
        max_radar_points=int(cfg_base.dataset.max_radar_points),
        max_depth=float(cfg_base.dataset.max_depth),
        min_depth=float(cfg_base.dataset.min_depth),
        augmentation=False,
    )
    idxs = np.linspace(0, len(ds) - 1, args.n).astype(int).tolist()

    results = {}
    results["Baseline"] = evaluate(m_base, ds, idxs, ranges)
    del m_base
    torch.cuda.empty_cache()

    # ── Specialists ──
    for lb in labels:
        for ck in args.specialist_ckpts:
            run = os.path.join(args.specialist_root, f"fusion_only_bin_{lb}")
            if not os.path.exists(os.path.join(run, ck)):
                continue
            print(f"\n=== {lb} @ {ck} ===")
            m, _ = load_model(run, ck)
            results[f"{lb}@{ck}"] = evaluate(m, ds, idxs, ranges)
            del m
            torch.cuda.empty_cache()

    # ── Report ──
    print("\n" + "=" * 78)
    print("MAE table (rows=model, cols=depth range on LiDAR GT)")
    print("=" * 78)
    print(f"{'Setting':<30}" + "  ".join(f"{r:>14}" for r in range_names))
    print("-" * 78)
    for name, mae_by_range in results.items():
        row = f"{name:<30}"
        for r in range_names:
            row += f"  {mae_by_range[r]:>14.4f}"
        print(row)

    # ── Head-to-head per bin ──
    print("\n" + "=" * 78)
    print("Head-to-head: specialist X in bin X vs Baseline in bin X")
    print("=" * 78)
    for lb, r in zip(labels, range_names):
        b_mae = results["Baseline"][r]
        print(f"\n  bin {lb} {r}:")
        print(f"    baseline mae:              {b_mae:.4f}")
        for ck in args.specialist_ckpts:
            key = f"{lb}@{ck}"
            if key not in results: continue
            s_mae = results[key][r]
            delta = s_mae - b_mae
            mark = "✓ 이김" if delta < 0 else "✗ 짐"
            print(f"    {lb}@{ck:<14} mae: {s_mae:.4f}   Δ = {delta:+.4f}  ({mark})")


if __name__ == "__main__":
    main()
