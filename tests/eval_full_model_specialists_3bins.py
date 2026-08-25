"""Re-eval full-model specialists (shape_edge_res_bin_{near,mid,far}) on
test split with 3 depth bins [0,20)/[20,50)/[50,100)m + overall.

Also includes:
  • Baseline @ e46
  • Oracle combination (each pixel → its own bin's specialist prediction)
"""
import argparse, os, sys, json, time
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


def load_model(run_dir, ckpt):
    cfg = OmegaConf.load(os.path.join(run_dir, "config.yaml"))
    m = _build_model(cfg, float(cfg.dataset.max_depth), int(cfg.dataset.max_radar_points))
    sd = torch.load(os.path.join(run_dir, ckpt), map_location=device, weights_only=False)
    m.load_state_dict(sd["model"])
    print(f"  loaded {run_dir}/{ckpt}  epoch={sd.get('epoch','?')}")
    return m.eval().to(device)


def per_range(pred, gt, mask, ranges):
    out = {}
    for (lo, hi) in ranges:
        m = mask & (gt >= lo) & (gt < hi) & (gt > 0)
        if m.any():
            err = np.abs(pred[m] - gt[m])
            out[f"[{lo},{hi})m"] = (float(err.sum()), int(m.sum()))
        else:
            out[f"[{lo},{hi})m"] = (0.0, 0)
    m = mask & (gt > 0) & (gt < 100.0)
    if m.any():
        err = np.abs(pred[m] - gt[m])
        out["0-100m"] = (float(err.sum()), int(m.sum()))
    else:
        out["0-100m"] = (0.0, 0)
    return out


def evaluate_single(model, ds, indices, ranges):
    """Standard eval — one model, all pixels."""
    acc = defaultdict(lambda: [0.0, 0])
    all_preds = {}  # cache for oracle combine
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
            all_preds[int(i)] = pred_np
    return {k: (v[0]/v[1] if v[1]>0 else float("nan")) for k, v in acc.items()}, all_preds


def evaluate_oracle_combine(preds_by_model, gts_by_idx, ranges, bin_edges=(0, 20, 50, 100)):
    """Oracle combine: each pixel routed to its GT-bin's specialist prediction.

    preds_by_model: dict of {model_name: {idx: pred_np}}
    gts_by_idx:     dict of {idx: (gt_lidar_np, mask_lidar_np, gt_dense_np)}
    """
    # Specialist name → bin index
    spec_by_bin = {0: "Near spec", 1: "Mid spec", 2: "Far spec"}
    acc = defaultdict(lambda: [0.0, 0])
    for idx in tqdm(sorted(gts_by_idx.keys()), desc="oracle-combine", leave=False):
        gt_np, mask_np, gt_dense_np = gts_by_idx[idx]
        # Build per-pixel routed prediction
        combined = np.zeros_like(gt_np)
        # Route by GT dense bin (each pixel → its bin's specialist)
        for bi, spec in spec_by_bin.items():
            lo = bin_edges[bi]; hi = bin_edges[bi+1]
            pix_in_bin = (gt_dense_np >= lo) & (gt_dense_np < hi)
            if bi == 2:  # far includes ≥ upper edge (max_depth fills)
                pix_in_bin = pix_in_bin | (gt_dense_np >= hi)
            combined[pix_in_bin] = preds_by_model[spec][idx][pix_in_bin]
        r = per_range(combined, gt_np, mask_np, ranges)
        for k, (se, n) in r.items():
            acc[k][0] += se; acc[k][1] += n
    return {k: (v[0]/v[1] if v[1]>0 else float("nan")) for k, v in acc.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", default="output/analysis/eval_full_model_specialists_test.json")
    args = ap.parse_args()

    ranges = [(0.0, 20.0), (20.0, 50.0), (50.0, 100.0)]
    range_names = [f"[{lo},{hi})m" for (lo, hi) in ranges] + ["0-100m"]
    OUT = "output"

    runs = [
        ("Baseline",   f"{OUT}/shape_lidar_grad_shape_edge_res"),
        ("Near spec",  f"{OUT}/shape_edge_res_bin_near"),
        ("Mid spec",   f"{OUT}/shape_edge_res_bin_mid"),
        ("Far spec",   f"{OUT}/shape_edge_res_bin_far"),
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
    print(f"Eval on {args.split} split — {len(idxs):,} samples")

    all_results = {}
    all_preds = {}

    # Cache GT for oracle combine
    print("\nCaching GT tensors...")
    gts = {}
    for i in tqdm(idxs, leave=False):
        s = ds[int(i)]
        gts[int(i)] = (
            s["depth_gt_lidar"][0].cpu().numpy(),
            s["valid_mask_lidar"][0].cpu().numpy().astype(bool),
            s["depth_gt_dense"][0, 0].cpu().numpy(),
        )

    for name, run_dir in runs:
        print("\n" + "=" * 80); print(f"### {name}"); print("=" * 80)
        t0 = time.time()
        m = load_model(run_dir, "best.pt")
        res, preds = evaluate_single(m, ds, idxs, ranges)
        all_results[name] = res
        all_preds[name] = preds
        print(f"  done in {time.time()-t0:.0f}s")
        del m; torch.cuda.empty_cache()

    # Oracle combine using specialists
    print("\n" + "=" * 80); print("### Oracle Combined (3 specialists, per-pixel GT routing)"); print("=" * 80)
    all_results["Oracle Combined"] = evaluate_oracle_combine(
        all_preds, gts, ranges,
    )

    # Save JSON
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    # Strip preds from saved JSON — too large
    to_save = {"split": args.split, "n_samples": len(idxs), "results": all_results}
    with open(args.out, "w") as f:
        json.dump(to_save, f, indent=2)
    print(f"\nSaved: {args.out}")

    # Report
    print("\n" + "=" * 100)
    print(f"MAE table ({args.split} split, {len(idxs):,} samples)")
    print("=" * 100)
    print(f"{'Setting':<24}" + "  ".join(f"{r:>14}" for r in range_names))
    print("-" * 100)
    for name, res in all_results.items():
        row = f"{name:<24}"
        for r in range_names:
            row += f"  {res[r]:>14.4f}"
        print(row)

    print("\n" + "=" * 100)
    print("Δ vs Baseline (negative = better)")
    print("=" * 100)
    base = all_results["Baseline"]
    print(f"{'Setting':<24}" + "  ".join(f"{r:>18}" for r in range_names))
    for name, res in all_results.items():
        if name == "Baseline": continue
        row = f"{name:<24}"
        for r in range_names:
            d = res[r] - base[r]
            pct = 100 * d / base[r]
            row += f"  {d:>+8.4f} ({pct:+6.1f}%)"
        print(row)


if __name__ == "__main__":
    main()
