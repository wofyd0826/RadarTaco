"""Full test-split evaluation on baseline + selected MoE models.

Modes:
  - Normal (baseline and MoE self-routing)
  - Oracle (per-token majority-vote GT routing, MoE only)

Ranges: [0,20)/[20,50)/[50,100)m + overall 0-100m.
"""
import argparse, os, sys, json, time
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from tqdm import tqdm

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, ROOT)

from src.dataset.nuscenes import NuScenesRadarDepthDataset
from src.model.radar_fusion import MoEFusionBlock
from scripts.train import _build_model


device = "cuda" if torch.cuda.is_available() else "cpu"


def load_model(run_dir, ckpt):
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


def per_range(pred, gt, mask, ranges):
    out = {}
    for (lo, hi) in ranges:
        m = mask & (gt >= lo) & (gt < hi) & (gt > 0)
        if m.any():
            err = np.abs(pred[m] - gt[m])
            out[f"[{lo},{hi})m"] = (float(err.sum()), int(m.sum()))
        else:
            out[f"[{lo},{hi})m"] = (0.0, 0)
    # overall 0-100m
    m = mask & (gt > 0) & (gt < 100.0)
    if m.any():
        err = np.abs(pred[m] - gt[m])
        out["0-100m"] = (float(err.sum()), int(m.sum()))
    else:
        out["0-100m"] = (0.0, 0)
    return out


def evaluate(model, ds, indices, ranges, mode, bins=None):
    """mode: 'normal' | 'oracle'"""
    moe_blocks = [blk for blk in
                  (model.radar_fusion.node_blocks[2], model.radar_fusion.edge_blocks[2])
                  if isinstance(blk, MoEFusionBlock)]
    current_gt = {}
    hooks = []
    if mode == "oracle" and moe_blocks:
        def make_hook(blk):
            def _h(_m, _i, out):
                gt = current_gt.get(id(blk))
                if gt is None: return out
                forced = torch.full_like(out, -1e4)
                forced.scatter_(1, gt.unsqueeze(1), 1e4)
                return forced
            return _h
        for blk in moe_blocks:
            hooks.append(blk.router.register_forward_hook(make_hook(blk)))

    acc = defaultdict(lambda: [0.0, 0])
    with torch.no_grad():
        for i in tqdm(indices, desc=mode, leave=False):
            s = ds[int(i)]
            rgb = s["rgb_norm"].unsqueeze(0).to(device)
            rp = s["radar_points"].unsqueeze(0).to(device)
            rm = s["radar_mask"].unsqueeze(0).to(device)
            if mode == "oracle" and moe_blocks:
                dgd = s["depth_gt_dense"].unsqueeze(0).to(device)
                current_gt[id(moe_blocks[0])] = compute_token_gt(dgd, (29, 50), bins)
                current_gt[id(moe_blocks[1])] = compute_token_gt(dgd, (15, 25), bins)
            out = model(rgb, rp, rm)
            pred = out["depth"] if isinstance(out, dict) else out
            pred_np = pred[0, 0].cpu().numpy()
            gt_np = s["depth_gt_lidar"][0].cpu().numpy()
            mask_np = s["valid_mask_lidar"][0].cpu().numpy().astype(bool)
            r = per_range(pred_np, gt_np, mask_np, ranges)
            for k, (se, n) in r.items():
                acc[k][0] += se; acc[k][1] += n

    for h in hooks: h.remove()
    current_gt.clear()
    return {k: (v[0]/v[1] if v[1]>0 else float("nan")) for k, v in acc.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--n", type=int, default=None, help="None = full split")
    ap.add_argument("--out", default="output/analysis/eval_full_test.json")
    args = ap.parse_args()

    ranges = [(0.0, 20.0), (20.0, 50.0), (50.0, 100.0)]
    range_names = [f"[{lo},{hi})m" for (lo, hi) in ranges] + ["0-100m"]
    bins = [0.0, 20.0, 50.0, 100.0]
    OUT = "output"

    runs = [
        # (name, run_dir, ckpt, is_moe)
        ("Baseline @ e46",              f"{OUT}/shape_lidar_grad_shape_edge_res",       "best.pt", False),
        ("no-shared v2 stage1 best",    f"{OUT}/radartaco_moe_v2/radartaco_moe_stage1_v2", "best.pt", True),
        ("no-shared v2 stage2 best",    f"{OUT}/radartaco_moe_v2/radartaco_moe_stage2_v2", "best.pt", True),
        ("warm25 stage1 best",          f"{OUT}/radartaco_moe_stage1_warm25",           "best.pt", True),
        ("warm25 stage2 best",          f"{OUT}/radartaco_moe_stage2_warm25",           "best.pt", True),
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
    print(f"Eval on {args.split} split — {len(idxs):,} of {len(ds):,} samples")

    all_results = {}
    for name, run_dir, ckpt, is_moe in runs:
        print("\n" + "=" * 80); print(f"### {name}"); print("=" * 80)
        t0 = time.time()
        m = load_model(run_dir, ckpt)
        res = {}
        res["Normal"] = evaluate(m, ds, idxs, ranges, "normal")
        print(f"  Normal done in {time.time()-t0:.0f}s")
        if is_moe:
            t1 = time.time()
            res["Oracle"] = evaluate(m, ds, idxs, ranges, "oracle", bins=bins)
            print(f"  Oracle done in {time.time()-t1:.0f}s")
        all_results[name] = res
        del m; torch.cuda.empty_cache()

    # Save JSON
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"split": args.split, "n_samples": len(idxs), "results": all_results}, f, indent=2)
    print(f"\nSaved: {args.out}")

    # Print tables
    print("\n" + "=" * 100)
    print(f"MAE table ({args.split} split, {len(idxs):,} samples)")
    print("=" * 100)
    print(f"{'Setting':<32}" + "  ".join(f"{r:>12}" for r in range_names))
    print("-" * 100)
    for name, res in all_results.items():
        for mode, mae in res.items():
            row = f"{name[:28]:<28} {mode:<4}"
            for r in range_names:
                row += f"  {mae[r]:>12.4f}"
            print(row)

    print("\n" + "=" * 100)
    print("Δ vs Baseline Normal (negative = better)")
    print("=" * 100)
    base = all_results["Baseline @ e46"]["Normal"]
    print(f"{'Setting':<32}" + "  ".join(f"{r:>16}" for r in range_names))
    for name, res in all_results.items():
        if name.startswith("Baseline") and len(res) == 1: continue
        for mode, mae in res.items():
            if name.startswith("Baseline"): continue
            row = f"{name[:28]:<28} {mode:<4}"
            for r in range_names:
                d = mae[r] - base[r]
                pct = 100*d/base[r]
                row += f"  {d:>+7.4f} ({pct:+5.1f}%)"
            print(row)


if __name__ == "__main__":
    main()
