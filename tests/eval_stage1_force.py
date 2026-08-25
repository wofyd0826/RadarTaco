"""Force-expert eval on warm25 STAGE 1 checkpoints (no pixmask + pixmask v1)."""
import argparse, os, sys
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
    return out


def eval_force(model, ds, indices, ranges, force_idx):
    hooks = []
    def make_hook(idx):
        def _h(_m, _i, out):
            forced = torch.full_like(out, -1e4)
            forced[:, idx] = 1e4
            return forced
        return _h
    for blk in (model.radar_fusion.node_blocks[2], model.radar_fusion.edge_blocks[2]):
        if isinstance(blk, MoEFusionBlock):
            hooks.append(blk.router.register_forward_hook(make_hook(force_idx)))

    acc = defaultdict(lambda: [0.0, 0])
    with torch.no_grad():
        for i in tqdm(indices, desc=f"force={force_idx}", leave=False):
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
    for h in hooks: h.remove()
    return {k: (v[0]/v[1] if v[1]>0 else float("nan")) for k, v in acc.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    args = ap.parse_args()

    ranges = [(0.0, 20.0), (20.0, 50.0), (50.0, 100.0)]
    range_names = [f"[{lo},{hi})m" for (lo, hi) in ranges]
    labels = ["near", "mid", "far"]
    OUT = "output"

    runs = [
        ("warm25 stage1 best",     f"{OUT}/radartaco_moe_stage1_warm25",           "best.pt"),
        ("pixmask v1 stage1 best", f"{OUT}/radartaco_moe_stage1_warm25_pixmask",   "best.pt"),
    ]

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

    per_exp = {}
    for name, run_dir, ckpt in runs:
        print("\n" + "=" * 80); print(f"### {name}"); print("=" * 80)
        m = load_model(run_dir, ckpt)
        res = {}
        for k, lb in enumerate(labels):
            print(f"-- Force {lb} --")
            res[f"Force-{lb}"] = eval_force(m, ds, idxs, ranges, force_idx=k)
        per_exp[name] = res
        del m; torch.cuda.empty_cache()

    print("\n" + "=" * 90)
    print("Force-expert MAE (val 200 samples)")
    print("=" * 90)
    print(f"{'Setting':<28}" + "  ".join(f"{r:>14}" for r in range_names))
    print("-" * 90)
    print(f"{'Baseline @ e46':<28}" + "  ".join(f"{BASELINE_MAE[r]:>14.4f}" for r in range_names))
    for name, res in per_exp.items():
        for lb in labels:
            row = f"{name[:15]} Force-{lb:<4}"
            for r in range_names:
                row += f"  {res[f'Force-{lb}'][r]:>14.4f}"
            print(row)

    print("\n" + "=" * 90)
    print("Head-to-head — Force expert (own bin) vs Baseline (same bin)")
    print("=" * 90)
    for name, res in per_exp.items():
        print(f"\n### {name}")
        for k, (lb, r) in enumerate(zip(labels, range_names)):
            b_mae = BASELINE_MAE[r]
            f_mae = res[f"Force-{lb}"][r]
            delta = f_mae - b_mae
            pct = 100 * delta / b_mae
            mark = "✓ 이김" if delta < 0 else "✗ 짐"
            print(f"  {lb} bin {r}:  baseline={b_mae:.4f}   Force={f_mae:.4f}   Δ={delta:+.4f} ({pct:+.1f}%)  {mark}")


if __name__ == "__main__":
    main()
