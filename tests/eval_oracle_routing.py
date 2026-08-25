"""Oracle-routing eval: each token forced to its majority-vote GT bin expert.

Upper bound the router could ever reach — reveals the intrinsic per-expert
specialization quality once routing is perfect.

For each sample:
  1. Compute per-pixel bin from depth_gt_dense
  2. For each MoE block, majority-vote-downsample to that block's grid size
  3. Install router-forward hook that returns one-hot logits at GT bin per token
  4. Standard forward → per-bin MAE
"""
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


def compute_token_gt(depth_gt_dense, tok_hw, bins):
    """Per-token majority-vote GT bin (matches MoEFusionBlock._depth_to_bin
    + adaptive_avg_pool → argmax logic)."""
    K = len(bins) - 1
    edges = torch.tensor(bins, device=depth_gt_dense.device, dtype=torch.float32)
    ds = depth_gt_dense.squeeze(1)                                   # (B, H, W)
    pix_bin = torch.zeros_like(ds, dtype=torch.long)
    for i in range(K):
        pix_bin = torch.where((ds >= float(edges[i])) & (ds < float(edges[i+1])),
                              pix_bin.new_tensor(i), pix_bin)
    pix_bin = torch.where(ds >= float(edges[-1]),
                          pix_bin.new_tensor(K-1), pix_bin)
    onehot = F.one_hot(pix_bin, num_classes=K).permute(0, 3, 1, 2).float()  # (B,K,H,W)
    frac = F.adaptive_avg_pool2d(onehot, tok_hw)                            # (B,K,Ht,Wt)
    return frac.argmax(dim=1)                                                # (B,Ht,Wt)


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


def evaluate_oracle(model, ds, indices, ranges, bins, mode="normal"):
    """mode: 'normal' (router self-routes), 'oracle' (per-token GT forcing)."""
    # Identify MoE blocks
    moe_blocks = []
    for blk in (model.radar_fusion.node_blocks[2], model.radar_fusion.edge_blocks[2]):
        if isinstance(blk, MoEFusionBlock):
            moe_blocks.append(blk)

    K = len(bins) - 1
    current_gt = {}  # id(block) -> (B,Ht,Wt) tensor

    def make_hook(block):
        def _hook(_mod, _inp, out):
            gt = current_gt.get(id(block))
            if gt is None:
                return out
            # out shape (B, K, Ht, Wt) — replace with one-hot at gt bin
            forced = torch.full_like(out, -1e4)
            # scatter 1e4 at (B, gt, Ht, Wt)
            forced.scatter_(1, gt.unsqueeze(1), 1e4)
            return forced
        return _hook

    hooks = []
    if mode == "oracle":
        for blk in moe_blocks:
            hooks.append(blk.router.register_forward_hook(make_hook(blk)))

    acc = defaultdict(lambda: [0.0, 0])
    with torch.no_grad():
        for i in tqdm(indices, desc=mode, leave=False):
            s = ds[int(i)]
            rgb = s["rgb_norm"].unsqueeze(0).to(device)
            rp = s["radar_points"].unsqueeze(0).to(device)
            rm = s["radar_mask"].unsqueeze(0).to(device)

            if mode == "oracle":
                dgd = s["depth_gt_dense"].unsqueeze(0).to(device)
                for blk in moe_blocks:
                    # Match each block's spatial resolution — need to run a dummy
                    # forward pass or know the shape a priori. Level 2 node = (29,50),
                    # edge = (15,25) for 900x1600 input. Compute from feature stride.
                    # Actually easier: temporarily run to get shape, then set GT.
                    pass
                # Compute expected token grid shape from a first forward pass
                # (or hardcode L2 shapes). Simplest: hardcode.
                # L2 node = /32, L2 edge = /64 on 900x1600 → (29, 50) and (15, 25)
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
    ap.add_argument("--n", type=int, default=200)
    args = ap.parse_args()

    ranges = [(0.0, 20.0), (20.0, 50.0), (50.0, 100.0)]
    range_names = [f"[{lo},{hi})m" for (lo, hi) in ranges]
    bins = [0.0, 20.0, 50.0, 100.0]
    OUT = "output"

    runs = [
        ("warm25 stage1 best",        f"{OUT}/radartaco_moe_stage1_warm25",           "best.pt"),
        ("warm25 stage2 best",        f"{OUT}/radartaco_moe_stage2_warm25",           "best.pt"),
        ("pixmask v1 stage1 best",    f"{OUT}/radartaco_moe_stage1_warm25_pixmask",   "best.pt"),
        ("pixmask v1 stage2 best",    f"{OUT}/radartaco_moe_stage2_warm25_pixmask",   "best.pt"),
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

    results = {"Baseline @ e46 (cached)": BASELINE_MAE}
    for name, run_dir, ckpt in runs:
        print(f"\n=== {name} — Normal ===")
        m = load_model(run_dir, ckpt)
        results[f"{name} Normal"] = evaluate_oracle(m, ds, idxs, ranges, bins, mode="normal")
        print(f"\n=== {name} — Oracle routing ===")
        results[f"{name} Oracle"] = evaluate_oracle(m, ds, idxs, ranges, bins, mode="oracle")
        del m; torch.cuda.empty_cache()

    print("\n" + "=" * 90)
    print("MAE table (val 200 samples, per-bin LiDAR GT)")
    print("=" * 90)
    print(f"{'Setting':<38}" + "  ".join(f"{r:>14}" for r in range_names))
    print("-" * 90)
    for name, res in results.items():
        row = f"{name:<38}"
        for r in range_names:
            row += f"  {res[r]:>14.4f}"
        print(row)

    print("\n" + "=" * 90)
    print("Δ vs Baseline (negative = better)")
    print("=" * 90)
    base = BASELINE_MAE
    print(f"{'Setting':<38}" + "  ".join(f"{r:>18}" for r in range_names))
    print("-" * 90)
    for name, res in results.items():
        if name.startswith("Baseline"): continue
        row = f"{name:<38}"
        for r in range_names:
            d = res[r] - base[r]
            pct = 100 * d / base[r]
            row += f"  {d:>+8.4f} ({pct:+5.1f}%)"
        print(row)


if __name__ == "__main__":
    main()
