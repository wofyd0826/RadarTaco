"""Cascade re-routing at inference time.

For each sample, run FIVE forwards on the trained model:
  1. Normal          — router's own gate.
  2. CascadedHard    — force router logits to one-hot(argmax(pixel_frac(depth_1)))
                       where depth_1 is the model's own first-pass prediction.
                       Router is effectively replaced by "bucketize model's own depth".
  3. CascadedSoft    — force router logits to log(pixel_frac(depth_1)),
                       i.e., soft gate driven by the model's own first-pass
                       bucketized fraction.
  4. OracleHard      — force router to one-hot(argmax(pixel_frac(gt))).
  5. OracleSoft      — force router to log(pixel_frac(gt)).

Question: does routing informed by the model's own first-pass depth
prediction (Cascaded*) yield better final depth than the router's own gate
(Normal)? If yes, cascade routing is worth training.

Reference: Oracle* is the upper bound (perfect routing given GT).

Usage:
  python tests/analyze_cascade_reroute.py --run-dir output/radartaco_moe_stage2_v6c_confshared
"""
import argparse, json, os, sys
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


def load_model(run_dir, ckpt="best.pt"):
    cfg = OmegaConf.load(os.path.join(run_dir, "config.yaml"))
    m = _build_model(cfg, float(cfg.dataset.max_depth), int(cfg.dataset.max_radar_points))
    sd = torch.load(os.path.join(run_dir, ckpt), map_location=device, weights_only=False)
    m.load_state_dict(sd["model"])
    print(f"  loaded {run_dir}/{ckpt}  epoch={sd.get('epoch','?')}")
    return m.eval().to(device), cfg


def _collect_moe_blocks(model):
    out = []
    for l in range(len(model.radar_fusion.node_blocks)):
        nb = model.radar_fusion.node_blocks[l]
        eb = model.radar_fusion.edge_blocks[l]
        if isinstance(nb, MoEFusionBlock): out.append((f"L{l}_node", nb))
        if isinstance(eb, MoEFusionBlock): out.append((f"L{l}_edge", eb))
    return out


def compute_pixel_bin_fraction(depth, tok_hw, bins, return_type="soft"):
    """Given (B, 1, H, W) depth, bucketize to K bins and pool to tok_hw
    grid. return_type: 'hard' → (B, H, W) int64, 'soft' → (B, K, H, W)."""
    K = len(bins) - 1
    edges = torch.tensor(list(bins), device=depth.device, dtype=torch.float32)
    ds = depth.squeeze(1)
    pix_bin = torch.zeros_like(ds, dtype=torch.long)
    for i in range(K):
        pix_bin = torch.where(
            (ds >= float(edges[i])) & (ds < float(edges[i + 1])),
            pix_bin.new_tensor(i), pix_bin)
    pix_bin = torch.where(ds >= float(edges[-1]),
                          pix_bin.new_tensor(K - 1), pix_bin)
    onehot = F.one_hot(pix_bin, num_classes=K).permute(0, 3, 1, 2).float()
    frac = F.adaptive_avg_pool2d(onehot, tok_hw)
    if return_type == "soft":
        return frac
    return frac.argmax(dim=1)


def install_router_forcing(moe_blocks, holder, bins):
    """Router hook that forces logits based on `holder["mode"]` and
    `holder["depth_src"]`.
      mode = 'hard'/'soft' : bucketize holder['depth_src'] to logits.
      mode = None          : passthrough.
    """
    hooks = []
    def _hook(_m, _i, out):
        mode = holder.get("mode")
        depth_src = holder.get("depth_src")
        if mode is None or depth_src is None:
            return out
        tok_hw = out.shape[-2:]
        if mode == "soft":
            frac = compute_pixel_bin_fraction(depth_src, tok_hw, bins,
                                              return_type="soft")
            return torch.log(frac.clamp_min(1e-8))
        # hard
        tok_gt = compute_pixel_bin_fraction(depth_src, tok_hw, bins,
                                            return_type="hard")
        forced = torch.full_like(out, -1e4)
        forced.scatter_(1, tok_gt.unsqueeze(1), 1e4)
        return forced
    for _, blk in moe_blocks:
        hooks.append(blk.router.register_forward_hook(_hook))
    return hooks


def per_range_l1(pred, gt, mask, ranges):
    out = {}
    for (lo, hi) in ranges:
        m = mask & (gt >= lo) & (gt < hi) & (gt > 0)
        if m.any():
            e = np.abs(pred[m] - gt[m])
            out[f"[{lo},{hi})m"] = (float(e.sum()), int(m.sum()))
        else:
            out[f"[{lo},{hi})m"] = (0.0, 0)
    for cap in (100.0,):
        m = mask & (gt > 0) & (gt < cap)
        if m.any():
            e = np.abs(pred[m] - gt[m])
            out[f"0-{cap:g}m"] = (float(e.sum()), int(m.sum()))
        else:
            out[f"0-{cap:g}m"] = (0.0, 0)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default="output/radartaco_moe_stage2_v6c_confshared")
    ap.add_argument("--ckpt", default="best.pt")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--split", default="test")
    ap.add_argument("--out-json", default=None)
    args = ap.parse_args()

    tag = os.path.basename(args.run_dir.rstrip("/"))
    if args.out_json is None:
        args.out_json = f"output/analysis/cascade_reroute_{tag}.json"

    model, cfg = load_model(args.run_dir, args.ckpt)
    router_bins = list(cfg.model.moe_bins)

    split_file = os.path.join(cfg.dataset.data_root, "splits", f"{args.split}.txt")
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
    n = min(args.n, len(ds))
    idxs = list(range(n)) if n == len(ds) else np.linspace(0, len(ds)-1, n).astype(int).tolist()
    print(f"{args.split} — {len(idxs):,}/{len(ds):,} samples")

    moe_blocks = _collect_moe_blocks(model)
    print(f"MoE blocks: {[nm for nm, _ in moe_blocks]}   router_bins={router_bins}\n")

    ranges = [(0.0, 20.0), (20.0, 50.0), (50.0, 100.0)]
    range_names = [f"[{lo},{hi})m" for lo, hi in ranges] + ["0-100m"]

    modes = ("Normal", "CascadedHard", "CascadedSoft", "OracleHard", "OracleSoft")
    accs = {m: defaultdict(lambda: [0.0, 0]) for m in modes}

    holder = {"mode": None, "depth_src": None}
    hooks = install_router_forcing(moe_blocks, holder, router_bins)

    try:
        with torch.no_grad():
            for i in tqdm(idxs, desc="samples", leave=False):
                s = ds[int(i)]
                rgb = s["rgb_norm"].unsqueeze(0).to(device)
                rp  = s["radar_points"].unsqueeze(0).to(device)
                rm  = s["radar_mask"].unsqueeze(0).to(device)
                dgd = s["depth_gt_dense"].unsqueeze(0).to(device)
                gt_np = s["depth_gt_lidar"][0].cpu().numpy()
                mask_np = s["valid_mask_lidar"][0].cpu().numpy().astype(bool)

                # 1. Normal — router as-is.
                holder["mode"] = None; holder["depth_src"] = None
                out = model(rgb, rp, rm)
                depth_normal_t = (out["depth"] if isinstance(out, dict) else out).detach()
                depth_normal = depth_normal_t[0, 0].cpu().numpy()
                for k, (se, n_) in per_range_l1(depth_normal, gt_np, mask_np, ranges).items():
                    accs["Normal"][k][0] += se; accs["Normal"][k][1] += n_

                # 2. CascadedHard — force router with depth_normal bucketized.
                holder["mode"] = "hard"; holder["depth_src"] = depth_normal_t
                out = model(rgb, rp, rm)
                depth_chard = (out["depth"] if isinstance(out, dict) else out)[0, 0].cpu().numpy()
                for k, (se, n_) in per_range_l1(depth_chard, gt_np, mask_np, ranges).items():
                    accs["CascadedHard"][k][0] += se; accs["CascadedHard"][k][1] += n_

                # 3. CascadedSoft — force router with depth_normal's pixel_frac.
                holder["mode"] = "soft"; holder["depth_src"] = depth_normal_t
                out = model(rgb, rp, rm)
                depth_csoft = (out["depth"] if isinstance(out, dict) else out)[0, 0].cpu().numpy()
                for k, (se, n_) in per_range_l1(depth_csoft, gt_np, mask_np, ranges).items():
                    accs["CascadedSoft"][k][0] += se; accs["CascadedSoft"][k][1] += n_

                # 4. OracleHard — GT bin argmax.
                holder["mode"] = "hard"; holder["depth_src"] = dgd
                out = model(rgb, rp, rm)
                depth_ohard = (out["depth"] if isinstance(out, dict) else out)[0, 0].cpu().numpy()
                for k, (se, n_) in per_range_l1(depth_ohard, gt_np, mask_np, ranges).items():
                    accs["OracleHard"][k][0] += se; accs["OracleHard"][k][1] += n_

                # 5. OracleSoft — GT pixel_frac.
                holder["mode"] = "soft"; holder["depth_src"] = dgd
                out = model(rgb, rp, rm)
                depth_osoft = (out["depth"] if isinstance(out, dict) else out)[0, 0].cpu().numpy()
                for k, (se, n_) in per_range_l1(depth_osoft, gt_np, mask_np, ranges).items():
                    accs["OracleSoft"][k][0] += se; accs["OracleSoft"][k][1] += n_

                holder["mode"] = None; holder["depth_src"] = None
    finally:
        for h in hooks:
            h.remove()

    # ---- Report ----
    def _mae(mode, k):
        se, n = accs[mode][k]
        return (se / n) if n > 0 else float("nan"), n

    print("\n" + "=" * 120)
    print(f"Cascade re-routing  —  {args.run_dir}")
    print("=" * 120)
    print(f"Samples: {len(idxs):,}\n")

    print(f"{'Mode':<16}" + "  ".join(f"{r:>12}" for r in range_names))
    print("-" * 100)
    for mode in modes:
        row = f"{mode:<16}"
        for r in range_names:
            mae, _ = _mae(mode, r)
            row += f"  {mae:>12.4f}"
        print(row)

    # Δ vs Normal
    print("\nΔ vs Normal (negative = better)")
    print("-" * 100)
    print(f"{'Mode':<16}" + "  ".join(f"{r:>12}" for r in range_names))
    for mode in modes:
        if mode == "Normal":
            continue
        row = f"{mode:<16}"
        for r in range_names:
            mae, _   = _mae(mode, r)
            mae_n, _ = _mae("Normal", r)
            d = mae - mae_n
            pct = 100 * d / mae_n if mae_n > 0 else 0
            row += f"  {d:>+7.4f}({pct:+.1f}%)"
        print(row)

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    dump = {
        "run_dir": args.run_dir, "ckpt": args.ckpt,
        "n_samples": len(idxs), "router_bins": router_bins,
        "mae": {mode: {r: {"mae": (accs[mode][r][0]/accs[mode][r][1]
                                    if accs[mode][r][1] > 0 else None),
                            "n": accs[mode][r][1]}
                       for r in range_names}
                for mode in modes},
    }
    with open(args.out_json, "w") as f:
        json.dump(dump, f, indent=2)
    print(f"\nSaved → {args.out_json}")


if __name__ == "__main__":
    main()
