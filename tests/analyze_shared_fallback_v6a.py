"""Diagnose whether v6a's shared expert acts as a real fallback for
low-confidence tokens.

Per sample, run 3 forward passes on v6a stage-2 with the L2 MoE blocks
patched to yield:
  Normal    : spec·α + shared·(1−α)      — the trained inference formula
  SharedOnly: gate=0, shared_weight=1    — feat + shared_delta (no specialists)
  SpecOnly  : spec (renormalized top-k), shared_weight=0

For each token grid (L=2, node + edge blocks) we capture α from the
Normal pass, then bucket pixels by their parent-token α and compute
per-bucket MAE on the LiDAR ground truth for each mode.

If shared is a real fallback:
  • at low α (uncertain), SharedOnly ≲ SpecOnly
  • at high α (confident), SharedOnly > SpecOnly
  • Normal should track min(SharedOnly, SpecOnly) — i.e. the α-mix wins.

Usage:
  python tests/analyze_shared_fallback_v6a.py --n 500
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


def make_patched_forward(orig_forward, mode: str, alpha_holder: dict, block_name: str):
    """Return a `forward` closure for MoEFusionBlock that overrides gate
    and shared_weight per mode.

    mode ∈ {'normal', 'shared_only', 'spec_only'}
    """
    def patched(self, feat, kv, radar_x_orig, radar_mask, image_w,
                depth_gt_dense=None, teacher_force=True, return_shared_only=False):
        B, C, H, W = feat.shape
        logits = self.router(feat)                                # (B, K, H, W)

        # Self-route only (we're in eval / stage 2).
        if self.router_gt_type == "overlap":
            probs = torch.sigmoid(logits)
            probs = probs / probs.sum(dim=1, keepdim=True).clamp_min(1e-8)
        else:
            probs = F.softmax(logits, dim=1)
        if self.top_k is not None and self.top_k < self.n_experts:
            _, idx = probs.topk(self.top_k, dim=1)
            keep = torch.zeros_like(probs).scatter_(1, idx, 1.0)
            gate = probs * keep
            gate = gate / gate.sum(dim=1, keepdim=True).clamp_min(1e-8)
        else:
            gate = probs

        # α from the (post-top_k) gate — same as forward().
        max_gate = gate.max(dim=1, keepdim=True).values           # (B, 1, H, W)
        K = self.n_experts
        alpha = ((K * max_gate - 1.0) / (K - 1)).clamp(0.0, 1.0)  # (B, 1, H, W)

        # Record α ONCE per (sample, block); the value is the same for all
        # 3 modes because it depends only on logits.
        if mode == "normal":
            alpha_holder[block_name] = alpha.detach().cpu()

        # Mode-specific gate / shared_weight.
        if mode == "normal":
            spec_gate = gate * alpha
            shared_w = 1.0 - alpha
        elif mode == "spec_only":
            spec_gate = gate                # unchanged, no α scaling
            shared_w = torch.zeros_like(alpha)
        elif mode == "shared_only":
            spec_gate = torch.zeros_like(gate)
            shared_w = torch.ones_like(alpha)
        else:
            raise ValueError(mode)

        mixed = feat
        for e_i, expert in enumerate(self.experts):
            sel = spec_gate[:, e_i] > 0
            if not bool(sel.any()):
                continue
            dense = bool(sel.all())
            out_e = expert(feat, kv, radar_x_orig, radar_mask, image_w,
                           sel=None if dense else sel)
            mixed = mixed + spec_gate[:, e_i:e_i+1] * (out_e - feat)

        if self.shared is not None:
            shared_delta = (self.shared(feat, kv, radar_x_orig, radar_mask, image_w)
                            - feat)
            mixed = mixed + shared_w * shared_delta

        return mixed, logits, None
    return patched


def upsample_alpha_to_pixel(alpha_bhw, target_hw):
    """(B, 1, h, w) → (B, H, W) via nearest upsampling."""
    up = F.interpolate(alpha_bhw, size=target_hw, mode="nearest")
    return up.squeeze(1)                                        # (B, H, W)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default="output/radartaco_moe_stage2_v6a_confshared")
    ap.add_argument("--ckpt", default="best.pt")
    ap.add_argument("--n", type=int, default=500,
                    help="Number of test samples to evaluate.")
    ap.add_argument("--split", default="test")
    ap.add_argument("--out-json", default="output/analysis/shared_fallback_v6a.json")
    args = ap.parse_args()

    # Fixed α bins for pixel bucketing.
    bins_alpha = [0.0, 0.1, 0.3, 0.5, 0.7, 0.85, 0.95, 1.001]
    bin_labels = [f"[{bins_alpha[i]:.2f},{bins_alpha[i+1]:.2f})"
                  for i in range(len(bins_alpha) - 1)]

    model, cfg = load_model(args.run_dir, args.ckpt)

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
    print(f"{args.split} — {len(idxs):,}/{len(ds):,} samples\n")

    moe_blocks = _collect_moe_blocks(model)
    print(f"MoE blocks: {[n for n, _ in moe_blocks]}\n")

    # Per-α-bucket accumulators, per mode.
    # Also stratify by depth range (near/mid/far) so we can see if fallback
    # helps different depths differently.
    depth_ranges = [(0.0, 20.0), (20.0, 50.0), (50.0, 100.0), (0.0, 100.0)]
    range_names = [f"[{lo:g},{hi:g})m" for lo, hi in depth_ranges]

    # {mode: {(bin_idx, range_name): [abs_err_sum, count]}}
    accs = {m: defaultdict(lambda: [0.0, 0])
            for m in ("normal", "spec_only", "shared_only")}
    # α distribution: {bin_idx: pixel_count} for the block averaged α
    alpha_dist = defaultdict(int)

    alpha_holder = {}

    def install(mode: str):
        orig_forwards = []
        for name, blk in moe_blocks:
            orig = blk.forward
            orig_forwards.append((blk, orig))
            blk.forward = make_patched_forward(orig, mode, alpha_holder, name).__get__(blk, type(blk))
        return orig_forwards

    def uninstall(orig_forwards):
        for blk, orig in orig_forwards:
            blk.forward = orig

    modes = ("normal", "spec_only", "shared_only")

    with torch.no_grad():
        for i in tqdm(idxs, desc="samples", leave=False):
            s = ds[int(i)]
            rgb = s["rgb_norm"].unsqueeze(0).to(device)
            rp  = s["radar_points"].unsqueeze(0).to(device)
            rm  = s["radar_mask"].unsqueeze(0).to(device)
            gt_np = s["depth_gt_lidar"][0].cpu().numpy()
            mask_np = s["valid_mask_lidar"][0].cpu().numpy().astype(bool)
            H_full, W_full = gt_np.shape

            preds = {}
            for mode in modes:
                if mode == "normal":
                    alpha_holder.clear()
                orig = install(mode)
                out = model(rgb, rp, rm)
                uninstall(orig)
                depth = (out["depth"] if isinstance(out, dict) else out)[0, 0].cpu().numpy()
                preds[mode] = depth


            # α at pixel resolution — upsample each block's α to full res,
            # then average. Blocks can be at different grids (L2_node vs L2_edge).
            alpha_up = []
            for nm, _ in moe_blocks:
                if nm not in alpha_holder:
                    continue
                up = upsample_alpha_to_pixel(alpha_holder[nm], (H_full, W_full))  # (1, H, W)
                alpha_up.append(up)
            if not alpha_up:
                continue
            alpha_pix = torch.stack(alpha_up, dim=0).mean(dim=0)[0].numpy()  # (H, W)

            # Bin pixels by α (only where LiDAR is valid).
            valid = mask_np & (gt_np > 0)
            for bi in range(len(bins_alpha) - 1):
                lo, hi = bins_alpha[bi], bins_alpha[bi+1]
                sel_alpha = valid & (alpha_pix >= lo) & (alpha_pix < hi)
                if not sel_alpha.any():
                    continue
                # α distribution — count all valid pixels in this bin.
                alpha_dist[bi] += int(sel_alpha.sum())
                for (dlo, dhi), rname in zip(depth_ranges, range_names):
                    sel = sel_alpha & (gt_np >= dlo) & (gt_np < dhi)
                    if not sel.any():
                        continue
                    for mode in modes:
                        err = np.abs(preds[mode][sel] - gt_np[sel])
                        accs[mode][(bi, rname)][0] += float(err.sum())
                        accs[mode][(bi, rname)][1] += int(sel.sum())

    # ---- Report ----
    def _mae(mode, bi, rname):
        se, n = accs[mode][(bi, rname)]
        return (se / n) if n > 0 else float("nan"), n

    # Header
    print("\n" + "=" * 120)
    print("Shared fallback analysis — v6a stage 2")
    print("=" * 120)
    print(f"Samples: {len(idxs):,}   α bins from block-averaged pixel α")
    print()

    # α distribution
    total_pix = sum(alpha_dist.values())
    print("α distribution (fraction of valid pixels per bucket)")
    print("-" * 60)
    for bi, lbl in enumerate(bin_labels):
        n = alpha_dist.get(bi, 0)
        frac = 100.0 * n / max(total_pix, 1)
        print(f"  {lbl:>18}  n={n:>10,}  ({frac:5.2f}%)")
    print()

    # MAE table for each depth range
    for rname in range_names:
        print(f"MAE per α bucket  —  depth range {rname}")
        print("-" * 100)
        print(f"{'α bucket':>18} {'Normal':>12} {'SpecOnly':>12} {'SharedOnly':>12}"
              f" {'Δ(Sh−Sp)':>12} {'n_pix':>12}")
        for bi, lbl in enumerate(bin_labels):
            mae_n, n_n = _mae("normal", bi, rname)
            mae_s, _   = _mae("spec_only", bi, rname)
            mae_sh, _  = _mae("shared_only", bi, rname)
            if n_n == 0:
                continue
            delta = mae_sh - mae_s
            print(f"{lbl:>18} {mae_n:>12.4f} {mae_s:>12.4f} {mae_sh:>12.4f}"
                  f" {delta:>+12.4f} {n_n:>12,}")
        print()

    # Persist
    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    dump = {
        "run_dir": args.run_dir,
        "ckpt": args.ckpt,
        "n_samples": len(idxs),
        "bins_alpha": bins_alpha,
        "alpha_dist": {str(bi): int(alpha_dist[bi]) for bi in alpha_dist},
        "mae": {
            mode: {
                f"{bin_labels[bi]}|{rname}": {"mae": (accs[mode][(bi, rname)][0] /
                                                      accs[mode][(bi, rname)][1])
                                              if accs[mode][(bi, rname)][1] > 0
                                              else None,
                                              "n": accs[mode][(bi, rname)][1]}
                for bi in range(len(bin_labels))
                for rname in range_names
            }
            for mode in modes
        },
    }
    with open(args.out_json, "w") as f:
        json.dump(dump, f, indent=2)
    print(f"Saved → {args.out_json}")


if __name__ == "__main__":
    main()
