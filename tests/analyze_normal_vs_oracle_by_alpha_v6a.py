"""Compare Normal vs OracleHard vs OracleSoft, bucketed by v6a's α.

Per sample, 3 forwards on v6a stage-2:
  1. Normal      : real v6a forward (confidence gating), captures α at L=2 MoE
                   blocks and captures Normal depth prediction.
  2. OracleHard  : router logits FORCED to one-hot(GT bin) via hook; the block
                   otherwise uses its natural forward (still confidence-gated
                   at inference). Captures Oracle-hard depth prediction.
  3. OracleSoft  : router logits FORCED to log(frac) via hook. Captures
                   Oracle-soft depth prediction.

Bucketing key: α from the Normal pass (upsampled to pixel res). We compare
per-bucket MAE across all three modes. This answers: at tokens where Normal
is uncertain (low α), does Oracle routing recover the gap?

Usage:
  python tests/analyze_normal_vs_oracle_by_alpha_v6a.py --n 1000
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


def compute_token_gt(depth_gt_dense, tok_hw, bins, return_type="hard"):
    K = len(bins) - 1
    edges = torch.tensor(bins, device=depth_gt_dense.device, dtype=torch.float32)
    ds = depth_gt_dense.squeeze(1)
    pix_bin = torch.zeros_like(ds, dtype=torch.long)
    for i in range(K):
        pix_bin = torch.where((ds >= float(edges[i])) & (ds < float(edges[i+1])),
                              pix_bin.new_tensor(i), pix_bin)
    pix_bin = torch.where(ds >= float(edges[-1]), pix_bin.new_tensor(K-1), pix_bin)
    onehot = F.one_hot(pix_bin, num_classes=K).permute(0, 3, 1, 2).float()
    frac = F.adaptive_avg_pool2d(onehot, tok_hw)
    if return_type == "soft":
        return frac
    if return_type == "both":
        return frac, frac.argmax(dim=1)
    return frac.argmax(dim=1)


def install_alpha_capture(moe_blocks, holder):
    """Patch MoE block forward to record α (post-top_k gate max) per block
    into `holder` while leaving semantics unchanged (v6a confidence gating)."""
    orig_forwards = []
    for name, blk in moe_blocks:
        orig = blk.forward
        orig_forwards.append((blk, orig))

        def make(name_, blk_, orig_):
            def patched(self, feat, kv, radar_x_orig, radar_mask, image_w,
                        depth_gt_dense=None, teacher_force=True, return_shared_only=False):
                B, C, H, W = feat.shape
                logits = self.router(feat)
                # Match block's inference path — self-routing, no teacher force.
                probs = F.softmax(logits, dim=1)
                if self.top_k is not None and self.top_k < self.n_experts:
                    _, idx = probs.topk(self.top_k, dim=1)
                    keep = torch.zeros_like(probs).scatter_(1, idx, 1.0)
                    gate = probs * keep
                    gate = gate / gate.sum(dim=1, keepdim=True).clamp_min(1e-8)
                else:
                    gate = probs
                max_gate = gate.max(dim=1, keepdim=True).values
                K = self.n_experts
                alpha = ((K * max_gate - 1.0) / (K - 1)).clamp(0.0, 1.0)
                holder[name_] = alpha.detach().cpu()
                # Delegate to original forward for the actual computation
                # (keeps v6a's confidence-gated mixing intact).
                return orig_(feat, kv, radar_x_orig, radar_mask, image_w,
                             depth_gt_dense=depth_gt_dense,
                             teacher_force=teacher_force,
                             return_shared_only=return_shared_only)
            return patched

        blk.forward = make(name, blk, orig).__get__(blk, type(blk))
    return orig_forwards


def install_oracle_hook(moe_blocks, dgd_holder, bins):
    """Register a forward_hook on each MoE block's router to force logits
    to reflect the depth-derived GT distribution.

    dgd_holder["mode"] ∈ {'hard', 'soft'}, dgd_holder["dgd"] = (B,1,H,W) dense GT.
    """
    hooks = []
    def _hook(_m, _i, out):
        mode = dgd_holder.get("mode")
        dgd = dgd_holder.get("dgd")
        if mode is None or dgd is None:
            return out
        if mode == "soft":
            frac = compute_token_gt(dgd, out.shape[-2:], bins, return_type="soft")
            return torch.log(frac.clamp_min(1e-8))
        tok_gt = compute_token_gt(dgd, out.shape[-2:], bins, return_type="hard")
        forced = torch.full_like(out, -1e4)
        forced.scatter_(1, tok_gt.unsqueeze(1), 1e4)
        return forced
    for _, blk in moe_blocks:
        hooks.append(blk.router.register_forward_hook(_hook))
    return hooks


def upsample_alpha_to_pixel(alpha_bhw, target_hw):
    up = F.interpolate(alpha_bhw, size=target_hw, mode="nearest")
    return up.squeeze(1)


def uninstall(orig_forwards):
    for blk, orig in orig_forwards:
        blk.forward = orig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default="output/radartaco_moe_stage2_v6a_confshared")
    ap.add_argument("--ckpt", default="best.pt")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--split", default="test")
    ap.add_argument("--out-json", default="output/analysis/normal_vs_oracle_by_alpha_v6a.json")
    args = ap.parse_args()

    bins_alpha = [0.0, 0.1, 0.3, 0.5, 0.7, 0.85, 0.95, 1.001]
    bin_labels = [f"[{bins_alpha[i]:.2f},{bins_alpha[i+1]:.2f})"
                  for i in range(len(bins_alpha) - 1)]

    model, cfg = load_model(args.run_dir, args.ckpt)
    router_bins = list(cfg.model.moe_bins)      # e.g. [0, 20, 50, 100]

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
    print(f"MoE blocks: {[nm for nm, _ in moe_blocks]}   router_bins={router_bins}\n")

    depth_ranges = [(0.0, 20.0), (20.0, 50.0), (50.0, 100.0), (0.0, 100.0)]
    range_names = [f"[{lo:g},{hi:g})m" for lo, hi in depth_ranges]

    modes = ("Normal", "OracleHard", "OracleSoft")
    accs = {m: defaultdict(lambda: [0.0, 0]) for m in modes}
    alpha_dist = defaultdict(int)

    alpha_holder = {}
    dgd_holder = {"dgd": None, "mode": None}

    # Install router hook ONCE — it inspects dgd_holder["mode"] each call and
    # passes through when mode is None.
    oracle_hooks = install_oracle_hook(moe_blocks, dgd_holder, router_bins)

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
                H_full, W_full = gt_np.shape
                dgd_holder["dgd"] = dgd

                preds = {}
                # ---- Normal (with α capture) ----
                dgd_holder["mode"] = None
                alpha_holder.clear()
                orig = install_alpha_capture(moe_blocks, alpha_holder)
                out = model(rgb, rp, rm)
                uninstall(orig)
                preds["Normal"] = (out["depth"] if isinstance(out, dict) else out)[0, 0].cpu().numpy()

                # ---- OracleHard ----
                dgd_holder["mode"] = "hard"
                out = model(rgb, rp, rm)
                preds["OracleHard"] = (out["depth"] if isinstance(out, dict) else out)[0, 0].cpu().numpy()

                # ---- OracleSoft ----
                dgd_holder["mode"] = "soft"
                out = model(rgb, rp, rm)
                preds["OracleSoft"] = (out["depth"] if isinstance(out, dict) else out)[0, 0].cpu().numpy()

                dgd_holder["mode"] = None

                # α at pixel resolution (mean over blocks).
                alpha_up = []
                for nm, _ in moe_blocks:
                    if nm not in alpha_holder:
                        continue
                    alpha_up.append(upsample_alpha_to_pixel(alpha_holder[nm], (H_full, W_full)))
                if not alpha_up:
                    continue
                alpha_pix = torch.stack(alpha_up, dim=0).mean(dim=0)[0].numpy()

                valid = mask_np & (gt_np > 0)
                for bi in range(len(bins_alpha) - 1):
                    lo, hi = bins_alpha[bi], bins_alpha[bi+1]
                    sel_alpha = valid & (alpha_pix >= lo) & (alpha_pix < hi)
                    if not sel_alpha.any():
                        continue
                    alpha_dist[bi] += int(sel_alpha.sum())
                    for (dlo, dhi), rname in zip(depth_ranges, range_names):
                        sel = sel_alpha & (gt_np >= dlo) & (gt_np < dhi)
                        if not sel.any():
                            continue
                        for mode in modes:
                            err = np.abs(preds[mode][sel] - gt_np[sel])
                            accs[mode][(bi, rname)][0] += float(err.sum())
                            accs[mode][(bi, rname)][1] += int(sel.sum())
    finally:
        for h in oracle_hooks:
            h.remove()

    # ---- Report ----
    def _mae(mode, bi, rname):
        se, n = accs[mode][(bi, rname)]
        return (se / n) if n > 0 else float("nan"), n

    print("\n" + "=" * 120)
    print("Normal vs Oracle by α (v6a stage 2)  —  bucketing key = Normal α")
    print("=" * 120)
    print(f"Samples: {len(idxs):,}\n")

    total_pix = sum(alpha_dist.values())
    print("α distribution")
    print("-" * 60)
    for bi, lbl in enumerate(bin_labels):
        n = alpha_dist.get(bi, 0)
        frac = 100.0 * n / max(total_pix, 1)
        print(f"  {lbl:>18}  n={n:>12,}  ({frac:5.2f}%)")
    print()

    for rname in range_names:
        print(f"MAE per α bucket  —  depth range {rname}")
        print("-" * 110)
        print(f"{'α bucket':>18} {'Normal':>12} {'OracleHard':>12} {'OracleSoft':>12}"
              f" {'Δ(OH−N)':>12} {'Δ(OS−N)':>12} {'n_pix':>12}")
        for bi, lbl in enumerate(bin_labels):
            mae_n, n_n   = _mae("Normal", bi, rname)
            mae_oh, _    = _mae("OracleHard", bi, rname)
            mae_os, _    = _mae("OracleSoft", bi, rname)
            if n_n == 0:
                continue
            d_oh = mae_oh - mae_n
            d_os = mae_os - mae_n
            print(f"{lbl:>18} {mae_n:>12.4f} {mae_oh:>12.4f} {mae_os:>12.4f}"
                  f" {d_oh:>+12.4f} {d_os:>+12.4f} {n_n:>12,}")
        print()

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
