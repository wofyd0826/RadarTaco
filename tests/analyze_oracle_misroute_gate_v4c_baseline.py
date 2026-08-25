"""Oracle misroute gate — v4c specialists vs baseline (single fusion).

Two SEPARATE models:
  - v4c stage 2   : MoE (3 specs, top_k=2), NO shared. Router assigns per token.
  - baseline      : single fusion (no MoE, no routing). "Generalist" model.

Per sample, forward BOTH models. Then compute:
  - Normal_v4c        : v4c's own prediction (its normal inference)
  - Baseline          : baseline's prediction (no routing to worry about)
  - OracleMisrouteGate: for each pixel, if v4c-router-argmax == GT bin argmax
                        → use v4c prediction; else use baseline prediction.
  - PerPixelOracle    : per-pixel min error between v4c and baseline.

Bucket pixels by v4c's α. Question: does a perfect misroute-detector, sending
misrouted-token pixels to a SEPARATE baseline model, help more than it did
in v6c (where spec/shared were same-model with shared encoder/decoder)?

Usage:
  python tests/analyze_oracle_misroute_gate_v4c_baseline.py --n 1000
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


def install_capture(moe_blocks, alpha_holder, argmax_holder):
    """Patch MoE forward to also capture α and router-argmax per block.
    Passes through to the block's ORIGINAL forward for the actual computation
    (so v4c behavior is unchanged)."""
    orig_forwards = []
    for name, blk in moe_blocks:
        orig = blk.forward
        orig_forwards.append((blk, orig))

        def make(name_, blk_, orig_):
            def patched(self, feat, kv, radar_x_orig, radar_mask, image_w,
                        depth_gt_dense=None, teacher_force=True, return_shared_only=False):
                B, C, H, W = feat.shape
                logits = self.router(feat)
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
                alpha_holder[name_]  = alpha.detach().cpu()
                argmax_holder[name_] = probs.argmax(dim=1).detach().cpu()   # top-1
                return orig_(feat, kv, radar_x_orig, radar_mask, image_w,
                             depth_gt_dense=depth_gt_dense,
                             teacher_force=teacher_force,
                             return_shared_only=return_shared_only)
            return patched

        blk.forward = make(name, blk, orig).__get__(blk, type(blk))
    return orig_forwards


def uninstall(orig_forwards):
    for blk, orig in orig_forwards:
        blk.forward = orig


def upsample_to_pixel(bhw, target_hw, mode="nearest"):
    up = F.interpolate(bhw.float(), size=target_hw, mode=mode)
    return up.squeeze(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v4c-dir",      default="output/radartaco_moe_stage2_v4c")
    ap.add_argument("--baseline-dir", default="output/shape_lidar_grad_shape_edge_res_fix")
    ap.add_argument("--ckpt", default="best.pt")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--split", default="test")
    ap.add_argument("--out-json", default="output/analysis/oracle_misroute_gate_v4c_baseline.json")
    args = ap.parse_args()

    bins_alpha = [0.0, 0.1, 0.3, 0.5, 0.7, 0.85, 0.95, 1.001]
    bin_labels = [f"[{bins_alpha[i]:.2f},{bins_alpha[i+1]:.2f})"
                  for i in range(len(bins_alpha) - 1)]

    print("v4c:"); model_v4c,  cfg_v4c  = load_model(args.v4c_dir, args.ckpt)
    print("baseline:"); model_bs, cfg_bs = load_model(args.baseline_dir, args.ckpt)

    router_bins = list(cfg_v4c.model.moe_bins)

    split_file = os.path.join(cfg_v4c.dataset.data_root, "splits", f"{args.split}.txt")
    ds = NuScenesRadarDepthDataset(
        data_root=cfg_v4c.dataset.data_root, split_file=split_file,
        dense_gt_dir=cfg_v4c.dataset.dense_gt_dir,
        radar_3d_dir=cfg_v4c.dataset.radar_3d_dir,
        night_ids_file=cfg_v4c.dataset.night_ids_file,
        max_radar_points=int(cfg_v4c.dataset.max_radar_points),
        max_depth=float(cfg_v4c.dataset.max_depth),
        min_depth=float(cfg_v4c.dataset.min_depth),
        augmentation=False,
    )
    n = min(args.n, len(ds))
    idxs = list(range(n)) if n == len(ds) else np.linspace(0, len(ds)-1, n).astype(int).tolist()
    print(f"\n{args.split} — {len(idxs):,}/{len(ds):,} samples\n")

    moe_blocks = _collect_moe_blocks(model_v4c)
    print(f"v4c MoE blocks: {[nm for nm, _ in moe_blocks]}   router_bins={router_bins}\n")

    depth_ranges = [(0.0, 20.0), (20.0, 50.0), (50.0, 100.0), (0.0, 100.0)]
    range_names = [f"[{lo:g},{hi:g})m" for lo, hi in depth_ranges]

    modes = ("v4c", "Baseline", "OracleMisrouteGate", "PerPixelOracle")
    accs = {m: defaultdict(lambda: [0.0, 0]) for m in modes}
    alpha_dist = defaultdict(int)
    shared_pixel_by_alpha = defaultdict(int)

    alpha_holder, argmax_holder = {}, {}

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

            # ---- v4c forward (with α + argmax capture) ----
            alpha_holder.clear(); argmax_holder.clear()
            orig = install_capture(moe_blocks, alpha_holder, argmax_holder)
            out_v4c = model_v4c(rgb, rp, rm)
            uninstall(orig)
            depth_v4c = (out_v4c["depth"] if isinstance(out_v4c, dict) else out_v4c)[0, 0].cpu().numpy()

            # ---- baseline forward ----
            out_bs = model_bs(rgb, rp, rm)
            depth_bs = (out_bs["depth"] if isinstance(out_bs, dict) else out_bs)[0, 0].cpu().numpy()

            # α at pixel res (mean over v4c MoE blocks).
            alpha_up = []
            for nm, _ in moe_blocks:
                if nm not in alpha_holder:
                    continue
                alpha_up.append(upsample_to_pixel(alpha_holder[nm], (H_full, W_full)))
            if not alpha_up:
                continue
            alpha_pix = torch.stack(alpha_up, dim=0).mean(dim=0)[0].numpy()

            # Per-token is_correct (v4c router argmax vs GT bin argmax).
            # A pixel is "correct" iff ALL v4c blocks are correct there.
            is_correct_up = None
            for nm, _ in moe_blocks:
                if nm not in argmax_holder:
                    continue
                token_pred = argmax_holder[nm]
                gt_hw = token_pred.shape[-2:]
                gt_hard = compute_token_gt(dgd.cpu(), gt_hw, router_bins, return_type="hard")
                is_correct = (token_pred == gt_hard).float().unsqueeze(1)
                up = upsample_to_pixel(is_correct, (H_full, W_full))[0].numpy() > 0.5
                is_correct_up = up if is_correct_up is None else (is_correct_up & up)

            # OracleMisrouteGate: v4c if correct, baseline if misrouted.
            depth_omg = np.where(is_correct_up, depth_v4c, depth_bs)
            # PerPixelOracle: choose the per-pixel better prediction.
            err_v4c = np.abs(depth_v4c - gt_np)
            err_bs  = np.abs(depth_bs  - gt_np)
            depth_ppo = np.where(err_v4c <= err_bs, depth_v4c, depth_bs)

            preds = {"v4c": depth_v4c, "Baseline": depth_bs,
                     "OracleMisrouteGate": depth_omg,
                     "PerPixelOracle": depth_ppo}

            valid = mask_np & (gt_np > 0)
            for bi in range(len(bins_alpha) - 1):
                lo, hi = bins_alpha[bi], bins_alpha[bi+1]
                sel_alpha = valid & (alpha_pix >= lo) & (alpha_pix < hi)
                if not sel_alpha.any():
                    continue
                alpha_dist[bi] += int(sel_alpha.sum())
                shared_pixel_by_alpha[bi] += int((sel_alpha & (~is_correct_up)).sum())
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

    def _weighted_mae(mode, rname):
        num, den = 0.0, 0
        for bi in range(len(bin_labels)):
            se, n = accs[mode][(bi, rname)]
            num += se; den += n
        return num / max(den, 1)

    print("\n" + "=" * 140)
    print("Oracle misroute-gate: v4c specialists vs baseline (SEPARATE models)")
    print("=" * 140)
    print(f"Samples: {len(idxs):,}\n")

    total_pix = sum(alpha_dist.values())
    print("α distribution (from v4c router) + baseline-routing frac under OracleMisrouteGate")
    print("-" * 90)
    print(f"  {'α bucket':>18}  {'n_pix':>12}  {'%':>6}  {'baseline_pix':>14}  {'baseline_frac':>14}")
    for bi, lbl in enumerate(bin_labels):
        n = alpha_dist.get(bi, 0)
        if n == 0:
            continue
        frac = 100.0 * n / max(total_pix, 1)
        sp = shared_pixel_by_alpha.get(bi, 0)
        sfrac = 100.0 * sp / max(n, 1)
        print(f"  {lbl:>18}  {n:>12,}  {frac:5.2f}%  {sp:>14,}  {sfrac:>13.2f}%")
    print()

    for rname in range_names:
        print(f"MAE per α bucket  —  {rname}")
        print("-" * 130)
        print(f"  {'α bucket':>18} {'v4c':>10} {'Baseline':>10} {'OracleMisr':>12} {'PerPixOra':>11}"
              f" {'Δ(OM−v4c)':>12} {'Δ(PP−v4c)':>12} {'n_pix':>12}")
        for bi, lbl in enumerate(bin_labels):
            mae_v, n_v   = _mae("v4c", bi, rname)
            mae_b, _     = _mae("Baseline", bi, rname)
            mae_om, _    = _mae("OracleMisrouteGate", bi, rname)
            mae_pp, _    = _mae("PerPixelOracle", bi, rname)
            if n_v == 0:
                continue
            d_om = mae_om - mae_v
            d_pp = mae_pp - mae_v
            print(f"  {lbl:>18} {mae_v:>10.4f} {mae_b:>10.4f} {mae_om:>12.4f} {mae_pp:>11.4f}"
                  f" {d_om:>+12.4f} {d_pp:>+12.4f} {n_v:>12,}")
        print()

    print("=" * 100)
    print("Aggregate MAE (0-100m, weighted by pixels)")
    print("=" * 100)
    for m in modes:
        print(f"  {m:>22s}: {_weighted_mae(m, '[0,100)m'):.4f} m")

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    dump = {
        "v4c_dir": args.v4c_dir, "baseline_dir": args.baseline_dir, "ckpt": args.ckpt,
        "n_samples": len(idxs),
        "bins_alpha": bins_alpha,
        "alpha_dist": {str(bi): int(alpha_dist[bi]) for bi in alpha_dist},
        "baseline_pixel_by_alpha": {str(bi): int(v) for bi, v in shared_pixel_by_alpha.items()},
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
        "aggregate_0_100m": {m: _weighted_mae(m, "[0,100)m") for m in modes},
    }
    with open(args.out_json, "w") as f:
        json.dump(dump, f, indent=2)
    print(f"\nSaved → {args.out_json}")


if __name__ == "__main__":
    main()
