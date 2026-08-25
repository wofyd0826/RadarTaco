"""Compare v6c's α-gate vs an oracle misroute-based gate.

Per sample, 3 forwards on v6c stage-2:
  1. SpecOnly    : specialists via router (no α scaling, no shared)
  2. SharedOnly  : shared only (no specialists)
  3. Normal      : v6c real inference (α-mix of spec and shared)

Additionally, using the Normal-pass router argmax vs GT bin argmax per token,
compute the "oracle misroute gate":
  • For each token, is_correct = (router_argmax == gt_argmax)
  • Upsample to pixel resolution.
  • For each pixel:
      correct token → use SpecOnly prediction
      misrouted    → use SharedOnly prediction
  This is the depth that a PERFECT misroute detector would deliver.

Also compute the per-pixel oracle upper bound:
  • For each pixel, take min(|SpecOnly−GT|, |SharedOnly−GT|)
  • This is the best any binary spec/shared gate could achieve, ignoring
    routing correctness — pure per-pixel choose-the-better ensemble.

Bucket pixels by α (from Normal). Compare MAE across:
  - Normal (α gate)
  - OracleMisrouteGate (spec if correct, shared if wrong)
  - SpecOnly
  - SharedOnly
  - PerPixelOracle (choose min error per pixel)

Usage:
  python tests/analyze_oracle_misroute_gate_v6c.py --n 1000
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


def make_patched_forward(mode, alpha_holder, argmax_holder, block_name):
    """MoE forward wrapper with three modes: 'normal' (v6c real), 'spec_only',
    'shared_only'. Also captures α and router argmax in 'normal' mode."""
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

        if mode == "normal":
            alpha_holder[block_name] = alpha.detach().cpu()
            # Router argmax = top-1 bin the router thinks the token belongs to.
            argmax_holder[block_name] = probs.argmax(dim=1).detach().cpu()

        if mode == "normal":
            spec_gate = gate * alpha
            shared_w = 1.0 - alpha
        elif mode == "spec_only":
            spec_gate = gate
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


def install(mode, alpha_holder, argmax_holder, moe_blocks):
    orig_forwards = []
    for name, blk in moe_blocks:
        orig = blk.forward
        orig_forwards.append((blk, orig))
        blk.forward = make_patched_forward(
            mode, alpha_holder, argmax_holder, name).__get__(blk, type(blk))
    return orig_forwards


def uninstall(orig_forwards):
    for blk, orig in orig_forwards:
        blk.forward = orig


def upsample_to_pixel(bhw, target_hw, mode="nearest"):
    """(B, 1 or K, h, w) → (B, target_h, target_w) via nearest for hard/binary."""
    up = F.interpolate(bhw.float(), size=target_hw, mode=mode)
    return up.squeeze(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default="output/radartaco_moe_stage2_v6c_confshared")
    ap.add_argument("--ckpt", default="best.pt")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--split", default="test")
    ap.add_argument("--out-json", default="output/analysis/oracle_misroute_gate_v6c.json")
    args = ap.parse_args()

    bins_alpha = [0.0, 0.1, 0.3, 0.5, 0.7, 0.85, 0.95, 1.001]
    bin_labels = [f"[{bins_alpha[i]:.2f},{bins_alpha[i+1]:.2f})"
                  for i in range(len(bins_alpha) - 1)]

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
    print(f"{args.split} — {len(idxs):,}/{len(ds):,} samples\n")

    moe_blocks = _collect_moe_blocks(model)
    print(f"MoE blocks: {[nm for nm, _ in moe_blocks]}   router_bins={router_bins}\n")

    depth_ranges = [(0.0, 20.0), (20.0, 50.0), (50.0, 100.0), (0.0, 100.0)]
    range_names = [f"[{lo:g},{hi:g})m" for lo, hi in depth_ranges]

    modes = ("Normal", "SpecOnly", "SharedOnly",
             "OracleMisrouteGate", "PerPixelOracle")
    accs = {m: defaultdict(lambda: [0.0, 0]) for m in modes}
    alpha_dist = defaultdict(int)
    # Track how many pixels the OracleMisrouteGate sent to shared.
    shared_pixel_by_alpha = defaultdict(int)

    alpha_holder = {}
    argmax_holder = {}

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

            preds = {}
            for m in ("normal", "spec_only", "shared_only"):
                if m == "normal":
                    alpha_holder.clear()
                    argmax_holder.clear()
                orig = install(m, alpha_holder, argmax_holder, moe_blocks)
                out = model(rgb, rp, rm)
                uninstall(orig)
                depth = (out["depth"] if isinstance(out, dict) else out)[0, 0].cpu().numpy()
                key = {"normal": "Normal", "spec_only": "SpecOnly",
                       "shared_only": "SharedOnly"}[m]
                preds[key] = depth

            # α at pixel res (mean over blocks).
            alpha_up = []
            for nm, _ in moe_blocks:
                if nm not in alpha_holder:
                    continue
                alpha_up.append(upsample_to_pixel(alpha_holder[nm], (H_full, W_full)))
            if not alpha_up:
                continue
            alpha_pix = torch.stack(alpha_up, dim=0).mean(dim=0)[0].numpy()

            # Per-token is_correct (router argmax vs GT bin argmax), then upsample
            # to pixel. When multiple blocks exist (L2_node + L2_edge with different
            # grids), a pixel is "correct" only if ALL blocks are correct there.
            is_correct_up = None
            for nm, _ in moe_blocks:
                if nm not in argmax_holder:
                    continue
                token_pred = argmax_holder[nm]                             # (B, h, w)
                gt_hw = token_pred.shape[-2:]
                gt_hard = compute_token_gt(dgd.cpu(), gt_hw, router_bins,
                                           return_type="hard")             # (B, h, w)
                is_correct = (token_pred == gt_hard).float().unsqueeze(1)  # (B, 1, h, w)
                up = upsample_to_pixel(is_correct, (H_full, W_full))[0].numpy() > 0.5
                is_correct_up = up if is_correct_up is None else (is_correct_up & up)

            # OracleMisrouteGate: spec if correct, else shared.
            preds["OracleMisrouteGate"] = np.where(
                is_correct_up, preds["SpecOnly"], preds["SharedOnly"])

            # PerPixelOracle: per-pixel min error between spec and shared.
            err_spec   = np.abs(preds["SpecOnly"]   - gt_np)
            err_shared = np.abs(preds["SharedOnly"] - gt_np)
            preds["PerPixelOracle"] = np.where(err_spec <= err_shared,
                                                preds["SpecOnly"],
                                                preds["SharedOnly"])

            valid = mask_np & (gt_np > 0)
            for bi in range(len(bins_alpha) - 1):
                lo, hi = bins_alpha[bi], bins_alpha[bi+1]
                sel_alpha = valid & (alpha_pix >= lo) & (alpha_pix < hi)
                if not sel_alpha.any():
                    continue
                alpha_dist[bi] += int(sel_alpha.sum())
                # How often does OracleMisrouteGate send to shared in this α bucket?
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

    print("\n" + "=" * 140)
    print("Oracle misroute-gate vs α-gate  —  v6c stage 2")
    print("=" * 140)
    print(f"Samples: {len(idxs):,}\n")

    total_pix = sum(alpha_dist.values())
    print("α distribution and shared-routing fraction under OracleMisrouteGate")
    print("-" * 80)
    print(f"  {'α bucket':>18}  {'n_pix':>12}  {'%':>6}  {'shared_pix':>12}  {'shared_frac':>12}")
    for bi, lbl in enumerate(bin_labels):
        n = alpha_dist.get(bi, 0)
        if n == 0:
            continue
        frac = 100.0 * n / max(total_pix, 1)
        sp = shared_pixel_by_alpha.get(bi, 0)
        sfrac = 100.0 * sp / max(n, 1)
        print(f"  {lbl:>18}  {n:>12,}  {frac:5.2f}%  {sp:>12,}  {sfrac:>11.2f}%")
    print()

    # Per depth range MAE table
    for rname in range_names:
        print(f"MAE per α bucket  —  {rname}")
        print("-" * 140)
        print(f"  {'α bucket':>18} {'Normal':>10} {'SpecOnly':>10} {'SharedOnly':>11}"
              f" {'OracleMisr':>12} {'PerPixOra':>11} "
              f" {'Δ(OM−N)':>10} {'Δ(PP−N)':>10} {'n_pix':>12}")
        for bi, lbl in enumerate(bin_labels):
            mae_n, n_n  = _mae("Normal", bi, rname)
            mae_s, _    = _mae("SpecOnly", bi, rname)
            mae_sh, _   = _mae("SharedOnly", bi, rname)
            mae_om, _   = _mae("OracleMisrouteGate", bi, rname)
            mae_pp, _   = _mae("PerPixelOracle", bi, rname)
            if n_n == 0:
                continue
            d_om = mae_om - mae_n
            d_pp = mae_pp - mae_n
            print(f"  {lbl:>18} {mae_n:>10.4f} {mae_s:>10.4f} {mae_sh:>11.4f}"
                  f" {mae_om:>12.4f} {mae_pp:>11.4f} "
                  f" {d_om:>+10.4f} {d_pp:>+10.4f} {n_n:>12,}")
        print()

    # Aggregate summary — weighted by pixel counts (0-100m).
    def _weighted_mae(mode, rname):
        num, den = 0.0, 0
        for bi in range(len(bin_labels)):
            se, n = accs[mode][(bi, rname)]
            num += se; den += n
        return num / max(den, 1)

    print("=" * 100)
    print("Aggregate MAE (0-100m, weighted by pixels)")
    print("=" * 100)
    for m in modes:
        print(f"  {m:>22s}: {_weighted_mae(m, '[0,100)m'):.4f} m")

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    dump = {
        "run_dir": args.run_dir,
        "ckpt": args.ckpt,
        "n_samples": len(idxs),
        "bins_alpha": bins_alpha,
        "alpha_dist": {str(bi): int(alpha_dist[bi]) for bi in alpha_dist},
        "shared_pixel_by_alpha": {str(bi): int(v) for bi, v in shared_pixel_by_alpha.items()},
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
