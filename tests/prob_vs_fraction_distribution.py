"""Compare model's predicted probability distribution vs target pixel_fraction.

For each token in L2 node grid, capture:
  probs (K=3)  — softmax of router logits (Normal pass)
  frac  (K=3)  — target pixel_fraction (aggregated GT bins)
  max_prob, max_frac  — concentration of each distribution
  entropy_prob, entropy_frac
  KL(frac || probs), TV distance, L1 distance

Then aggregate:
  1. Histogram of max_prob vs max_frac (are they equally sharp?)
  2. Cross-tab of pure/mixed status (predicted vs actual)
  3. Mean |probs - frac| per bin of max_frac (which target regimes suffer?)
  4. Scatter of predicted vs true entropy
"""
import argparse
import io
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from tqdm import tqdm

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, ROOT)

from src.dataset.nuscenes import NuScenesRadarDepthDataset   # noqa: E402
from src.model.radar_fusion import MoEFusionBlock            # noqa: E402
from scripts.train import _build_model                       # noqa: E402


device = "cuda" if torch.cuda.is_available() else "cpu"
BINS = [0.0, 20.0, 50.0, 100.0]
K = 3


def load_model(run_dir, ckpt="best.pt"):
    cfg = OmegaConf.load(os.path.join(run_dir, "config.yaml"))
    m = _build_model(cfg, float(cfg.dataset.max_depth),
                     int(cfg.dataset.max_radar_points))
    sd = torch.load(os.path.join(run_dir, ckpt),
                    map_location=device, weights_only=False)
    m.load_state_dict(sd["model"])
    return m.eval().to(device)


def pixel_fraction(dgd, tok_hw):
    """Target: per-token pixel_fraction across K bins. (B, K, H_tok, W_tok)."""
    edges = torch.tensor(BINS, device=dgd.device, dtype=torch.float32)
    ds = dgd.squeeze(1)
    pix_bin = torch.zeros_like(ds, dtype=torch.long)
    for i in range(K):
        pix_bin = torch.where((ds >= float(edges[i])) & (ds < float(edges[i+1])),
                              pix_bin.new_tensor(i), pix_bin)
    pix_bin = torch.where(ds >= float(edges[-1]),
                          pix_bin.new_tensor(K-1), pix_bin)
    onehot = F.one_hot(pix_bin, num_classes=K).permute(0, 3, 1, 2).float()
    return F.adaptive_avg_pool2d(onehot, tok_hw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--ckpt", default="best.pt")
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--split", default="test")
    ap.add_argument("--out-dir", default="output/analysis/router_cost")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    base_cfg = OmegaConf.load(
        "output/shape_lidar_grad_shape_edge_res/config.yaml")
    split_file = os.path.join(base_cfg.dataset.data_root, "splits",
                              f"{args.split}.txt")
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
    idxs = list(range(n)) if n == len(ds) else \
           np.linspace(0, len(ds)-1, n).astype(int).tolist()
    print(f"{args.split} — {len(idxs):,}/{len(ds):,} samples")

    m = load_model(args.run, args.ckpt)
    all_probs = []      # (K, N_total)
    all_frac = []       # (K, N_total)

    t0 = time.time()
    with torch.no_grad():
        for i in tqdm(idxs, desc="probs"):
            s = ds[int(i)]
            rgb = s["rgb_norm"].unsqueeze(0).to(device)
            rp  = s["radar_points"].unsqueeze(0).to(device)
            rm  = s["radar_mask"].unsqueeze(0).to(device)
            dgd = s["depth_gt_dense"].unsqueeze(0).to(device)

            out = m(rgb, rp, rm)
            rl = out["router_logits"][0]                # (1, K, H, W) — L2 node
            tok_hw = tuple(rl.shape[-2:])
            probs = rl.softmax(dim=1)[0]                # (K, H, W)
            frac = pixel_fraction(dgd, tok_hw)[0]       # (K, H, W)

            all_probs.append(probs.reshape(K, -1).cpu().numpy())
            all_frac.append(frac.reshape(K, -1).cpu().numpy())
    print(f"  ({time.time()-t0:.0f}s forward)")

    probs = np.concatenate(all_probs, axis=1)           # (K, N)
    frac = np.concatenate(all_frac, axis=1)             # (K, N)
    N = probs.shape[1]
    print(f"  tokens: {N:,}")

    # Derived quantities
    max_p = probs.max(axis=0)
    max_f = frac.max(axis=0)
    argmax_p = probs.argmax(axis=0)
    argmax_f = frac.argmax(axis=0)
    l1 = np.abs(probs - frac).sum(axis=0)               # (N,), in [0, 2]
    # Entropy (natural log; use base-3 to normalize to [0, 1])
    def entropy(p):
        return -np.sum(p * np.log(np.clip(p, 1e-12, 1.0)), axis=0) / np.log(K)
    ent_p = entropy(probs)
    ent_f = entropy(frac)
    kl = np.sum(frac * (np.log(np.clip(frac, 1e-12, 1.0)) -
                        np.log(np.clip(probs, 1e-12, 1.0))), axis=0)

    npz_path = os.path.join(args.out_dir, f"{args.tag}.npz")
    np.savez_compressed(npz_path,
                        probs=probs, frac=frac,
                        max_p=max_p, max_f=max_f,
                        argmax_p=argmax_p, argmax_f=argmax_f,
                        l1=l1, ent_p=ent_p, ent_f=ent_f, kl=kl)
    print(f"  Saved: {npz_path}")

    # ================= Text summary =================
    buf = io.StringIO()
    def _p(msg=""):
        print(msg); buf.write(msg + "\n")

    _p("=" * 96)
    _p(f"Prob vs Fraction distribution comparison — {args.tag}")
    _p(f"  run: {args.run}   split: {args.split}   samples: {len(idxs):,}")
    _p(f"  tokens (L2 node): {N:,}")
    _p("=" * 96)

    # -- 1. max_prob vs max_frac histograms
    _p("\n-- 1. Distribution of max_prob (model) vs max_frac (target) --")
    edges = [0.34, 0.50, 0.70, 0.85, 0.95, 0.999, 1.001]
    names = ["0.34-0.50", "0.50-0.70", "0.70-0.85", "0.85-0.95", "0.95-1.00", "==1.00"]
    _p(f"  {'range':<12s}  {'max_prob %':>12s}  {'max_frac %':>12s}  {'diff':>8s}")
    for i, name in enumerate(names):
        lo, hi = edges[i], edges[i+1]
        fp = float(((max_p >= lo) & (max_p < hi)).mean() * 100)
        ff = float(((max_f >= lo) & (max_f < hi)).mean() * 100)
        _p(f"  {name:<12s}  {fp:>11.2f}%  {ff:>11.2f}%  "
           f"{fp - ff:>+7.2f}%")
    _p(f"  Mean max_prob: {max_p.mean():.4f}   Mean max_frac: {max_f.mean():.4f}")

    # -- 2. Cross-tab: predicted pure/mixed vs target pure/mixed
    _p("\n-- 2. Cross-tab: PURE (max ≥ 0.95) vs MIXED (max < 0.95) --")
    p_pure = max_p >= 0.95
    f_pure = max_f >= 0.95
    tab = np.zeros((2, 2), dtype=np.int64)
    for pi in (0, 1):
        for fi in (0, 1):
            tab[pi, fi] = int(((p_pure == bool(1-pi)) & (f_pure == bool(1-fi))).sum())
    _p(f"                     target PURE     target MIXED     total")
    _p(f"  pred PURE      {tab[0, 0]:>12,d}  {tab[0, 1]:>15,d}  {tab[0].sum():>10,d}")
    _p(f"  pred MIXED     {tab[1, 0]:>12,d}  {tab[1, 1]:>15,d}  {tab[1].sum():>10,d}")
    _p(f"  total          {tab[:,0].sum():>12,d}  {tab[:,1].sum():>15,d}  {N:>10,d}")

    # -- 3. L1 by target-max bucket (where does the model miss?)
    _p("\n-- 3. L1 (|probs − frac|) by target max_frac bucket --")
    _p(f"  {'target max_frac':<20s}  {'n_tokens':>12s}  {'L1 mean':>10s}  "
       f"{'L1 median':>10s}  {'argmax match %':>14s}")
    for lo, hi in [(0.34, 0.50), (0.50, 0.70), (0.70, 0.85), (0.85, 0.95),
                    (0.95, 1.00), (1.00, 1.01)]:
        m = (max_f >= lo) & (max_f < hi)
        n = int(m.sum())
        if n == 0: continue
        l1_m = float(l1[m].mean())
        l1_med = float(np.median(l1[m]))
        acc = float((argmax_p[m] == argmax_f[m]).mean() * 100)
        tag = f"[{lo:.2f}, {hi:.2f})"
        _p(f"  {tag:<20s}  {n:>12,d}  {l1_m:>9.4f}  {l1_med:>9.4f}  "
           f"{acc:>13.2f}%")

    # -- 4. Entropy correlation
    _p("\n-- 4. Predicted vs target entropy (normalized to [0, 1]) --")
    _p(f"  Mean entropy — probs: {ent_p.mean():.4f}   frac: {ent_f.mean():.4f}")
    # Cross-tab low/mid/high entropy
    ent_buckets = [(0.0, 0.1), (0.1, 0.3), (0.3, 0.6), (0.6, 1.0)]
    ent_names = ["very-sharp [0, 0.1)", "sharp [0.1, 0.3)",
                 "mixed [0.3, 0.6)", "flat [0.6, 1.0]"]
    _p(f"  {'bucket':<24s}  {'model %':>10s}  {'target %':>10s}  {'diff':>8s}")
    for (lo, hi), name in zip(ent_buckets, ent_names):
        mp = float(((ent_p >= lo) & (ent_p < hi)).mean() * 100)
        mf = float(((ent_f >= lo) & (ent_f < hi)).mean() * 100)
        _p(f"  {name:<24s}  {mp:>9.2f}%  {mf:>9.2f}%  {mp-mf:>+7.2f}%")

    # -- 5. Mean predicted probs conditioned on target bucket
    _p("\n-- 5. Mean predicted (p_near, p_mid, p_far) by target bin category --")
    _p("   PURE target: max_frac ≥ 0.95  |  MIXED: 0.34-0.95")
    _p(f"  {'category':<28s}  {'n':>10s}  {'mean p_near':>12s}  "
       f"{'mean p_mid':>10s}  {'mean p_far':>10s}")
    for target_bin, name in enumerate(["near", "mid", "far"]):
        for pure_flag in (True, False):
            m = (argmax_f == target_bin) & (f_pure if pure_flag else ~f_pure)
            n = int(m.sum())
            if n == 0: continue
            mp_near = float(probs[0, m].mean())
            mp_mid = float(probs[1, m].mean())
            mp_far = float(probs[2, m].mean())
            tag = f"{name:<4s} + {'PURE' if pure_flag else 'MIXED':<5s}"
            _p(f"  {tag:<28s}  {n:>10,d}  {mp_near:>11.4f}  "
               f"{mp_mid:>10.4f}  {mp_far:>10.4f}")

    # -- 6. When target is MIXED, is model too sharp or too flat?
    _p("\n-- 6. Sharpness comparison on MIXED tokens (target max_frac 0.5-0.95) --")
    m = (max_f >= 0.50) & (max_f < 0.95)
    n = int(m.sum())
    if n > 0:
        too_sharp = float((max_p[m] > max_f[m]).mean() * 100)
        too_flat = float((max_p[m] < max_f[m]).mean() * 100)
        _p(f"  Mixed tokens: {n:,}")
        _p(f"  Model MORE confident than target (too sharp): {too_sharp:.2f}%")
        _p(f"  Model LESS confident than target (too flat):  {too_flat:.2f}%")
        _p(f"  Mean max_prob on mixed: {float(max_p[m].mean()):.4f}   "
           f"Mean max_frac on mixed: {float(max_f[m].mean()):.4f}")

    txt_path = os.path.join(args.out_dir, f"{args.tag}.txt")
    with open(txt_path, "w") as f:
        f.write(buf.getvalue())
    print(f"\nSaved: {txt_path}")


if __name__ == "__main__":
    main()
