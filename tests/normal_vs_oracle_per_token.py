"""Per-token per-mode error capture for Normal / OracleHard / OracleSoft.

For each sample, forward the model 3 times:
  Pass 0 — Normal        (self-routed by learned router; top_k=2 softmax)
  Pass 1 — OracleHard    (all router logits forced to one_hot(gt_bin))
  Pass 2 — OracleSoft    (all router logits forced to log(pixel_fraction))

Per token in the L2 NODE grid (29×50), compute:
  err_normal / err_hard / err_soft — mean |pred - gt| over LiDAR pixels
  gt_bin, depth, bound              — token-level GT statistics
  actual_pred                       — Normal router's argmax
  valid_pixel_count                 — LiDAR pixel count per token (for weighting)

The critical distinction from router_cost_analysis.py:
  • That script forces the WHOLE image to a single expert per pass, so
    per-token error for token t comes from a different forward pass than
    for token t'. The decoder (UNet) is spatially non-local, so this does
    NOT equal per-token gt-routed error.
  • This script uses PER-TOKEN gates within a single forward — matching
    exactly what eval_v3.py OracleHard / OracleSoft do, so aggregated
    per-token errors reproduce the eval MAE numbers.

Saves an npz plus a text summary.
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
    print(f"  loaded {run_dir}/{ckpt}  epoch={sd.get('epoch','?')}")
    return m.eval().to(device)


def compute_token_gt(dgd, tok_hw, return_type="hard"):
    """Compute per-token GT bin (hard) or pixel_fraction (soft).
    Returns:
      hard → (B, H_tok, W_tok) int64 argmax
      soft → (B, K, H_tok, W_tok) float pixel fraction
      both → (frac, hard)
    """
    edges = torch.tensor(BINS, device=dgd.device, dtype=torch.float32)
    ds = dgd.squeeze(1)
    pix_bin = torch.zeros_like(ds, dtype=torch.long)
    for i in range(K):
        pix_bin = torch.where((ds >= float(edges[i])) & (ds < float(edges[i+1])),
                              pix_bin.new_tensor(i), pix_bin)
    pix_bin = torch.where(ds >= float(edges[-1]),
                          pix_bin.new_tensor(K-1), pix_bin)
    onehot = F.one_hot(pix_bin, num_classes=K).permute(0, 3, 1, 2).float()
    frac = F.adaptive_avg_pool2d(onehot, tok_hw)                 # (B, K, H, W)
    if return_type == "soft":
        return frac
    hard = frac.argmax(dim=1)
    if return_type == "hard":
        return hard
    return frac, hard


def per_token_stats(dgd, tok_hw):
    """Return (gt_bin, mean_depth, boundary_score) — all (B, H, W)."""
    frac, gt_bin = compute_token_gt(dgd, tok_hw, return_type="both")
    bound = 1.0 - frac.max(dim=1).values
    mean_d = F.adaptive_avg_pool2d(dgd, tok_hw).squeeze(1)
    return gt_bin, mean_d, bound


def per_token_err(pred, gt, valid_mask, tok_hw):
    """Mean |pred-gt| over valid LiDAR pixels in each token's receptive field."""
    err = (pred - gt).abs() * valid_mask
    err_pooled = F.adaptive_avg_pool2d(err, tok_hw).squeeze(1)
    valid_pool = F.adaptive_avg_pool2d(valid_mask, tok_hw).squeeze(1)
    out = err_pooled / valid_pool.clamp_min(1e-8)
    return torch.where(valid_pool > 0, out, torch.full_like(out, float("nan")))


def per_token_valid_count(mask_lidar, tok_hw):
    """Per-token count of valid LiDAR pixels."""
    cnt = F.adaptive_avg_pool2d(mask_lidar, tok_hw).squeeze(1) * \
           float(mask_lidar.shape[-2] * mask_lidar.shape[-1]) / \
           float(tok_hw[0] * tok_hw[1])
    return cnt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="Model run_dir (MoE, top_k=2 ideal)")
    ap.add_argument("--tag", required=True, help="Filename tag: <tag>.{npz,txt}")
    ap.add_argument("--ckpt", default="best.pt")
    ap.add_argument("--n", type=int, default=None, help="Sample count (default: full)")
    ap.add_argument("--split", default="test")
    ap.add_argument("--out-dir", default="output/analysis/router_cost")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Dataset (use base config for path fields; augmentation off)
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

    # MoE blocks
    node_blocks = [(f"L{l}_node", m.radar_fusion.node_blocks[l])
                   for l in range(len(m.radar_fusion.node_blocks))
                   if isinstance(m.radar_fusion.node_blocks[l], MoEFusionBlock)]
    edge_blocks = [(f"L{l}_edge", m.radar_fusion.edge_blocks[l])
                   for l in range(len(m.radar_fusion.edge_blocks))
                   if isinstance(m.radar_fusion.edge_blocks[l], MoEFusionBlock)]
    all_moe = node_blocks + edge_blocks
    assert node_blocks, "No node MoE blocks — needed for per-token grid."
    node_block = node_blocks[0][1]
    top_k = node_block.top_k
    print(f"  Analyzing node block {node_blocks[0][0]} (top_k={top_k})")
    if top_k is None or top_k >= K:
        print("  WARN: model uses full soft gate (no top-k). OracleSoft and "
              "Normal will use the same gate mechanism — OracleSoft ≡ eval's "
              "OracleSoft still, since the *values* differ (fraction vs "
              "learned softmax).")

    # Mode-switch hook shared across all MoE routers. dgd_holder is set per
    # sample; mode ∈ {None, 'hard', 'soft'}.
    dgd_holder = {"dgd": None, "mode": None}
    def _hook(_m, _i, out):
        mode = dgd_holder["mode"]
        dgd = dgd_holder["dgd"]
        if mode is None or dgd is None:
            return out
        if mode == "soft":
            frac = compute_token_gt(dgd, out.shape[-2:], return_type="soft")
            return torch.log(frac.clamp_min(1e-8))
        tok_gt = compute_token_gt(dgd, out.shape[-2:], return_type="hard")
        forced = torch.full_like(out, -1e4)
        forced.scatter_(1, tok_gt.unsqueeze(1), 1e4)
        return forced
    hooks = [blk.router.register_forward_hook(_hook) for _, blk in all_moe]

    all_gt_bin = []
    all_actual_pred = []
    all_depth = []
    all_bound = []
    all_err_normal = []
    all_err_hard = []
    all_err_soft = []
    all_valid_cnt = []

    t0 = time.time()
    try:
        with torch.no_grad():
            for i in tqdm(idxs, desc="per-mode"):
                s = ds[int(i)]
                rgb = s["rgb_norm"].unsqueeze(0).to(device)
                rp  = s["radar_points"].unsqueeze(0).to(device)
                rm  = s["radar_mask"].unsqueeze(0).to(device)
                dgd = s["depth_gt_dense"].unsqueeze(0).to(device)
                gt_lidar = s["depth_gt_lidar"].unsqueeze(0).to(device)
                mask_lidar = s["valid_mask_lidar"].unsqueeze(0).to(device).float()
                dgd_holder["dgd"] = dgd

                # Pass 1 — Normal
                dgd_holder["mode"] = None
                out_n = m(rgb, rp, rm)
                depth_n = out_n["depth"] if isinstance(out_n, dict) else out_n
                rl = out_n["router_logits"][0]              # (1, K, H_tok, W_tok)
                tok_hw = tuple(rl.shape[-2:])
                actual_pred = rl.softmax(dim=1).argmax(dim=1).squeeze(0).cpu().numpy()

                gt_bin, mean_d, bound = per_token_stats(dgd, tok_hw)
                err_n = per_token_err(depth_n, gt_lidar, mask_lidar, tok_hw)
                vcnt = per_token_valid_count(mask_lidar, tok_hw)

                # Pass 2 — OracleHard
                dgd_holder["mode"] = "hard"
                out_h = m(rgb, rp, rm)
                depth_h = out_h["depth"] if isinstance(out_h, dict) else out_h
                err_h = per_token_err(depth_h, gt_lidar, mask_lidar, tok_hw)

                # Pass 3 — OracleSoft
                dgd_holder["mode"] = "soft"
                out_s = m(rgb, rp, rm)
                depth_s = out_s["depth"] if isinstance(out_s, dict) else out_s
                err_s = per_token_err(depth_s, gt_lidar, mask_lidar, tok_hw)

                all_gt_bin.append(gt_bin[0].cpu().numpy().ravel().astype(np.uint8))
                all_actual_pred.append(actual_pred.ravel().astype(np.uint8))
                all_depth.append(mean_d[0].cpu().numpy().ravel().astype(np.float32))
                all_bound.append(bound[0].cpu().numpy().ravel().astype(np.float32))
                all_err_normal.append(err_n[0].cpu().numpy().ravel().astype(np.float32))
                all_err_hard.append(err_h[0].cpu().numpy().ravel().astype(np.float32))
                all_err_soft.append(err_s[0].cpu().numpy().ravel().astype(np.float32))
                all_valid_cnt.append(vcnt[0].cpu().numpy().ravel().astype(np.float32))
    finally:
        for h in hooks: h.remove()
        dgd_holder.clear()

    print(f"  ({time.time()-t0:.0f}s forward — 3 passes/sample)")

    gt = np.concatenate(all_gt_bin)
    pred = np.concatenate(all_actual_pred)
    depth = np.concatenate(all_depth)
    bound = np.concatenate(all_bound)
    en = np.concatenate(all_err_normal)
    eh = np.concatenate(all_err_hard)
    es = np.concatenate(all_err_soft)
    vc = np.concatenate(all_valid_cnt)
    N = len(gt)
    valid = np.isfinite(en) & np.isfinite(eh) & np.isfinite(es) & (vc > 0)
    n_valid = int(valid.sum())
    print(f"  Total tokens: {N:,}  valid: {n_valid:,} ({100*n_valid/N:.1f}%)")

    # Save arrays
    npz_path = os.path.join(args.out_dir, f"{args.tag}.npz")
    np.savez_compressed(npz_path,
                        gt=gt, actual_pred=pred, depth=depth, bound=bound,
                        err_normal=en, err_hard=eh, err_soft=es,
                        valid_pixel_count=vc)
    print(f"  Saved: {npz_path}")

    # ================== Text summary ==================
    buf = io.StringIO()
    def _p(msg=""):
        print(msg); buf.write(msg + "\n")

    pix = vc[valid].sum()
    mae_n = float((en[valid] * vc[valid]).sum() / pix)
    mae_h = float((eh[valid] * vc[valid]).sum() / pix)
    mae_s = float((es[valid] * vc[valid]).sum() / pix)
    _p("=" * 96)
    _p(f"Per-token per-mode analysis — {args.tag}")
    _p(f"  run: {args.run}   split: {args.split}   samples: {len(idxs):,}")
    _p(f"  valid tokens: {n_valid:,}/{N:,} ({100*n_valid/N:.1f}%)  "
       f"valid pixels: {pix:,.0f}")
    _p(f"  Pixel-weighted MAE:  Normal={mae_n:.4f}  "
       f"OracleHard={mae_h:.4f}  OracleSoft={mae_s:.4f}")
    _p(f"  Gap Normal→OracleHard:  {mae_n - mae_h:+.4f} m")
    _p(f"  Gap OracleHard→OracleSoft:  {mae_h - mae_s:+.4f} m")
    _p("=" * 96)

    def _decompose(header, mode_a, mode_b, ea, eb):
        _p(f"\n===== {header} — decomposition (Σ(err_a − err_b) × pix, per bucket) =====")
        gap = (ea - eb) * vc
        v = valid
        tot = float(gap[v].sum())
        _p(f"  Total gap:  {tot:+.1f} m·pix  "
           f"(mean per pixel: {tot/pix:+.4f} m)")

        # (i) depth range
        _p("\n  -- by depth range --")
        _p(f"  {'range':>14s}  {'tokens':>10s}  {'MAE_a':>8s}  "
           f"{'MAE_b':>8s}  {'Δ':>8s}  {'Σgap':>12s}  {'%':>7s}")
        pos_tot = float(gap[v & (gap > 0)].sum())
        edges = [0, 10, 15, 20, 22, 25, 30, 40, 45, 50, 55, 60, 80, 100]
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = v & (depth >= lo) & (depth < hi)
            n = int(m.sum())
            if n == 0: continue
            p = float(vc[m].sum())
            ma = float((ea[m] * vc[m]).sum() / max(p, 1e-9))
            mb = float((eb[m] * vc[m]).sum() / max(p, 1e-9))
            gs = float(gap[m].sum())
            pct = 100 * gs / max(pos_tot if pos_tot != 0 else -tot, 1e-9)
            _p(f"  [{lo:>4d},{hi:>4d})m  {n:>10,d}  {ma:>7.3f}  "
               f"{mb:>7.3f}  {(ma-mb):>+7.3f}  {gs:>+12,.1f}  {pct:>6.2f}%")

        # (ii) boundary bucket
        _p("\n  -- by boundary score (bound = 1 − max pixel_frac) --")
        _p(f"  {'bucket':<26s}  {'tokens':>10s}  {'MAE_a':>8s}  "
           f"{'MAE_b':>8s}  {'Δ':>8s}  {'Σgap':>12s}  {'%':>7s}")
        for name, lo, hi in [
            ("PURE       [0.00,0.05)", 0.00, 0.05),
            ("SLIGHT MIX [0.05,0.15)", 0.05, 0.15),
            ("MODERATE   [0.15,0.30)", 0.15, 0.30),
            ("HIGH MIX   [0.30,0.70)", 0.30, 0.70),
        ]:
            m = v & (bound >= lo) & (bound < hi)
            n = int(m.sum())
            if n == 0: continue
            p = float(vc[m].sum())
            ma = float((ea[m] * vc[m]).sum() / max(p, 1e-9))
            mb = float((eb[m] * vc[m]).sum() / max(p, 1e-9))
            gs = float(gap[m].sum())
            pct = 100 * gs / max(pos_tot if pos_tot != 0 else -tot, 1e-9)
            _p(f"  {name:<26s}  {n:>10,d}  {ma:>7.3f}  {mb:>7.3f}  "
               f"{(ma-mb):>+7.3f}  {gs:>+12,.1f}  {pct:>6.2f}%")

        # (iii) misroute type (using Normal's actual_pred vs gt)
        _p("\n  -- by Normal routing outcome (pred vs gt) --")
        _p(f"  {'type':<32s}  {'tokens':>10s}  {'MAE_a':>8s}  "
           f"{'MAE_b':>8s}  {'Δ':>8s}  {'Σgap':>12s}  {'%':>7s}")
        diff = np.abs(pred.astype(np.int64) - gt.astype(np.int64))
        for name, cond in [
            ("Correct (pred==gt)", diff == 0),
            ("Adjacent misroute (|Δ|=1)", diff == 1),
            ("Cross-bin misroute (|Δ|=2)", diff == 2),
        ]:
            m = v & cond
            n = int(m.sum())
            if n == 0: continue
            p = float(vc[m].sum())
            ma = float((ea[m] * vc[m]).sum() / max(p, 1e-9))
            mb = float((eb[m] * vc[m]).sum() / max(p, 1e-9))
            gs = float(gap[m].sum())
            pct = 100 * gs / max(pos_tot if pos_tot != 0 else -tot, 1e-9)
            _p(f"  {name:<32s}  {n:>10,d}  {ma:>7.3f}  {mb:>7.3f}  "
               f"{(ma-mb):>+7.3f}  {gs:>+12,.1f}  {pct:>6.2f}%")

        # (iv) distance to nearest bin edge
        _p("\n  -- by distance to nearest bin edge (20m or 50m) --")
        _p(f"  {'edge_dist':<14s}  {'tokens':>10s}  {'MAE_a':>8s}  "
           f"{'MAE_b':>8s}  {'Δ':>8s}  {'Σgap':>12s}  {'%':>7s}")
        ed = np.minimum(np.abs(depth - 20.0), np.abs(depth - 50.0))
        for lo, hi in [(0, 1), (1, 3), (3, 5), (5, 10), (10, 30), (30, 100)]:
            m = v & (ed >= lo) & (ed < hi)
            n = int(m.sum())
            if n == 0: continue
            p = float(vc[m].sum())
            ma = float((ea[m] * vc[m]).sum() / max(p, 1e-9))
            mb = float((eb[m] * vc[m]).sum() / max(p, 1e-9))
            gs = float(gap[m].sum())
            pct = 100 * gs / max(pos_tot if pos_tot != 0 else -tot, 1e-9)
            _p(f"  ±{lo:>3d}-{hi:<3d}m       {n:>10,d}  {ma:>7.3f}  "
               f"{mb:>7.3f}  {(ma-mb):>+7.3f}  {gs:>+12,.1f}  {pct:>6.2f}%")

    _decompose("Normal vs OracleHard", "Normal", "OracleHard", en, eh)
    _decompose("OracleHard vs OracleSoft", "OracleHard", "OracleSoft", eh, es)
    _decompose("Normal vs OracleSoft", "Normal", "OracleSoft", en, es)

    txt_path = os.path.join(args.out_dir, f"{args.tag}.txt")
    with open(txt_path, "w") as f:
        f.write(buf.getvalue())
    print(f"\nSaved: {txt_path}")


if __name__ == "__main__":
    main()
