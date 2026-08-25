"""Per-token routing cost analysis — how much depth error does each routing
decision actually cause?

For each sample, forward the model 4 times:
  Pass 0 — Actual routing (self-routed by learned router)
  Pass 1 — Force ALL tokens to expert 0 (near)
  Pass 2 — Force ALL tokens to expert 1 (mid)
  Pass 3 — Force ALL tokens to expert 2 (far)

Per token in the L2 NODE grid (29×50 by default), compute:
  • per_expert_err[K] — mean |pred - gt| over LiDAR valid pixels in the
                        token's receptive field, for each expert alone
  • actual_err        — same but for the actual (mixed) routing
  • best_expert       — argmin(per_expert_err) → the depth-optimal choice
  • gt_bin            — majority-vote pixel bin (existing router CE target)
  • actual_pred_bin   — router's argmax at this token
  • depth, bound      — token mean depth + boundary score (as in router_diag)

Saves an npz for offline analysis, plus prints an aggregated summary:
  • Routing cost (= actual_err − per_expert_err[best]) by depth range
  • Whether GT bin = task-optimal bin (does CE target match task?)
  • Cost of cross-bin vs adjacent-bin misroutes

This directly answers: "does misrouting a 21m token to near expert really
cost much depth error?" — and, more generally, WHERE in depth space each
routing mistake actually hurts.

CAVEAT — GroupNorm side-effect:
  Each expert uses `_masked_group_norm` normalized over the tokens routed
  to it. Forcing one expert on ALL tokens changes the normalization stats
  vs the actual sparse-dispatch run. So `per_expert_err[k]` is a slight
  approximation of "what expert k would have predicted for this token
  in the real routing regime" (differences are typically small, but can
  produce tiny anomalies like negative routing cost in aggregates).
"""
import argparse, os, sys, time
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
BINS = [0.0, 20.0, 50.0, 100.0]
LABELS = ["near", "mid", "far"]
K = 3


def load_model(run_dir, ckpt="best.pt"):
    cfg = OmegaConf.load(os.path.join(run_dir, "config.yaml"))
    m = _build_model(cfg, float(cfg.dataset.max_depth), int(cfg.dataset.max_radar_points))
    sd = torch.load(os.path.join(run_dir, ckpt), map_location=device, weights_only=False)
    m.load_state_dict(sd["model"])
    print(f"  loaded {run_dir}/{ckpt}  epoch={sd.get('epoch','?')}")
    return m.eval().to(device)


def per_token_stats(depth_gt_dense, tok_hw, bins):
    """Return (gt_bin, mean_depth, boundary_score) — all (B, H, W)."""
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
    gt_bin = frac.argmax(dim=1)
    bound = 1.0 - frac.max(dim=1).values
    mean_d = F.adaptive_avg_pool2d(ds.unsqueeze(1), tok_hw).squeeze(1)
    return gt_bin, mean_d, bound


def per_token_err(pred, gt, valid_mask, tok_hw):
    """pred / gt: (B, 1, H, W).  valid_mask: (B, 1, H, W) float{0,1}.
    Returns (B, H_tok, W_tok) mean |pred - gt| over valid pixels per token.
    Tokens with 0 valid pixels → NaN (caller must mask them out)."""
    err = (pred - gt).abs() * valid_mask                        # (B, 1, H, W)
    err_pooled = F.adaptive_avg_pool2d(err, tok_hw).squeeze(1)  # (B, H_tok, W_tok)
    valid_pool = F.adaptive_avg_pool2d(valid_mask, tok_hw).squeeze(1)
    # adaptive_avg_pool computes mean over ALL pixels in the receptive field
    # (including masked-out zeros), so we need to renormalize:
    #   per_token_err = mean(|err|*valid) / mean(valid) = mean(|err|) over VALID pixels
    out = err_pooled / valid_pool.clamp_min(1e-8)
    # Mark tokens with no valid pixels as NaN.
    out = torch.where(valid_pool > 0, out, torch.full_like(out, float("nan")))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="Model run_dir (should be MoE)")
    ap.add_argument("--tag", required=True, help="Filename tag: <tag>.{npz,txt}")
    ap.add_argument("--ckpt", default="best.pt")
    ap.add_argument("--n", type=int, default=None, help="Sample count (default: full)")
    ap.add_argument("--split", default="test")
    ap.add_argument("--out-dir", default="output/analysis/router_cost")
    args = ap.parse_args()

    OUT = "output"
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
    print(f"{args.split} — {len(idxs):,}/{len(ds):,} samples")

    m = load_model(args.run, args.ckpt)

    # Focus on L2 node block for per-token analysis (matches router_diagnostics).
    moe_blocks = [(f"L{l}_node", m.radar_fusion.node_blocks[l])
                  for l in range(len(m.radar_fusion.node_blocks))
                  if isinstance(m.radar_fusion.node_blocks[l], MoEFusionBlock)]
    edge_moe = [(f"L{l}_edge", m.radar_fusion.edge_blocks[l])
                for l in range(len(m.radar_fusion.edge_blocks))
                if isinstance(m.radar_fusion.edge_blocks[l], MoEFusionBlock)]
    all_moe = moe_blocks + edge_moe
    assert moe_blocks, "No node MoE blocks — this analysis needs at least one."
    node_block = moe_blocks[0][1]
    print(f"  Analyzing node block {moe_blocks[0][0]} (top_k={node_block.top_k})")

    # Hook holder: 'mode' → None (actual) | 0/1/2 (force expert k on ALL routers).
    holder = {"force_expert": None}
    def make_hook():
        def _h(_m, _i, out):
            k = holder["force_expert"]
            if k is None:
                return out
            forced = torch.full_like(out, -1e4)
            forced[:, k, :, :] = 1e4
            return forced
        return _h
    hooks = [blk.router.register_forward_hook(make_hook()) for _, blk in all_moe]

    # Buffers for per-token arrays (concatenate at end).
    all_gt_bin = []
    all_actual_pred = []
    all_depth = []
    all_bound = []
    all_actual_err = []          # (B, H_tok, W_tok)
    all_expert_err = []          # (K, B, H_tok, W_tok)
    all_valid_pixel_count = []   # per-token valid pixel count (for weighting later)

    t0 = time.time()
    try:
        with torch.no_grad():
            for i in tqdm(idxs, desc="cost analysis"):
                s = ds[int(i)]
                rgb = s["rgb_norm"].unsqueeze(0).to(device)
                rp  = s["radar_points"].unsqueeze(0).to(device)
                rm  = s["radar_mask"].unsqueeze(0).to(device)
                dgd = s["depth_gt_dense"].unsqueeze(0).to(device)   # (1, 1, H, W)
                gt_lidar = s["depth_gt_lidar"].unsqueeze(0).to(device)  # (1, 1, H, W)
                mask_lidar = s["valid_mask_lidar"].unsqueeze(0).to(device).float()  # (1, 1, H, W)

                # Determine node block's token grid from a probe forward.
                # We do the "actual" pass first and read router_logits shape.
                holder["force_expert"] = None
                out = m(rgb, rp, rm)
                actual_depth = out["depth"] if isinstance(out, dict) else out  # (1,1,H,W)
                rl = out["router_logits"][0]                                    # (1, K, h, w)
                tok_hw = tuple(rl.shape[-2:])
                actual_pred = rl.softmax(dim=1).argmax(dim=1).squeeze(0).cpu().numpy()  # (h,w)

                # Per-token GT stats.
                gt_bin, mean_d, bound = per_token_stats(dgd, tok_hw, BINS)
                gt_bin_np = gt_bin[0].cpu().numpy()
                depth_np = mean_d[0].cpu().numpy()
                bound_np = bound[0].cpu().numpy()

                # Per-token error under ACTUAL routing.
                actual_err = per_token_err(actual_depth, gt_lidar, mask_lidar, tok_hw)
                actual_err_np = actual_err[0].cpu().numpy()

                # Per-token valid pixel count (for later weighting / filtering).
                valid_cnt = F.adaptive_avg_pool2d(mask_lidar, tok_hw).squeeze(1) * \
                            float(mask_lidar.shape[-2] * mask_lidar.shape[-1]) / \
                            float(tok_hw[0] * tok_hw[1])
                valid_cnt_np = valid_cnt[0].cpu().numpy()

                # Force each expert in turn.
                expert_err_stack = np.full((K, *tok_hw), np.nan, dtype=np.float32)
                for k in range(K):
                    holder["force_expert"] = k
                    out_k = m(rgb, rp, rm)
                    depth_k = out_k["depth"] if isinstance(out_k, dict) else out_k
                    err_k = per_token_err(depth_k, gt_lidar, mask_lidar, tok_hw)
                    expert_err_stack[k] = err_k[0].cpu().numpy()
                holder["force_expert"] = None

                all_gt_bin.append(gt_bin_np.ravel().astype(np.uint8))
                all_actual_pred.append(actual_pred.ravel().astype(np.uint8))
                all_depth.append(depth_np.ravel().astype(np.float32))
                all_bound.append(bound_np.ravel().astype(np.float32))
                all_actual_err.append(actual_err_np.ravel().astype(np.float32))
                all_expert_err.append(expert_err_stack.reshape(K, -1).astype(np.float32))
                all_valid_pixel_count.append(valid_cnt_np.ravel().astype(np.float32))
    finally:
        for h in hooks: h.remove()

    print(f"  ({time.time()-t0:.0f}s forward)")

    gt = np.concatenate(all_gt_bin)
    ap_arr = np.concatenate(all_actual_pred)
    depth = np.concatenate(all_depth)
    bound = np.concatenate(all_bound)
    actual_err = np.concatenate(all_actual_err)
    expert_err = np.concatenate(all_expert_err, axis=1)   # (K, N_total)
    valid_cnt = np.concatenate(all_valid_pixel_count)
    N = len(gt)
    has_lidar = ~np.isnan(actual_err)
    print(f"  Total tokens: {N:,}  (with LiDAR pixels: {has_lidar.sum():,}, "
          f"{100*has_lidar.mean():.1f}%)")

    # Task-optimal expert per token (min error). Tokens with no LiDAR pixels
    # have all-NaN expert_err; nanargmin would raise on those, so set them
    # to a sentinel (0) and rely on `has_lidar` mask downstream.
    has_expert_data = ~np.all(np.isnan(expert_err), axis=0)           # (N,)
    best_expert = np.zeros(N, dtype=np.uint8)
    best_err = np.full(N, np.nan, dtype=np.float32)
    if has_expert_data.any():
        best_expert[has_expert_data] = np.nanargmin(
            expert_err[:, has_expert_data], axis=0).astype(np.uint8)
        best_err[has_expert_data] = np.nanmin(
            expert_err[:, has_expert_data], axis=0)
    routing_regret = actual_err - best_err                            # NaN where no LiDAR

    # ================= Summary =================
    import io
    buf = io.StringIO()
    def _p(msg=""):
        print(msg); buf.write(msg + "\n")

    _p("=" * 96)
    _p(f"Per-token routing cost analysis — {args.tag}")
    _p(f"  run: {args.run}   split: {args.split}   n_samples: {len(idxs)}")
    _p(f"  tokens with valid LiDAR: {has_lidar.sum():,}/{N:,} "
       f"({100*has_lidar.mean():.1f}%)")
    _p("=" * 96)

    # ---- 1. Does GT bin (CE target) equal task-optimal expert? ----
    valid = has_lidar
    n_v = int(valid.sum())
    gt_equals_task = (gt[valid] == best_expert[valid])
    pct_agree = 100 * gt_equals_task.mean()
    _p("\n-- 1. Does CE target (GT majority-vote bin) match task-optimal expert? --")
    _p(f"  Overall agreement: {pct_agree:.2f}%  ({gt_equals_task.sum():,}/{n_v:,})")
    _p("  → When these disagree, CE loss is teaching the router the WRONG thing")
    _p("    from a depth-error perspective.")
    _p("  Disagreement breakdown (rows=GT bin, cols=task-optimal):")
    _p("       " + "  ".join(f"{l:>10}" for l in LABELS))
    for i in range(K):
        row = f"  {LABELS[i]:<4}"
        for j in range(K):
            n = ((gt[valid] == i) & (best_expert[valid] == j)).sum()
            row += f"  {n:>10,}"
        _p(row)

    # ---- 2. Actual routing cost by depth range ----
    _p("\n-- 2. Routing cost (actual_err − best_expert_err) by depth range --")
    _p("  Reports mean per-token gap between actual routing and task-optimal.")
    _p(f"{'depth range':<15} {'n_tokens':>10} {'actual':>8} {'best_e':>8} "
       f"{'cost':>8} {'wrong%':>8}")
    for lo, hi in [(0, 10), (10, 15), (15, 20), (20, 25), (25, 30),
                   (30, 40), (40, 50), (50, 55), (55, 60), (60, 80), (80, 100)]:
        m = valid & (depth >= lo) & (depth < hi)
        n = int(m.sum())
        if n == 0: continue
        a = float(np.nanmean(actual_err[m]))
        b = float(np.nanmean(best_err[m]))
        c = a - b
        w = 100 * (ap_arr[m] != gt[m]).mean()
        _p(f"  [{lo:>3},{hi:>4})m  {n:>10,}  {a:>7.3f}  {b:>7.3f}  {c:>7.3f}  {w:>7.2f}%")

    # ---- 3. Cost by boundary status ----
    _p("\n-- 3. Routing cost by boundary status (bound = 1 − max bin frac) --")
    _p(f"{'bound bucket':<20} {'n_tokens':>10} {'actual':>8} {'best_e':>8} {'cost':>8}")
    for lo, hi, name in [(0.0, 0.05, "PURE      "),
                         (0.05, 0.15, "SLIGHT MIX"),
                         (0.15, 0.30, "MODERATE  "),
                         (0.30, 0.70, "HIGH MIX  ")]:
        m = valid & (bound >= lo) & (bound < hi)
        n = int(m.sum())
        if n == 0: continue
        a = float(np.nanmean(actual_err[m]))
        b = float(np.nanmean(best_err[m]))
        _p(f"  {name} [{lo:.2f},{hi:.2f})  {n:>10,}  {a:>7.3f}  {b:>7.3f}  {a-b:>7.3f}")

    # ---- 4. Cost by edge distance ----
    dist = np.minimum(np.abs(depth - 20), np.abs(depth - 50))
    _p("\n-- 4. Routing cost by distance from nearest bin edge (20m or 50m) --")
    _p(f"{'edge_dist':<15} {'n_tokens':>10} {'actual':>8} {'best_e':>8} {'cost':>8}")
    for lo, hi in [(0, 1), (1, 3), (3, 5), (5, 10), (10, 30), (30, 100)]:
        m = valid & (dist >= lo) & (dist < hi)
        n = int(m.sum())
        if n == 0: continue
        a = float(np.nanmean(actual_err[m]))
        b = float(np.nanmean(best_err[m]))
        _p(f"  ±{lo:>3}-{hi:<4}m   {n:>10,}  {a:>7.3f}  {b:>7.3f}  {a-b:>7.3f}")

    # ---- 5. Cost of wrong routing by type (adjacent vs cross) ----
    _p("\n-- 5. Cost of wrong routing (actual_pred ≠ gt) by error type --")
    wrong = valid & (ap_arr != gt)
    adjacent = (
        ((gt == 0) & (ap_arr == 1)) | ((gt == 1) & (ap_arr == 0)) |
        ((gt == 1) & (ap_arr == 2)) | ((gt == 2) & (ap_arr == 1))
    )
    cross = ((gt == 0) & (ap_arr == 2)) | ((gt == 2) & (ap_arr == 0))
    for name, cond in [("Correct routing   ", valid & ~wrong),
                       ("Adjacent misroute ", wrong & adjacent),
                       ("Cross-bin misroute", wrong & cross)]:
        n = int(cond.sum())
        if n == 0: continue
        a = float(np.nanmean(actual_err[cond]))
        b = float(np.nanmean(best_err[cond]))
        _p(f"  {name}  n={n:>10,}  actual_err={a:.3f}  best_err={b:.3f}  "
           f"cost={a-b:.3f}")

    # ---- 6. Per-expert error profile ----
    _p("\n-- 6. Per-expert error by token's GT bin (validating expert specialization) --")
    _p("  Values are mean per-token |pred - gt| when only that expert is used.")
    _p(f"{'GT bin':<10}" + "  ".join(f"{'e_'+l:>8}" for l in LABELS)
       + f"  {'n':>10}")
    for i in range(K):
        m = valid & (gt == i)
        n = int(m.sum())
        if n == 0: continue
        row = f"  {LABELS[i]:<8}"
        for k in range(K):
            row += f"  {float(np.nanmean(expert_err[k, m])):>7.3f}"
        row += f"  {n:>10,}"
        _p(row)

    # ---- 7. Overlap-bin simulation refined by actual cost ----
    _p("\n-- 7. Overlap-bin (edge-adjacent forgive) — depth-cost weighted --")
    _p("  If we forgive adjacent misroutes on tokens within ±W of an edge,")
    _p("  cost gain = sum of (actual_err − best_err) recovered.")
    total_cost = float(np.nansum(actual_err[valid] - best_err[valid]))
    _p(f"  Total current routing cost across valid tokens: {total_cost:,.1f} m·tokens")
    for w in [3, 5, 7, 10, 15]:
        near_edge = (np.abs(depth - 20) < w) | (np.abs(depth - 50) < w)
        m = valid & wrong & adjacent & near_edge
        recovered = float(np.nansum(actual_err[m] - best_err[m]))
        pct = 100 * recovered / max(total_cost, 1e-6)
        _p(f"  ±{w:>3}m overlap: {int(m.sum()):>10,} tokens forgiven, "
           f"cost recovered = {recovered:>9.1f} m·tokens ({pct:5.1f}% of total)")

    # ---- Save ----
    os.makedirs(args.out_dir, exist_ok=True)
    npz_path = os.path.join(args.out_dir, f"{args.tag}.npz")
    np.savez_compressed(npz_path,
                        gt=gt, actual_pred=ap_arr, depth=depth, bound=bound,
                        actual_err=actual_err, expert_err=expert_err,
                        best_expert=best_expert, best_err=best_err,
                        valid_pixel_count=valid_cnt)
    txt_path = os.path.join(args.out_dir, f"{args.tag}.txt")
    with open(txt_path, "w") as f:
        f.write(buf.getvalue())
    _p(f"\nSaved raw: {npz_path}")
    _p(f"Saved txt: {txt_path}")


if __name__ == "__main__":
    main()
