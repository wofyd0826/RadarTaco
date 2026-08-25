"""Precisely decompose Normal vs OracleHard MAE gap by token attributes.

Given the router_cost_analysis npz with per-token:
  - actual_err   : Normal routing error (top_k=2 softmax)
  - expert_err[3]: error under each expert if forced
  - gt           : majority-vote bin (also = OracleHard's gate)
  - depth        : token center depth
  - bound        : 1 - max(pixel_fraction), boundary/mixing score
  - actual_pred  : what Normal actually chose (argmax)

Per-token OracleHard error := expert_err[gt].
Per-token Normal→OracleHard gap := actual_err - expert_err[gt].

We split the total gap by:
  (i)   depth range        (which distances hurt the most?)
  (ii)  bound / mixing bucket  (are boundary tokens the culprits?)
  (iii) misroute type       (correct / adjacent / cross-bin)
  (iv)  depth distance to nearest bin edge (20m / 50m)
  (v)   top-N single tokens (are a few extremes dominating?)

Prints a text summary. Also saves grouped stats to output/analysis/router_cost/
gap_decomposition_v4c_stage2.txt.
"""
import argparse
import os
import sys

import numpy as np


def _fmt_bar(v: float, vmax: float, width: int = 20) -> str:
    n = int(round(width * (v / vmax))) if vmax > 0 else 0
    return "#" * n


def analyze(npz_path: str, out_path: str) -> None:
    d = np.load(npz_path)
    gt = d["gt"].astype(np.int64)
    pred = d["actual_pred"].astype(np.int64)
    depth = d["depth"].astype(np.float32)
    bound = d["bound"].astype(np.float32)
    actual_err = d["actual_err"].astype(np.float32)
    expert_err = d["expert_err"].astype(np.float32)          # (3, N)
    best_expert = d["best_expert"].astype(np.int64)
    n_pix = d["valid_pixel_count"].astype(np.float32)        # (N,) — pixels/token

    N = gt.shape[0]
    valid = np.isfinite(actual_err) & np.all(np.isfinite(expert_err), axis=0)
    valid &= (n_pix > 0)
    n_valid = int(valid.sum())
    total_valid_pix = float(n_pix[valid].sum())

    # OracleHard error = force expert=gt bin's expert.
    oracle_err = np.take_along_axis(expert_err, gt[None, :], axis=0).squeeze(0)  # (N,)
    gap = actual_err - oracle_err          # positive → Normal loses to OracleHard
    # Pixel-weighted contribution: MAE aggregates per-pixel, so per-token
    # contribution to overall MAE is err × pixel_count. This is what makes
    # the analysis reconcile with the eval numbers.
    gap_w = gap * n_pix
    actual_err_w = actual_err * n_pix
    oracle_err_w = oracle_err * n_pix

    lines = []
    def log(msg: str = ""):
        print(msg); lines.append(msg)

    log("=" * 96)
    log(f"Normal → OracleHard gap decomposition — {os.path.basename(npz_path)}")
    log(f"  Tokens with finite errors: {n_valid:,}/{N:,} ({100*n_valid/N:.1f}%)")
    log(f"  Total valid LiDAR pixels: {total_valid_pix:,.0f}")
    total_gap_w = float(gap_w[valid].sum())
    pixmae_normal = float(actual_err_w[valid].sum()) / max(total_valid_pix, 1e-9)
    pixmae_oracle = float(oracle_err_w[valid].sum()) / max(total_valid_pix, 1e-9)
    log(f"  Pixel-weighted Normal MAE:      {pixmae_normal:.4f} m")
    log(f"  Pixel-weighted OracleHard MAE:  {pixmae_oracle:.4f} m")
    log(f"  Δ (Normal − OracleHard):        {pixmae_normal - pixmae_oracle:+.4f} m")
    log(f"  Total pixel-weighted gap:       {total_gap_w:+,.1f} m·pix")
    log(f"  (Eval-reported: Normal 1.9124, OracleHard 1.6406, Δ +0.2718 m)")
    log("=" * 96)

    # ------------------------------------------------------------------
    # (i) Depth range
    # ------------------------------------------------------------------
    log("\n-- 1. Gap by depth range --")
    edges = [0, 10, 15, 20, 22, 25, 30, 40, 45, 50, 55, 60, 80, 100]
    log(f"{'range':>14s}  {'n_tokens':>10s}  {'act_err':>8s}  "
        f"{'orc_err':>8s}  {'gap':>8s}  {'Σgap':>10s}  {'%Σgap':>7s}")
    per_range = []
    total_pos_gap = float(gap_w[valid & (gap_w > 0)].sum())
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = valid & (depth >= lo) & (depth < hi)
        n = int(m.sum())
        if n == 0:
            continue
        pix = float(n_pix[m].sum())
        ae = float(actual_err_w[m].sum()) / max(pix, 1e-9)
        oe = float(oracle_err_w[m].sum()) / max(pix, 1e-9)
        g = ae - oe
        gs = float(gap_w[m].sum())
        per_range.append((lo, hi, n, ae, oe, g, gs))
        pct = 100 * gs / max(total_pos_gap, 1e-9)
        log(f"[{lo:>4d},{hi:>4d})m  {n:>10,d}  {ae:>7.3f}  "
            f"{oe:>7.3f}  {g:>+7.3f}  {gs:>+10,.1f}  {pct:>6.2f}%")

    # ------------------------------------------------------------------
    # (ii) Boundary / mixing bucket
    # ------------------------------------------------------------------
    log("\n-- 2. Gap by boundary score (bound = 1 − max pixel_fraction) --")
    log(f"{'bucket':<28s}  {'n_tokens':>10s}  {'act_err':>8s}  "
        f"{'orc_err':>8s}  {'gap':>8s}  {'Σgap':>10s}  {'%Σgap':>7s}")
    bkts = [
        ("PURE       [0.00,0.05)", 0.00, 0.05),
        ("SLIGHT MIX [0.05,0.15)", 0.05, 0.15),
        ("MODERATE   [0.15,0.30)", 0.15, 0.30),
        ("HIGH MIX   [0.30,0.70)", 0.30, 0.70),
    ]
    for name, lo, hi in bkts:
        m = valid & (bound >= lo) & (bound < hi)
        n = int(m.sum())
        if n == 0:
            continue
        pix = float(n_pix[m].sum())
        ae = float(actual_err_w[m].sum()) / max(pix, 1e-9)
        oe = float(oracle_err_w[m].sum()) / max(pix, 1e-9)
        g = ae - oe
        gs = float(gap_w[m].sum())
        pct = 100 * gs / max(total_pos_gap, 1e-9)
        log(f"{name:<28s}  {n:>10,d}  {ae:>7.3f}  {oe:>7.3f}  "
            f"{g:>+7.3f}  {gs:>+10,.1f}  {pct:>6.2f}%")

    # ------------------------------------------------------------------
    # (iii) Misroute type: correct / adjacent / cross-bin
    # ------------------------------------------------------------------
    log("\n-- 3. Gap by routing outcome (pred vs gt) --")
    diff = np.abs(pred - gt)
    types = [
        ("Correct routing (pred==gt)", diff == 0),
        ("Adjacent misroute (|Δ|=1)", diff == 1),
        ("Cross-bin misroute (|Δ|=2)", diff == 2),
    ]
    log(f"{'type':<32s}  {'n_tokens':>10s}  {'act_err':>8s}  "
        f"{'orc_err':>8s}  {'gap':>8s}  {'Σgap':>10s}  {'%Σgap':>7s}")
    for name, cond in types:
        m = valid & cond
        n = int(m.sum())
        if n == 0:
            continue
        pix = float(n_pix[m].sum())
        ae = float(actual_err_w[m].sum()) / max(pix, 1e-9)
        oe = float(oracle_err_w[m].sum()) / max(pix, 1e-9)
        g = ae - oe
        gs = float(gap_w[m].sum())
        pct = 100 * gs / max(total_pos_gap, 1e-9)
        log(f"{name:<32s}  {n:>10,d}  {ae:>7.3f}  {oe:>7.3f}  "
            f"{g:>+7.3f}  {gs:>+10,.1f}  {pct:>6.2f}%")

    # Correct-routing decomposition: even if pred==gt, actual (top_k=2 softmax
    # blend) may differ from expert_err[gt] because top_k=2 mixes two experts.
    log("\n   Note: 'Correct routing' rows can have non-zero gap because Normal "
        "uses top_k=2 softmax mixing (not one_hot). Positive value = mixing HURTS "
        "vs pure expert_gt; negative = mixing HELPS.")

    # ------------------------------------------------------------------
    # (iv) Distance to nearest bin edge (20 or 50 m)
    # ------------------------------------------------------------------
    log("\n-- 4. Gap by distance to nearest bin edge (20m or 50m) --")
    d20 = np.abs(depth - 20.0)
    d50 = np.abs(depth - 50.0)
    ed = np.minimum(d20, d50)
    edge_buckets = [(0, 1), (1, 3), (3, 5), (5, 10), (10, 30), (30, 100)]
    log(f"{'edge_dist':<14s}  {'n_tokens':>10s}  {'act_err':>8s}  "
        f"{'orc_err':>8s}  {'gap':>8s}  {'Σgap':>10s}  {'%Σgap':>7s}")
    for lo, hi in edge_buckets:
        m = valid & (ed >= lo) & (ed < hi)
        n = int(m.sum())
        if n == 0:
            continue
        pix = float(n_pix[m].sum())
        ae = float(actual_err_w[m].sum()) / max(pix, 1e-9)
        oe = float(oracle_err_w[m].sum()) / max(pix, 1e-9)
        g = ae - oe
        gs = float(gap_w[m].sum())
        pct = 100 * gs / max(total_pos_gap, 1e-9)
        tag = f"±{lo:>3d}-{hi:<3d}m"
        log(f"{tag:<14s}  {n:>10,d}  {ae:>7.3f}  {oe:>7.3f}  "
            f"{g:>+7.3f}  {gs:>+10,.1f}  {pct:>6.2f}%")

    # ------------------------------------------------------------------
    # (v) Top-1000 tokens by pixel-weighted gap (are extremes dominating?)
    # ------------------------------------------------------------------
    log("\n-- 5. Concentration of gap in top tokens (pixel-weighted) --")
    g_valid = gap_w[valid]
    total_valid_gap = float(g_valid[g_valid > 0].sum())
    for topN in (100, 1000, 10000, 100000):
        # take topN largest positive gaps
        if topN > g_valid.size:
            continue
        top = np.partition(g_valid, -topN)[-topN:]
        pos = top[top > 0]
        share = 100 * float(pos.sum()) / max(total_valid_gap, 1e-9)
        log(f"  top {topN:>6,d} tokens: contain {share:>6.2f}% of total positive gap  "
            f"(mean top-gap = {float(pos.mean()):.3f} m/tok, n_pos={pos.size})")

    # ------------------------------------------------------------------
    # (vi) 2D: depth × edge_dist joint slicing (quick view of concentration)
    # ------------------------------------------------------------------
    log("\n-- 6. Where does most of the gap live (depth × edge_dist)? --")
    log("   Rows: gt bin,  Cols: |depth-nearest_edge| bucket")
    edge_bkt = np.digitize(ed, [1, 3, 5, 10, 30])  # 0..5
    edge_names = ["0-1", "1-3", "3-5", "5-10", "10-30", "30+"]
    for b in range(3):
        m0 = valid & (gt == b)
        row = []
        for eb in range(len(edge_names)):
            m = m0 & (edge_bkt == eb)
            gs = float(gap_w[m].sum()) if m.any() else 0.0
            row.append(gs)
        pct_row = [100 * v / max(total_pos_gap, 1e-9) for v in row]
        log(f"  gt={['near', 'mid', 'far'][b]:<4s}  " +
            "  ".join(f"{edge_names[i]:>6s}: {row[i]:>+8,.0f} ({pct_row[i]:>5.1f}%)"
                     for i in range(len(edge_names))))

    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nSaved: {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz",
                    default="output/analysis/router_cost/v4c_stage2.npz")
    ap.add_argument("--out",
                    default="output/analysis/router_cost/"
                            "gap_decomposition_v4c_stage2.txt")
    args = ap.parse_args()
    analyze(args.npz, args.out)


if __name__ == "__main__":
    main()
