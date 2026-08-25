"""Analyze WHICH tokens the router finds hard, and WHICH cause the most error.

Two axes:
  HARD to predict:
    - L1(probs, frac) : how far model's distribution is from target
    - argmax mismatch : model picks wrong bin
    - low max_prob    : model itself signals uncertainty
    - top-2 pair mismatch : model can't even identify the ambiguity zone

  COSTLY (cause error):
    - mae            : per-token depth error
    - mae * valid_pix: contribution to overall MAE

Cross-analysis:
  - Are the HARD tokens also the COSTLY tokens?
  - Distribution of costly tokens by depth, boundary, position, etc.
  - Do argmax-correct tokens ever cause big MAE? (top-2 weighting failure)

Reads: output/analysis/router_cost/v4c_stage2_per_token_dump.npz
Writes: output/analysis/router_cost/hard_and_costly_v4c.txt
"""
import io
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, ROOT)

BINS = [0.0, 20.0, 50.0, 100.0]
K = 3


def main():
    path = "output/analysis/router_cost/v4c_stage2_per_token_dump.npz"
    d = np.load(path, allow_pickle=True)

    sample_idx = d['sample_idx']
    row = d['token_row']
    col = d['token_col']
    probs = d['probs']                    # (N, 3)
    frac = d['frac']                      # (N, 3)
    mae = d['mae']                        # (N,)  NaN where no LiDAR
    depth = d['depth']                    # per-token mean dense-GT depth
    vc = d['valid_pix_count']             # LiDAR pixels per token
    argmax_p = d['argmax_p'].astype(np.int64)
    argmax_f = d['argmax_f'].astype(np.int64)
    sample_ids = d['sample_ids']
    grid_hw = d['grid_hw']; image_hw = d['image_hw']

    N = len(mae)
    valid = np.isfinite(mae) & (vc > 0)
    print(f"tokens: {N:,}  valid (with LiDAR): {valid.sum():,}  "
          f"grid={tuple(grid_hw)}  image={tuple(image_hw)}")

    # Derived
    max_p = probs.max(axis=1)
    max_f = frac.max(axis=1)
    l1 = np.abs(probs - frac).sum(axis=1)     # [0, 2]
    bound = 1.0 - max_f                       # target boundary score
    entropy = -(probs * np.log(probs.clip(1e-12, 1.0))).sum(axis=1) / np.log(K)

    def top2(a):
        idx = np.argsort(-a, axis=1)[:, :2]
        return np.sort(idx, axis=1)
    top2_p = top2(probs); top2_f = top2(frac)
    top2_match = (top2_p[:, 0] == top2_f[:, 0]) & (top2_p[:, 1] == top2_f[:, 1])
    argmax_match = (argmax_p == argmax_f)

    # Pixel-weighted MAE contribution
    mae_pix = mae * vc     # NaN * anything = NaN, but valid mask handles it
    total_pix = float(vc[valid].sum())
    total_mae_pix = float(mae_pix[valid].sum())
    pix_weighted_mae = total_mae_pix / total_pix
    print(f"Pixel-weighted MAE across valid: {pix_weighted_mae:.4f} m "
          f"({total_pix:,.0f} pixels)")

    buf = io.StringIO()
    def _p(msg=""):
        print(msg); buf.write(msg + "\n")

    _p("=" * 96)
    _p(f"HARD vs COSTLY token analysis — v4c stage2")
    _p(f"  {valid.sum():,} valid tokens ({100*valid.mean():.1f}%)  "
       f"{total_pix:,.0f} LiDAR pixels  overall MAE {pix_weighted_mae:.4f} m")
    _p("=" * 96)

    # ================================================================
    # PART 1 — Where does MAE (costly tokens) live?
    # ================================================================
    _p("\n" + "=" * 96)
    _p("PART 1 — WHERE MAE COMES FROM (costly tokens)")
    _p("=" * 96)

    # Top-N tokens by MAE * pixel count
    _p("\n-- 1a. Concentration: top-N tokens' share of total MAE·pixels --")
    contrib = np.where(valid, mae_pix, 0.0)
    order = np.argsort(-contrib)
    for topN in (100, 1000, 10000, 100000, 500000):
        share = 100 * contrib[order[:topN]].sum() / total_mae_pix
        _p(f"  top {topN:>7,d} tokens contain {share:>6.2f}% of total MAE·pix "
           f"(that's {100*topN/valid.sum():.2f}% of tokens)")

    # By depth range
    _p("\n-- 1b. MAE by depth range (pixel-weighted, share of total error) --")
    _p(f"  {'range':>14s}  {'n_tokens':>10s}  {'n_pix':>12s}  "
       f"{'MAE':>7s}  {'ΣMAE·pix':>14s}  {'%':>7s}")
    edges = [0, 10, 15, 20, 22, 25, 30, 40, 45, 50, 55, 60, 80, 100]
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = valid & (depth >= lo) & (depth < hi)
        n = int(m.sum())
        if n == 0: continue
        pix = float(vc[m].sum())
        mae_val = float((mae[m] * vc[m]).sum() / pix)
        contrib_val = float(contrib[m].sum())
        pct = 100 * contrib_val / total_mae_pix
        _p(f"  [{lo:>4d},{hi:>4d})m  {n:>10,d}  {pix:>12,.0f}  "
           f"{mae_val:>6.3f}  {contrib_val:>+13,.1f}  {pct:>6.2f}%")

    # By boundary status
    _p("\n-- 1c. MAE by boundary score (bound = 1 − max_frac) --")
    _p(f"  {'bucket':<26s}  {'n_tokens':>10s}  {'MAE':>7s}  "
       f"{'ΣMAE·pix':>14s}  {'%':>7s}")
    for name, lo, hi in [
        ("PURE       [0.00,0.05)", 0.00, 0.05),
        ("SLIGHT MIX [0.05,0.15)", 0.05, 0.15),
        ("MODERATE   [0.15,0.30)", 0.15, 0.30),
        ("HIGH MIX   [0.30,0.70)", 0.30, 0.70),
    ]:
        m = valid & (bound >= lo) & (bound < hi)
        n = int(m.sum())
        if n == 0: continue
        pix = float(vc[m].sum())
        mae_val = float((mae[m] * vc[m]).sum() / pix)
        contrib_val = float(contrib[m].sum())
        pct = 100 * contrib_val / total_mae_pix
        _p(f"  {name:<26s}  {n:>10,d}  {mae_val:>6.3f}  "
           f"{contrib_val:>+13,.1f}  {pct:>6.2f}%")

    # By routing outcome
    _p("\n-- 1d. MAE by routing outcome (top-1 match) --")
    _p(f"  {'type':<32s}  {'n_tokens':>10s}  {'MAE':>7s}  "
       f"{'ΣMAE·pix':>14s}  {'%':>7s}")
    for name, cond in [
        ("Correct (argmax match)", argmax_match),
        ("Adjacent misroute (|Δ|=1)", np.abs(argmax_p - argmax_f) == 1),
        ("Cross-bin misroute (|Δ|=2)", np.abs(argmax_p - argmax_f) == 2),
    ]:
        m = valid & cond
        n = int(m.sum())
        if n == 0: continue
        pix = float(vc[m].sum())
        mae_val = float((mae[m] * vc[m]).sum() / pix)
        contrib_val = float(contrib[m].sum())
        pct = 100 * contrib_val / total_mae_pix
        _p(f"  {name:<32s}  {n:>10,d}  {mae_val:>6.3f}  "
           f"{contrib_val:>+13,.1f}  {pct:>6.2f}%")

    # By row (image position — top=sky, bottom=road)
    _p("\n-- 1e. MAE by row band (image position, top=0 rows=hood/road) --")
    _p(f"  {'row bucket':<14s}  {'~pixel-y':<14s}  {'n_tokens':>10s}  "
       f"{'MAE':>7s}  {'ΣMAE·pix':>14s}  {'%':>7s}")
    H_img, W_img = int(image_hw[0]), int(image_hw[1])
    H_tok = int(grid_hw[0])
    row_edges = [0, 8, 13, 15, 17, 20, 25, 29]
    for lo, hi in zip(row_edges[:-1], row_edges[1:]):
        m = valid & (row >= lo) & (row < hi)
        n = int(m.sum())
        if n == 0: continue
        pix = float(vc[m].sum())
        mae_val = float((mae[m] * vc[m]).sum() / pix)
        contrib_val = float(contrib[m].sum())
        pct = 100 * contrib_val / total_mae_pix
        y0 = lo * H_img // H_tok; y1 = hi * H_img // H_tok
        tag = f"rows [{lo},{hi})"
        pytag = f"[y {y0}-{y1})"
        _p(f"  {tag:<14s}  {pytag:<14s}  {n:>10,d}  {mae_val:>6.3f}  "
           f"{contrib_val:>+13,.1f}  {pct:>6.2f}%")

    # ================================================================
    # PART 2 — Where does the router find it HARD?
    # ================================================================
    _p("\n" + "=" * 96)
    _p("PART 2 — WHERE THE ROUTER STRUGGLES (hard tokens)")
    _p("=" * 96)

    # Distribution of "hard" indicators
    _p("\n-- 2a. Router 'hard' indicators (over valid tokens) --")
    n_v = int(valid.sum())
    _p(f"  L1 (probs, frac):")
    for thr in [0.10, 0.30, 0.50, 1.00]:
        cnt = int((valid & (l1 >= thr)).sum())
        _p(f"    L1 ≥ {thr:.2f}: {cnt:>10,d}  ({100*cnt/n_v:>6.2f}%)")
    _p(f"  Argmax mismatch: {(valid & ~argmax_match).sum():>10,d}  "
       f"({100*(valid & ~argmax_match).sum()/n_v:>6.2f}%)")
    _p(f"  Top-2 pair mismatch: {(valid & ~top2_match).sum():>10,d}  "
       f"({100*(valid & ~top2_match).sum()/n_v:>6.2f}%)")
    _p(f"  Low confidence (max_prob < 0.70): {(valid & (max_p < 0.70)).sum():>10,d}  "
       f"({100*(valid & (max_p < 0.70)).sum()/n_v:>6.2f}%)")

    # By target boundary status (which target-shapes are hard)
    _p("\n-- 2b. Router hardness by TARGET boundary status --")
    _p(f"  {'bucket':<26s}  {'n':>10s}  {'L1 mean':>8s}  {'argmax %':>10s}  "
       f"{'top2 pair %':>12s}  {'ent_p':>7s}  {'ent_f':>7s}")
    for name, lo, hi in [
        ("PURE       [0.00,0.05)", 0.00, 0.05),
        ("SLIGHT MIX [0.05,0.15)", 0.05, 0.15),
        ("MODERATE   [0.15,0.30)", 0.15, 0.30),
        ("HIGH MIX   [0.30,0.70)", 0.30, 0.70),
    ]:
        m = valid & (bound >= lo) & (bound < hi)
        n = int(m.sum())
        if n == 0: continue
        l1_m = float(l1[m].mean())
        am_pct = float(100 * argmax_match[m].mean())
        t2_pct = float(100 * top2_match[m].mean())
        e_p = float(entropy[m].mean())
        # target entropy
        ent_f_arr = -(frac[m] * np.log(frac[m].clip(1e-12, 1.0))).sum(axis=1) / np.log(K)
        e_f = float(ent_f_arr.mean())
        _p(f"  {name:<26s}  {n:>10,d}  {l1_m:>7.4f}  {am_pct:>9.2f}%  "
           f"{t2_pct:>11.2f}%  {e_p:>6.3f}  {e_f:>6.3f}")

    # By depth range
    _p("\n-- 2c. Router hardness by depth range --")
    _p(f"  {'range':>14s}  {'n':>10s}  {'L1 mean':>8s}  {'argmax %':>10s}  "
       f"{'top2 pair %':>12s}  {'max_p mean':>11s}")
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = valid & (depth >= lo) & (depth < hi)
        n = int(m.sum())
        if n == 0: continue
        l1_m = float(l1[m].mean())
        am_pct = float(100 * argmax_match[m].mean())
        t2_pct = float(100 * top2_match[m].mean())
        mp_m = float(max_p[m].mean())
        _p(f"  [{lo:>4d},{hi:>4d})m  {n:>10,d}  {l1_m:>7.4f}  {am_pct:>9.2f}%  "
           f"{t2_pct:>11.2f}%  {mp_m:>10.4f}")

    # ================================================================
    # PART 3 — INTERSECTION: are hard tokens the same as costly tokens?
    # ================================================================
    _p("\n" + "=" * 96)
    _p("PART 3 — INTERSECTION: HARD × COSTLY")
    _p("=" * 96)

    _p("\n-- 3a. Rank correlations (over valid tokens) --")
    v_l1 = l1[valid]; v_mae = mae[valid]
    v_bound = bound[valid]
    v_maxp = max_p[valid]
    # sample subset for speed
    if v_l1.size > 500_000:
        sel = np.random.RandomState(0).choice(v_l1.size, 500_000, replace=False)
        v_l1_s = v_l1[sel]; v_mae_s = v_mae[sel]
        v_bound_s = v_bound[sel]; v_maxp_s = v_maxp[sel]
    else:
        v_l1_s, v_mae_s, v_bound_s, v_maxp_s = v_l1, v_mae, v_bound, v_maxp
    def _corr(x, y):
        return float(np.corrcoef(x, y)[0, 1])
    _p(f"  Pearson r(MAE, L1):     {_corr(v_mae_s, v_l1_s):.4f}")
    _p(f"  Pearson r(MAE, bound):  {_corr(v_mae_s, v_bound_s):.4f}")
    _p(f"  Pearson r(MAE, -max_p): {_corr(v_mae_s, -v_maxp_s):.4f}")

    _p("\n-- 3b. MAE by L1 bucket (hard by distribution mismatch → costly?) --")
    _p(f"  {'L1 bucket':<18s}  {'n':>10s}  {'MAE':>7s}  {'ΣMAE·pix':>14s}  {'%':>7s}")
    for lo, hi in [(0.00, 0.10), (0.10, 0.30), (0.30, 0.60), (0.60, 1.00), (1.00, 2.01)]:
        m = valid & (l1 >= lo) & (l1 < hi)
        n = int(m.sum())
        if n == 0: continue
        pix = float(vc[m].sum())
        mae_val = float((mae[m] * vc[m]).sum() / pix)
        contrib_val = float(contrib[m].sum())
        pct = 100 * contrib_val / total_mae_pix
        _p(f"  L1 [{lo:.2f},{hi:.2f})  {n:>10,d}  {mae_val:>6.3f}  "
           f"{contrib_val:>+13,.1f}  {pct:>6.2f}%")

    _p("\n-- 3c. MAE by max_prob (router confidence) --")
    _p(f"  {'confidence':<28s}  {'n':>10s}  {'MAE':>7s}  {'ΣMAE·pix':>14s}  {'%':>7s}")
    for name, lo, hi in [
        ("SATURATED     [1.0]      ", 0.99999, 1.001),
        ("near-sat      [0.95,1.0) ", 0.95, 0.99999),
        ("confident     [0.70,0.95)", 0.70, 0.95),
        ("uncertain     [0.34,0.70)", 0.0, 0.70),
    ]:
        m = valid & (max_p >= lo) & (max_p < hi)
        n = int(m.sum())
        if n == 0: continue
        pix = float(vc[m].sum())
        mae_val = float((mae[m] * vc[m]).sum() / pix)
        contrib_val = float(contrib[m].sum())
        pct = 100 * contrib_val / total_mae_pix
        _p(f"  {name:<28s}  {n:>10,d}  {mae_val:>6.3f}  "
           f"{contrib_val:>+13,.1f}  {pct:>6.2f}%")

    _p("\n-- 3d. 2D join: L1 × routing outcome (hard AND misroute)? --")
    _p(f"  {'L1 bucket':<12s}  {'routing':<32s}  {'n':>10s}  "
       f"{'MAE':>7s}  {'ΣMAE·pix':>14s}  {'%':>7s}")
    for name_l1, lo_l1, hi_l1 in [
        ("L1<0.1",  0.00, 0.10),
        ("L1<0.5",  0.10, 0.50),
        ("L1<1.0",  0.50, 1.00),
        ("L1>=1.0", 1.00, 2.01),
    ]:
        for name_r, cond in [
            ("Correct",            argmax_match),
            ("Adjacent misroute",  np.abs(argmax_p - argmax_f) == 1),
            ("Cross-bin misroute", np.abs(argmax_p - argmax_f) == 2),
        ]:
            m = valid & (l1 >= lo_l1) & (l1 < hi_l1) & cond
            n = int(m.sum())
            if n == 0: continue
            pix = float(vc[m].sum())
            mae_val = float((mae[m] * vc[m]).sum() / pix)
            contrib_val = float(contrib[m].sum())
            pct = 100 * contrib_val / total_mae_pix
            _p(f"  {name_l1:<12s}  {name_r:<32s}  {n:>10,d}  {mae_val:>6.3f}  "
               f"{contrib_val:>+13,.1f}  {pct:>6.2f}%")

    _p("\n-- 3e. Are the top-100 highest-MAE tokens ALSO hard? --")
    top_mae_idx = order[:100]
    _p(f"  Among the 100 costliest tokens (by mae·pix):")
    _p(f"    mean MAE:              {mae[top_mae_idx].mean():>7.3f} m")
    _p(f"    mean L1:               {l1[top_mae_idx].mean():>7.3f}")
    _p(f"    mean max_prob:         {max_p[top_mae_idx].mean():>7.3f}")
    _p(f"    mean bound (=1-maxf):  {bound[top_mae_idx].mean():>7.3f}")
    _p(f"    argmax match:          {100*argmax_match[top_mae_idx].mean():>6.2f}%")
    _p(f"    top-2 pair match:      {100*top2_match[top_mae_idx].mean():>6.2f}%")
    _p(f"    mean depth:            {depth[top_mae_idx].mean():>7.2f} m")
    # Depth distribution of top tokens
    bin_counts = np.bincount(np.digitize(depth[top_mae_idx],
                                          [10, 20, 30, 50, 80, 100]),
                             minlength=7)
    _p(f"    depth histogram (bins <10, 10-20, 20-30, 30-50, 50-80, 80-100, >100): "
       f"{bin_counts.tolist()}")

    # ================================================================
    # PART 4 — WORST samples (per-sample MAE)
    # ================================================================
    _p("\n" + "=" * 96)
    _p("PART 4 — WORST SAMPLES (per-sample MAE contribution)")
    _p("=" * 96)

    _p("\n-- 4a. Per-sample pixel-weighted MAE distribution --")
    n_samples = len(sample_ids)
    sample_mae_num = np.zeros(n_samples, dtype=np.float64)
    sample_pix = np.zeros(n_samples, dtype=np.float64)
    v_mae_pix = np.where(valid, mae_pix, 0.0)
    v_pix = np.where(valid, vc, 0.0)
    np.add.at(sample_mae_num, sample_idx, v_mae_pix)
    np.add.at(sample_pix, sample_idx, v_pix)
    sample_mae = sample_mae_num / np.maximum(sample_pix, 1e-9)
    _p(f"  n samples: {n_samples:,}")
    _p(f"  mean sample MAE: {sample_mae.mean():.4f}  "
       f"median: {np.median(sample_mae):.4f}  "
       f"p95: {np.percentile(sample_mae, 95):.4f}  "
       f"p99: {np.percentile(sample_mae, 99):.4f}  "
       f"max: {sample_mae.max():.4f}")

    _p("\n-- 4b. Top-10 worst samples (by mean MAE per sample) --")
    _p(f"  {'sample_id':<80s}  {'sample MAE':>12s}  {'n_valid_pix':>12s}")
    worst = np.argsort(-sample_mae)[:10]
    for s in worst:
        sid = str(sample_ids[s])[:78]
        _p(f"  {sid:<80s}  {sample_mae[s]:>11.3f}  {sample_pix[s]:>12,.0f}")

    _p("\n-- 4c. Contribution concentration --")
    contrib_sample = sample_mae * sample_pix
    total = contrib_sample.sum()
    order_s = np.argsort(-contrib_sample)
    for topN in (10, 100, 500):
        share = 100 * contrib_sample[order_s[:topN]].sum() / total
        _p(f"  Top {topN:>4d} samples ({100*topN/n_samples:.2f}% of samples) "
           f"contain {share:>5.2f}% of total MAE·pix")

    out = "output/analysis/router_cost/hard_and_costly_v4c.txt"
    with open(out, "w") as f:
        f.write(buf.getvalue())
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
