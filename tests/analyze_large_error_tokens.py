"""Deep dive on tokens with catastrophic error (≥10m) — what are they?

Compares distribution of characteristics between:
  ALL valid tokens (baseline)
  Large-error tokens (err_normal ≥ 10m)
  Extreme-error tokens (err_normal ≥ 20m)

Dimensions:
  A. Router behavior (max_prob, L1, top-2 pair match)
  B. Target characteristics (bound, target shape, depth, entropy)
  C. Position (row, col in router grid → image y, x)
  D. Routing outcome (pred vs gt, misroute type)
  E. Would OracleSoft still fail?
  F. Sample-level clustering (which scenes dominate?)
  G. Top-20 worst tokens with sample IDs + regions for visual inspection

Reads:
  output/analysis/router_cost/v4c_stage2_per_mode.npz
  output/analysis/router_cost/v4c_stage2_per_token_dump.npz
"""
import io
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, ROOT)


def main():
    perm = np.load("output/analysis/router_cost/v4c_stage2_per_mode.npz",
                   allow_pickle=True)
    dump = np.load("output/analysis/router_cost/v4c_stage2_per_token_dump.npz",
                   allow_pickle=True)
    assert perm["err_normal"].shape == dump["mae"].shape

    err_n = perm["err_normal"]
    err_h = perm["err_hard"]
    err_s = perm["err_soft"]
    vc = perm["valid_pixel_count"]
    gt = perm["gt"]
    pred = perm["actual_pred"]
    bound = perm["bound"]

    probs = dump["probs"]
    frac = dump["frac"]
    depth = dump["depth"]
    sample_idx = dump["sample_idx"]
    row = dump["token_row"]
    col = dump["token_col"]
    sample_ids = dump["sample_ids"]
    grid_hw = dump["grid_hw"]
    image_hw = dump["image_hw"]

    max_p = probs.max(axis=1)
    max_f = frac.max(axis=1)
    l1 = np.abs(probs - frac).sum(axis=1)
    entropy_f = -(frac * np.log(frac.clip(1e-12, 1.0))).sum(axis=1) / np.log(3)

    def top2(a):
        idx = np.argsort(-a, axis=1)[:, :2]
        return np.sort(idx, axis=1)
    top2_p = top2(probs); top2_f = top2(frac)
    top2_match = (top2_p[:, 0] == top2_f[:, 0]) & (top2_p[:, 1] == top2_f[:, 1])

    valid = np.isfinite(err_n) & np.isfinite(err_s) & (vc > 0)
    n_v = int(valid.sum())

    # Three cohorts
    all_mask = valid
    large_mask = valid & (err_n >= 10.0)
    extreme_mask = valid & (err_n >= 20.0)

    buf = io.StringIO()
    def _p(msg=""):
        print(msg); buf.write(msg + "\n")

    _p("=" * 96)
    _p("LARGE-ERROR TOKEN ANALYSIS — v4c stage2")
    _p(f"  All valid:       {int(all_mask.sum()):>10,d}")
    _p(f"  Large err (≥10m):{int(large_mask.sum()):>10,d}  "
       f"({100*large_mask.sum()/n_v:.2f}%)")
    _p(f"  Extreme (≥20m):  {int(extreme_mask.sum()):>10,d}  "
       f"({100*extreme_mask.sum()/n_v:.2f}%)")
    _p("=" * 96)

    def _profile(cohort_mask, label):
        m = cohort_mask
        n = int(m.sum())
        if n == 0: return
        pix = float(vc[m].sum())
        _p(f"\n--- {label}  n={n:,}  pixels={pix:,.0f} ---")
        _p(f"  Mean MAE (Normal):     {float(err_n[m].mean()):>7.3f}")
        _p(f"  Mean MAE (OracleHard): {float(err_h[m].mean()):>7.3f}")
        _p(f"  Mean MAE (OracleSoft): {float(err_s[m].mean()):>7.3f}")
        _p(f"  Recoverable per-token: {float((err_n[m]-err_s[m]).mean()):>+7.3f}  "
           f"(how much OracleSoft would save)")
        _p(f"  Mean depth:            {float(depth[m].mean()):>7.2f}")
        _p(f"  Mean bound:            {float(bound[m].mean()):>7.4f}")
        _p(f"  Mean max_prob:         {float(max_p[m].mean()):>7.4f}")
        _p(f"  Mean max_frac:         {float(max_f[m].mean()):>7.4f}")
        _p(f"  Mean L1(probs, frac):  {float(l1[m].mean()):>7.4f}")
        _p(f"  Mean entropy(frac):    {float(entropy_f[m].mean()):>7.4f}")
        _p(f"  Argmax match:          {100*float((pred[m]==gt[m]).mean()):>6.2f}%")
        _p(f"  Top-2 pair match:      {100*float(top2_match[m].mean()):>6.2f}%")

    _profile(all_mask, "ALL VALID")
    _profile(large_mask, "LARGE ERROR (≥10m)")
    _profile(extreme_mask, "EXTREME (≥20m)")

    # =========================================================
    # Detailed distributions for large-error
    # =========================================================
    _p("\n" + "=" * 96)
    _p("DETAILED DISTRIBUTIONS — LARGE ERROR (≥10m) tokens")
    _p("=" * 96)

    m = large_mask
    _p(f"\n-- Depth range --")
    for lo, hi in [(0, 10), (10, 15), (15, 20), (20, 25), (25, 30),
                    (30, 40), (40, 50), (50, 60), (60, 80), (80, 100)]:
        mm = m & (depth >= lo) & (depth < hi)
        n = int(mm.sum())
        if n == 0: continue
        pct = 100 * n / m.sum()
        mae_mean = float(err_n[mm].mean())
        _p(f"  [{lo:>3d},{hi:>3d})m: {n:>7,d} ({pct:>5.2f}%)  mean_MAE={mae_mean:.1f}m")

    _p(f"\n-- Row band (image position) --")
    H_img = int(image_hw[0]); H_tok = int(grid_hw[0])
    for lo, hi in [(0, 8), (8, 13), (13, 15), (15, 17), (17, 20), (20, 25), (25, 29)]:
        mm = m & (row >= lo) & (row < hi)
        n = int(mm.sum())
        if n == 0: continue
        pct = 100 * n / m.sum()
        mae_mean = float(err_n[mm].mean())
        y0 = lo * H_img // H_tok; y1 = hi * H_img // H_tok
        _p(f"  rows [{lo:>2d},{hi:>2d}) y=[{y0:>3d},{y1:>3d}): "
           f"{n:>7,d} ({pct:>5.2f}%)  mean_MAE={mae_mean:.1f}m")

    _p(f"\n-- Bound bucket --")
    for name, lo, hi in [("PURE      [0.00,0.05)", 0.00, 0.05),
                          ("SLIGHT    [0.05,0.15)", 0.05, 0.15),
                          ("MODERATE  [0.15,0.30)", 0.15, 0.30),
                          ("HIGH      [0.30,0.70)", 0.30, 0.70)]:
        mm = m & (bound >= lo) & (bound < hi)
        n = int(mm.sum())
        if n == 0: continue
        pct = 100 * n / m.sum()
        mae_mean = float(err_n[mm].mean())
        _p(f"  {name:<24s}: {n:>7,d} ({pct:>5.2f}%)  mean_MAE={mae_mean:.1f}m")

    _p(f"\n-- Confidence bucket --")
    for name, lo, hi in [("SATURATED  [1.00]     ", 0.9999, 1.001),
                          ("near-sat   [0.95,1.0)", 0.95, 0.9999),
                          ("confident  [0.70,0.95)", 0.70, 0.95),
                          ("uncertain  [0.34,0.70)", 0.0, 0.70)]:
        mm = m & (max_p >= lo) & (max_p < hi)
        n = int(mm.sum())
        if n == 0: continue
        pct = 100 * n / m.sum()
        mae_mean = float(err_n[mm].mean())
        _p(f"  {name:<26s}: {n:>7,d} ({pct:>5.2f}%)  mean_MAE={mae_mean:.1f}m")

    _p(f"\n-- Routing outcome --")
    diff = np.abs(pred.astype(np.int64) - gt.astype(np.int64))
    for name, cond in [("Correct (argmax match)", pred == gt),
                        ("Adjacent misroute", diff == 1),
                        ("Cross-bin misroute", diff == 2)]:
        mm = m & cond
        n = int(mm.sum())
        pct = 100 * n / m.sum()
        if n == 0: continue
        mae_mean = float(err_n[mm].mean())
        _p(f"  {name:<26s}: {n:>7,d} ({pct:>5.2f}%)  mean_MAE={mae_mean:.1f}m")

    _p(f"\n-- (target argmax → pred argmax) confusion --")
    labels = ["near", "mid", "far"]
    for tk in range(3):
        for pk in range(3):
            mm = m & (gt == tk) & (pred == pk)
            n = int(mm.sum())
            if n == 0: continue
            pct = 100 * n / m.sum()
            mae_mean = float(err_n[mm].mean())
            oracle_soft = float(err_s[mm].mean())
            tag = f"{labels[tk]:<4s} → {labels[pk]:<4s}"
            _p(f"  {tag:<15s}: {n:>7,d} ({pct:>5.2f}%)  "
               f"mean_MAE={mae_mean:>5.1f}  soft_MAE={oracle_soft:>5.1f}")

    # Would OracleSoft still fail?
    _p(f"\n-- Would OracleSoft fix these? --")
    for oracle_thr in [1.0, 2.0, 5.0, 10.0]:
        still_wrong = m & (err_s >= oracle_thr)
        n = int(still_wrong.sum())
        pct = 100 * n / m.sum()
        _p(f"  OracleSoft would STILL be ≥{oracle_thr:>4.0f}m: {n:>7,d} ({pct:>5.2f}%)")
    fixed = m & (err_s < 1.0)
    _p(f"  OracleSoft would fix to <1m:             {int(fixed.sum()):>7,d} "
       f"({100*fixed.sum()/m.sum():>5.2f}%)")

    # =========================================================
    # Sample-level clustering
    # =========================================================
    _p("\n" + "=" * 96)
    _p("SAMPLE-LEVEL CLUSTERING")
    _p("=" * 96)
    per_sample = np.bincount(sample_idx[m], minlength=len(sample_ids))
    _p(f"\n  Total samples: {len(sample_ids):,}")
    _p(f"  Samples with any large-err token: "
       f"{int((per_sample > 0).sum()):,} "
       f"({100*(per_sample>0).sum()/len(sample_ids):.1f}%)")
    _p(f"  Mean large-err tokens per sample: {per_sample.mean():.2f}")
    _p(f"  Max in a single sample: {per_sample.max():,}")

    # Top 10 samples by large-error count
    order = np.argsort(-per_sample)
    _p(f"\n  Top 10 samples by large-err token count:")
    for i in order[:10]:
        if per_sample[i] == 0: break
        _p(f"    {str(sample_ids[i])[:78]:<80s}  {int(per_sample[i]):>5d} tokens")

    # =========================================================
    # Top-20 worst individual tokens with region info
    # =========================================================
    _p("\n" + "=" * 96)
    _p("TOP-20 WORST INDIVIDUAL TOKENS (for visual inspection)")
    _p("=" * 96)
    W_img = int(image_hw[1]); W_tok = int(grid_hw[1])
    err_n_ranked = np.where(valid, err_n, -np.inf)
    top = np.argsort(-err_n_ranked)[:20]
    _p(f"\n  {'#':<3s}  {'sample_id':<60s}  {'row,col':<8s}  "
       f"{'y_range':<12s}  {'x_range':<12s}  "
       f"{'depth':>6s}  {'MAE_N':>7s}  {'MAE_S':>7s}  {'gt':<5s}  {'pred':<5s}  {'max_p':>6s}")
    for rank, i in enumerate(top):
        s_id = str(sample_ids[sample_idx[i]])[:58]
        r, c = int(row[i]), int(col[i])
        y0 = r * H_img // H_tok; y1 = (r+1) * H_img // H_tok
        x0 = c * W_img // W_tok; x1 = (c+1) * W_img // W_tok
        _p(f"  {rank+1:<3d}  {s_id:<60s}  {r},{c:<5d}  "
           f"[{y0:>3d},{y1:<3d})  [{x0:>4d},{x1:<4d})  "
           f"{depth[i]:>5.1f}m  {err_n[i]:>6.1f}m  {err_s[i]:>6.1f}m  "
           f"{['near','mid','far'][gt[i]]:<5s}  {['near','mid','far'][pred[i]]:<5s}  "
           f"{max_p[i]:>6.3f}")

    out = "output/analysis/router_cost/large_error_tokens_v4c.txt"
    with open(out, "w") as f:
        f.write(buf.getvalue())
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
