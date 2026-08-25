"""Redo the hard × costly analysis using ROUTABLE error, not raw MAE.

Routable error = err_normal − err_soft (per token, pixel-weighted).
  Positive: Normal WORSE than OracleSoft → routing/mixing can be improved
  Negative: Normal BETTER than OracleSoft → nothing to fix here (bonus)

We ignore the raw MAE that Oracle already suffers from (=feature/expert
limit), and focus on what BETTER ROUTING could recover.

Reads:
  output/analysis/router_cost/v4c_stage2_per_mode.npz     (per-mode errors)
  output/analysis/router_cost/v4c_stage2_per_token_dump.npz (probs, positions)

Writes:
  output/analysis/router_cost/routable_error_v4c.txt
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
    dump = np.load(
        "output/analysis/router_cost/v4c_stage2_per_token_dump.npz",
        allow_pickle=True)
    perm = np.load(
        "output/analysis/router_cost/v4c_stage2_per_mode.npz",
        allow_pickle=True)

    # Verify same order/length
    assert dump['mae'].shape == perm['err_normal'].shape, \
        (dump['mae'].shape, perm['err_normal'].shape)

    # Per-token
    err_n = perm['err_normal']
    err_h = perm['err_hard']
    err_s = perm['err_soft']
    vc = perm['valid_pixel_count']
    gt_b = perm['gt']
    pred_b = perm['actual_pred']

    # From dump
    probs = dump['probs']
    frac = dump['frac']
    depth = dump['depth']
    sample_idx = dump['sample_idx']
    row = dump['token_row']
    col = dump['token_col']
    argmax_p = dump['argmax_p'].astype(np.int64)
    argmax_f = dump['argmax_f'].astype(np.int64)
    grid_hw = dump['grid_hw']; image_hw = dump['image_hw']
    sample_ids = dump['sample_ids']

    N = err_n.shape[0]
    valid = np.isfinite(err_n) & np.isfinite(err_s) & np.isfinite(err_h) & (vc > 0)
    n_v = int(valid.sum())

    # Gaps
    gap_hard = err_n - err_h          # Normal − OracleHard
    gap_soft = err_n - err_s          # Normal − OracleSoft (routable ceiling)

    # Pixel-weighted totals
    pix = float(vc[valid].sum())
    gap_h_total = float(((gap_hard * vc)[valid]).sum())
    gap_s_total = float(((gap_soft * vc)[valid]).sum())
    mae_n = float(((err_n * vc)[valid]).sum() / pix)
    mae_h = float(((err_h * vc)[valid]).sum() / pix)
    mae_s = float(((err_s * vc)[valid]).sum() / pix)

    max_p = probs.max(axis=1)
    max_f = frac.max(axis=1)
    l1 = np.abs(probs - frac).sum(axis=1)
    bound = 1.0 - max_f

    def top2(a):
        idx = np.argsort(-a, axis=1)[:, :2]
        return np.sort(idx, axis=1)
    top2_p = top2(probs); top2_f = top2(frac)
    top2_match = (top2_p[:, 0] == top2_f[:, 0]) & (top2_p[:, 1] == top2_f[:, 1])
    argmax_match = (argmax_p == argmax_f)

    buf = io.StringIO()
    def _p(msg=""):
        print(msg); buf.write(msg + "\n")

    _p("=" * 96)
    _p("ROUTABLE ERROR ANALYSIS — v4c stage2")
    _p(f"  valid tokens: {n_v:,}   pixels: {pix:,.0f}")
    _p(f"  Pixel-weighted MAE:")
    _p(f"    Normal      = {mae_n:.4f}  m")
    _p(f"    OracleHard  = {mae_h:.4f}  m  (gap to Normal: {mae_n - mae_h:+.4f})")
    _p(f"    OracleSoft  = {mae_s:.4f}  m  (gap to Normal: {mae_n - mae_s:+.4f})")
    _p(f"  Total routable gap:")
    _p(f"    Σ(N − H)·pix = {gap_h_total:+,.1f}  → {gap_h_total/pix:+.4f} m/pix")
    _p(f"    Σ(N − S)·pix = {gap_s_total:+,.1f}  → {gap_s_total/pix:+.4f} m/pix")
    _p(f"  Note: negative-gap tokens = Normal already ≥ Oracle (bonus).")
    _p("=" * 96)

    # -------- Positive vs negative gap accounting --------
    _p("\n-- 0. Sign distribution of routable gap (Normal − OracleSoft) --")
    for name, cond in [
        ("Normal WORSE than OracleSoft (gap>0)", valid & (gap_soft > 0)),
        ("Normal ≈ OracleSoft (|gap|<1e-6)", valid & (np.abs(gap_soft) < 1e-6)),
        ("Normal BETTER than OracleSoft (gap<0)", valid & (gap_soft < 0)),
    ]:
        n = int(cond.sum())
        pix_b = float(vc[cond].sum())
        gap_b = float((gap_soft[cond] * vc[cond]).sum())
        _p(f"  {name:<44s}  n={n:>10,d}  pix={pix_b:>12,.0f}  Σgap·pix={gap_b:+13,.1f}")

    # From here on, focus on tokens with POSITIVE gap (recoverable error).
    pos = valid & (gap_soft > 0)
    pos_pix = float(vc[pos].sum())
    pos_gap = float((gap_soft[pos] * vc[pos]).sum())
    _p(f"\n  → Restricting downstream analysis to POSITIVE-gap tokens:")
    _p(f"    n={int(pos.sum()):,}  pixels={pos_pix:,.0f}  total recoverable Σgap·pix={pos_gap:+.1f}")
    _p(f"    Recoverable per-pixel MAE reduction if fully realized: "
       f"{pos_gap/pix:+.4f} m")

    def _table(header, buckets, name_fmt, weight_source):
        """Print a table of gap contribution by bucket."""
        _p(f"\n{header}")
        _p(f"  {'bucket':<32s}  {'n':>10s}  {'MAE_N':>7s}  {'MAE_S':>7s}  "
           f"{'gap':>7s}  {'Σgap·pix':>14s}  {'%':>7s}")
        for name, mask in buckets:
            m = pos & mask
            n = int(m.sum())
            if n == 0: continue
            p_ = float(vc[m].sum())
            mae_n_b = float((err_n[m] * vc[m]).sum() / p_)
            mae_s_b = float((err_s[m] * vc[m]).sum() / p_)
            gs = float((gap_soft[m] * vc[m]).sum())
            pct = 100 * gs / pos_gap
            _p(f"  {name:<32s}  {n:>10,d}  {mae_n_b:>6.3f}  {mae_s_b:>6.3f}  "
               f"{mae_n_b-mae_s_b:>+6.3f}  {gs:>+13,.1f}  {pct:>6.2f}%")

    # -------- 1. Depth range --------
    edges = [0, 10, 15, 20, 22, 25, 30, 40, 45, 50, 55, 60, 80, 100]
    depth_buckets = [(f"[{lo:>3d},{hi:>3d})m", (depth >= lo) & (depth < hi))
                     for lo, hi in zip(edges[:-1], edges[1:])]
    _table("-- 1. Recoverable gap by depth range --", depth_buckets,
           None, None)

    # -------- 2. Boundary --------
    bound_buckets = [
        ("PURE       [0.00,0.05)", (bound >= 0.00) & (bound < 0.05)),
        ("SLIGHT MIX [0.05,0.15)", (bound >= 0.05) & (bound < 0.15)),
        ("MODERATE   [0.15,0.30)", (bound >= 0.15) & (bound < 0.30)),
        ("HIGH MIX   [0.30,0.70)", (bound >= 0.30) & (bound < 0.70)),
    ]
    _table("-- 2. Recoverable gap by target boundary --", bound_buckets,
           None, None)

    # -------- 3. Routing outcome --------
    route_buckets = [
        ("Correct (argmax match)", argmax_match),
        ("Adjacent misroute (|Δ|=1)", np.abs(argmax_p - argmax_f) == 1),
        ("Cross-bin misroute (|Δ|=2)", np.abs(argmax_p - argmax_f) == 2),
    ]
    _table("-- 3. Recoverable gap by routing outcome --", route_buckets,
           None, None)

    # -------- 4. Confidence --------
    conf_buckets = [
        ("SATURATED    [1.0]",     max_p >= 0.99999),
        ("near-sat    [0.95,1.0)", (max_p >= 0.95) & (max_p < 0.99999)),
        ("confident   [0.70,0.95)", (max_p >= 0.70) & (max_p < 0.95)),
        ("uncertain   [0.34,0.70)", max_p < 0.70),
    ]
    _table("-- 4. Recoverable gap by router confidence --", conf_buckets,
           None, None)

    # -------- 5. L1 bucket --------
    l1_buckets = [
        ("L1 [0.00,0.10)", (l1 >= 0.00) & (l1 < 0.10)),
        ("L1 [0.10,0.30)", (l1 >= 0.10) & (l1 < 0.30)),
        ("L1 [0.30,0.60)", (l1 >= 0.30) & (l1 < 0.60)),
        ("L1 [0.60,1.00)", (l1 >= 0.60) & (l1 < 1.00)),
        ("L1 [1.00,2.01)", (l1 >= 1.00) & (l1 < 2.01)),
    ]
    _table("-- 5. Recoverable gap by L1(probs, frac) --", l1_buckets,
           None, None)

    # -------- 6. Top-2 pair match --------
    _table("-- 6. Recoverable gap by top-2 pair match --", [
        ("Top-2 pair match", top2_match),
        ("Top-2 pair mismatch", ~top2_match),
    ], None, None)

    # -------- 7. Row band --------
    H_img, W_img = int(image_hw[0]), int(image_hw[1])
    H_tok = int(grid_hw[0])
    row_edges = [0, 8, 13, 15, 17, 20, 25, 29]
    row_buckets = []
    for lo, hi in zip(row_edges[:-1], row_edges[1:]):
        y0 = lo * H_img // H_tok; y1 = hi * H_img // H_tok
        row_buckets.append((f"rows [{lo},{hi}) y=[{y0},{y1})", (row >= lo) & (row < hi)))
    _table("-- 7. Recoverable gap by row band --", row_buckets, None, None)

    # -------- 8. 2D join: L1 × routing outcome --------
    _p("\n-- 8. L1 × routing outcome (2D) --")
    _p(f"  {'L1 bucket':<14s}  {'routing':<28s}  {'n':>10s}  "
       f"{'gap':>7s}  {'Σgap·pix':>14s}  {'%':>7s}")
    l1_bins = [
        ("L1<0.1",   (l1 >= 0.00) & (l1 < 0.10)),
        ("L1<0.5",   (l1 >= 0.10) & (l1 < 0.50)),
        ("L1<1.0",   (l1 >= 0.50) & (l1 < 1.00)),
        ("L1>=1.0",  l1 >= 1.00),
    ]
    for l1_name, l1_cond in l1_bins:
        for r_name, r_cond in route_buckets:
            m = pos & l1_cond & r_cond
            n = int(m.sum())
            if n == 0: continue
            p_ = float(vc[m].sum())
            mae_n_b = float((err_n[m] * vc[m]).sum() / p_)
            mae_s_b = float((err_s[m] * vc[m]).sum() / p_)
            gs = float((gap_soft[m] * vc[m]).sum())
            pct = 100 * gs / pos_gap
            _p(f"  {l1_name:<14s}  {r_name:<28s}  {n:>10,d}  "
               f"{mae_n_b-mae_s_b:>+6.3f}  {gs:>+13,.1f}  {pct:>6.2f}%")

    # -------- 9. Correlation between gap and hardness --------
    _p("\n-- 9. Correlation between routable gap and hardness (positive gap only) --")
    sel = np.where(pos)[0]
    if sel.size > 500_000:
        sel = np.random.RandomState(0).choice(sel, 500_000, replace=False)
    def _corr(x, y):
        return float(np.corrcoef(x, y)[0, 1])
    _p(f"  Pearson r(gap_soft, L1):       {_corr(gap_soft[sel], l1[sel]):.4f}")
    _p(f"  Pearson r(gap_soft, bound):    {_corr(gap_soft[sel], bound[sel]):.4f}")
    _p(f"  Pearson r(gap_soft, -max_p):   {_corr(gap_soft[sel], -max_p[sel]):.4f}")

    # -------- 10. Concentration of gap in top-N tokens --------
    _p("\n-- 10. Concentration of routable gap in top-N tokens --")
    contrib = np.where(pos, gap_soft * vc, 0.0)
    order = np.argsort(-contrib)
    for topN in (100, 1000, 10000, 100000, 500000):
        share = 100 * contrib[order[:topN]].sum() / pos_gap
        _p(f"  top {topN:>7,d} tokens contain {share:>6.2f}% of positive gap "
           f"({100*topN/n_v:.2f}% of valid tokens)")

    # -------- 11. Characterize the top-100 gap tokens --------
    _p("\n-- 11. Top-100 routable-gap tokens: profile --")
    top = order[:100]
    _p(f"  Mean gap:         {gap_soft[top].mean():>7.3f} m")
    _p(f"  Mean MAE Normal:  {err_n[top].mean():>7.3f} m")
    _p(f"  Mean MAE Soft:    {err_s[top].mean():>7.3f} m")
    _p(f"  Mean L1:          {l1[top].mean():>7.3f}")
    _p(f"  Mean max_prob:    {max_p[top].mean():>7.3f}")
    _p(f"  Mean bound:       {bound[top].mean():>7.3f}")
    _p(f"  Mean depth:       {depth[top].mean():>7.2f} m")
    _p(f"  argmax match:     {100*argmax_match[top].mean():>6.2f}%")
    _p(f"  top-2 pair match: {100*top2_match[top].mean():>6.2f}%")
    depth_hist = np.bincount(np.digitize(depth[top], [10, 20, 30, 50, 80, 100]),
                              minlength=7).tolist()
    _p(f"  depth histogram (<10, 10-20, 20-30, 30-50, 50-80, 80-100, >100): {depth_hist}")

    out = "output/analysis/router_cost/routable_error_v4c.txt"
    with open(out, "w") as f:
        f.write(buf.getvalue())
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
