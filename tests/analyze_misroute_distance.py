"""Analyze MISROUTED tokens by how BADLY they're misrouted.

"Misroute distance" = how far the token's depth is from the assigned
expert's bin range.
  Expert bins: near [0, 20), mid [20, 50), far [50, 100).
  Example: token at 21m routed to near expert → distance = 21 - 20 = 1m
           (aptly wrong — depth just 1m past the boundary)
  Example: token at 50m routed to near expert → distance = 50 - 20 = 30m
           (badly wrong — depth 30m past the boundary)

For misrouted tokens only (pred != gt), analyze:
  1. Distribution of misroute distances
  2. Correlation with resulting depth MAE
  3. Characterize tokens with SMALL misroute distance vs LARGE
  4. Where do the "badly misrouted" tokens live?
"""
import io
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, ROOT)


BIN_RANGES = [(0.0, 20.0), (20.0, 50.0), (50.0, 100.0)]  # near, mid, far


def misroute_distance(depth: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """For each token, distance from depth to the assigned expert's range.
    0 if depth is inside the assigned bin's range."""
    lo = np.array([r[0] for r in BIN_RANGES])[pred]
    hi = np.array([r[1] for r in BIN_RANGES])[pred]
    return np.maximum(0.0, np.maximum(lo - depth, depth - hi))


def main():
    perm = np.load("output/analysis/router_cost/v4c_stage2_per_mode.npz",
                   allow_pickle=True)
    dump = np.load("output/analysis/router_cost/v4c_stage2_per_token_dump.npz",
                   allow_pickle=True)

    err_n = perm["err_normal"]
    err_h = perm["err_hard"]
    err_s = perm["err_soft"]
    vc = perm["valid_pixel_count"]
    gt = perm["gt"].astype(np.int64)
    pred = perm["actual_pred"].astype(np.int64)
    bound = perm["bound"]

    probs = dump["probs"]
    frac = dump["frac"]
    depth = dump["depth"]
    sample_idx = dump["sample_idx"]
    row = dump["token_row"]
    col = dump["token_col"]
    sample_ids = dump["sample_ids"]
    grid_hw = dump["grid_hw"]; image_hw = dump["image_hw"]

    max_p = probs.max(axis=1)
    l1 = np.abs(probs - frac).sum(axis=1)

    valid = np.isfinite(err_n) & np.isfinite(err_s) & (vc > 0)
    misroute = valid & (pred != gt)

    md = misroute_distance(depth, pred)                       # misroute distance
    # For tokens where pred == gt: distance = 0 (mostly), or small (if
    # token depth falls slightly outside assigned bin due to majority-vote
    # rounding). We ignore those here since focus is misrouted.

    n_v = int(valid.sum())
    n_mis = int(misroute.sum())

    total_normal_mae_pix = float((err_n[valid] * vc[valid]).sum())
    total_recov_pix = float(
        np.where((err_n - err_s) > 0, (err_n - err_s) * vc, 0)[valid].sum())

    buf = io.StringIO()
    def _p(msg=""):
        print(msg); buf.write(msg + "\n")

    _p("=" * 96)
    _p("MISROUTED TOKEN — misroute distance analysis")
    _p(f"  valid tokens: {n_v:,}")
    _p(f"  MISROUTED:    {n_mis:,} ({100*n_mis/n_v:.2f}%)  ← analysis focus")
    _p(f"  Total Normal MAE·pix (all valid):    {total_normal_mae_pix:>15,.1f}")
    _p(f"  Total positive recoverable Σ·pix:    {total_recov_pix:>15,.1f}")
    _p("=" * 96)

    # ================================================================
    # PART 1 — Distribution of misroute distances
    # ================================================================
    _p("\n" + "=" * 96)
    _p("PART 1 — HOW BADLY ARE MISROUTED TOKENS WRONG?")
    _p("  misroute_distance = how far depth is past assigned expert's bin edge")
    _p("=" * 96)

    _p(f"\n  Misroute distance stats (over {n_mis:,} misrouted tokens):")
    md_mis = md[misroute]
    _p(f"    mean: {md_mis.mean():.2f}m   median: {np.median(md_mis):.2f}m   "
       f"p25: {np.percentile(md_mis, 25):.2f}   p75: {np.percentile(md_mis, 75):.2f}   "
       f"max: {md_mis.max():.2f}")

    _p(f"\n  Misroute distance buckets:")
    _p(f"  {'bucket':<16s}  {'n_tokens':>10s}  {'% of misr':>10s}  "
       f"{'mean_MAE_N':>12s}  {'mean_MAE_S':>12s}  {'recov/tok':>10s}  "
       f"{'ΣMAE·pix':>14s}  {'% total MAE':>12s}  {'% recov':>10s}")

    md_edges = [0.0, 1.0, 2.0, 5.0, 10.0, 20.0, 1e6]
    md_labels = ["0-1m", "1-2m", "2-5m", "5-10m", "10-20m", "≥20m"]
    for i in range(len(md_labels)):
        lo, hi = md_edges[i], md_edges[i+1]
        m = misroute & (md >= lo) & (md < hi)
        n = int(m.sum())
        if n == 0: continue
        pct_misr = 100 * n / n_mis
        mae_n = float(err_n[m].mean())
        mae_s = float(err_s[m].mean())
        recov = float((err_n[m] - err_s[m]).mean())
        contrib_mae = float((err_n[m] * vc[m]).sum())
        contrib_recov = float(np.where(err_n[m] > err_s[m],
                                        (err_n[m]-err_s[m])*vc[m], 0).sum())
        pct_mae = 100 * contrib_mae / total_normal_mae_pix
        pct_recov = 100 * contrib_recov / total_recov_pix
        _p(f"  {md_labels[i]:<16s}  {n:>10,d}  {pct_misr:>9.2f}%  "
           f"{mae_n:>11.3f}  {mae_s:>11.3f}  {recov:>+9.3f}  "
           f"{contrib_mae:>+14,.1f}  {pct_mae:>11.2f}%  {pct_recov:>9.2f}%")

    # ================================================================
    # PART 2 — MAE correlates with misroute distance?
    # ================================================================
    _p("\n" + "=" * 96)
    _p("PART 2 — MAE vs MISROUTE DISTANCE")
    _p("=" * 96)
    _p(f"\n  Correlation (Pearson) over misrouted tokens:")
    _p(f"    r(misroute_dist, MAE_normal): {np.corrcoef(md_mis, err_n[misroute])[0,1]:.4f}")
    _p(f"    r(misroute_dist, MAE_soft):   {np.corrcoef(md_mis, err_s[misroute])[0,1]:.4f}")
    _p(f"    r(misroute_dist, recov gap):  "
       f"{np.corrcoef(md_mis, (err_n-err_s)[misroute])[0,1]:.4f}")

    # ================================================================
    # PART 3 — Split by misroute type (adjacent vs cross-bin)
    # ================================================================
    _p("\n" + "=" * 96)
    _p("PART 3 — BY MISROUTE TYPE (adjacent vs cross-bin)")
    _p("=" * 96)

    diff = np.abs(pred - gt)
    for tname, cond in [("Adjacent misroute (|Δ|=1)", diff == 1),
                         ("Cross-bin misroute (|Δ|=2)", diff == 2)]:
        m = misroute & cond
        n = int(m.sum())
        if n == 0: continue
        md_sub = md[m]
        _p(f"\n  {tname}  ({n:,} tokens)")
        _p(f"    mean dist: {md_sub.mean():.2f}m  median: {np.median(md_sub):.2f}m  "
           f"max: {md_sub.max():.2f}")
        _p(f"    {'bucket':<16s}  {'n':>10s}  {'% of type':>10s}  "
           f"{'mean_MAE':>9s}  {'mean_recov':>10s}")
        for i in range(len(md_labels)):
            lo, hi = md_edges[i], md_edges[i+1]
            mm = m & (md >= lo) & (md < hi)
            n2 = int(mm.sum())
            if n2 == 0: continue
            pct = 100 * n2 / n
            mae_v = float(err_n[mm].mean())
            recov = float((err_n[mm] - err_s[mm]).mean())
            _p(f"    {md_labels[i]:<16s}  {n2:>10,d}  {pct:>9.2f}%  "
               f"{mae_v:>8.3f}  {recov:>+9.3f}")

    # ================================================================
    # PART 4 — Which tokens get BADLY misrouted?
    # ================================================================
    _p("\n" + "=" * 96)
    _p("PART 4 — CHARACTERIZE BADLY MISROUTED TOKENS (dist ≥ 10m)")
    _p("=" * 96)

    badly = misroute & (md >= 10.0)
    aptly = misroute & (md < 2.0)
    _p(f"\n  Badly misrouted (dist ≥ 10m):   {int(badly.sum()):>7,d}")
    _p(f"  Aptly misrouted (dist < 2m):    {int(aptly.sum()):>7,d}")

    def _stat(m, name):
        n = int(m.sum())
        if n == 0: return
        _p(f"\n  --- {name}  n={n:,} ---")
        _p(f"    Mean MAE_normal:   {float(err_n[m].mean()):>7.3f}")
        _p(f"    Mean MAE_soft:     {float(err_s[m].mean()):>7.3f}")
        _p(f"    Mean depth:        {float(depth[m].mean()):>7.2f}")
        _p(f"    Mean bound:        {float(bound[m].mean()):>7.4f}")
        _p(f"    Mean max_prob:     {float(max_p[m].mean()):>7.4f}")
        _p(f"    Mean L1(p,f):      {float(l1[m].mean()):>7.4f}")
        # Cross-bin fraction
        d = np.abs(pred[m] - gt[m])
        adj_pct = 100 * (d == 1).mean()
        cross_pct = 100 * (d == 2).mean()
        _p(f"    Adjacent:{adj_pct:.1f}%  Cross-bin:{cross_pct:.1f}%")

    _stat(aptly, "APTLY misrouted (dist < 2m)")
    _stat(badly, "BADLY misrouted (dist ≥ 10m)")

    # Where do badly-misrouted tokens live?
    _p(f"\n  BADLY misrouted — depth distribution:")
    for lo, hi in [(0, 20), (20, 50), (50, 100)]:
        m = badly & (depth >= lo) & (depth < hi)
        n = int(m.sum())
        if n == 0: continue
        pct = 100 * n / badly.sum()
        _p(f"    depth [{lo:>3d},{hi:>3d}) (gt={['near','mid','far'][gt[m][0] if n>0 else 0]}): "
           f"{n:>7,d} ({pct:>5.2f}%)")

    _p(f"\n  BADLY misrouted — (gt → pred) confusion:")
    labels = ["near", "mid", "far"]
    for tk in range(3):
        for pk in range(3):
            if tk == pk: continue
            m = badly & (gt == tk) & (pred == pk)
            n = int(m.sum())
            if n == 0: continue
            pct = 100 * n / badly.sum()
            mae_v = float(err_n[m].mean())
            md_m = float(md[m].mean())
            _p(f"    {labels[tk]:<4s} → {labels[pk]:<4s}: {n:>7,d} ({pct:>5.2f}%)  "
               f"mean_dist={md_m:.1f}m  mean_MAE={mae_v:.1f}m")

    _p(f"\n  BADLY misrouted — row band (image position):")
    H_img = int(image_hw[0]); H_tok = int(grid_hw[0])
    for lo, hi in [(0, 8), (8, 13), (13, 15), (15, 17), (17, 20), (20, 29)]:
        m = badly & (row >= lo) & (row < hi)
        n = int(m.sum())
        if n == 0: continue
        pct = 100 * n / badly.sum()
        y0 = lo * H_img // H_tok; y1 = hi * H_img // H_tok
        _p(f"    rows [{lo:>2d},{hi:>2d}) y=[{y0:>3d},{y1:>3d}): {n:>7,d} ({pct:>5.2f}%)")

    _p(f"\n  BADLY misrouted — col band (image position):")
    W_img = int(image_hw[1]); W_tok = int(grid_hw[1])
    for lo, hi in [(0, 10), (10, 20), (20, 30), (30, 40), (40, 50)]:
        m = badly & (col >= lo) & (col < hi)
        n = int(m.sum())
        if n == 0: continue
        pct = 100 * n / badly.sum()
        x0 = lo * W_img // W_tok; x1 = hi * W_img // W_tok
        _p(f"    cols [{lo:>2d},{hi:>2d}) x=[{x0:>4d},{x1:>4d}): {n:>7,d} ({pct:>5.2f}%)")

    _p(f"\n  BADLY misrouted — bound bucket:")
    for name, lo, hi in [("PURE      [0.00,0.05)", 0.00, 0.05),
                          ("SLIGHT    [0.05,0.15)", 0.05, 0.15),
                          ("MODERATE  [0.15,0.30)", 0.15, 0.30),
                          ("HIGH      [0.30,0.70)", 0.30, 0.70)]:
        m = badly & (bound >= lo) & (bound < hi)
        n = int(m.sum())
        if n == 0: continue
        pct = 100 * n / badly.sum()
        _p(f"    {name:<24s}: {n:>7,d} ({pct:>5.2f}%)")

    _p(f"\n  BADLY misrouted — confidence bucket:")
    for name, lo, hi in [("SATURATED [1.00]     ", 0.9999, 1.001),
                          ("near-sat  [0.95,1.0)", 0.95, 0.9999),
                          ("confident [0.70,0.95)", 0.70, 0.95),
                          ("uncertain [0.34,0.70)", 0.0, 0.70)]:
        m = badly & (max_p >= lo) & (max_p < hi)
        n = int(m.sum())
        if n == 0: continue
        pct = 100 * n / badly.sum()
        _p(f"    {name:<24s}: {n:>7,d} ({pct:>5.2f}%)")

    # ================================================================
    # Top-20 worst BADLY misrouted (for visual inspection)
    # ================================================================
    _p("\n" + "=" * 96)
    _p("PART 5 — TOP-20 BADLY MISROUTED TOKENS (dist ≥ 10m, sorted by dist)")
    _p("=" * 96)
    md_ranked = np.where(badly, md, -np.inf)
    top = np.argsort(-md_ranked)[:20]
    _p(f"\n  {'#':<3s}  {'sample_id':<58s}  {'r,c':<7s}  "
       f"{'x_range':<12s}  {'depth':>7s}  {'gt→pred':<12s}  "
       f"{'dist':>6s}  {'MAE':>6s}  {'max_p':>6s}")
    for rank, i in enumerate(top):
        s_id = str(sample_ids[sample_idx[i]])[:56]
        r, c = int(row[i]), int(col[i])
        x0 = c * W_img // W_tok; x1 = (c+1) * W_img // W_tok
        gt_name = ["near", "mid", "far"][gt[i]]
        pr_name = ["near", "mid", "far"][pred[i]]
        _p(f"  {rank+1:<3d}  {s_id:<58s}  {r:>2d},{c:<2d}  "
           f"[{x0:>4d},{x1:<4d})  {depth[i]:>6.1f}m  {gt_name}→{pr_name:<5s}  "
           f"{md[i]:>5.1f}m  {err_n[i]:>5.1f}m  {max_p[i]:>6.3f}")

    out = "output/analysis/router_cost/misroute_distance_v4c.txt"
    with open(out, "w") as f:
        f.write(buf.getvalue())
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
