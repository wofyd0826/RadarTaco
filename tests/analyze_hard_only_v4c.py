"""HARD-only analysis of router hardness × routable error.

Deliberately IGNORES all soft/distribution signals:
  • No L1(probs, frac)
  • No top-2 pair match
  • No OracleSoft comparison
  • No max_prob confidence bucketing

Uses ONLY:
  • argmax_p    — router's argmax (which single expert Normal would pick)
  • argmax_f    — target argmax (majority-vote bin from pixel_fraction)
  • err_normal  — actual per-token MAE under Normal routing
  • err_hard    — per-token MAE under OracleHard (per-token one_hot(gt))
  • gap_hard    — err_normal − err_hard (routable via better argmax alone)

"Hard for router" definition (all binary, from argmax only):
  HARD  ↔  argmax_p ≠ argmax_f  (misroute)
     — split into ADJACENT (|Δ|=1) and CROSS (|Δ|=2)

"Costly" = pixel-weighted gap_hard, positive-only, share of total_pos.
"""
import io
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, ROOT)


def main():
    dump = np.load(
        "output/analysis/router_cost/v4c_stage2_per_token_dump.npz",
        allow_pickle=True)
    perm = np.load(
        "output/analysis/router_cost/v4c_stage2_per_mode.npz",
        allow_pickle=True)

    err_n = perm['err_normal']
    err_h = perm['err_hard']
    vc = perm['valid_pixel_count']
    depth = dump['depth']
    argmax_p = dump['argmax_p'].astype(np.int64)
    argmax_f = dump['argmax_f'].astype(np.int64)
    row = dump['token_row']
    grid_hw = dump['grid_hw']; image_hw = dump['image_hw']

    N = err_n.shape[0]
    valid = np.isfinite(err_n) & np.isfinite(err_h) & (vc > 0)
    n_v = int(valid.sum())

    gap = err_n - err_h              # routable via argmax fix alone
    contrib = gap * vc               # m·pix per token
    total_pos = float(contrib[valid & (gap > 0)].sum())
    total_neg = float(contrib[valid & (gap < 0)].sum())
    pix = float(vc[valid].sum())

    match = (argmax_p == argmax_f)
    diff = np.abs(argmax_p - argmax_f)

    buf = io.StringIO()
    def _p(msg=""):
        print(msg); buf.write(msg + "\n")

    _p("=" * 96)
    _p("HARD-ONLY analysis — v4c stage2  (argmax match / Normal vs OracleHard)")
    _p(f"  valid tokens: {n_v:,}   valid pixels: {pix:,.0f}")
    _p(f"  Pixel-weighted MAE:  Normal={float((err_n*vc)[valid].sum()/pix):.4f}  "
       f"OracleHard={float((err_h*vc)[valid].sum()/pix):.4f}  "
       f"gap={float((gap*vc)[valid].sum()/pix):+.4f}")
    _p(f"  Positive-only total (recoverable via argmax fix):  "
       f"{total_pos:+,.1f} m·pix   → {total_pos/pix:+.4f} m/pix")
    _p(f"  Negative offset (Normal already ≥ Hard):           "
       f"{total_neg:+,.1f} m·pix   → {total_neg/pix:+.4f} m/pix")
    _p("=" * 96)

    pos = valid & (gap > 0)

    # =====================================================
    # PART A — WHERE ARE MISROUTES?  (router hardness view)
    # =====================================================
    _p("\n" + "=" * 96)
    _p("PART A — WHERE THE ROUTER GETS ARGMAX WRONG  (misroute distribution)")
    _p("=" * 96)

    # A.1  overall counts
    _p("\n-- A.1 Overall argmax outcome --")
    _p(f"  {'outcome':<32s}  {'n_tokens':>10s}  {'% of valid':>10s}")
    for name, cond in [
        ("Correct (argmax match)", valid & match),
        ("Adjacent misroute (|Δ|=1)", valid & (diff == 1)),
        ("Cross-bin misroute (|Δ|=2)", valid & (diff == 2)),
    ]:
        n = int(cond.sum())
        _p(f"  {name:<32s}  {n:>10,d}  {100*n/n_v:>9.2f}%")

    # A.2  misroute rate by depth
    _p("\n-- A.2 Misroute rate by depth range --")
    _p(f"  {'range':>14s}  {'tokens':>10s}  {'correct%':>10s}  "
       f"{'adj-mis%':>10s}  {'cross-mis%':>12s}")
    edges = [0, 10, 15, 20, 22, 25, 30, 40, 45, 50, 55, 60, 80, 100]
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = valid & (depth >= lo) & (depth < hi)
        n = int(m.sum())
        if n == 0: continue
        p_correct = float(100 * match[m].mean())
        p_adj = float(100 * (diff[m] == 1).mean())
        p_cross = float(100 * (diff[m] == 2).mean())
        _p(f"  [{lo:>4d},{hi:>4d})m  {n:>10,d}  {p_correct:>9.2f}%  "
           f"{p_adj:>9.2f}%  {p_cross:>11.2f}%")

    # A.3  misroute rate by target class (which gt bin gets misrouted the most)
    _p("\n-- A.3 Misroute rate by target bin --")
    names = ["near", "mid", "far"]
    _p(f"  {'target':<8s}  {'tokens':>10s}  {'correct%':>10s}  "
       f"{'→ mispred as (%)':<40s}")
    for k in range(3):
        m = valid & (argmax_f == k)
        n = int(m.sum())
        if n == 0: continue
        c = float(100 * match[m].mean())
        # pred distribution when misrouted
        wrong = m & ~match
        pd = {}
        for kp in range(3):
            if kp == k: continue
            pd[names[kp]] = int((wrong & (argmax_p == kp)).sum())
        dist = "  ".join([f"{names[kp]}: {100*pd[names[kp]]/max(n,1):.2f}%" for kp in range(3) if kp != k])
        _p(f"  {names[k]:<8s}  {n:>10,d}  {c:>9.2f}%  {dist:<40s}")

    # A.4  row band
    _p("\n-- A.4 Misroute rate by image row band --")
    H_img = int(image_hw[0]); H_tok = int(grid_hw[0])
    _p(f"  {'rows':<10s}  {'y range':<14s}  {'tokens':>10s}  {'correct%':>10s}  "
       f"{'adj-mis%':>10s}  {'cross-mis%':>12s}")
    row_edges = [0, 8, 13, 15, 17, 20, 25, 29]
    for lo, hi in zip(row_edges[:-1], row_edges[1:]):
        m = valid & (row >= lo) & (row < hi)
        n = int(m.sum())
        if n == 0: continue
        y0 = lo * H_img // H_tok; y1 = hi * H_img // H_tok
        pc = float(100 * match[m].mean())
        pa = float(100 * (diff[m] == 1).mean())
        px = float(100 * (diff[m] == 2).mean())
        _p(f"  [{lo:>2d},{hi:>2d})  [{y0:>3d},{y1:>3d})    {n:>10,d}  "
           f"{pc:>9.2f}%  {pa:>9.2f}%  {px:>11.2f}%")

    # =====================================================
    # PART B — WHERE DOES THE HARD GAP LIVE?  (costly view)
    # =====================================================
    _p("\n" + "=" * 96)
    _p("PART B — WHERE HARD-GAP (routable-via-argmax) COMES FROM")
    _p("=" * 96)

    _p("\n-- B.1 Gap by routing outcome --")
    _p(f"  {'outcome':<32s}  {'n':>10s}  {'MAE_N':>7s}  {'MAE_H':>7s}  "
       f"{'gap':>7s}  {'Σgap·pix':>14s}  {'%':>7s}")
    for name, cond in [
        ("Correct (argmax match)", match),
        ("Adjacent misroute (|Δ|=1)", diff == 1),
        ("Cross-bin misroute (|Δ|=2)", diff == 2),
    ]:
        m = pos & cond
        n = int(m.sum())
        if n == 0: continue
        p_ = float(vc[m].sum())
        mn = float((err_n[m] * vc[m]).sum() / p_)
        mh = float((err_h[m] * vc[m]).sum() / p_)
        gs = float((gap[m] * vc[m]).sum())
        pct = 100 * gs / total_pos
        _p(f"  {name:<32s}  {n:>10,d}  {mn:>6.3f}  {mh:>6.3f}  "
           f"{mn-mh:>+6.3f}  {gs:>+13,.1f}  {pct:>6.2f}%")

    _p("\n-- B.2 Gap by depth range --")
    _p(f"  {'range':>14s}  {'n':>10s}  {'MAE_N':>7s}  {'MAE_H':>7s}  "
       f"{'gap':>7s}  {'Σgap·pix':>14s}  {'%':>7s}")
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = pos & (depth >= lo) & (depth < hi)
        n = int(m.sum())
        if n == 0: continue
        p_ = float(vc[m].sum())
        mn = float((err_n[m] * vc[m]).sum() / p_)
        mh = float((err_h[m] * vc[m]).sum() / p_)
        gs = float((gap[m] * vc[m]).sum())
        pct = 100 * gs / total_pos
        _p(f"  [{lo:>4d},{hi:>4d})m  {n:>10,d}  {mn:>6.3f}  {mh:>6.3f}  "
           f"{mn-mh:>+6.3f}  {gs:>+13,.1f}  {pct:>6.2f}%")

    _p("\n-- B.3 Gap by target bin --")
    _p(f"  {'target':<8s}  {'n':>10s}  {'MAE_N':>7s}  {'MAE_H':>7s}  "
       f"{'gap':>7s}  {'Σgap·pix':>14s}  {'%':>7s}")
    for k in range(3):
        m = pos & (argmax_f == k)
        n = int(m.sum())
        if n == 0: continue
        p_ = float(vc[m].sum())
        mn = float((err_n[m] * vc[m]).sum() / p_)
        mh = float((err_h[m] * vc[m]).sum() / p_)
        gs = float((gap[m] * vc[m]).sum())
        pct = 100 * gs / total_pos
        _p(f"  {names[k]:<8s}  {n:>10,d}  {mn:>6.3f}  {mh:>6.3f}  "
           f"{mn-mh:>+6.3f}  {gs:>+13,.1f}  {pct:>6.2f}%")

    _p("\n-- B.4 Gap by row band --")
    _p(f"  {'rows':<10s}  {'y range':<14s}  {'n':>10s}  {'gap':>7s}  "
       f"{'Σgap·pix':>14s}  {'%':>7s}")
    for lo, hi in zip(row_edges[:-1], row_edges[1:]):
        m = pos & (row >= lo) & (row < hi)
        n = int(m.sum())
        if n == 0: continue
        p_ = float(vc[m].sum())
        mn = float((err_n[m] * vc[m]).sum() / p_)
        mh = float((err_h[m] * vc[m]).sum() / p_)
        gs = float((gap[m] * vc[m]).sum())
        pct = 100 * gs / total_pos
        y0 = lo * H_img // H_tok; y1 = hi * H_img // H_tok
        _p(f"  [{lo:>2d},{hi:>2d})  [{y0:>3d},{y1:>3d})    {n:>10,d}  "
           f"{mn-mh:>+6.3f}  {gs:>+13,.1f}  {pct:>6.2f}%")

    # =====================================================
    # PART C — INTERSECTION: HARD × COSTLY
    # =====================================================
    _p("\n" + "=" * 96)
    _p("PART C — HARD (argmax mismatch) × COSTLY (gap contribution)")
    _p("=" * 96)

    _p("\n-- C.1 2D join: routing outcome × depth range --")
    _p(f"  {'depth':<14s}  {'routing':<28s}  {'n':>10s}  "
       f"{'gap':>7s}  {'Σgap·pix':>14s}  {'%':>7s}")
    depth_groups = [
        ("<15m", (depth < 15)),
        ("15-25m (near-mid bd)", (depth >= 15) & (depth < 25)),
        ("25-45m", (depth >= 25) & (depth < 45)),
        ("45-60m (mid-far bd)", (depth >= 45) & (depth < 60)),
        ("60-80m", (depth >= 60) & (depth < 80)),
        (">=80m", depth >= 80),
    ]
    for d_name, d_cond in depth_groups:
        for r_name, r_cond in [
            ("Correct", match),
            ("Adjacent misroute", diff == 1),
            ("Cross-bin misroute", diff == 2),
        ]:
            m = pos & d_cond & r_cond
            n = int(m.sum())
            if n == 0: continue
            p_ = float(vc[m].sum())
            mn = float((err_n[m] * vc[m]).sum() / p_)
            mh = float((err_h[m] * vc[m]).sum() / p_)
            gs = float((gap[m] * vc[m]).sum())
            pct = 100 * gs / total_pos
            _p(f"  {d_name:<14s}  {r_name:<28s}  {n:>10,d}  "
               f"{mn-mh:>+6.3f}  {gs:>+13,.1f}  {pct:>6.2f}%")

    _p("\n-- C.2 Misroute-only summary (what fixing all misroutes recovers) --")
    m = pos & ~match
    n = int(m.sum())
    p_ = float(vc[m].sum())
    gs = float((gap[m] * vc[m]).sum())
    _p(f"  All misroutes (positive-gap only): n={n:,}  "
       f"pix={p_:,.0f}  Σgap·pix={gs:+.1f}  "
       f"= {100*gs/total_pos:.2f}% of total_pos")
    m = pos & match
    n = int(m.sum())
    p_ = float(vc[m].sum())
    gs = float((gap[m] * vc[m]).sum())
    _p(f"  Correct routing (positive-gap only): n={n:,}  "
       f"pix={p_:,.0f}  Σgap·pix={gs:+.1f}  "
       f"= {100*gs/total_pos:.2f}% of total_pos")
    _p("  (Correct-routing positive gap comes from Normal's top_k=2 soft mix "
       "hurting vs OracleHard's per-token one_hot — this is the mixing tax.)")

    _p("\n-- C.3 Concentration in top-N tokens --")
    order = np.argsort(-contrib)
    for topN in (100, 1000, 10000, 100000, 500000):
        share = 100 * contrib[order[:topN]].sum() / total_pos
        _p(f"  top {topN:>7,d} tokens contain {share:>6.2f}% of total_pos "
           f"({100*topN/n_v:.2f}% of valid)")

    _p("\n-- C.4 Top-100 most costly tokens: hardness profile --")
    top = order[:100]
    _p(f"  Mean gap:       {gap[top].mean():>7.3f} m")
    _p(f"  Mean MAE_N:     {err_n[top].mean():>7.3f} m")
    _p(f"  Mean MAE_H:     {err_h[top].mean():>7.3f} m")
    _p(f"  Mean depth:     {depth[top].mean():>7.2f} m")
    _p(f"  argmax match:   {100*match[top].mean():>6.2f}%")
    _p(f"  adjacent misr:  {100*(diff[top]==1).mean():>6.2f}%")
    _p(f"  cross-bin misr: {100*(diff[top]==2).mean():>6.2f}%")
    dh = np.bincount(np.digitize(depth[top], [10, 20, 30, 50, 80, 100]),
                     minlength=7).tolist()
    _p(f"  depth histogram (<10, 10-20, 20-30, 30-50, 50-80, 80-100, >100): {dh}")

    out = "output/analysis/router_cost/hard_only_v4c.txt"
    with open(out, "w") as f:
        f.write(buf.getvalue())
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
