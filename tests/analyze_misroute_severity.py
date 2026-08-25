"""Analyze misrouted tokens' error severity distribution + mix-ratio breakdown.

For each MISROUTED token (pred != gt in Normal), bucket by per-token MAE
severity. Then decompose within each mix (bound) bucket.

Reads:
  output/analysis/router_cost/v4c_stage2_per_mode.npz

Writes:
  output/analysis/router_cost/misroute_severity_v4c.txt
"""
import io
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, ROOT)


def main():
    d = np.load("output/analysis/router_cost/v4c_stage2_per_mode.npz",
                allow_pickle=True)
    err_n = d["err_normal"]
    err_h = d["err_hard"]
    err_s = d["err_soft"]
    vc = d["valid_pixel_count"]
    gt = d["gt"]
    pred = d["actual_pred"]
    bound = d["bound"]
    depth = d["depth"]

    N = err_n.shape[0]
    valid = np.isfinite(err_n) & np.isfinite(err_h) & np.isfinite(err_s) & (vc > 0)
    misroute = valid & (pred != gt)
    correct = valid & (pred == gt)
    all_pix = float(vc[valid].sum())

    # Severity buckets (per-token MAE)
    sev_edges = [0.0, 1.0, 2.0, 5.0, 10.0, 20.0, 1e9]
    sev_labels = ["< 1m", "1-2m", "2-5m", "5-10m", "10-20m", "≥ 20m"]

    # Mix buckets (bound = 1 - max_frac)
    mix_edges = [(0.0, 0.05, "PURE"),
                 (0.05, 0.15, "SLIGHT"),
                 (0.15, 0.30, "MODERATE"),
                 (0.30, 0.70, "HIGH")]

    buf = io.StringIO()
    def _p(msg=""):
        print(msg); buf.write(msg + "\n")

    total_normal_mae_pix = float((err_n[valid] * vc[valid]).sum())
    total_recov_pix = float(((err_n - err_s)[valid] * vc[valid]).sum())
    total_recov_pos_pix = float(np.where((err_n - err_s) > 0, (err_n - err_s) * vc, 0)[valid].sum())

    _p("=" * 96)
    _p("Misroute severity analysis — v4c stage2")
    _p(f"  valid tokens: {int(valid.sum()):,}  pixels: {all_pix:,.0f}")
    _p(f"  correct routing:  {int(correct.sum()):,} ({100*correct.sum()/valid.sum():.2f}%)")
    _p(f"  MISROUTED:        {int(misroute.sum()):,} ({100*misroute.sum()/valid.sum():.2f}%)")
    _p(f"  Total Normal MAE·pix:  {total_normal_mae_pix:>15,.1f}  (=1.92 m avg)")
    _p(f"  Total recoverable Σ:   {total_recov_pix:>15,.1f}  (Normal-OracleSoft)")
    _p(f"  Total pos-only recov:  {total_recov_pos_pix:>15,.1f}")
    _p("=" * 96)

    # ============================================================
    # PART 1 — Severity distribution among MISROUTED tokens
    # ============================================================
    _p("\n" + "=" * 96)
    _p("PART 1 — SEVERITY DISTRIBUTION OF MISROUTED TOKENS")
    _p("=" * 96)
    _p(f"\n  Per-token MAE (Normal) among misrouted tokens ({int(misroute.sum()):,}):")
    _p(f"  {'severity':<10s}  {'n_tokens':>10s}  {'% of misr':>10s}  "
       f"{'mean MAE':>9s}  {'ΣMAE·pix':>14s}  {'% total MAE':>12s}  "
       f"{'ΣRecov·pix':>14s}  {'% recov':>10s}")
    for i in range(len(sev_labels)):
        lo, hi = sev_edges[i], sev_edges[i+1]
        m = misroute & (err_n >= lo) & (err_n < hi)
        n = int(m.sum())
        if n == 0:
            _p(f"  {sev_labels[i]:<10s}  {0:>10,d}")
            continue
        mean_mae = float(err_n[m].mean())
        contrib_pix = float((err_n[m] * vc[m]).sum())
        pct_total = 100 * contrib_pix / total_normal_mae_pix
        recov_pix = float(((err_n[m] - err_s[m]) * vc[m]).sum())
        pct_recov = 100 * recov_pix / total_recov_pos_pix
        pct_misr = 100 * n / misroute.sum()
        _p(f"  {sev_labels[i]:<10s}  {n:>10,d}  {pct_misr:>9.2f}%  "
           f"{mean_mae:>8.3f}  {contrib_pix:>+14,.1f}  {pct_total:>11.2f}%  "
           f"{recov_pix:>+14,.1f}  {pct_recov:>9.2f}%")

    # Same for CORRECTLY-routed tokens (baseline)
    _p(f"\n  For comparison — CORRECTLY routed tokens ({int(correct.sum()):,}):")
    _p(f"  {'severity':<10s}  {'n_tokens':>10s}  {'% of corr':>10s}  "
       f"{'mean MAE':>9s}  {'ΣMAE·pix':>14s}  {'% total MAE':>12s}")
    for i in range(len(sev_labels)):
        lo, hi = sev_edges[i], sev_edges[i+1]
        m = correct & (err_n >= lo) & (err_n < hi)
        n = int(m.sum())
        if n == 0: continue
        mean_mae = float(err_n[m].mean())
        contrib_pix = float((err_n[m] * vc[m]).sum())
        pct_total = 100 * contrib_pix / total_normal_mae_pix
        pct_corr = 100 * n / correct.sum()
        _p(f"  {sev_labels[i]:<10s}  {n:>10,d}  {pct_corr:>9.2f}%  "
           f"{mean_mae:>8.3f}  {contrib_pix:>+14,.1f}  {pct_total:>11.2f}%")

    # ============================================================
    # PART 2 — Severity × mix ratio (misrouted only)
    # ============================================================
    _p("\n" + "=" * 96)
    _p("PART 2 — MISROUTE SEVERITY × MIX RATIO")
    _p("=" * 96)

    for mlo, mhi, mname in mix_edges:
        m_all = valid & (bound >= mlo) & (bound < mhi)
        m_misr = misroute & (bound >= mlo) & (bound < mhi)
        m_corr = correct & (bound >= mlo) & (bound < mhi)
        n_all = int(m_all.sum())
        n_misr = int(m_misr.sum())
        n_corr = int(m_corr.sum())
        if n_all == 0: continue
        mix_pix = float(vc[m_all].sum())
        mix_mae_pix = float((err_n[m_all] * vc[m_all]).sum())
        _p(f"\n  --- {mname:<9s} [{mlo:.2f},{mhi:.2f})  "
           f"n_tokens={n_all:,}  ({100*n_all/valid.sum():.1f}% of valid)  "
           f"---")
        _p(f"    Misroute rate: {100*n_misr/max(n_all,1):.2f}% "
           f"({n_misr:,}/{n_all:,})")
        _p(f"    Mean MAE (all): {mix_mae_pix/max(mix_pix,1e-9):.3f} m  "
           f"(contribution: {100*mix_mae_pix/total_normal_mae_pix:.1f}% of total)")
        # severity table within this mix
        _p(f"    Severity distribution of MISROUTED in this bucket:")
        _p(f"    {'severity':<10s}  {'n_tokens':>10s}  {'% of misr':>10s}  "
           f"{'ΣMAE·pix':>14s}  {'% mix MAE':>10s}  {'% total MAE':>12s}")
        for i in range(len(sev_labels)):
            lo, hi = sev_edges[i], sev_edges[i+1]
            m = m_misr & (err_n >= lo) & (err_n < hi)
            n = int(m.sum())
            if n == 0: continue
            contrib = float((err_n[m] * vc[m]).sum())
            pct_misr = 100 * n / max(n_misr, 1)
            pct_mix = 100 * contrib / max(mix_mae_pix, 1e-9)
            pct_tot = 100 * contrib / total_normal_mae_pix
            _p(f"    {sev_labels[i]:<10s}  {n:>10,d}  {pct_misr:>9.2f}%  "
               f"{contrib:>+14,.1f}  {pct_mix:>9.2f}%  {pct_tot:>11.2f}%")

    # ============================================================
    # PART 3 — Are the catastrophic misroutes concentrated somewhere?
    # ============================================================
    _p("\n" + "=" * 96)
    _p("PART 3 — CATASTROPHIC MISROUTES (≥ 10m) — WHO ARE THEY?")
    _p("=" * 96)
    catastrophic = misroute & (err_n >= 10.0)
    n_cat = int(catastrophic.sum())
    _p(f"\n  Total catastrophic misroutes: {n_cat:,} "
       f"({100*n_cat/misroute.sum():.2f}% of misroutes, "
       f"{100*n_cat/valid.sum():.2f}% of all valid)")
    _p(f"\n  Depth distribution:")
    for lo, hi in [(0, 10), (10, 20), (20, 30), (30, 50), (50, 80), (80, 100)]:
        m = catastrophic & (depth >= lo) & (depth < hi)
        n = int(m.sum())
        if n == 0: continue
        pct = 100 * n / max(n_cat, 1)
        mean_mae = float(err_n[m].mean())
        _p(f"    [{lo:>3d},{hi:>3d})m: {n:>7,d} ({pct:>5.2f}%)  mean_MAE={mean_mae:.1f}")

    _p(f"\n  Misroute type (adjacent vs cross-bin):")
    diff = np.abs(pred.astype(np.int64) - gt.astype(np.int64))
    for name, cond in [("Adjacent (|Δ|=1)", diff == 1),
                        ("Cross-bin (|Δ|=2)", diff == 2)]:
        m = catastrophic & cond
        n = int(m.sum())
        pct = 100 * n / max(n_cat, 1)
        if n == 0: continue
        mean_mae = float(err_n[m].mean())
        _p(f"    {name:<20s}: {n:>7,d} ({pct:>5.2f}%)  mean_MAE={mean_mae:.1f}")

    _p(f"\n  Bound distribution:")
    for mlo, mhi, mname in mix_edges:
        m = catastrophic & (bound >= mlo) & (bound < mhi)
        n = int(m.sum())
        if n == 0: continue
        pct = 100 * n / max(n_cat, 1)
        mean_mae = float(err_n[m].mean())
        _p(f"    {mname:<10s}: {n:>7,d} ({pct:>5.2f}%)  mean_MAE={mean_mae:.1f}")

    out = "output/analysis/router_cost/misroute_severity_v4c.txt"
    with open(out, "w") as f:
        f.write(buf.getvalue())
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
