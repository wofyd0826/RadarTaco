"""Deep dive: WHERE does the model fail to match the distribution?

Beyond argmax accuracy — what SHAPE of distribution errors is the model making?
  A. Overall bin-wise bias (model over/under-predicting each bin on avg)
  B. Mean predicted distribution by target 'shape class':
       - pure_near / pure_mid / pure_far
       - near-mid mix, mid-far mix, near-far mix (cross-bin)
       - tri-mix
  C. Cross-tab of target shape class vs predicted shape class
  D. Systematic residual per (target_bin, predicted_bin)
  E. KL asymmetry — is model spreading too much or missing mass?
  F. Where in image (depth × row) does distribution fail most?
"""
import io
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, ROOT)

BINS = [0.0, 20.0, 50.0, 100.0]
K = 3
LABELS = ["near", "mid", "far"]


def classify_shape(dist, pure_thr=0.90, trivial_thr=0.05):
    """Classify a K=3 distribution into 8 shape classes.

    Returns integer 0..7:
      0..2: pure_near/mid/far (max >= pure_thr)
      3..5: mix_near_mid / mix_mid_far / mix_near_far  (two bins share mass)
      6:    tri_mix (all three non-trivial)
      7:    unclassified (shouldn't happen with these thresholds)
    """
    N, _ = dist.shape
    out = np.zeros(N, dtype=np.int8)
    maxi = dist.argmax(axis=1)
    maxv = dist.max(axis=1)
    # pure
    pure = maxv >= pure_thr
    out[pure] = maxi[pure]                         # 0..2

    # mixed
    non_pure = ~pure
    if non_pure.any():
        # How many bins > trivial_thr
        active = dist > trivial_thr
        n_active = active.sum(axis=1)
        two = non_pure & (n_active == 2)
        three = non_pure & (n_active >= 3)

        if two.any():
            # Identify which two bins are active
            # Encode active mask into class: 011=near+mid=3, 110=mid+far=4, 101=near+far=5
            am = active[two]
            # order [near, mid, far]
            near_mid = am[:, 0] & am[:, 1] & ~am[:, 2]
            mid_far  = ~am[:, 0] & am[:, 1] & am[:, 2]
            near_far = am[:, 0] & ~am[:, 1] & am[:, 2]
            two_idx = np.where(two)[0]
            out[two_idx[near_mid]] = 3
            out[two_idx[mid_far]]  = 4
            out[two_idx[near_far]] = 5
            # Edge cases fall through as class 7 (unclassified)

        if three.any():
            out[three] = 6
    return out


SHAPE_NAMES = ['pure_near', 'pure_mid', 'pure_far',
               'mix_near+mid', 'mix_mid+far', 'mix_near+far',
               'tri_mix', 'other']


def main():
    d = np.load("output/analysis/router_cost/v4c_stage2_per_token_dump.npz",
                allow_pickle=True)
    probs = d['probs']
    frac = d['frac']
    depth = d['depth']
    vc = d['valid_pix_count']
    row = d['token_row']
    mae = d['mae']
    valid = np.isfinite(mae) & (vc > 0)

    max_p = probs.max(axis=1)
    max_f = frac.max(axis=1)

    buf = io.StringIO()
    def _p(msg=""):
        print(msg); buf.write(msg + "\n")

    N_v = int(valid.sum())
    _p("=" * 96)
    _p(f"DISTRIBUTION ERROR ANALYSIS — v4c stage2")
    _p(f"  valid tokens: {N_v:,}")
    _p("=" * 96)

    # ============================================================
    # A. Overall bin-wise bias
    # ============================================================
    _p("\n" + "=" * 96)
    _p("A. OVERALL BIN-WISE BIAS")
    _p("=" * 96)
    _p(f"\n  Mean values across all valid tokens:")
    _p(f"  {'bin':<8s}  {'mean probs':>12s}  {'mean frac':>12s}  {'bias (p-f)':>12s}")
    for k, name in enumerate(LABELS):
        mp = float(probs[valid, k].mean())
        mf = float(frac[valid, k].mean())
        _p(f"  {name:<8s}  {mp:>11.4f}  {mf:>11.4f}  {mp-mf:>+11.4f}")

    # ============================================================
    # B. Mean predicted distribution by target shape class
    # ============================================================
    _p("\n" + "=" * 96)
    _p("B. MEAN PREDICTED DISTRIBUTION BY TARGET SHAPE CLASS")
    _p("=" * 96)
    target_shape = classify_shape(frac)
    _p(f"\n  {'target shape':<15s}  {'n':>10s}  {'% of tok':>9s}  "
       f"{'mean p_near':>12s}  {'mean p_mid':>11s}  {'mean p_far':>11s}  "
       f"{'target mean':<24s}")
    for cls in range(8):
        m = valid & (target_shape == cls)
        n = int(m.sum())
        if n == 0: continue
        pn = float(probs[m, 0].mean())
        pm_ = float(probs[m, 1].mean())
        pf = float(probs[m, 2].mean())
        tn = float(frac[m, 0].mean())
        tm = float(frac[m, 1].mean())
        tf_ = float(frac[m, 2].mean())
        tgt_str = f"({tn:.2f}, {tm:.2f}, {tf_:.2f})"
        pct = 100 * n / N_v
        _p(f"  {SHAPE_NAMES[cls]:<15s}  {n:>10,d}  {pct:>8.2f}%  "
           f"{pn:>11.4f}  {pm_:>10.4f}  {pf:>10.4f}  {tgt_str:<24s}")

    # ============================================================
    # C. Cross-tab: target shape class × predicted shape class
    # ============================================================
    _p("\n" + "=" * 96)
    _p("C. SHAPE-CLASS CONFUSION (target × predicted)")
    _p("=" * 96)
    pred_shape = classify_shape(probs)
    _p(f"\n  Row = TARGET class, Col = PREDICTED class. Values are % of row (recall)")
    header_left = "target vs predicted"
    _p(f"  {header_left:<20s}  " + "  ".join(f"{n[:9]:>9s}" for n in SHAPE_NAMES) + f"  {'n_row':>10s}")
    for tc in range(7):
        row_mask = valid & (target_shape == tc)
        n_row = int(row_mask.sum())
        if n_row == 0: continue
        parts = [SHAPE_NAMES[tc][:20].ljust(20)]
        for pc in range(8):
            frac_pc = 100 * ((pred_shape[row_mask] == pc).sum() / n_row)
            parts.append(f"{frac_pc:>9.2f}")
        parts.append(f"{n_row:>10,d}")
        _p("  " + "  ".join(parts))

    # ============================================================
    # D. Systematic residual per (target argmax, predicted mass)
    # ============================================================
    _p("\n" + "=" * 96)
    _p("D. WHERE MASS LEAKS — mean residual (probs − frac) by target argmax")
    _p("=" * 96)
    argmax_f = frac.argmax(axis=1)
    argmax_p = probs.argmax(axis=1)
    _p(f"\n  Split by target argmax and MIXED vs PURE:")
    _p(f"  {'target':<12s}  {'target|mixed':<14s}  {'n':>10s}  "
       f"{'res_near':>10s}  {'res_mid':>10s}  {'res_far':>10s}")
    for tgt_k, name in enumerate(LABELS):
        for pure_flag in (True, False):
            m = valid & (argmax_f == tgt_k) & \
                ((max_f >= 0.95) if pure_flag else (max_f < 0.95))
            n = int(m.sum())
            if n == 0: continue
            r_n = float((probs[m, 0] - frac[m, 0]).mean())
            r_m = float((probs[m, 1] - frac[m, 1]).mean())
            r_f = float((probs[m, 2] - frac[m, 2]).mean())
            tag = f"{name}"
            purity = "PURE" if pure_flag else "MIXED"
            _p(f"  {tag:<12s}  {purity:<14s}  {n:>10,d}  "
               f"{r_n:>+9.4f}  {r_m:>+9.4f}  {r_f:>+9.4f}")

    # ============================================================
    # E. KL asymmetry — spread too much or miss mass?
    # ============================================================
    _p("\n" + "=" * 96)
    _p("E. KL ASYMMETRY — DOES MODEL SPREAD TOO MUCH OR MISS MASS?")
    _p("=" * 96)
    p_ = probs.clip(1e-12, 1.0)
    f_ = frac.clip(1e-12, 1.0)
    kl_fp = (frac * (np.log(f_) - np.log(p_))).sum(axis=1)   # KL(f || p): penalty for missing mass where target has
    kl_pf = (probs * (np.log(p_) - np.log(f_))).sum(axis=1)  # KL(p || f): penalty for spreading to where target has none
    _p(f"\n  Overall mean:")
    _p(f"    KL(target || pred) = {float(kl_fp[valid].mean()):.4f}  "
       f"(model MISSING mass at target's peak)")
    _p(f"    KL(pred || target) = {float(kl_pf[valid].mean()):.4f}  "
       f"(model SPREADING mass where target has none)")

    _p(f"\n  By target shape:")
    _p(f"  {'target shape':<15s}  {'n':>10s}  {'KL(f||p)':>10s}  "
       f"{'KL(p||f)':>10s}  {'ratio p/f':>10s}")
    for cls in range(7):
        m = valid & (target_shape == cls)
        n = int(m.sum())
        if n == 0: continue
        kfp = float(kl_fp[m].mean())
        kpf = float(kl_pf[m].mean())
        _p(f"  {SHAPE_NAMES[cls]:<15s}  {n:>10,d}  {kfp:>9.4f}  "
           f"{kpf:>9.4f}  {kpf/max(kfp, 1e-9):>9.3f}")

    # ============================================================
    # F. Where does distribution fail — by depth × row band
    # ============================================================
    _p("\n" + "=" * 96)
    _p("F. WHERE THE DISTRIBUTION FAILS — L1 by depth × row")
    _p("=" * 96)
    l1 = np.abs(probs - frac).sum(axis=1)
    _p(f"\n  Mean L1 by depth range (all valid):")
    _p(f"  {'depth':<14s}  {'n':>10s}  {'mean L1':>8s}  {'median L1':>10s}  {'p95 L1':>8s}")
    for lo, hi in zip([0, 10, 20, 30, 50, 80],
                       [10, 20, 30, 50, 80, 100]):
        m = valid & (depth >= lo) & (depth < hi)
        n = int(m.sum())
        if n == 0: continue
        l1_m = float(l1[m].mean())
        l1_med = float(np.median(l1[m]))
        l1_p95 = float(np.percentile(l1[m], 95))
        _p(f"  [{lo:>3d},{hi:>3d})m  {n:>10,d}  {l1_m:>7.4f}  {l1_med:>9.4f}  {l1_p95:>7.4f}")

    _p(f"\n  Mean L1 by row band (image position):")
    _p(f"  {'rows':<14s}  {'n':>10s}  {'mean L1':>8s}  {'median L1':>10s}")
    for lo, hi in zip([0, 8, 13, 15, 17, 20, 25],
                       [8, 13, 15, 17, 20, 25, 29]):
        m = valid & (row >= lo) & (row < hi)
        n = int(m.sum())
        if n == 0: continue
        l1_m = float(l1[m].mean())
        l1_med = float(np.median(l1[m]))
        _p(f"  rows [{lo:>2d},{hi:>2d})  {n:>10,d}  {l1_m:>7.4f}  {l1_med:>9.4f}")

    # ============================================================
    # G. Sharpness vs target sharpness — full 2D
    # ============================================================
    _p("\n" + "=" * 96)
    _p("G. SHARPNESS: (target max_frac) → (predicted max_prob)")
    _p("=" * 96)
    _p(f"\n  Mean max_prob by target max_frac bucket:")
    _p(f"  {'target max_frac':<18s}  {'n':>10s}  {'mean max_prob':>15s}  "
       f"{'target max mean':>17s}  {'model - target':>15s}")
    for lo, hi in [(0.34, 0.50), (0.50, 0.70), (0.70, 0.85),
                    (0.85, 0.95), (0.95, 1.00), (1.00, 1.01)]:
        m = valid & (max_f >= lo) & (max_f < hi)
        n = int(m.sum())
        if n == 0: continue
        mp = float(max_p[m].mean())
        mf = float(max_f[m].mean())
        tag = f"[{lo:.2f}, {hi:.2f})"
        _p(f"  {tag:<18s}  {n:>10,d}  {mp:>14.4f}  {mf:>16.4f}  {mp-mf:>+14.4f}")

    # ============================================================
    # H. High-L1 tokens — WHAT does model output?
    # ============================================================
    _p("\n" + "=" * 96)
    _p("H. HIGH-L1 (≥1.0) TOKENS — sample of what model does wrong")
    _p("=" * 96)
    bad = valid & (l1 >= 1.0)
    n_bad = int(bad.sum())
    _p(f"\n  {n_bad:,} high-L1 tokens.")
    # Categorize by (target_argmax → pred_argmax)
    _p(f"\n  Confusion breakdown (target argmax → pred argmax) among high-L1:")
    _p(f"  {'flow':<20s}  {'n':>10s}  {'% of high-L1':>12s}  "
       f"{'mean target':<30s}  {'mean pred':<30s}")
    for tk in range(3):
        for pk in range(3):
            m = bad & (argmax_f == tk) & (argmax_p == pk)
            n = int(m.sum())
            if n == 0: continue
            tgt_s = f"({frac[m, 0].mean():.2f}, {frac[m, 1].mean():.2f}, {frac[m, 2].mean():.2f})"
            prd_s = f"({probs[m, 0].mean():.2f}, {probs[m, 1].mean():.2f}, {probs[m, 2].mean():.2f})"
            tag = f"{LABELS[tk]}→{LABELS[pk]}"
            pct = 100 * n / n_bad
            _p(f"  {tag:<20s}  {n:>10,d}  {pct:>11.2f}%  "
               f"{tgt_s:<30s}  {prd_s:<30s}")

    out = "output/analysis/router_cost/distribution_errors_v4c.txt"
    with open(out, "w") as f:
        f.write(buf.getvalue())
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
