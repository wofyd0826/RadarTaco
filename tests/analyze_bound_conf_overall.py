"""Compare mean bound / max_prob across token groups."""
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, ROOT)


def bin_of(depth):
    return np.clip(
        np.digitize(depth, [20.0, 50.0]), 0, 2).astype(np.int64)


def main():
    d = np.load("output/analysis/router_cost/v4c_stage2_per_mode.npz",
                allow_pickle=True)
    err_n = d["err_normal"]
    err_s = d["err_soft"]
    vc = d["valid_pixel_count"]
    gt = d["gt"]
    pred = d["actual_pred"]
    bound = d["bound"]
    depth = d["depth"]
    dump = np.load("output/analysis/router_cost/v4c_stage2_per_token_dump.npz",
                   allow_pickle=True)
    max_prob = dump["probs"].max(axis=1)

    valid = np.isfinite(err_n) & np.isfinite(err_s) & (vc > 0)
    correct = valid & (pred == gt)
    misroute = valid & (pred != gt)

    # Misroute distance calc (same as prior analysis)
    edges = np.array([[0.0, 20.0], [20.0, 50.0], [50.0, 100.0]])
    lo = edges[pred, 0]
    hi = edges[pred, 1]
    md = np.maximum(0.0, np.maximum(lo - depth, depth - hi))

    aptly = misroute & (md < 2.0)
    badly = misroute & (md >= 10.0)
    mid_mis = misroute & (md >= 2.0) & (md < 10.0)

    def stats(mask, name):
        n = int(mask.sum())
        if n == 0:
            print(f"  {name:<28s}  n=0")
            return
        mb = float(bound[mask].mean())
        mp = float(max_prob[mask].mean())
        # standard deviation for context
        sb = float(bound[mask].std())
        sp = float(max_prob[mask].std())
        pct = 100 * n / valid.sum()
        print(f"  {name:<28s}  n={n:>10,}  ({pct:>5.2f}%)  "
              f"bound: {mb:.3f} ± {sb:.3f}   "
              f"max_prob: {mp:.3f} ± {sp:.3f}")

    print("=" * 96)
    print("Mean bound / max_prob across token groups (v4c stage2)")
    print("=" * 96)

    print("\n[Overall]")
    stats(valid, "ALL valid")

    print("\n[Routing outcome]")
    stats(correct, "CORRECT (pred == gt)")
    stats(misroute, "MISROUTE (pred != gt)")
    stats(aptly, "  MISROUTE — aptly (<2m)")
    stats(mid_mis, "  MISROUTE — 2-10m")
    stats(badly, "  MISROUTE — badly (>=10m)")

    print("\n[By target bound bucket, over ALL valid]")
    for lo_b, hi_b, name in [(0.0, 0.05, "PURE     "),
                              (0.05, 0.15, "SLIGHT   "),
                              (0.15, 0.30, "MODERATE "),
                              (0.30, 0.70, "HIGH     ")]:
        m = valid & (bound >= lo_b) & (bound < hi_b)
        stats(m, name)

    print("\n[By gt bin, over ALL valid]")
    stats(valid & (gt == 0), "gt = near [0,20)")
    stats(valid & (gt == 1), "gt = mid  [20,50)")
    stats(valid & (gt == 2), "gt = far  [50,100)")

    print("\n[Correct vs misroute within each gt bin]")
    for g, name in [(0, "near"), (1, "mid"), (2, "far")]:
        stats(valid & (gt == g) & correct, f"gt={name}  CORRECT")
        stats(valid & (gt == g) & misroute, f"gt={name}  MISROUTE")


if __name__ == "__main__":
    main()
