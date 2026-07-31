"""Depth-specialist verification summary.

Compares:
  - baseline           : shape_lidar_grad_shape_edge_res (loss on all depths)
  - specialist NEAR    : loss restricted to GT ∈ [0, 20)
  - specialist MID     : loss restricted to GT ∈ [20, 50)
  - specialist FAR     : loss restricted to GT ∈ [50, 100)
  - oracle combine     : each pixel routed to the specialist whose training
                         bin contains its GT depth (upper-bound of MoE)

Reads eval_independent/metrics.json for each run.
"""
import json


OUT = "/workspace/RadarTaco/output"
RUNS = {
    "baseline":       f"{OUT}/shape_lidar_grad_shape_edge_res/eval_independent/metrics.json",
    "near [0,20)":    f"{OUT}/shape_edge_res_bin_near/eval_independent/metrics.json",
    "mid  [20,50)":   f"{OUT}/shape_edge_res_bin_mid/eval_independent/metrics.json",
    "far  [50,100)":  f"{OUT}/shape_edge_res_bin_far/eval_independent/metrics.json",
    "oracle combine": f"{OUT}/depth_oracle_combine/eval_independent_test_route_dense/metrics.json",
}
RANGES = ("0-10m", "10-20m", "20-30m", "30-40m", "40-50m",
          "50-60m", "60-70m", "70-80m", "80-90m", "90-100m")
BIN_MAP = {
    "near": ("0-10m", "10-20m"),
    "mid":  ("20-30m", "30-40m", "40-50m"),
    "far":  ("50-60m", "60-70m", "70-80m", "80-90m", "90-100m"),
}

runs = {n: json.load(open(p)) for n, p in RUNS.items()}
bl = runs["baseline"]["per_range"]


# ── Table 1: per-range MAE, all runs ────────────────────────────────────
print("[Per-range MAE]  (bold = specialist trained on this range)")
print(f"  {'range':<8} {'baseline':>10} {'near':>10} {'mid':>10} {'far':>10} {'oracle':>10}")
print("  " + "-" * 66)
NEAR_R = BIN_MAP["near"]; MID_R = BIN_MAP["mid"]; FAR_R = BIN_MAP["far"]
for r in RANGES:
    vals = {}
    for name in ("baseline", "near [0,20)", "mid  [20,50)", "far  [50,100)", "oracle combine"):
        vals[name] = runs[name]["per_range"][r]["mae"]
    def mk(v, own):
        return f"*{v:>9.4f}" if own else f"{v:>10.4f}"
    print(f"  {r:<8} "
          f"{vals['baseline']:>10.4f} "
          f"{mk(vals['near [0,20)'], r in NEAR_R):>10} "
          f"{mk(vals['mid  [20,50)'], r in MID_R):>10} "
          f"{mk(vals['far  [50,100)'], r in FAR_R):>10} "
          f"{vals['oracle combine']:>10.4f}")


# ── Table 2: specialist vs baseline in own bin ─────────────────────────
print("\n\n[Specialist vs baseline — each in its OWN training bin]")
for name, spec_key, rlist in [
    ("NEAR", "near [0,20)",  NEAR_R),
    ("MID",  "mid  [20,50)", MID_R),
    ("FAR",  "far  [50,100)", FAR_R),
]:
    spec = runs[spec_key]["per_range"]
    print(f"\n  {name} bin ({rlist[0]} — {rlist[-1]})")
    print(f"    {'range':<8} {'baseline':>10} {'specialist':>12} {'Δ':>10} {'%':>7}")
    for r in rlist:
        b = bl[r]["mae"]; s = spec[r]["mae"]
        print(f"    {r:<8} {b:>10.4f} {s:>12.4f} {s-b:>+10.4f} {100*(s-b)/b:>+6.1f}%")
    b_avg = sum(bl[r]["mae"] for r in rlist) / len(rlist)
    s_avg = sum(spec[r]["mae"] for r in rlist) / len(rlist)
    print(f"    {'avg':<8} {b_avg:>10.4f} {s_avg:>12.4f} {s_avg-b_avg:>+10.4f} "
          f"{100*(s_avg-b_avg)/b_avg:>+6.1f}%")


# ── Table 3: oracle combine vs baseline aggregate ───────────────────────
print("\n\n[Oracle combine vs baseline]  (upper bound of depth-routed MoE)")
print(f"  {'range':<8} {'baseline':>10} {'oracle':>10} {'Δ':>10} {'%':>7}")
for agg in ("0-50m", "0-70m", "0-80m", "0-100m"):
    b = runs["baseline"]["overall"][agg]["mae"]
    o = runs["oracle combine"]["overall"][agg]["mae"]
    print(f"  {agg:<8} {b:>10.4f} {o:>10.4f} {o-b:>+10.4f} {100*(o-b)/b:>+6.1f}%")


# ── Table 4: catastrophic degradation OUTSIDE own bin ───────────────────
print("\n\n[Specialist OUTSIDE its bin]  (no supervision → catastrophic)")
for name, spec_key, own in [
    ("NEAR", "near [0,20)",  set(NEAR_R)),
    ("MID",  "mid  [20,50)", set(MID_R)),
    ("FAR",  "far  [50,100)", set(FAR_R)),
]:
    spec = runs[spec_key]["per_range"]
    print(f"\n  {name} specialist:")
    for r in RANGES:
        if r in own: continue
        s = spec[r]["mae"]; b = bl[r]["mae"]
        print(f"    {r:<8} baseline={b:>8.4f}  specialist={s:>8.4f}  Δ={s-b:>+7.2f} "
              f"({100*(s-b)/b:>+6.1f}%)")
