"""Compare per-tag specialists against the all-data baseline (same recipe).

Baseline: output/shape_lidar_grad_shape_edge_res/
    trained on all 28k train samples, evaluated on each tag test subset via
    eval_test_<tag>/metrics.json.

Specialists: output/specialist_<tag>/
    trained ONLY on the matching tag subset, evaluated on that tag test
    subset via eval_independent/metrics.json.

For each tag we compare MAE / RMSE / rel at the 0-80m depth range (paper
convention). The delta answers: "does per-tag specialization beat learning
from all data?"
"""
import json
import os


OUT_ROOT = "/workspace/RadarTaco/output"
BASELINE = f"{OUT_ROOT}/shape_lidar_grad_shape_edge_res"
TAGS = ("day_clear", "day_rain", "night_clear", "night_rain")
TAG_SIZES = {  # test subset sizes for context
    "day_clear": 4427, "day_rain": 963, "night_clear": 455, "night_rain": 163,
}
TRAIN_SIZES = {
    "day_clear": 19685, "day_rain": 5060, "night_clear": 2863, "night_rain": 522,
}
RANGES = ("0-50m", "0-70m", "0-80m", "0-100m")


def load(path):
    with open(path) as f:
        return json.load(f)


def main():
    print(f"{'Tag':<14}{'Train n':>9}{'Test n':>8}"
          f"   {'Baseline':>10}    {'Specialist':>12}    {'Δ MAE':>10} {'%':>7}   Verdict")
    print("-" * 100)
    rows = []
    for tag in TAGS:
        b = load(f"{BASELINE}/eval_test_{tag}/metrics.json")["overall"]
        s = load(f"{OUT_ROOT}/specialist_{tag}/eval_independent/metrics.json")["overall"]
        b80 = b["0-80m"]["mae"]
        s80 = s["0-80m"]["mae"]
        delta = s80 - b80
        pct = 100 * delta / b80
        verdict = "specialist worse ❌" if delta > 0 else "specialist better ✅"
        rows.append((tag, b, s, b80, s80, delta, pct))
        print(f"{tag:<14}{TRAIN_SIZES[tag]:>9,}{TAG_SIZES[tag]:>8,}"
              f"   MAE={b80:>6.4f}    MAE={s80:>6.4f}    "
              f"{delta:+.4f}  {pct:+6.1f}%   {verdict}")

    # ---- All ranges, one table per tag ----
    for tag, b, s, b80, s80, d, pct in rows:
        print(f"\n=== {tag} (train n={TRAIN_SIZES[tag]:,}  test n={TAG_SIZES[tag]:,}) ===")
        print(f"  {'range':<7} {'Baseline MAE':>13} {'Specialist MAE':>15} {'Δ':>8} {'%':>7}"
              f"   {'Baseline RMSE':>13} {'Specialist RMSE':>15}")
        for r in RANGES:
            bm = b[r]["mae"]; sm = s[r]["mae"]
            br = b[r]["rmse"]; sr = s[r]["rmse"]
            print(f"  {r:<7} {bm:>13.4f} {sm:>15.4f} {sm-bm:>+8.4f} {100*(sm-bm)/bm:>+6.1f}%"
                  f"   {br:>13.4f} {sr:>15.4f}")

    # ---- Delta by tag rarity ----
    print("\n\n[Rarity vs specialization gain]")
    print(f"  {'Tag':<14}{'Train n':>9}  {'Δ MAE %':>9}")
    for tag, _, _, _, _, _, pct in sorted(rows, key=lambda x: -TRAIN_SIZES[x[0]]):
        print(f"  {tag:<14}{TRAIN_SIZES[tag]:>9,}  {pct:>+8.1f}%")


if __name__ == "__main__":
    main()
