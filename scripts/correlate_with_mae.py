#!/usr/bin/env python3
"""For all 6019 val samples, find which per-sample metrics correlate with
prediction MAE (Spearman, robust to outliers; Pearson for reference).
Joins all_val_lidar_vs_acc.csv with the official per_sample.csv to also
test against overall/0-80m/mae.
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVAL_DIR = os.path.join(ROOT, "output/baseline/eval_val")
A_CSV = os.path.join(EVAL_DIR, "all_val_lidar_vs_acc.csv")
B_CSV = os.path.join(EVAL_DIR, "per_sample.csv")
OUT_CSV = os.path.join(EVAL_DIR, "analysis_acc/correlations_with_mae.csv")

PREDICTORS = [
    "is_night",
    "lidar_n_gt", "lidar_gt_min", "lidar_gt_max", "lidar_gt_range",
    "lidar_n", "lidar_radar_mae",
    "acc_n_gt", "acc_gt_min", "acc_gt_max", "acc_gt_range",
    "acc_n", "acc_radar_mae",
    # also derived: ratio of how many of the radar's neighborhoods land on GT
    "lidar_n_per_gt", "acc_n_per_gt",
]

TARGETS = [
    "lidar_pred_mae",
    "acc_pred_mae",
    "overall/0-80m/mae",     # the official MAE from per_sample.csv
]


def main():
    a = pd.read_csv(A_CSV)
    b = pd.read_csv(B_CSV)[["sample_id", "overall/0-80m/mae"]]
    df = a.merge(b, on="sample_id", how="left")

    df["lidar_gt_range"] = df["lidar_gt_max"] - df["lidar_gt_min"]
    df["acc_gt_range"] = df["acc_gt_max"] - df["acc_gt_min"]
    df["lidar_n_per_gt"] = df["lidar_n"] / df["lidar_n_gt"].clip(lower=1)
    df["acc_n_per_gt"] = df["acc_n"] / df["acc_n_gt"].clip(lower=1)
    df["is_night"] = df["is_night"].astype(int)

    rows = []
    for tgt in TARGETS:
        for p in PREDICTORS:
            sub = df[[p, tgt]].dropna()
            if len(sub) < 10:
                continue
            sr, sp_pv = spearmanr(sub[p], sub[tgt])
            pr, pp_pv = pearsonr(sub[p], sub[tgt])
            rows.append({
                "target": tgt, "predictor": p, "n": len(sub),
                "spearman_rho": sr, "spearman_p": sp_pv,
                "pearson_r": pr, "pearson_p": pp_pv,
            })
    out = pd.DataFrame(rows)
    out["abs_rho"] = out["spearman_rho"].abs()
    out = out.sort_values(["target", "abs_rho"], ascending=[True, False])
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    out.drop(columns=["abs_rho"]).to_csv(OUT_CSV, index=False)

    for tgt in TARGETS:
        sub = out[out["target"] == tgt]
        print(f"\n=== correlation with {tgt} (n={sub['n'].iloc[0]}) ===")
        print(f"{'predictor':<22} {'spearman_rho':>14} {'spearman_p':>14} {'pearson_r':>11}")
        for _, r in sub.iterrows():
            mark = ""
            if abs(r["spearman_rho"]) >= 0.5:   mark = "  ***"
            elif abs(r["spearman_rho"]) >= 0.3: mark = "  **"
            elif abs(r["spearman_rho"]) >= 0.1: mark = "  *"
            print(f"{r['predictor']:<22} {r['spearman_rho']:>14.4f} "
                  f"{r['spearman_p']:>14.2e} {r['pearson_r']:>11.4f}{mark}")
    print(f"\nwrote {OUT_CSV}")


if __name__ == "__main__":
    sys.exit(main())
