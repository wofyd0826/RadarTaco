#!/usr/bin/env python3
"""Per-sample radar depth accuracy, restricted to GT pixels within 5 px of
the nearest radar projection (a colocated comparison, not the misleading
nearest-neighbor average over all GT).
"""
import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IN_CSV = os.path.join(ROOT, "output/baseline/eval_test/points_compare/all.csv")
OUT_CSV = os.path.join(ROOT, "output/baseline/eval_test/analysis/radar_accuracy_within5px.csv")
RADIUS_PX = 5.0


def main():
    df = pd.read_csv(IN_CSV)
    df = df.dropna(subset=["radar_pix_dist"])
    near = df[df["radar_pix_dist"] <= RADIUS_PX].copy()

    per_sample = (
        df.groupby(["label", "sample_id"])
        .agg(n_gt_total=("gt_depth", "size"))
        .reset_index()
    )
    near_summary = (
        near.groupby(["label", "sample_id"])
        .agg(
            n_gt_within5px=("gt_depth", "size"),
            radar_depth_mae=("radar_depth_abs_err", "mean"),
            radar_depth_rmse=("radar_depth_abs_err",
                              lambda s: float((s ** 2).mean() ** 0.5)),
            gt_depth_mean=("gt_depth", "mean"),
            radar_depth_mean=("nearest_radar_depth", "mean"),
            pred_mae_same=("pred_abs_err", "mean"),
        )
        .reset_index()
    )
    out = per_sample.merge(near_summary, on=["label", "sample_id"], how="left")
    out["coverage_pct"] = 100.0 * out["n_gt_within5px"].fillna(0) / out["n_gt_total"]
    out = out[[
        "label", "sample_id",
        "n_gt_total", "n_gt_within5px", "coverage_pct",
        "gt_depth_mean", "radar_depth_mean",
        "radar_depth_mae", "radar_depth_rmse",
        "pred_mae_same",
    ]]
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    out.to_csv(OUT_CSV, index=False)
    print(out.to_string(index=False))
    print(f"\nwrote {OUT_CSV}")


if __name__ == "__main__":
    sys.exit(main())
