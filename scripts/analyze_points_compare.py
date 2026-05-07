#!/usr/bin/env python3
"""Analyze points_compare/all.csv from two angles:

(1) Aggregate tables — pred error binned by gt_depth and radar_pix_dist;
    per-sample summary (depth distribution, radar coverage, radar's own
    error). Saved as CSVs + printed.

(2) Spatial 2D plots — per-sample image-space scatter of GT points
    colored by pred_abs_err, with radar points overlaid; plus error vs
    gt_depth and error vs radar_pix_dist scatter/binned curves for
    best vs worst.

Output: output/baseline/eval_test/analysis/
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IN_CSV = os.path.join(ROOT, "output/baseline/eval_test/points_compare/all.csv")
OUT_DIR = os.path.join(ROOT, "output/baseline/eval_test/analysis")
os.makedirs(OUT_DIR, exist_ok=True)

# nuScenes CAM_FRONT image size — used to fix axes for spatial plots.
IMG_W, IMG_H = 1600, 900


def aggregate_tables(df: pd.DataFrame):
    # Per-sample summary
    summary = df.groupby(["label", "sample_id"]).agg(
        n_gt=("gt_depth", "size"),
        gt_depth_mean=("gt_depth", "mean"),
        gt_depth_p90=("gt_depth", lambda s: float(np.percentile(s, 90))),
        pred_mae=("pred_abs_err", "mean"),
        pred_rmse=("pred_abs_err", lambda s: float(np.sqrt((s ** 2).mean()))),
        radar_pix_dist_mean=("radar_pix_dist", "mean"),
        radar_pix_dist_median=("radar_pix_dist", "median"),
        radar_depth_mae=("radar_depth_abs_err", "mean"),
    ).reset_index()
    summary.to_csv(os.path.join(OUT_DIR, "per_sample_summary.csv"), index=False)
    print("=== per-sample summary ===")
    print(summary.to_string(index=False))

    # Pred error binned by gt_depth (overall + per-sample)
    depth_bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 100]
    df["gt_depth_bin"] = pd.cut(df["gt_depth"], bins=depth_bins, right=False,
                                include_lowest=True)
    by_depth = df.groupby(["label", "gt_depth_bin"], observed=True).agg(
        n=("pred_abs_err", "size"),
        pred_mae=("pred_abs_err", "mean"),
        pred_rmse=("pred_abs_err", lambda s: float(np.sqrt((s ** 2).mean()))),
        radar_pix_dist_mean=("radar_pix_dist", "mean"),
        radar_depth_mae=("radar_depth_abs_err", "mean"),
    ).reset_index()
    by_depth["gt_depth_bin"] = by_depth["gt_depth_bin"].astype(str)
    by_depth.to_csv(os.path.join(OUT_DIR, "by_gt_depth_bin.csv"), index=False)
    print("\n=== pred error by gt_depth bin (per sample) ===")
    print(by_depth.to_string(index=False))

    # Pred error binned by radar_pix_dist (does pred get better when radar is closer?)
    rd_bins = [0, 25, 50, 100, 200, 400, 1000]
    df["radar_pix_dist_bin"] = pd.cut(df["radar_pix_dist"], bins=rd_bins,
                                      right=False, include_lowest=True)
    by_rd = df.groupby(["label", "radar_pix_dist_bin"], observed=True).agg(
        n=("pred_abs_err", "size"),
        pred_mae=("pred_abs_err", "mean"),
        gt_depth_mean=("gt_depth", "mean"),
    ).reset_index()
    by_rd["radar_pix_dist_bin"] = by_rd["radar_pix_dist_bin"].astype(str)
    by_rd.to_csv(os.path.join(OUT_DIR, "by_radar_pix_dist_bin.csv"), index=False)
    print("\n=== pred error by radar_pix_dist bin (per sample) ===")
    print(by_rd.to_string(index=False))


def spatial_plots(df: pd.DataFrame):
    """One figure per sample: image-space scatter of GT colored by pred_abs_err,
    radar points overlaid as black crosses sized by their own depth error."""
    labels = ["best_1", "best_2", "worst_1", "worst_2"]
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    vmax = 5.0  # color cap for pred_abs_err — keeps best-samples readable
    for ax, label in zip(axes.flat, labels):
        sub = df[df["label"] == label]
        sc = ax.scatter(sub["x"], sub["y"], c=sub["pred_abs_err"],
                        cmap="inferno", s=3, vmin=0, vmax=vmax)
        # radar points: dedup (same nearest_radar_x/y appears for many gt rows)
        rad = sub.dropna(subset=["nearest_radar_x"]).drop_duplicates(
            subset=["nearest_radar_x", "nearest_radar_y"])
        ax.scatter(rad["nearest_radar_x"], rad["nearest_radar_y"],
                   facecolors="none", edgecolors="cyan", s=40, linewidths=1.0,
                   label=f"radar (n={len(rad)})")
        ax.set_xlim(0, IMG_W); ax.set_ylim(IMG_H, 0)
        ax.set_aspect("equal")
        mae = sub["pred_abs_err"].mean()
        ax.set_title(f"{label}  MAE={mae:.2f}m  n_gt={len(sub)}")
        ax.legend(loc="lower right", fontsize=8)
        plt.colorbar(sc, ax=ax, fraction=0.03, label="pred_abs_err (m, capped 5)")
    fig.suptitle("GT pixels colored by pred error; cyan = radar projections", y=1.0)
    fig.tight_layout()
    p = os.path.join(OUT_DIR, "spatial_pred_error.png")
    fig.savefig(p, dpi=120, bbox_inches="tight"); plt.close(fig)
    print(f"\nwrote {p}")


def relationship_plots(df: pd.DataFrame):
    """pred_abs_err vs gt_depth and vs radar_pix_dist — best/worst overlaid."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = {"best_1": "#1f77b4", "best_2": "#17becf",
              "worst_1": "#d62728", "worst_2": "#ff7f0e"}

    # (a) error vs gt_depth — binned mean
    depth_edges = np.linspace(0, 100, 21)
    centers = 0.5 * (depth_edges[:-1] + depth_edges[1:])
    ax = axes[0]
    for label in ["best_1", "best_2", "worst_1", "worst_2"]:
        sub = df[df["label"] == label]
        idx = np.digitize(sub["gt_depth"], depth_edges) - 1
        means = [sub["pred_abs_err"][idx == i].mean() if (idx == i).any()
                 else np.nan for i in range(len(centers))]
        ax.plot(centers, means, "-o", color=colors[label], label=label, ms=4)
    ax.set_xlabel("gt_depth (m)"); ax.set_ylabel("pred MAE (m)")
    ax.set_title("Pred error vs GT depth"); ax.grid(alpha=0.3); ax.legend()

    # (b) error vs radar_pix_dist — binned mean
    rd_edges = np.array([0, 25, 50, 75, 100, 150, 200, 300, 400, 600, 1000])
    centers2 = 0.5 * (rd_edges[:-1] + rd_edges[1:])
    ax = axes[1]
    for label in ["best_1", "best_2", "worst_1", "worst_2"]:
        sub = df[df["label"] == label].dropna(subset=["radar_pix_dist"])
        idx = np.digitize(sub["radar_pix_dist"], rd_edges) - 1
        means = [sub["pred_abs_err"][idx == i].mean() if (idx == i).any()
                 else np.nan for i in range(len(centers2))]
        ax.plot(centers2, means, "-o", color=colors[label], label=label, ms=4)
    ax.set_xlabel("nearest radar pix dist (px)"); ax.set_ylabel("pred MAE (m)")
    ax.set_title("Pred error vs distance to nearest radar"); ax.grid(alpha=0.3); ax.legend()

    fig.tight_layout()
    p = os.path.join(OUT_DIR, "error_vs_depth_and_radar.png")
    fig.savefig(p, dpi=120, bbox_inches="tight"); plt.close(fig)
    print(f"wrote {p}")

    # (c) gt depth distribution histograms — best vs worst
    fig, ax = plt.subplots(figsize=(8, 4))
    for label in ["best_1", "best_2", "worst_1", "worst_2"]:
        sub = df[df["label"] == label]
        ax.hist(sub["gt_depth"], bins=np.linspace(0, 100, 41),
                histtype="step", linewidth=1.5, color=colors[label], label=label)
    ax.set_xlabel("gt_depth (m)"); ax.set_ylabel("count")
    ax.set_title("GT depth distribution per sample"); ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    p = os.path.join(OUT_DIR, "gt_depth_hist.png")
    fig.savefig(p, dpi=120, bbox_inches="tight"); plt.close(fig)
    print(f"wrote {p}")


def main():
    df = pd.read_csv(IN_CSV)
    aggregate_tables(df)
    spatial_plots(df)
    relationship_plots(df)
    print(f"\nall outputs → {OUT_DIR}")


if __name__ == "__main__":
    sys.exit(main())
