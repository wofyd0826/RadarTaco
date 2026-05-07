#!/usr/bin/env python3
"""Analyze the val 10-best / 10-worst points_compare_acc/all.csv.

Outputs (in output/baseline/eval_val/analysis_acc/):
  - per_sample_summary.csv
  - by_gt_depth_bin.csv
  - by_radar_pix_dist_bin.csv
  - radar_accuracy_within5px.csv     (per-radar, GT pixels within 5 px)
  - error_vs_depth.png               (best-group vs worst-group binned curves)
  - error_vs_radar_dist.png
  - gt_depth_hist.png
  - spatial_grid_best.png            (4×3 panels of best samples)
  - spatial_grid_worst.png
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVAL_DIR = os.path.join(ROOT, "output/baseline/eval_val")
IN_CSV = os.path.join(EVAL_DIR, "points_compare_acc/all.csv")
OUT_DIR = os.path.join(EVAL_DIR, "analysis_acc")
RADIUS_PX = 5.0
IMG_W, IMG_H = 1600, 900


def is_best(label: str) -> bool: return label.startswith("best_")
def is_worst(label: str) -> bool: return label.startswith("worst_")


def aggregate_tables(df: pd.DataFrame):
    summary = df.groupby(["label", "sample_id"]).agg(
        n_gt=("gt_depth", "size"),
        gt_depth_mean=("gt_depth", "mean"),
        gt_depth_p90=("gt_depth", lambda s: float(np.percentile(s, 90))),
        pred_mae=("pred_abs_err", "mean"),
        pred_rmse=("pred_abs_err", lambda s: float(np.sqrt((s ** 2).mean()))),
        radar_pix_dist_median=("radar_pix_dist", "median"),
    ).reset_index().sort_values(["label"])
    summary.to_csv(os.path.join(OUT_DIR, "per_sample_summary.csv"), index=False)
    print("=== per-sample summary ===")
    print(summary.to_string(index=False))

    df["group"] = np.where(df["label"].str.startswith("best_"), "best", "worst")
    depth_bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 100]
    df["gt_depth_bin"] = pd.cut(df["gt_depth"], bins=depth_bins, right=False, include_lowest=True)
    by_depth = df.groupby(["group", "gt_depth_bin"], observed=True).agg(
        n=("pred_abs_err", "size"),
        pred_mae=("pred_abs_err", "mean"),
        pred_rmse=("pred_abs_err", lambda s: float(np.sqrt((s ** 2).mean()))),
    ).reset_index()
    by_depth["gt_depth_bin"] = by_depth["gt_depth_bin"].astype(str)
    by_depth.to_csv(os.path.join(OUT_DIR, "by_gt_depth_bin.csv"), index=False)
    print("\n=== pred error by gt_depth bin (group means) ===")
    print(by_depth.to_string(index=False))

    rd_bins = [0, 25, 50, 100, 200, 400, 1000]
    df["radar_pix_dist_bin"] = pd.cut(df["radar_pix_dist"], bins=rd_bins, right=False,
                                      include_lowest=True)
    by_rd = df.groupby(["group", "radar_pix_dist_bin"], observed=True).agg(
        n=("pred_abs_err", "size"),
        pred_mae=("pred_abs_err", "mean"),
        gt_depth_mean=("gt_depth", "mean"),
    ).reset_index()
    by_rd["radar_pix_dist_bin"] = by_rd["radar_pix_dist_bin"].astype(str)
    by_rd.to_csv(os.path.join(OUT_DIR, "by_radar_pix_dist_bin.csv"), index=False)
    print("\n=== pred error by radar_pix_dist bin (group means) ===")
    print(by_rd.to_string(index=False))


def radar_within5(df: pd.DataFrame):
    near = df.dropna(subset=["radar_pix_dist"])
    near = near[near["radar_pix_dist"] <= RADIUS_PX]
    per_sample = df.groupby(["label", "sample_id"]).agg(n_gt_total=("gt_depth", "size")).reset_index()
    rsum = near.groupby(["label", "sample_id"]).agg(
        n_within5px=("gt_depth", "size"),
        gt_depth_mean=("gt_depth", "mean"),
        radar_depth_mean=("nearest_radar_depth", "mean"),
        radar_depth_mae=("radar_depth_abs_err", "mean"),
        radar_depth_rmse=("radar_depth_abs_err",
                          lambda s: float((s ** 2).mean() ** 0.5)),
        pred_mae_same=("pred_abs_err", "mean"),
    ).reset_index()
    out = per_sample.merge(rsum, on=["label", "sample_id"], how="left")
    out["coverage_pct"] = 100.0 * out["n_within5px"].fillna(0) / out["n_gt_total"]
    out = out.sort_values("label")
    out.to_csv(os.path.join(OUT_DIR, "radar_accuracy_within5px.csv"), index=False)
    print("\n=== radar accuracy within 5 px (per sample, depth_acc GT) ===")
    print(out.to_string(index=False))

    # group-level rollup
    near["group"] = np.where(near["label"].str.startswith("best_"), "best", "worst")
    g = near.groupby("group").agg(
        n=("gt_depth", "size"),
        radar_depth_mae=("radar_depth_abs_err", "mean"),
        pred_mae_same=("pred_abs_err", "mean"),
        gt_depth_mean=("gt_depth", "mean"),
    ).reset_index()
    print("\n=== group rollup (radar within 5 px) ===")
    print(g.to_string(index=False))


def relationship_plots(df: pd.DataFrame):
    df["group"] = np.where(df["label"].str.startswith("best_"), "best", "worst")
    colors = {"best": "#1f77b4", "worst": "#d62728"}

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    depth_edges = np.linspace(0, 100, 21)
    centers = 0.5 * (depth_edges[:-1] + depth_edges[1:])
    ax = axes[0]
    for g in ("best", "worst"):
        sub = df[df["group"] == g]
        idx = np.digitize(sub["gt_depth"], depth_edges) - 1
        means = [sub["pred_abs_err"][idx == i].mean() if (idx == i).any() else np.nan
                 for i in range(len(centers))]
        ax.plot(centers, means, "-o", color=colors[g], label=f"{g} (n={len(sub)})", ms=4)
    ax.set_xlabel("gt_depth (m)"); ax.set_ylabel("pred MAE (m)")
    ax.set_title("Pred error vs GT depth (best vs worst groups)")
    ax.grid(alpha=0.3); ax.legend()

    rd_edges = np.array([0, 25, 50, 75, 100, 150, 200, 300, 400, 600, 1000])
    centers2 = 0.5 * (rd_edges[:-1] + rd_edges[1:])
    ax = axes[1]
    for g in ("best", "worst"):
        sub = df[df["group"] == g].dropna(subset=["radar_pix_dist"])
        idx = np.digitize(sub["radar_pix_dist"], rd_edges) - 1
        means = [sub["pred_abs_err"][idx == i].mean() if (idx == i).any() else np.nan
                 for i in range(len(centers2))]
        ax.plot(centers2, means, "-o", color=colors[g], label=g, ms=4)
    ax.set_xlabel("nearest radar pix dist (px)"); ax.set_ylabel("pred MAE (m)")
    ax.set_title("Pred error vs distance to nearest radar")
    ax.grid(alpha=0.3); ax.legend()

    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "error_vs_depth_and_radar.png"), dpi=120,
                bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    bins = np.linspace(0, 100, 41)
    for g in ("best", "worst"):
        sub = df[df["group"] == g]
        ax.hist(sub["gt_depth"], bins=bins, histtype="step", linewidth=1.5,
                color=colors[g], label=f"{g} (n={len(sub)})")
    ax.set_xlabel("gt_depth (m)"); ax.set_ylabel("count")
    ax.set_title("GT depth distribution (best vs worst groups)")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "gt_depth_hist.png"), dpi=120, bbox_inches="tight")
    plt.close(fig)


def spatial_grid(df: pd.DataFrame, group: str, vmax: float):
    labels = sorted([lab for lab in df["label"].unique() if lab.startswith(f"{group}_")])
    if not labels:
        return
    cols = 5
    rows = int(np.ceil(len(labels) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 2.5 * rows),
                             squeeze=False)
    for ax, label in zip(axes.flat, labels):
        sub = df[df["label"] == label]
        sc = ax.scatter(sub["x"], sub["y"], c=sub["pred_abs_err"],
                        cmap="inferno", s=2, vmin=0, vmax=vmax)
        rad = sub.dropna(subset=["nearest_radar_x"]).drop_duplicates(
            subset=["nearest_radar_x", "nearest_radar_y"])
        ax.scatter(rad["nearest_radar_x"], rad["nearest_radar_y"],
                   facecolors="none", edgecolors="cyan", s=20, linewidths=0.7)
        ax.set_xlim(0, IMG_W); ax.set_ylim(IMG_H, 0); ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        mae = sub["pred_abs_err"].mean()
        ax.set_title(f"{label}  MAE={mae:.2f}m", fontsize=9)
    for ax in axes.flat[len(labels):]:
        ax.axis("off")
    fig.suptitle(f"{group} group — GT pixels colored by pred error (cap {vmax} m); cyan = radar",
                 y=1.0)
    fig.colorbar(sc, ax=axes.ravel().tolist(), fraction=0.02, label="pred_abs_err (m)")
    p = os.path.join(OUT_DIR, f"spatial_grid_{group}.png")
    fig.savefig(p, dpi=110, bbox_inches="tight"); plt.close(fig)
    print(f"wrote {p}")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    df = pd.read_csv(IN_CSV)
    aggregate_tables(df)
    radar_within5(df)
    relationship_plots(df)
    spatial_grid(df, "best", vmax=2.0)
    spatial_grid(df, "worst", vmax=15.0)
    print(f"\nall outputs → {OUT_DIR}")


if __name__ == "__main__":
    sys.exit(main())
