#!/usr/bin/env python3
"""Scatter + binned-mean curve of (lidar/acc)_gt_min and gt_max vs the
matching pred MAE, over all 6019 val samples."""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IN_CSV = os.path.join(ROOT, "output/baseline/eval_val/all_val_lidar_vs_acc.csv")
OUT_PNG = os.path.join(ROOT, "output/baseline/eval_val/analysis_acc/mae_vs_gt_minmax.png")


def panel(ax, x, y, xlabel, ylabel, title_prefix):
    rho, _ = spearmanr(x, y)
    ax.scatter(x, y, s=2, alpha=0.15, color="#1f77b4")
    edges = np.linspace(x.min(), x.max(), 21)
    centers = 0.5 * (edges[:-1] + edges[1:])
    idx = np.digitize(x, edges) - 1
    means = [y[idx == i].mean() if (idx == i).any() else np.nan
             for i in range(len(centers))]
    ax.plot(centers, means, "-", color="red", lw=2, label="binned mean")
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    ax.set_title(f"{title_prefix}  (Spearman ρ={rho:+.3f})", fontsize=10)
    ax.grid(alpha=0.3); ax.legend(fontsize=8, loc="upper right")


def main():
    df = pd.read_csv(IN_CSV).dropna(subset=["lidar_pred_mae", "acc_pred_mae"])
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    panel(axes[0, 0],
          df["lidar_gt_min"].values, df["lidar_pred_mae"].values,
          "lidar_gt_min (m)", "lidar_pred_mae (m)",
          "lidar GT min depth → lidar pred MAE")
    panel(axes[0, 1],
          df["lidar_gt_max"].values, df["lidar_pred_mae"].values,
          "lidar_gt_max (m)", "lidar_pred_mae (m)",
          "lidar GT max depth → lidar pred MAE")
    panel(axes[1, 0],
          df["acc_gt_min"].values, df["acc_pred_mae"].values,
          "acc_gt_min (m)", "acc_pred_mae (m)",
          "acc GT min depth → acc pred MAE")
    panel(axes[1, 1],
          df["acc_gt_max"].values, df["acc_pred_mae"].values,
          "acc_gt_max (m)", "acc_pred_mae (m)",
          "acc GT max depth → acc pred MAE")

    fig.suptitle(f"MAE vs GT depth extrema (n={len(df)} val samples)", y=1.0)
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=120, bbox_inches="tight")
    print(f"wrote {OUT_PNG}")


if __name__ == "__main__":
    sys.exit(main())
