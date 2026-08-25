"""Diagnose depth_acc as a stripe-suppression / metric supervision source.

Quantifies:
  • Coverage statistics — overall, per-row, per-depth-range
  • Acc vs single-LiDAR disagreement — proxy for dynamic-object ghost
  • Acc vs dense_gt (DAv2) disagreement — proxy for DAv2 ratio error
  • Vertical-pair density of acc — feasibility of ∂y direct supervision
  • Stripe-prone region coverage — how much of the LiDAR-row dilation
    band depth_acc actually fills (= how much it kills the
    supervision-source mismatch that creates stripes)

Outputs (default at output/diagnose_depth_acc/):
  • summary.txt
  • coverage_per_row.png
  • disagreement_distributions.png
  • viz_<idx>.png  (N panel comparisons for sample inspection)
"""
import argparse
import os
import sys
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scipy.ndimage as nd

DATA_ROOT = "/data/public/nuScenes/derived"
# train split has derived files (test split is scene_10000+, no derived data).
DEFAULT_SPLIT = f"{DATA_ROOT}/splits/train.txt"
OUT_DIR = "/workspace/RadarTaco/output/diagnose_depth_acc"


def load_depth(path):
    img = np.array(Image.open(path))
    if img.dtype == np.uint16:
        return img.astype(np.float32) / 256.0
    return img.astype(np.float32)


def main(n_samples, n_viz, split_path):
    os.makedirs(OUT_DIR, exist_ok=True)

    with open(split_path) as f:
        # split files are tab-separated; first column is sample_id
        all_samples = [l.strip().split("\t")[0] for l in f if l.strip()]
    samples = all_samples[:n_samples]
    print(f"split: {split_path} ({len(all_samples)} total, using first {len(samples)})")

    cov = {k: [] for k in ("lidar", "acc", "interp", "dense")}
    diff_acc_vs_lidar = []
    diff_acc_vs_dense = []
    diff_dense_vs_lidar = []  # for direct comparison with mask experiments
    acc_row_dens_sum = np.zeros(900)
    n_row_samples = 0
    vp_acc = []           # vertical pair valid fraction (per sample)
    stripe_band_acc = []  # acc coverage inside lidar-row ±4 dilation
    stripe_band_lidar = [] # baseline: lidar coverage inside the same band

    # depth-range bucketed coverage of acc
    range_bins = [(0, 10), (10, 20), (20, 30), (30, 50), (50, 70), (70, 100)]
    range_total_acc = {r: 0 for r in range_bins}
    range_total_pix = {r: 0 for r in range_bins}

    viz_indices = list(range(min(n_viz, len(samples))))
    viz_data = []

    used = 0
    for i, sid in enumerate(samples):
        try:
            lidar  = load_depth(f"{DATA_ROOT}/depth_lidar/{sid}.png")
            acc    = load_depth(f"{DATA_ROOT}/depth_acc/{sid}.png")
            interp = load_depth(f"{DATA_ROOT}/depth_interp/{sid}.png")
            dense  = load_depth(f"{DATA_ROOT}/depth_refined_grid/{sid}.png")
        except (FileNotFoundError, OSError):
            continue
        used += 1
        m_lidar, m_acc, m_interp, m_dense = (lidar > 0.1, acc > 0.1, interp > 0.1, dense > 0.1)

        cov["lidar"].append(m_lidar.mean());  cov["acc"].append(m_acc.mean())
        cov["interp"].append(m_interp.mean()); cov["dense"].append(m_dense.mean())

        acc_row_dens_sum += m_acc.mean(axis=1)
        n_row_samples += 1

        both_al = m_lidar & m_acc
        if both_al.any():
            diff_acc_vs_lidar.append(np.abs(acc[both_al] - lidar[both_al]))
        both_ad = m_acc & m_dense
        if both_ad.any():
            diff_acc_vs_dense.append(np.abs(acc[both_ad] - dense[both_ad]))
        both_dl = m_dense & m_lidar
        if both_dl.any():
            diff_dense_vs_lidar.append(np.abs(dense[both_dl] - lidar[both_dl]))

        # vertical pair density of acc
        vp = m_acc[:-1, :] & m_acc[1:, :]
        vp_acc.append(vp.mean())

        # stripe-prone region: ±4 row dilation of LiDAR mask
        lidar_band = nd.maximum_filter(m_lidar.astype(np.uint8), size=(9, 1)).astype(bool)
        if lidar_band.any():
            band_n = lidar_band.sum()
            stripe_band_acc.append((m_acc & lidar_band).sum() / band_n)
            stripe_band_lidar.append((m_lidar & lidar_band).sum() / band_n)

        # depth-range bucketed acc coverage (denominator = pixels with ANY GT)
        # Use acc itself as the depth proxy for binning
        for r in range_bins:
            in_range = (acc >= r[0]) & (acc < r[1]) & m_acc
            # density relative to dense GT validity in that range
            dense_in_range = (dense >= r[0]) & (dense < r[1]) & m_dense
            range_total_acc[r] += in_range.sum()
            range_total_pix[r] += dense_in_range.sum()

        if i in viz_indices:
            viz_data.append((sid, lidar, acc, interp, dense, m_lidar, m_acc, m_dense, lidar_band))

    print(f"\nused {used}/{len(samples)} samples (others missing files)")

    # -------- statistics --------
    def _q(arr, qs=(0.5, 0.9, 0.99)):
        a = np.concatenate(arr)
        return {q: np.quantile(a, q) for q in qs}, a.mean(), a.max()

    print("\n=== Coverage (%) ===")
    for k in ("lidar", "acc", "interp", "dense"):
        v = np.array(cov[k]) * 100
        print(f"  {k:<7}  mean={v.mean():>6.2f}  std={v.std():>5.2f}  range=[{v.min():.2f}, {v.max():.2f}]")

    qs, mean_v, max_v = _q(diff_acc_vs_lidar)
    print(f"\n=== |acc − lidar| (m), pairs where both valid — DYNAMIC GHOST proxy ===")
    print(f"  mean={mean_v:.3f}  q50={qs[0.5]:.3f}  q90={qs[0.9]:.3f}  q99={qs[0.99]:.3f}  max={max_v:.2f}")
    a_all = np.concatenate(diff_acc_vs_lidar)
    for th in (0.5, 1.0, 2.0, 5.0, 10.0):
        print(f"    fraction with |acc-lidar| > {th:>4.1f}m : {(a_all > th).mean()*100:>5.2f}%")

    qs, mean_v, max_v = _q(diff_acc_vs_dense)
    print(f"\n=== |acc − dense_gt| (m), pairs where both valid — DAv2 ERROR proxy ===")
    print(f"  mean={mean_v:.3f}  q50={qs[0.5]:.3f}  q90={qs[0.9]:.3f}  q99={qs[0.99]:.3f}  max={max_v:.2f}")
    a_all = np.concatenate(diff_acc_vs_dense)
    for th in (0.5, 1.0, 2.0, 5.0, 10.0):
        print(f"    fraction with |acc-dense| > {th:>4.1f}m : {(a_all > th).mean()*100:>5.2f}%")

    qs, mean_v, max_v = _q(diff_dense_vs_lidar)
    print(f"\n=== |dense_gt − lidar| (m), pairs where both valid — BASELINE comparison ===")
    print(f"  mean={mean_v:.3f}  q50={qs[0.5]:.3f}  q90={qs[0.9]:.3f}  q99={qs[0.99]:.3f}  max={max_v:.2f}")

    print(f"\n=== Vertical pair valid fraction (∂y direct supervision feasibility) ===")
    print(f"  acc       : {np.mean(vp_acc)*100:.2f}%  (single sample mean)")

    print(f"\n=== LiDAR-row ±4 stripe-prone band ===")
    print(f"  pixels in band:        ~1.9% of image (single LiDAR dilated)")
    print(f"  acc coverage in band:  {np.mean(stripe_band_acc)*100:.2f}%")
    print(f"  lidar coverage in band:{np.mean(stripe_band_lidar)*100:.2f}%  (= sparse seeds)")
    print(f"  → acc fills {np.mean(stripe_band_acc)/max(np.mean(stripe_band_lidar),1e-6):.1f}× more pixels in the stripe band")

    print(f"\n=== Depth-range bucketed acc/dense ratio (relative coverage) ===")
    for r in range_bins:
        if range_total_pix[r] > 0:
            ratio = range_total_acc[r] / range_total_pix[r] * 100
            print(f"  {r[0]:>3}-{r[1]:<3}m : acc-valid = {ratio:>5.1f}% of dense-valid pixels")

    # -------- visualizations --------
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.plot(acc_row_dens_sum / n_row_samples * 100)
    ax.set_xlabel("row (y)"); ax.set_ylabel("acc valid (%)")
    ax.set_title(f"depth_acc coverage per row (mean over {n_row_samples} samples)")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/coverage_per_row.png", dpi=120)
    plt.close()
    print(f"\nsaved {OUT_DIR}/coverage_per_row.png")

    # Disagreement histograms
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.6))
    for ax, name, data in zip(
        axes,
        ["|acc − lidar|  (ghost)", "|acc − dense_gt|  (DAv2 err)", "|dense_gt − lidar|  (baseline)"],
        [np.concatenate(diff_acc_vs_lidar),
         np.concatenate(diff_acc_vs_dense),
         np.concatenate(diff_dense_vs_lidar)],
    ):
        ax.hist(np.clip(data, 0, 10), bins=80, alpha=0.85)
        ax.set_xlabel("m"); ax.set_yscale("log")
        ax.set_title(name); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/disagreement_distributions.png", dpi=120)
    plt.close()
    print(f"saved {OUT_DIR}/disagreement_distributions.png")

    # Per-sample visualizations
    for k, (sid, lidar, acc, interp, dense, m_lidar, m_acc, m_dense, band) in enumerate(viz_data):
        fig, axes = plt.subplots(2, 4, figsize=(18, 7))
        for ax, img, mask, title in zip(
            axes[0],
            [lidar, acc, interp, dense],
            [m_lidar, m_acc, m_interp := (interp > 0.1), m_dense],
            ["depth_lidar (single)", "depth_acc", "depth_interp", "depth_refined_grid"],
        ):
            disp = np.where(mask, img, np.nan)
            im = ax.imshow(disp, cmap="turbo", vmin=0, vmax=80)
            ax.set_title(title); ax.axis("off")
        # Bottom row: disagreement heatmaps + stripe band
        # |acc - lidar| on shared pixels
        diff_al = np.where(m_lidar & m_acc, np.abs(acc - lidar), np.nan)
        diff_ad = np.where(m_acc & m_dense, np.abs(acc - dense), np.nan)
        # Stripe band visualization
        band_vis = band.astype(np.float32)
        # Profile of mean depth per row (for stripe diagnosis)
        axes[1, 0].imshow(diff_al, cmap="hot", vmin=0, vmax=10)
        axes[1, 0].set_title("|acc − lidar|  (ghost)"); axes[1, 0].axis("off")
        axes[1, 1].imshow(diff_ad, cmap="hot", vmin=0, vmax=10)
        axes[1, 1].set_title("|acc − dense|  (DAv2 err)"); axes[1, 1].axis("off")
        axes[1, 2].imshow(band_vis, cmap="gray")
        axes[1, 2].set_title(f"LiDAR ±4 row dilation band\n(acc fills {((m_acc & band).sum()/max(band.sum(),1))*100:.1f}%)")
        axes[1, 2].axis("off")
        # Mean depth per row (where any GT exists)
        rows = np.arange(lidar.shape[0])
        prof_lidar = np.where(m_lidar.any(axis=1), np.nanmean(np.where(m_lidar, lidar, np.nan), axis=1), np.nan)
        prof_acc   = np.where(m_acc.any(axis=1),   np.nanmean(np.where(m_acc, acc, np.nan), axis=1), np.nan)
        prof_dense = np.nanmean(np.where(m_dense, dense, np.nan), axis=1)
        axes[1, 3].plot(prof_lidar, rows, label="lidar", lw=1.2, color="tab:blue")
        axes[1, 3].plot(prof_acc, rows, label="acc", lw=1.2, color="tab:orange", alpha=0.8)
        axes[1, 3].plot(prof_dense, rows, label="dense", lw=1.2, color="tab:green", alpha=0.7)
        axes[1, 3].invert_yaxis(); axes[1, 3].set_xlabel("mean depth (m)")
        axes[1, 3].set_title("per-row mean depth"); axes[1, 3].legend(loc="lower right", fontsize=8)
        axes[1, 3].grid(True, alpha=0.3)
        fig.suptitle(sid, fontsize=10)
        plt.tight_layout()
        plt.savefig(f"{OUT_DIR}/viz_{k}.png", dpi=110, bbox_inches="tight")
        plt.close()
    print(f"saved {len(viz_data)} per-sample visualizations to {OUT_DIR}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=200, help="number of samples for statistics")
    parser.add_argument("--viz", type=int, default=5, help="number of samples to visualize")
    parser.add_argument("--split", default=DEFAULT_SPLIT)
    args = parser.parse_args()
    main(args.n, args.viz, args.split)
