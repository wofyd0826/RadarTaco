"""Verify: is out-of-bin specialist error the cause of Oracle-Normal gap?

Per sample, forward K=3 times with each specialist ALONE (gate one-hot on k,
no shared), measure per-pixel error under each specialist. Bucket pixels by
their token's GT bin, and compute an error matrix:

  error_matrix[k][j] = mean |depth_spec_k(pixel) − gt(pixel)|
                       for pixels whose parent token's GT bin is j

If out-of-bin specialists produce large errors (hypothesis true):
  - diagonal (k == j) should be SMALL
  - off-diagonal (k != j) should be LARGE

Also compute Oracle-Normal gap decomposition:
  For each token, actual mix uses specialist(s) chosen by router.
  If misrouted, we can compare `actual specialist(s) output` vs `correct
  specialist output` to see per-misroute penalty.

Usage:
  python tests/analyze_expert_out_of_bin_penalty_v6c.py --n 500
"""
import argparse, json, os, sys
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from tqdm import tqdm

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, ROOT)

from src.dataset.nuscenes import NuScenesRadarDepthDataset
from src.model.radar_fusion import MoEFusionBlock
from scripts.train import _build_model

device = "cuda" if torch.cuda.is_available() else "cpu"


def load_model(run_dir, ckpt="best.pt"):
    cfg = OmegaConf.load(os.path.join(run_dir, "config.yaml"))
    m = _build_model(cfg, float(cfg.dataset.max_depth), int(cfg.dataset.max_radar_points))
    sd = torch.load(os.path.join(run_dir, ckpt), map_location=device, weights_only=False)
    m.load_state_dict(sd["model"])
    print(f"  loaded {run_dir}/{ckpt}  epoch={sd.get('epoch','?')}")
    return m.eval().to(device), cfg


def _collect_moe_blocks(model):
    out = []
    for l in range(len(model.radar_fusion.node_blocks)):
        nb = model.radar_fusion.node_blocks[l]
        eb = model.radar_fusion.edge_blocks[l]
        if isinstance(nb, MoEFusionBlock): out.append((f"L{l}_node", nb))
        if isinstance(eb, MoEFusionBlock): out.append((f"L{l}_edge", eb))
    return out


def compute_pixel_bin(depth_gt_dense, bins):
    """Per-pixel bin index (B, H, W) int64."""
    edges = torch.tensor(bins, device=depth_gt_dense.device, dtype=torch.float32)
    ds = depth_gt_dense.squeeze(1)
    pix_bin = torch.zeros_like(ds, dtype=torch.long)
    for i in range(len(bins) - 1):
        pix_bin = torch.where((ds >= float(edges[i])) & (ds < float(edges[i+1])),
                              pix_bin.new_tensor(i), pix_bin)
    pix_bin = torch.where(ds >= float(edges[-1]),
                          pix_bin.new_tensor(len(bins) - 2), pix_bin)
    return pix_bin


def make_patched_one_hot(spec_idx, K):
    """MoE forward with gate one-hot on `spec_idx`, no shared, no α scaling."""
    def patched(self, feat, kv, radar_x_orig, radar_mask, image_w,
                depth_gt_dense=None, teacher_force=True, return_shared_only=False,
                **kwargs):
        B, C, H, W = feat.shape
        # Force one-hot gate on spec_idx.
        gate = torch.zeros(B, self.n_experts, H, W, device=feat.device, dtype=feat.dtype)
        gate[:, spec_idx] = 1.0

        mixed = feat
        # Only run the selected specialist.
        expert = self.experts[spec_idx]
        out_e = expert(feat, kv, radar_x_orig, radar_mask, image_w, sel=None)
        mixed = mixed + (out_e - feat)

        # shared disabled entirely (no forward, no add).
        logits = self.router(feat)   # still compute for interface consistency
        return mixed, logits, None
    return patched


def install_one_hot(spec_idx, K, moe_blocks):
    orig_forwards = []
    for name, blk in moe_blocks:
        orig = blk.forward
        orig_forwards.append((blk, orig))
        blk.forward = make_patched_one_hot(spec_idx, K).__get__(blk, type(blk))
    return orig_forwards


def uninstall(orig_forwards):
    for blk, orig in orig_forwards:
        blk.forward = orig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default="output/radartaco_moe_stage2_v6c_confshared")
    ap.add_argument("--ckpt", default="best.pt")
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--split", default="test")
    ap.add_argument("--out-json", default="output/analysis/expert_out_of_bin_penalty_v6c.json")
    args = ap.parse_args()

    model, cfg = load_model(args.run_dir, args.ckpt)
    router_bins = list(cfg.model.moe_bins)
    K = len(router_bins) - 1
    labels = ["near", "mid", "far"] if K == 3 else [f"bin{k}" for k in range(K)]

    split_file = os.path.join(cfg.dataset.data_root, "splits", f"{args.split}.txt")
    ds = NuScenesRadarDepthDataset(
        data_root=cfg.dataset.data_root, split_file=split_file,
        dense_gt_dir=cfg.dataset.dense_gt_dir,
        radar_3d_dir=cfg.dataset.radar_3d_dir,
        night_ids_file=cfg.dataset.night_ids_file,
        max_radar_points=int(cfg.dataset.max_radar_points),
        max_depth=float(cfg.dataset.max_depth),
        min_depth=float(cfg.dataset.min_depth),
        augmentation=False,
    )
    n = min(args.n, len(ds))
    idxs = list(range(n)) if n == len(ds) else np.linspace(0, len(ds)-1, n).astype(int).tolist()
    print(f"\n{args.split} — {len(idxs):,}/{len(ds):,} samples\n")

    moe_blocks = _collect_moe_blocks(model)
    print(f"MoE blocks: {[nm for nm, _ in moe_blocks]}   K={K}\n")

    # error_matrix[k][j] accumulator: (sum_abs_err, count) over pixels with
    # per-PIXEL bin=j when specialist k was the one running.
    err_mat = np.zeros((K, K, 2))   # (sum_l1, count)

    with torch.no_grad():
        for i in tqdm(idxs, desc="samples", leave=False):
            s = ds[int(i)]
            rgb = s["rgb_norm"].unsqueeze(0).to(device)
            rp  = s["radar_points"].unsqueeze(0).to(device)
            rm  = s["radar_mask"].unsqueeze(0).to(device)
            dgd = s["depth_gt_dense"].unsqueeze(0).to(device)
            gt_np = s["depth_gt_lidar"][0].cpu().numpy()
            mask_np = s["valid_mask_lidar"][0].cpu().numpy().astype(bool)
            H_full, W_full = gt_np.shape

            # Per-pixel bin from GT LIDAR (accurate, not pooled).
            pix_bin = np.zeros_like(gt_np, dtype=np.int64)
            edges = router_bins
            for k in range(K):
                lo, hi = edges[k], edges[k+1] if k < K-1 else float("inf")
                pix_bin = np.where((gt_np >= lo) & (gt_np < hi), k, pix_bin)
            pix_bin = np.where(gt_np >= edges[-1], K-1, pix_bin)

            valid = mask_np & (gt_np > 0)

            for spec_idx in range(K):
                orig = install_one_hot(spec_idx, K, moe_blocks)
                out = model(rgb, rp, rm)
                uninstall(orig)
                depth = (out["depth"] if isinstance(out, dict) else out)[0, 0].cpu().numpy()

                # For each per-pixel bin j, accumulate error of this specialist.
                err = np.abs(depth - gt_np)
                for j in range(K):
                    sel = valid & (pix_bin == j)
                    if sel.any():
                        err_mat[spec_idx, j, 0] += float(err[sel].sum())
                        err_mat[spec_idx, j, 1] += int(sel.sum())

    # Compute mean MAE.
    mae_mat = np.zeros((K, K))
    for k in range(K):
        for j in range(K):
            if err_mat[k, j, 1] > 0:
                mae_mat[k, j] = err_mat[k, j, 0] / err_mat[k, j, 1]
            else:
                mae_mat[k, j] = float("nan")

    # ---- Report ----
    print("\n" + "=" * 100)
    print("Specialist error matrix — mean |pred − gt| by (specialist_run × pixel_gt_bin)")
    print("v6c stage 2  —  each specialist run ALONE (one-hot gate, no shared)")
    print("=" * 100)
    print(f"Samples: {len(idxs):,}\n")

    print("Rows: specialist_k running.   Cols: pixel gt_bin j.")
    print("  Diagonal = specialist on its own bin.  Off-diagonal = out-of-bin.")
    print("-" * 80)
    header = f"  {'specialist':<15}" + "  ".join(f"{'gt=' + l:>14}" for l in labels)
    print(header)
    for k in range(K):
        row = f"  spec_{labels[k]:<10}" + "  ".join(
            f"{mae_mat[k, j]:>14.4f}" for j in range(K))
        print(row)

    # Diagonal vs off-diagonal summary.
    print("\nDiagonal (in-bin) vs Off-diagonal (out-of-bin) — per specialist")
    print("-" * 80)
    for k in range(K):
        diag = mae_mat[k, k]
        off = [mae_mat[k, j] for j in range(K) if j != k]
        off_mean = float(np.nanmean(off))
        print(f"  spec_{labels[k]:<10}  in-bin MAE = {diag:.4f}   out-of-bin MAE = {off_mean:.4f}   ratio = {off_mean/diag:.2f}x")

    # Per-pixel gt bin distribution.
    total_pix = int(err_mat[0, :, 1].sum())  # any spec has same total
    print(f"\nPixel distribution by gt bin (total valid pixels: {total_pix:,}):")
    for j in range(K):
        n = int(err_mat[0, j, 1])
        pct = 100.0 * n / max(total_pix, 1)
        print(f"  gt={labels[j]:<10}: {n:>12,}  ({pct:>5.2f}%)")

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    dump = {
        "run_dir": args.run_dir,
        "ckpt": args.ckpt,
        "n_samples": len(idxs),
        "K": K,
        "labels": labels,
        "router_bins": router_bins,
        "error_matrix_mae":  mae_mat.tolist(),
        "error_matrix_sum":  err_mat[..., 0].tolist(),
        "error_matrix_count": err_mat[..., 1].astype(np.int64).tolist(),
    }
    with open(args.out_json, "w") as f:
        json.dump(dump, f, indent=2)
    print(f"\nSaved → {args.out_json}")


if __name__ == "__main__":
    main()
