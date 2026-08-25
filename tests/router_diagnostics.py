"""Router diagnostics — where does the router fail, and do stages agree?

Two diagnostics, one pass:

  Diagnostic 1 (label-noise vs. undertraining):
    Are wrong-routing tokens concentrated near bin boundaries (19-21m, 49-51m)?
    Plots wrong-token distributions over
      • token mean depth (with bin edges marked)
      • boundary score  = 1 - max(bin fraction) per token
                          (0 = pure single-bin, 0.67 = perfect 3-way mix)

  Diagnostic 2 (shared vs. distinct failure modes):
    Do stage 1 and stage 2 make the same mistakes on the same tokens?
    Reports Cohen's κ on their argmax predictions (per-class) and on the
    binary "correct/wrong" identity, plus the wrong-overlap breakdown.

Loads BOTH stage checkpoints, forwards each sample twice. Runs on full test
by default. Saves plots + .npz + text summary to output/analysis/router_diag/.

Usage
-----
  python tests/router_diagnostics.py \
      --tag v3 \
      --stage1 output/radartaco_moe_stage1_v3 \
      --stage2 output/radartaco_moe_stage2_v3
"""
import argparse, os, sys, time
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from tqdm import tqdm

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, ROOT)

from src.dataset.nuscenes import NuScenesRadarDepthDataset
from scripts.train import _build_model


device = "cuda" if torch.cuda.is_available() else "cpu"
BINS = [0.0, 20.0, 50.0, 100.0]
LABELS = ["near", "mid", "far"]
K = 3


def load_model(run_dir, ckpt="best.pt"):
    cfg = OmegaConf.load(os.path.join(run_dir, "config.yaml"))
    m = _build_model(cfg, float(cfg.dataset.max_depth), int(cfg.dataset.max_radar_points))
    sd = torch.load(os.path.join(run_dir, ckpt), map_location=device, weights_only=False)
    m.load_state_dict(sd["model"])
    print(f"  loaded {run_dir}/{ckpt}  epoch={sd.get('epoch','?')}")
    return m.eval().to(device)


def per_token_stats(depth_gt_dense, tok_hw, bins):
    """Return (gt_bin, mean_depth, boundary_score) each shape (B, H, W).

    gt_bin        : majority-vote bin over the token's receptive field
    mean_depth    : average pixel depth in the token (m)
    boundary_score: 1 - max bin fraction  (0 = pure, ≤0.67 = mixed)
    """
    edges = torch.tensor(bins, device=depth_gt_dense.device, dtype=torch.float32)
    ds = depth_gt_dense.squeeze(1)                          # (B, H_full, W_full)
    pix_bin = torch.zeros_like(ds, dtype=torch.long)
    for i in range(K):
        pix_bin = torch.where(
            (ds >= float(edges[i])) & (ds < float(edges[i+1])),
            pix_bin.new_tensor(i), pix_bin)
    pix_bin = torch.where(ds >= float(edges[-1]),
                          pix_bin.new_tensor(K-1), pix_bin)
    onehot = F.one_hot(pix_bin, num_classes=K).permute(0, 3, 1, 2).float()
    frac = F.adaptive_avg_pool2d(onehot, tok_hw)             # (B, K, H, W)
    gt_bin = frac.argmax(dim=1)                              # (B, H, W)
    boundary = 1.0 - frac.max(dim=1).values                  # (B, H, W)
    mean_d = F.adaptive_avg_pool2d(ds.unsqueeze(1), tok_hw).squeeze(1)
    return gt_bin, mean_d, boundary


def cohens_kappa(agree_matrix, N):
    po = np.trace(agree_matrix) / N
    p_r = agree_matrix.sum(axis=1) / N
    p_c = agree_matrix.sum(axis=0) / N
    pe = (p_r * p_c).sum()
    return po, pe, (po - pe) / (1 - pe) if pe < 1 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, help="e.g. v2, v3 — used in filenames")
    ap.add_argument("--stage1", required=True, help="stage1 run_dir")
    ap.add_argument("--stage2", required=True, help="stage2 run_dir")
    ap.add_argument("--ckpt", default="best.pt")
    ap.add_argument("--n", type=int, default=None, help="Sample count (default: all)")
    ap.add_argument("--split", default="test")
    ap.add_argument("--scale", choices=["node", "edge"], default="node",
                    help="L2 MoE block: node (29x50) or edge (15x25)")
    ap.add_argument("--out-dir", default="output/analysis/router_diag")
    args = ap.parse_args()

    OUT = "output"
    base_cfg = OmegaConf.load(f"{OUT}/shape_lidar_grad_shape_edge_res/config.yaml")
    split_file = os.path.join(base_cfg.dataset.data_root, "splits", f"{args.split}.txt")
    ds = NuScenesRadarDepthDataset(
        data_root=base_cfg.dataset.data_root, split_file=split_file,
        dense_gt_dir=base_cfg.dataset.dense_gt_dir,
        radar_3d_dir=base_cfg.dataset.radar_3d_dir,
        night_ids_file=base_cfg.dataset.night_ids_file,
        max_radar_points=int(base_cfg.dataset.max_radar_points),
        max_depth=float(base_cfg.dataset.max_depth),
        min_depth=float(base_cfg.dataset.min_depth),
        augmentation=False,
    )
    n = args.n or len(ds)
    idxs = list(range(n)) if n == len(ds) else np.linspace(0, len(ds)-1, n).astype(int).tolist()
    print(f"{args.split} — {len(idxs):,}/{len(ds):,} samples")

    m1 = load_model(args.stage1, args.ckpt)
    m2 = load_model(args.stage2, args.ckpt)

    tok_hw = (29, 50) if args.scale == "node" else (15, 25)
    block_idx = 0 if args.scale == "node" else 1

    # Collect per-token arrays (uint8/float32 to keep memory small).
    all_gt, all_s1, all_s2, all_depth, all_bound = [], [], [], [], []
    t0 = time.time()
    with torch.no_grad():
        for i in tqdm(idxs, desc=f"{args.tag}/{args.scale}"):
            s = ds[int(i)]
            rgb = s["rgb_norm"].unsqueeze(0).to(device)
            rp  = s["radar_points"].unsqueeze(0).to(device)
            rm  = s["radar_mask"].unsqueeze(0).to(device)
            dgd = s["depth_gt_dense"].unsqueeze(0).to(device)

            gt_bin, mean_d, bound = per_token_stats(dgd, tok_hw, BINS)

            out1 = m1(rgb, rp, rm)
            out2 = m2(rgb, rp, rm)
            r1 = out1["router_logits"][block_idx].softmax(dim=1).argmax(dim=1)
            r2 = out2["router_logits"][block_idx].softmax(dim=1).argmax(dim=1)

            all_gt.append(gt_bin[0].cpu().numpy().ravel().astype(np.uint8))
            all_s1.append(r1[0].cpu().numpy().ravel().astype(np.uint8))
            all_s2.append(r2[0].cpu().numpy().ravel().astype(np.uint8))
            all_depth.append(mean_d[0].cpu().numpy().ravel().astype(np.float32))
            all_bound.append(bound[0].cpu().numpy().ravel().astype(np.float32))

    print(f"  ({time.time()-t0:.0f}s forward)")

    gt = np.concatenate(all_gt); s1 = np.concatenate(all_s1); s2 = np.concatenate(all_s2)
    depth = np.concatenate(all_depth); bound = np.concatenate(all_bound)
    N = len(gt)

    # Build summary text — printed and saved.
    import io
    buf = io.StringIO()
    def _p(msg=""):
        print(msg); buf.write(msg + "\n")

    _p(f"\n{'='*72}")
    _p(f"Router diagnostics — {args.tag} ({args.scale}, {args.split} n={len(idxs):,})")
    _p(f"  tokens/sample = {tok_hw[0]*tok_hw[1]:,}   total tokens = {N:,}")
    _p(f"  stage1: {args.stage1}")
    _p(f"  stage2: {args.stage2}")
    _p("=" * 72)

    # ============ Diagnostic 1 ============
    s1_wrong = (s1 != gt); s2_wrong = (s2 != gt)
    both_wrong = s1_wrong & s2_wrong
    only_s1 = s1_wrong & ~s2_wrong
    only_s2 = ~s1_wrong & s2_wrong

    _p("\n--- Diagnostic 1: Wrong-routing depth distribution ---")
    _p(f"  s1 wrong: {s1_wrong.sum():>10,}  ({100*s1_wrong.mean():.2f}%)")
    _p(f"  s2 wrong: {s2_wrong.sum():>10,}  ({100*s2_wrong.mean():.2f}%)")
    _p(f"  both:     {both_wrong.sum():>10,}  ({100*both_wrong.mean():.2f}%)")

    # Zone at bin edges: within ±3m of 20m or 50m
    edge_zone = (np.abs(depth - 20) < 3) | (np.abs(depth - 50) < 3)
    _p(f"\n  Fraction of tokens in ±3m edge zone:")
    _p(f"    all tokens : {100*edge_zone.mean():5.2f}%")
    _p(f"    s1 wrong   : {100*edge_zone[s1_wrong].mean():5.2f}%")
    _p(f"    s2 wrong   : {100*edge_zone[s2_wrong].mean():5.2f}%")
    _p(f"    both wrong : {100*edge_zone[both_wrong].mean():5.2f}%")

    # Bimodal tokens: boundary_score > 0.3  (max-bin ≤ 70%)
    mixed = bound > 0.3
    _p(f"\n  Fraction of high-mix tokens (boundary_score > 0.3):")
    _p(f"    all tokens : {100*mixed.mean():5.2f}%")
    _p(f"    s1 wrong   : {100*mixed[s1_wrong].mean():5.2f}%")
    _p(f"    s2 wrong   : {100*mixed[s2_wrong].mean():5.2f}%")
    _p(f"    both wrong : {100*mixed[both_wrong].mean():5.2f}%")

    _p(f"\n  Wrong-token conditional accuracy (accuracy | mix bucket):")
    for lo, hi in [(0.0, 0.05), (0.05, 0.15), (0.15, 0.3), (0.3, 0.5), (0.5, 0.7)]:
        m = (bound >= lo) & (bound < hi)
        if m.sum() > 0:
            _p(f"    bound=[{lo:.2f},{hi:.2f}) n={m.sum():>10,}  "
               f"s1_acc={100*(1-s1_wrong[m].mean()):5.2f}%  "
               f"s2_acc={100*(1-s2_wrong[m].mean()):5.2f}%")

    # Plot
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    bins_hist = np.linspace(0, 100, 51)
    axes[0].hist(depth, bins=bins_hist, alpha=0.35, label=f"all ({N:,})", color="gray", density=True)
    axes[0].hist(depth[s1_wrong], bins=bins_hist, alpha=0.55,
                 label=f"s1 wrong ({s1_wrong.sum():,})", color="tab:blue", density=True)
    axes[0].hist(depth[s2_wrong], bins=bins_hist, alpha=0.55,
                 label=f"s2 wrong ({s2_wrong.sum():,})", color="tab:orange", density=True)
    for e in (20, 50):
        axes[0].axvline(e, ls="--", color="red", alpha=0.6, lw=1)
    axes[0].set_xlabel("Token mean depth (m)")
    axes[0].set_ylabel("density")
    axes[0].set_title(f"Diag 1a: wrong-token depth distribution ({args.tag})")
    axes[0].legend(loc="upper right", fontsize=9)

    bins_b = np.linspace(0, 0.67, 34)
    axes[1].hist(bound, bins=bins_b, alpha=0.35, label="all", color="gray", density=True)
    axes[1].hist(bound[s1_wrong], bins=bins_b, alpha=0.55,
                 label="s1 wrong", color="tab:blue", density=True)
    axes[1].hist(bound[s2_wrong], bins=bins_b, alpha=0.55,
                 label="s2 wrong", color="tab:orange", density=True)
    axes[1].set_xlabel("Boundary score  (1 − max bin fraction)")
    axes[1].set_ylabel("density")
    axes[1].set_title(f"Diag 1b: mix-degree of wrong tokens ({args.tag})")
    axes[1].legend(fontsize=9)
    plt.tight_layout()

    os.makedirs(args.out_dir, exist_ok=True)
    plot_path = os.path.join(args.out_dir, f"diag1_{args.tag}_{args.scale}.png")
    plt.savefig(plot_path, dpi=120, bbox_inches="tight")
    plt.close()
    _p(f"\n  Saved plot: {plot_path}")

    # ============ Diagnostic 2 ============
    _p("\n--- Diagnostic 2: Stage 1 vs Stage 2 agreement ---")

    # 3x3 agreement (rows = s1 pred, cols = s2 pred)
    agree = np.zeros((K, K), dtype=np.int64)
    for i in range(K):
        for j in range(K):
            agree[i, j] = ((s1 == i) & (s2 == j)).sum()

    _p("  Agreement matrix (rows=s1 pred, cols=s2 pred):")
    header = "         " + "  ".join(f"{l:>10}" for l in LABELS) + "  |  row sum"
    _p(header)
    for i in range(K):
        row = f"  {LABELS[i]:<6}" + "  ".join(f"{agree[i,j]:>10,}" for j in range(K))
        _p(row + f"  |  {agree[i,:].sum():>10,}")
    col_sums = "  colsum" + "  ".join(f"{agree[:,j].sum():>10,}" for j in range(K))
    _p(col_sums + f"  |  {agree.sum():>10,}")

    po, pe, kappa = cohens_kappa(agree, N)
    _p(f"\n  Overall agreement (s1_pred == s2_pred): {100*po:5.2f}%")
    _p(f"  Chance agreement:                       {100*pe:5.2f}%")
    _p(f"  Cohen's κ (per-class predictions):      {kappa:.3f}")

    _p(f"\n  Wrong-overlap breakdown:")
    _p(f"    both correct  : {(~s1_wrong & ~s2_wrong).sum():>10,}  ({100*(~s1_wrong & ~s2_wrong).mean():5.2f}%)")
    _p(f"    both wrong    : {both_wrong.sum():>10,}  ({100*both_wrong.mean():5.2f}%)")
    _p(f"    only s1 wrong : {only_s1.sum():>10,}  ({100*only_s1.mean():5.2f}%)")
    _p(f"    only s2 wrong : {only_s2.sum():>10,}  ({100*only_s2.mean():5.2f}%)")

    # Binary correct/wrong agreement — Cohen's κ on identity of errors
    binary_agree = np.array([
        [(~s1_wrong & ~s2_wrong).sum(), only_s2.sum()],
        [only_s1.sum(),                  both_wrong.sum()],
    ], dtype=np.int64)
    _, _, kappa_err = cohens_kappa(binary_agree, N)
    _p(f"  Cohen's κ (correct/wrong identity):     {kappa_err:.3f}")

    # Wrong-set overlap ratio (Jaccard)
    inter = both_wrong.sum()
    union = (s1_wrong | s2_wrong).sum()
    _p(f"  Jaccard(s1_wrong, s2_wrong):            {inter/max(union,1):.3f}")

    # Save raw arrays for offline re-analysis + summary
    npz_path = os.path.join(args.out_dir, f"router_diag_{args.tag}_{args.scale}.npz")
    np.savez_compressed(npz_path, gt=gt, s1=s1, s2=s2,
                        depth=depth, bound=bound)
    _p(f"\n  Saved raw:  {npz_path}")

    txt_path = os.path.join(args.out_dir, f"router_diag_{args.tag}_{args.scale}.txt")
    with open(txt_path, "w") as f:
        f.write(buf.getvalue())
    _p(f"  Saved text: {txt_path}")


if __name__ == "__main__":
    main()
