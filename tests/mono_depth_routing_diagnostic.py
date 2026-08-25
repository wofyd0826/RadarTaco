"""Step 1: Can mono depth (rel_depth) act as a router?

For every token in the L2 node grid (29 x 50), compute the mean of the
mono-depth PNG in that receptive field. Fit two thresholds T1, T2 on TRAIN
samples so that binning `rel_depth_mean` at [-inf, T1, T2, inf] best matches
the target argmax bin (from GT majority vote).

Then evaluate the fitted thresholds on the TEST set and compare accuracy
against v4c's learned router (90.83% overall for reference).

Only diagnostic — does not run the model. Output tells us whether
mono-depth-based routing is worth wiring into the MoE.
"""
import argparse
import io
import itertools
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, ROOT)

BINS = [0.0, 20.0, 50.0, 100.0]
K = 3


def pixel_fraction(depth_gt_dense: torch.Tensor, tok_hw):
    edges = torch.tensor(BINS, device=depth_gt_dense.device, dtype=torch.float32)
    ds = depth_gt_dense.squeeze(1)
    pix_bin = torch.zeros_like(ds, dtype=torch.long)
    for i in range(K):
        pix_bin = torch.where((ds >= float(edges[i])) & (ds < float(edges[i+1])),
                              pix_bin.new_tensor(i), pix_bin)
    pix_bin = torch.where(ds >= float(edges[-1]),
                          pix_bin.new_tensor(K-1), pix_bin)
    onehot = F.one_hot(pix_bin, num_classes=K).permute(0, 3, 1, 2).float()
    return F.adaptive_avg_pool2d(onehot, tok_hw)


def collect(data_root, split_file, rel_depth_dir, tok_hw, n_samples):
    """Return (rel_mean, gt_bin) arrays of size (N_samples * H_tok * W_tok,)."""
    lines = [l.strip() for l in open(split_file) if l.strip()]
    if n_samples < len(lines):
        idxs = np.linspace(0, len(lines) - 1, n_samples).astype(int)
        lines = [lines[i] for i in idxs]
    H_tok, W_tok = tok_hw

    rel_all = []
    gt_all = []
    for line in tqdm(lines, desc=os.path.basename(split_file)):
        sid = line.split("\t")[0]
        # Load rel_depth
        rp = os.path.join(data_root, rel_depth_dir, f"{sid}.png")
        if not os.path.exists(rp):
            continue
        rel = np.asarray(Image.open(rp), dtype=np.float32) / 1000.0    # (H, W)
        rel_t = torch.from_numpy(rel).unsqueeze(0).unsqueeze(0)         # (1,1,H,W)
        rel_tok = F.adaptive_avg_pool2d(rel_t, tok_hw).squeeze().numpy()  # (H_tok, W_tok)

        # Load GT depth for target bin (matches baseline `dense_gt_dir`).
        gt_path = os.path.join(data_root, "depth_edge_res", f"{sid}.png")
        if not os.path.exists(gt_path):
            continue
        gt = np.asarray(Image.open(gt_path)).astype(np.float32) / 256.0
        gt_t = torch.from_numpy(gt).unsqueeze(0).unsqueeze(0)
        frac = pixel_fraction(gt_t, tok_hw).squeeze(0).numpy()          # (K, H, W)
        gt_bin = frac.argmax(axis=0)                                    # (H, W)

        rel_all.append(rel_tok.ravel().astype(np.float32))
        gt_all.append(gt_bin.ravel().astype(np.uint8))

    return np.concatenate(rel_all), np.concatenate(gt_all)


def fit_thresholds(rel, gt, n_grid=100, verbose=True):
    """Grid-search over (T1, T2) to maximize per-token argmax agreement.

    We assume rel_depth is monotone (higher rel = nearer OR farther — we try both).
    Bin function: bin(x) = 0 if x < T1, 1 if T1 <= x < T2, 2 if x >= T2  (order A)
                  OR reversed order if higher rel = farther (order B).
    Pick the ordering + thresholds that maximize agreement.
    """
    q = np.linspace(0.01, 0.99, n_grid)
    ts = np.quantile(rel, q)          # candidate thresholds (values sit at data quantiles)

    best = (0.0, None, None, None)     # (acc, T1, T2, order)
    for order in ('A', 'B'):
        for i, t1 in enumerate(ts[:-1]):
            for t2 in ts[i+1:]:
                pred = np.where(rel < t1, 0,
                                np.where(rel < t2, 1, 2)).astype(np.uint8)
                if order == 'B':
                    pred = 2 - pred
                acc = float((pred == gt).mean())
                if acc > best[0]:
                    best = (acc, float(t1), float(t2), order)
        if verbose:
            print(f"  order={order} best-so-far acc={best[0]*100:.2f}% "
                  f"T1={best[1]:.4f} T2={best[2]:.4f}")
    return best


def apply(rel, gt, T1, T2, order):
    pred = np.where(rel < T1, 0, np.where(rel < T2, 1, 2)).astype(np.uint8)
    if order == 'B':
        pred = 2 - pred
    acc = float((pred == gt).mean())
    return acc, pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="/data/public/nuScenes/derived")
    ap.add_argument("--rel-depth-dir", default="relative_depth")
    ap.add_argument("--train-n", type=int, default=2000,
                    help="Number of train samples for threshold fit")
    ap.add_argument("--test-n", type=int, default=None,
                    help="Number of test samples for eval (default: all)")
    ap.add_argument("--out-dir", default="output/analysis/router_cost")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    tok_hw = (29, 50)
    train_split = os.path.join(args.data_root, "splits", "train.txt")
    test_split = os.path.join(args.data_root, "splits", "test.txt")

    print(f"=== Collecting TRAIN ({args.train_n} samples) ===")
    t0 = time.time()
    rel_tr, gt_tr = collect(args.data_root, train_split, args.rel_depth_dir,
                            tok_hw, args.train_n)
    print(f"  train tokens: {len(rel_tr):,}  ({time.time()-t0:.0f}s)")

    print(f"\n=== Fitting thresholds (grid search) ===")
    t0 = time.time()
    acc_fit, T1, T2, order = fit_thresholds(rel_tr, gt_tr, n_grid=200)
    print(f"  best fit acc = {acc_fit*100:.2f}%   T1={T1:.4f}  T2={T2:.4f}  order={order}")
    print(f"  ({time.time()-t0:.0f}s)")

    n_test = args.test_n or 100000
    print(f"\n=== Applying to TEST ({args.test_n or 'all'} samples) ===")
    t0 = time.time()
    with open(test_split) as f:
        n_test_total = sum(1 for l in f if l.strip())
    rel_te, gt_te = collect(args.data_root, test_split, args.rel_depth_dir,
                            tok_hw, args.test_n or n_test_total)
    print(f"  test tokens: {len(rel_te):,}  ({time.time()-t0:.0f}s)")

    acc_test, pred_test = apply(rel_te, gt_te, T1, T2, order)

    # Per-bin recall
    def _recall(pred, gt):
        return [100.0 * ((pred == k) & (gt == k)).sum() / max((gt == k).sum(), 1)
                for k in range(K)]
    rec_te = _recall(pred_test, gt_te)

    buf = io.StringIO()
    def _p(msg=""):
        print(msg); buf.write(msg + "\n")

    _p("=" * 96)
    _p("MONO-DEPTH ROUTING DIAGNOSTIC")
    _p(f"  rel_depth dir: {args.rel_depth_dir}   grid: {tok_hw}")
    _p("=" * 96)
    _p(f"\nFitted (TRAIN {args.train_n} samples): "
       f"T1={T1:.4f}  T2={T2:.4f}  order={order}")
    _p(f"  Train agreement:  {acc_fit*100:.2f}%")

    _p(f"\nTest agreement (n={args.test_n or 'all'} samples): "
       f"{acc_test*100:.2f}%")
    _p(f"  Per-class recall (test):")
    for k, name in enumerate(["near", "mid", "far"]):
        _p(f"    {name:<5s}  {rec_te[k]:>6.2f}%")

    # Baselines
    _p("\nReference points (for comparison):")
    _p(f"  v4c stage2 learned router L2_node accuracy: 90.61% "
       f"(near 95.1% / mid 84.5% / far 88.6%)")
    _p(f"  Random routing (uniform)                  : 33.33%")

    if acc_test > 0.93:
        _p(f"\n→ Mono-depth routing BEATS learned router — worth wiring into MoE.")
    elif acc_test > 0.85:
        _p(f"\n→ Mono-depth routing COMPARABLE to learned router. "
           f"Injecting as feature (not replacing) is the natural direction.")
    else:
        _p(f"\n→ Mono-depth routing UNDERPERFORMS the learned router.")

    # Confusion matrix
    _p("\nConfusion (target = GT bin, pred = mono-depth-derived):")
    _p(f"  {'  ':<8s}" + "".join(f"{n:>8s}" for n in ["near", "mid", "far"]))
    for k, tname in enumerate(["near", "mid", "far"]):
        row = []
        m = gt_te == k
        n_t = int(m.sum())
        for kp in range(K):
            n_p = int((m & (pred_test == kp)).sum())
            row.append(f"{100*n_p/max(n_t,1):>7.2f}%")
        _p(f"  {tname:<8s}" + " ".join(row) + f"   ({n_t:,} tokens)")

    out = os.path.join(args.out_dir, "mono_depth_routing.txt")
    with open(out, "w") as f:
        f.write(buf.getvalue())
    print(f"\nSaved: {out}")

    # Also save fit params for use in Step 2 (inference override)
    np.savez(os.path.join(args.out_dir, "mono_depth_routing_fit.npz"),
             T1=T1, T2=T2, order=order,
             train_acc=acc_fit, test_acc=acc_test)


if __name__ == "__main__":
    main()
