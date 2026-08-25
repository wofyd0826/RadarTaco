"""Measure how well v4c's PREDICTED depth (bucketized) matches the dense-GT
bin classification, per bin.

Not the router's classification — the model's own depth output, bucketized to
the same 0-20 / 20-50 / 50-100m bins, compared with the dense-GT bin.

For each test sample:
  1. Run model forward → depth_pred (B, 1, H, W)
  2. Pixel-level:
     • bin(pred) vs bin(dense_gt), per pixel (mask out zeros)
     • Confusion matrix K×K
  3. Token-level (router grid — L2_node grid ~29×50 for 900×1600 inputs):
     • bin(pred) argmax-pooled to token grid: majority-vote of pixel bins
     • bin(dense_gt) argmax-pooled likewise (this IS the router GT)
     • Confusion matrix at token level

Reports per-bin recall + precision + full confusion.

Usage:
  python tests/analyze_pred_depth_bin_accuracy_v4c.py --n 6008
"""
import argparse, os, sys, json
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
    return m.eval().to(device), cfg


def bucketize_depth(depth, bins):
    """depth: (H, W) or (N,) → int in [0, K-1]. Values ≥ last edge collapse
    to last bin; values < first edge (0 typically) also to first bin (only
    used inside `valid` mask so 0/invalid rows are filtered)."""
    K = len(bins) - 1
    edges = np.asarray(bins, dtype=np.float32)
    out = np.zeros_like(depth, dtype=np.int64)
    for i in range(K):
        out = np.where((depth >= edges[i]) & (depth < edges[i + 1]), i, out)
    out = np.where(depth >= edges[-1], K - 1, out)
    return out


def token_grid_from_dense(depth_bhw_torch, tok_hw, bins):
    """Majority-vote bin per token given a dense per-pixel depth (torch).
    Returns (H_tok, W_tok) int64. Uses adaptive_avg_pool2d on one-hot bins."""
    K = len(bins) - 1
    edges = torch.tensor(list(bins), device=depth_bhw_torch.device,
                          dtype=torch.float32)
    ds = depth_bhw_torch.squeeze(0).squeeze(0)                       # (H, W)
    pix_bin = torch.zeros_like(ds, dtype=torch.long)
    for i in range(K):
        pix_bin = torch.where(
            (ds >= float(edges[i])) & (ds < float(edges[i + 1])),
            pix_bin.new_tensor(i), pix_bin)
    pix_bin = torch.where(ds >= float(edges[-1]),
                          pix_bin.new_tensor(K - 1), pix_bin)
    onehot = F.one_hot(pix_bin, num_classes=K).permute(2, 0, 1).float() \
              .unsqueeze(0)                                          # (1, K, H, W)
    frac = F.adaptive_avg_pool2d(onehot, tok_hw)[0]                  # (K, H_tok, W_tok)
    return frac.argmax(dim=0).cpu().numpy()                          # (H_tok, W_tok)


def _find_router_grids(model):
    """Return list of (name, (H_tok, W_tok)) for each MoE router grid."""
    grids = []
    for l in range(len(model.radar_fusion.node_blocks)):
        for tag, blk in (("node", model.radar_fusion.node_blocks[l]),
                          ("edge", model.radar_fusion.edge_blocks[l])):
            if isinstance(blk, MoEFusionBlock):
                grids.append((f"L{l}_{tag}", blk))
    return grids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default="output/radartaco_moe_stage2_v4c")
    ap.add_argument("--ckpt", default="best.pt")
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--split", default="test")
    ap.add_argument("--out-json", default=None)
    args = ap.parse_args()

    tag = os.path.basename(args.run_dir.rstrip("/"))
    if args.out_json is None:
        args.out_json = f"output/analysis/pred_depth_bin_accuracy_{tag}.json"

    model, cfg = load_model(args.run_dir, args.ckpt)
    bins = list(cfg.model.moe_bins)                                   # e.g. [0, 20, 50, 100]
    K = len(bins) - 1
    print(f"bins={bins}   K={K}")

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
    n = args.n or len(ds)
    idxs = list(range(n)) if n == len(ds) else np.linspace(0, len(ds)-1, n).astype(int).tolist()
    print(f"{args.split} — {len(idxs):,}/{len(ds):,} samples\n")

    router_grids = _find_router_grids(model)
    print(f"Router grids: {[nm for nm, _ in router_grids]}")

    # Pixel-level confusion accumulator.
    pix_cm = np.zeros((K, K), dtype=np.int64)
    # Token-level confusion per router-grid block.
    tok_cms = {nm: np.zeros((K, K), dtype=np.int64) for nm, _ in router_grids}

    with torch.no_grad():
        for i in tqdm(idxs, desc="samples", leave=False):
            s = ds[int(i)]
            rgb = s["rgb_norm"].unsqueeze(0).to(device)
            rp  = s["radar_points"].unsqueeze(0).to(device)
            rm  = s["radar_mask"].unsqueeze(0).to(device)
            dgd = s["depth_gt_dense"].to(device)                       # (1, H, W)
            gt_np = dgd[0].cpu().numpy()                               # (H, W)

            out = model(rgb, rp, rm)
            depth_pred_t = (out["depth"] if isinstance(out, dict) else out)   # (1, 1, H, W)
            depth_pred = depth_pred_t[0, 0].cpu().numpy()              # (H, W)

            # ---- Pixel level ----
            valid_pix = (gt_np > 0) & np.isfinite(gt_np) & np.isfinite(depth_pred)
            if valid_pix.any():
                gt_bin  = bucketize_depth(gt_np, bins)
                prd_bin = bucketize_depth(depth_pred, bins)
                # Confusion count via bincount trick
                combo = gt_bin * K + prd_bin
                combo = combo[valid_pix]
                cnts = np.bincount(combo, minlength=K * K).reshape(K, K)
                pix_cm += cnts

            # ---- Token level (per router grid) ----
            # Do the same computation on a token grid via majority-vote pooling
            # of BOTH GT and PRED into bin fraction, then argmax.
            for nm, _ in router_grids:
                # We need the actual grid size. Use L2_node fixed 29x50 fallback,
                # but ideally derive from a forward. Instead, use router grid
                # inferred from the block's router logits shape via forward output.
                pass

            # Cleaner: derive token grids from the router_logits (already in out).
            if isinstance(out, dict) and out.get("router_logits"):
                for (nm, _), rl in zip(router_grids, out["router_logits"]):
                    _, _, H_tok, W_tok = rl.shape
                    tok_gt  = token_grid_from_dense(dgd.unsqueeze(0),
                                                     (H_tok, W_tok), bins)
                    tok_prd = token_grid_from_dense(depth_pred_t,
                                                     (H_tok, W_tok), bins)
                    combo = tok_gt * K + tok_prd
                    cnts = np.bincount(combo.ravel(), minlength=K * K).reshape(K, K)
                    tok_cms[nm] += cnts

    # ---- Report ----
    def report(name, cm):
        tot = cm.sum()
        acc = 100.0 * cm.trace() / max(tot, 1)
        print(f"\n{name}  —  total = {tot:,},  overall accuracy = {acc:.2f}%")
        labels = ["near 0-20m", "mid 20-50m", "far 50-100m"]
        print(f"  Confusion (rows=GT, cols=PRED):")
        print(f"    {'':<15} " + " ".join(f"{l:>18s}" for l in labels))
        for i, r in enumerate(labels):
            row = cm[i]
            tot_r = row.sum()
            cells = " ".join(f"{v:>10,}({100*v/max(tot_r,1):5.1f}%)" for v in row)
            print(f"    GT {r:<12} {cells}")
        print(f"\n  Per-GT recall (given true bin, how often pred matched):")
        for i, r in enumerate(labels):
            tot_r = cm[i].sum()
            rec = 100 * cm[i, i] / max(tot_r, 1)
            print(f"    {r:<15}  {cm[i,i]:>10,} / {int(tot_r):>10,}  =  {rec:6.2f}%")
        print(f"\n  Per-PRED precision (when pred says X, how often correct):")
        for i, r in enumerate(labels):
            tot_c = cm[:, i].sum()
            prec = 100 * cm[i, i] / max(tot_c, 1)
            print(f"    pred {r:<12}  {cm[i,i]:>10,} / {int(tot_c):>10,}  =  {prec:6.2f}%")
        return acc

    print("\n" + "=" * 100)
    print(f"PIXEL-level bin agreement (bucketize(pred_depth) vs bucketize(dense_gt))")
    print("=" * 100)
    pix_acc = report("PIXEL-level", pix_cm)

    print("\n" + "=" * 100)
    print(f"TOKEN-level bin agreement (majority-vote per router grid)")
    print("=" * 100)
    tok_accs = {}
    for nm, cm in tok_cms.items():
        tok_accs[nm] = report(f"TOKEN-level [{nm}]", cm)

    # Persist
    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    dump = {
        "run_dir": args.run_dir, "ckpt": args.ckpt, "bins": bins,
        "n_samples": len(idxs),
        "pixel_confusion": pix_cm.tolist(),
        "pixel_accuracy":  pix_acc,
        "token_confusion": {nm: cm.tolist() for nm, cm in tok_cms.items()},
        "token_accuracy":  tok_accs,
    }
    with open(args.out_json, "w") as f:
        json.dump(dump, f, indent=2)
    print(f"\nSaved → {args.out_json}")


if __name__ == "__main__":
    main()
