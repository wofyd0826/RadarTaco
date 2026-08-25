"""For warm25 stage-2 checkpoints (no pixmask + pixmask v1):
    1. Force each expert only → per-bin MAE vs cached baseline
    2. Measure router self-decision accuracy vs majority-vote GT bin
"""
import argparse, os, sys
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

# Cached baseline @ e46 (val 200, per-bin MAE)
BASELINE_MAE = {
    "[0.0,20.0)m": 0.7612,
    "[20.0,50.0)m": 3.6500,
    "[50.0,100.0)m": 8.7229,
}


def load_model(run_dir, ckpt):
    cfg = OmegaConf.load(os.path.join(run_dir, "config.yaml"))
    m = _build_model(cfg, float(cfg.dataset.max_depth), int(cfg.dataset.max_radar_points))
    sd = torch.load(os.path.join(run_dir, ckpt), map_location=device, weights_only=False)
    m.load_state_dict(sd["model"])
    print(f"  loaded {run_dir}/{ckpt}  epoch={sd.get('epoch','?')}")
    return m.eval().to(device)


def make_force_hook(expert_idx):
    def _hook(_mod, _inp, out):
        forced = torch.full_like(out, -1e4)
        forced[:, expert_idx] = 1e4
        return forced
    return _hook


def per_range(pred, gt, mask, ranges):
    out = {}
    for (lo, hi) in ranges:
        m = mask & (gt >= lo) & (gt < hi) & (gt > 0)
        if m.any():
            err = np.abs(pred[m] - gt[m])
            out[f"[{lo},{hi})m"] = (float(err.sum()), int(m.sum()))
        else:
            out[f"[{lo},{hi})m"] = (0.0, 0)
    return out


def evaluate_mae(model, ds, indices, ranges, force_idx=None):
    hooks = []
    if force_idx is not None:
        for blk in (model.radar_fusion.node_blocks[2], model.radar_fusion.edge_blocks[2]):
            if isinstance(blk, MoEFusionBlock):
                hooks.append(blk.router.register_forward_hook(make_force_hook(force_idx)))

    acc = defaultdict(lambda: [0.0, 0])
    with torch.no_grad():
        for i in tqdm(indices, desc=f"force={force_idx}", leave=False):
            s = ds[int(i)]
            rgb = s["rgb_norm"].unsqueeze(0).to(device)
            rp = s["radar_points"].unsqueeze(0).to(device)
            rm = s["radar_mask"].unsqueeze(0).to(device)
            out = model(rgb, rp, rm)
            pred = out["depth"] if isinstance(out, dict) else out
            pred_np = pred[0, 0].cpu().numpy()
            gt_np = s["depth_gt_lidar"][0].cpu().numpy()
            mask_np = s["valid_mask_lidar"][0].cpu().numpy().astype(bool)
            r = per_range(pred_np, gt_np, mask_np, ranges)
            for k, (se, n) in r.items():
                acc[k][0] += se; acc[k][1] += n
    for h in hooks: h.remove()
    return {k: (v[0]/v[1] if v[1]>0 else float("nan")) for k, v in acc.items()}


def router_accuracy(model, ds, indices, bins=(0.0, 20.0, 50.0, 100.0)):
    """For stage-2 model (self-routing): compare router argmax to
    majority-vote GT bin per token. Report overall accuracy + per-bin
    confusion (fraction of tokens with GT=b routed to each expert).
    """
    K = len(bins) - 1
    edges = torch.tensor(bins, device=device, dtype=torch.float32)
    confusion = torch.zeros(K, K, dtype=torch.long)   # rows=GT bin, cols=predicted bin

    with torch.no_grad():
        for i in tqdm(indices, desc="router-acc", leave=False):
            s = ds[int(i)]
            rgb = s["rgb_norm"].unsqueeze(0).to(device)
            rp = s["radar_points"].unsqueeze(0).to(device)
            rm = s["radar_mask"].unsqueeze(0).to(device)
            d_gt = s["depth_gt_dense"].unsqueeze(0).to(device)

            # Get router logits AND compute GT bin at same time
            # We need router_logits from the model in eval mode.
            # Trick: call model with depth_gt_dense passed as stage-1 GT
            # so it emits router_gts too.
            # But stage-2 model has moe_stage=2 → won't emit router_gts.
            # So we compute GT bin manually.
            # Router logits: run a forward and grab from output dict.
            out = model(rgb, rp, rm)
            rl = out["router_logits"][0]              # (1, K, H_tok, W_tok) — L2 node
            token_pred = rl.softmax(dim=1).argmax(dim=1).squeeze(0)   # (H_tok, W_tok)

            # Majority-vote GT bin per token (same logic as MoEFusionBlock)
            H_tok, W_tok = token_pred.shape
            ds_full = d_gt.squeeze(1)                                   # (1, H, W)
            pix_bin = torch.zeros_like(ds_full, dtype=torch.long)
            for k in range(K):
                pix_bin = torch.where((ds_full >= float(edges[k])) & (ds_full < float(edges[k+1])),
                                      pix_bin.new_tensor(k), pix_bin)
            pix_bin = torch.where(ds_full >= float(edges[-1]),
                                  pix_bin.new_tensor(K-1), pix_bin)
            onehot = F.one_hot(pix_bin, num_classes=K).permute(0, 3, 1, 2).float()
            frac = F.adaptive_avg_pool2d(onehot, (H_tok, W_tok))         # (1, K, H_tok, W_tok)
            token_gt = frac.argmax(dim=1).squeeze(0)                     # (H_tok, W_tok)

            # Update confusion
            for gt in range(K):
                for pr in range(K):
                    confusion[gt, pr] += ((token_gt == gt) & (token_pred == pr)).sum().item()

    tot = confusion.sum().item()
    correct = confusion.diagonal().sum().item()
    overall_acc = 100.0 * correct / max(tot, 1)

    print(f"\nOverall router accuracy: {overall_acc:.1f}%  (tokens={tot})")
    print(f"Confusion (rows=GT bin, cols=router prediction, values=token count):")
    labels = ["near", "mid", "far"]
    print(f"  {'':6s}" + "  ".join(f"{l:>10s}" for l in labels))
    for gt in range(K):
        row = f"  {labels[gt]:6s}"
        for pr in range(K):
            row += f"  {confusion[gt, pr].item():>10,}"
        print(row)

    # Per-bin recall (fraction of GT-b tokens correctly routed to b)
    print(f"\nPer-bin recall (of tokens whose GT-bin=b, fraction routed to b):")
    for b in range(K):
        n_b = confusion[b, :].sum().item()
        if n_b > 0:
            recall = 100.0 * confusion[b, b].item() / n_b
            print(f"  {labels[b]}: {recall:.1f}%  ({confusion[b,b].item():,} / {n_b:,})")
    return overall_acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    args = ap.parse_args()

    ranges = [(0.0, 20.0), (20.0, 50.0), (50.0, 100.0)]
    range_names = [f"[{lo},{hi})m" for (lo, hi) in ranges]
    labels = ["near", "mid", "far"]
    OUT = "output"

    experiments = [
        ("warm25 stage2 best",   f"{OUT}/radartaco_moe_stage2_warm25",           "best.pt"),
        ("pixmask v1 stage2 best", f"{OUT}/radartaco_moe_stage2_warm25_pixmask", "best.pt"),
    ]

    cfg = OmegaConf.load(f"{OUT}/shape_lidar_grad_shape_edge_res/config.yaml")
    split_file = os.path.join(cfg.dataset.data_root, "splits", "val.txt")
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
    idxs = np.linspace(0, len(ds)-1, args.n).astype(int).tolist()

    per_exp = {}
    for name, run_dir, ckpt in experiments:
        print("\n" + "=" * 80)
        print(f"### {name}")
        print("=" * 80)
        m = load_model(run_dir, ckpt)

        # 1. Router accuracy (stage 2 self-routing)
        print("\n-- Router accuracy vs majority-vote GT --")
        _ = router_accuracy(m, ds, idxs)

        # 2. Normal + Force per expert
        print("\n-- Normal routing --")
        res = {"Normal": evaluate_mae(m, ds, idxs, ranges)}
        for k, lb in enumerate(labels):
            print(f"-- Force {lb} --")
            res[f"Force-{lb}"] = evaluate_mae(m, ds, idxs, ranges, force_idx=k)
        per_exp[name] = res

        del m; torch.cuda.empty_cache()

    # ── Report ──
    print("\n\n" + "=" * 90)
    print("SUMMARY: Head-to-head — expert (own bin) vs Baseline (same bin)")
    print("=" * 90)
    for name, res in per_exp.items():
        print(f"\n### {name}")
        for k, (lb, r) in enumerate(zip(labels, range_names)):
            b_mae = BASELINE_MAE[r]
            f_mae = res[f"Force-{lb}"][r]
            n_mae = res["Normal"][r]
            print(f"  {lb} bin {r}:")
            print(f"    baseline mae:           {b_mae:.4f}")
            print(f"    Force {lb} expert only:  {f_mae:.4f}   Δ = {f_mae - b_mae:+.4f}  "
                  f"({'✓ 이김' if f_mae < b_mae else '✗ 짐'})")
            print(f"    Normal routing:          {n_mae:.4f}   Δ = {n_mae - b_mae:+.4f}  "
                  f"({'✓ 이김' if n_mae < b_mae else '✗ 짐'})")


if __name__ == "__main__":
    main()
