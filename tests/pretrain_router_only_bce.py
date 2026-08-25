"""Router-only pretraining with sigmoid + BCE loss (experiment A).

Same setup as tests/pretrain_router_only.py — baseline encoder frozen,
train only the two L2 MoE routers — BUT loss is BCE per bin (independent
sigmoid classifiers) instead of softmax + soft CE.

Same discrete bins [0, 20, 50, 100], same target pixel_fraction. Only
the *interpretation* and *loss* change:
  Original: softmax over K, treat as one probability distribution,
            loss = -Σ_k frac_k · log(softmax(logits)_k)
  This exp: sigmoid per k, treat as K independent binary probabilities,
            loss = -Σ_k [frac_k · log(σ(z_k)) + (1-frac_k) · log(1-σ(z_k))]

Purpose: isolate the effect of the loss function (softmax competition vs
independent sigmoids) from any label-definition change (overlap bins).
Comparable to baseline_encoder pretrain (overall_acc = 90.30% with soft CE).

Metrics reported (per-block, plus token-weighted overall):
  argmax_acc    — argmax(sigmoid) == argmax(frac). Directly comparable to
                  baseline_encoder's 90.3%.
  soft_l1       — mean |sigmoid - frac| (K-dim L1). Directly comparable to
                  soft_l1 seen elsewhere (v4c ~0.19).
  bce_loss      — mean BCE-with-logits over val
  per-bin recall(thr=0.5): sigmoid_k > 0.5 vs frac_k > 0.5, per k
  sig_sum_mean  — mean of Σ_k sigmoid_k, tracks whether outputs drift from
                  a sum-to-1 convex simplex (softmax gives 1; BCE has no
                  such constraint, but converges toward sum(target)=1 at
                  optimum)

Output layout mirrors pretrain_router_only.py:
  output/router_pretrain/<tag>/{best,last}.pt + history.json + log.txt
"""
import argparse
import json
import os
import sys
import time
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, ROOT)

from src.dataset.nuscenes import NuScenesRadarDepthDataset
from scripts.train import _build_model


BINS = [0.0, 20.0, 50.0, 100.0]
K = 3
LABELS = ["near", "mid", "far"]


def compute_frac(depth_gt_dense: torch.Tensor, tok_hw, bins) -> torch.Tensor:
    """Per-token pixel-bin fraction. (B, K, H_tok, W_tok) summing to 1 along K."""
    edges = torch.tensor(bins, device=depth_gt_dense.device, dtype=torch.float32)
    ds = depth_gt_dense.squeeze(1)
    pix_bin = torch.zeros_like(ds, dtype=torch.long)
    for i in range(K):
        pix_bin = torch.where(
            (ds >= float(edges[i])) & (ds < float(edges[i+1])),
            pix_bin.new_tensor(i), pix_bin)
    pix_bin = torch.where(ds >= float(edges[-1]),
                          pix_bin.new_tensor(K-1), pix_bin)
    onehot = F.one_hot(pix_bin, num_classes=K).permute(0, 3, 1, 2).float()
    return F.adaptive_avg_pool2d(onehot, tok_hw)


def make_router(ch: int, n_experts: int) -> nn.Module:
    """Same architecture as v4c mlp3x3 (K logits — interpreted as sigmoid here)."""
    h1 = min(256, ch // 2)
    h2 = h1 // 2
    return nn.Sequential(
        nn.Conv2d(ch, h1, 3, padding=1), nn.GELU(),
        nn.Conv2d(h1, h2, 3, padding=1), nn.GELU(),
        nn.Conv2d(h2, n_experts, 1),
    )


@torch.no_grad()
def evaluate(baseline, routers, captured, loader, device) -> Dict:
    """Per-block metrics: argmax_acc, soft_l1, bce_loss,
    per-bin recall/precision, sig_sum stats."""
    for r in routers.values():
        r.eval()
    correct = {n: 0 for n in routers}
    total = {n: 0 for n in routers}
    per_bin_tp = {n: torch.zeros(K, dtype=torch.long) for n in routers}
    per_bin_fp = {n: torch.zeros(K, dtype=torch.long) for n in routers}
    per_bin_fn = {n: torch.zeros(K, dtype=torch.long) for n in routers}
    per_bin_recall_argmax_correct = {n: torch.zeros(K, dtype=torch.long) for n in routers}
    per_bin_recall_argmax_total = {n: torch.zeros(K, dtype=torch.long) for n in routers}
    bce_sum = {n: 0.0 for n in routers}
    l1_sum = {n: 0.0 for n in routers}
    l1_ct = {n: 0 for n in routers}
    sig_sum_accum = {n: 0.0 for n in routers}
    sig_sum_ct = {n: 0 for n in routers}
    n_batches = 0

    for batch in loader:
        rgb = batch["rgb_norm"].to(device, non_blocking=True)
        rp = batch["radar_points"].to(device, non_blocking=True)
        rm = batch["radar_mask"].to(device, non_blocking=True)
        dgd = batch["depth_gt_dense"].to(device, non_blocking=True)

        _ = baseline(rgb, rp, rm)
        for name, router in routers.items():
            feat = captured[name]
            logits = router(feat)                             # (B, K, H, W)
            sig = torch.sigmoid(logits)                       # (B, K, H, W)
            frac = compute_frac(dgd, logits.shape[-2:], BINS) # (B, K, H, W)

            # ---- Argmax accuracy (for direct comparison to baseline 90.3%) ----
            gt = frac.argmax(dim=1)
            pred = logits.argmax(dim=1)   # = sig.argmax since sigmoid is monotonic
            correct[name] += (pred == gt).sum().item()
            total[name] += gt.numel()

            # Per-bin argmax-recall (like existing metric)
            for k in range(K):
                mask_k = (gt == k)
                per_bin_recall_argmax_total[name][k] += mask_k.sum().item()
                per_bin_recall_argmax_correct[name][k] += (mask_k & (pred == k)).sum().item()

            # ---- Multi-label per-bin (threshold 0.5) ----
            #   target positive if frac_k > 0.5 (majority of pixels in bin k)
            #   pred positive if sig_k > 0.5
            for k in range(K):
                t_pos = (frac[:, k] > 0.5)
                p_pos = (sig[:, k] > 0.5)
                per_bin_tp[name][k] += (t_pos & p_pos).sum().item()
                per_bin_fp[name][k] += (~t_pos & p_pos).sum().item()
                per_bin_fn[name][k] += (t_pos & ~p_pos).sum().item()

            # ---- BCE loss ----
            bce_sum[name] += F.binary_cross_entropy_with_logits(
                logits, frac, reduction="mean").item()

            # ---- soft L1 (|sigmoid - frac| summed across K, mean per token) ----
            l1_per_tok = (sig - frac).abs().sum(dim=1)   # (B, H, W)
            l1_sum[name] += float(l1_per_tok.sum().item())
            l1_ct[name] += int(l1_per_tok.numel())

            # ---- sum of sigmoids per token (should be near 1 at good calibration) ----
            sig_sum_per_tok = sig.sum(dim=1)
            sig_sum_accum[name] += float(sig_sum_per_tok.sum().item())
            sig_sum_ct[name] += int(sig_sum_per_tok.numel())
        n_batches += 1

    out = {"per_block": {}}
    tot_c = 0; tot_n = 0
    for name in routers:
        acc = 100.0 * correct[name] / max(total[name], 1)
        rec_argmax = [100.0 * per_bin_recall_argmax_correct[name][k].item()
                      / max(per_bin_recall_argmax_total[name][k].item(), 1)
                      for k in range(K)]
        # per-bin sigmoid recall/precision/f1 (threshold 0.5)
        tp = per_bin_tp[name].float()
        fp = per_bin_fp[name].float()
        fn = per_bin_fn[name].float()
        bin_recall = (100.0 * tp / (tp + fn).clamp_min(1)).tolist()
        bin_precision = (100.0 * tp / (tp + fp).clamp_min(1)).tolist()
        bin_f1 = [2 * p * r / max(p + r, 1e-9)
                  for p, r in zip(bin_precision, bin_recall)]
        out["per_block"][name] = {
            "argmax_acc": acc,
            "recall_argmax": rec_argmax,
            "tokens": total[name],
            "bce_loss": bce_sum[name] / max(n_batches, 1),
            "soft_l1": l1_sum[name] / max(l1_ct[name], 1),
            "sig_sum_mean": sig_sum_accum[name] / max(sig_sum_ct[name], 1),
            "bin_recall_thr0.5": bin_recall,
            "bin_precision_thr0.5": bin_precision,
            "bin_f1_thr0.5": bin_f1,
        }
        tot_c += correct[name]; tot_n += total[name]
    out["overall_argmax_acc"] = 100.0 * tot_c / max(tot_n, 1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline-run",
                    default="output/shape_lidar_grad_shape_edge_res_fix",
                    help="Baseline run dir with config.yaml + best.pt")
    ap.add_argument("--ckpt", default="best.pt")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=12)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--val-every", type=int, default=1)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--out-dir", default="output/router_pretrain")
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()

    # Per-tag subdirectory. Create it up-front so any early failure still
    # leaves an artifact on disk.
    args.out_dir = os.path.join(args.out_dir, args.tag)
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"Output dir: {args.out_dir}")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ============ Load baseline (frozen) ============
    cfg_path = os.path.join(args.baseline_run, "config.yaml")
    cfg = OmegaConf.load(cfg_path)
    print(f"Loading baseline from {args.baseline_run}")
    baseline = _build_model(cfg, float(cfg.dataset.max_depth),
                             int(cfg.dataset.max_radar_points))
    sd = torch.load(os.path.join(args.baseline_run, args.ckpt),
                    map_location=device, weights_only=False)
    baseline.load_state_dict(sd["model"])
    baseline = baseline.to(device).eval()
    for p in baseline.parameters():
        p.requires_grad = False
    print(f"  Baseline params (frozen): "
          f"{sum(p.numel() for p in baseline.parameters()):,}")

    # ============ Determine L2 block channel dims ============
    l2_node_block = baseline.radar_fusion.node_blocks[2]
    l2_edge_block = baseline.radar_fusion.edge_blocks[2]
    ch_node = l2_node_block.norm1.num_channels
    ch_edge = l2_edge_block.norm1.num_channels
    print(f"  L2 node ch={ch_node}, edge ch={ch_edge}")

    # ============ Create new routers (trainable) ============
    routers = {
        "node": make_router(ch_node, K).to(device),
        "edge": make_router(ch_edge, K).to(device),
    }
    n_router_params = sum(sum(p.numel() for p in r.parameters())
                          for r in routers.values())
    print(f"  New router params (trainable): {n_router_params:,}")

    # ============ Hooks: capture L2 block INPUTS ============
    captured: Dict[str, torch.Tensor] = {}
    def make_pre_hook(name):
        def _h(_m, args_):
            captured[name] = args_[0]
        return _h
    hook_node = l2_node_block.register_forward_pre_hook(make_pre_hook("node"))
    hook_edge = l2_edge_block.register_forward_pre_hook(make_pre_hook("edge"))

    # ============ Data ============
    ds_root = cfg.dataset.data_root
    def make_ds(split, aug):
        return NuScenesRadarDepthDataset(
            data_root=ds_root,
            split_file=os.path.join(ds_root, "splits", f"{split}.txt"),
            dense_gt_dir=cfg.dataset.dense_gt_dir,
            radar_3d_dir=cfg.dataset.radar_3d_dir,
            night_ids_file=cfg.dataset.night_ids_file,
            max_radar_points=int(cfg.dataset.max_radar_points),
            max_depth=float(cfg.dataset.max_depth),
            min_depth=float(cfg.dataset.min_depth),
            augmentation=aug,
        )
    train_ds = make_ds("train", aug=True)
    val_ds = make_ds("val", aug=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True,
                              persistent_workers=(args.num_workers > 0))
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=max(2, args.num_workers // 2),
                            pin_memory=True,
                            persistent_workers=(args.num_workers > 0))
    print(f"  Train samples: {len(train_ds):,}   Val samples: {len(val_ds):,}")

    # ============ Optimizer ============
    trainable = [p for r in routers.values() for p in r.parameters()]
    opt = torch.optim.Adam(trainable, lr=args.lr, weight_decay=args.weight_decay)

    log_lines = []
    history = []
    best_acc = 0.0
    def log(msg):
        print(msg); log_lines.append(msg)

    log(f"Config: LOSS=BCE  epochs={args.epochs}  batch_size={args.batch_size}  "
        f"lr={args.lr}  wd={args.weight_decay}")

    # ============ Training loop ============
    for epoch in range(args.epochs):
        for r in routers.values():
            r.train()
        t0 = time.time()
        train_losses = []
        pbar = tqdm(train_loader, desc=f"ep{epoch:02d} train", leave=False)
        for it, batch in enumerate(pbar):
            rgb = batch["rgb_norm"].to(device, non_blocking=True)
            rp = batch["radar_points"].to(device, non_blocking=True)
            rm = batch["radar_mask"].to(device, non_blocking=True)
            dgd = batch["depth_gt_dense"].to(device, non_blocking=True)

            with torch.no_grad():
                _ = baseline(rgb, rp, rm)
            feat_node = captured["node"].detach()
            feat_edge = captured["edge"].detach()

            logits_node = routers["node"](feat_node)
            logits_edge = routers["edge"](feat_edge)

            frac_node = compute_frac(dgd, logits_node.shape[-2:], BINS)
            frac_edge = compute_frac(dgd, logits_edge.shape[-2:], BINS)

            # BCE per bin (independent sigmoids). soft target (frac) natively
            # supported by BCE — see notes in module docstring.
            loss_node = F.binary_cross_entropy_with_logits(logits_node, frac_node)
            loss_edge = F.binary_cross_entropy_with_logits(logits_edge, frac_edge)
            loss = 0.5 * (loss_node + loss_edge)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            train_losses.append(loss.item())
            if (it + 1) % 100 == 0:
                pbar.set_postfix({"bce": f"{np.mean(train_losses[-100:]):.4f}"})
        train_time = time.time() - t0
        train_loss = float(np.mean(train_losses))

        entry = {
            "epoch": epoch,
            "train_bce": train_loss,
            "train_time_sec": train_time,
        }
        if (epoch + 1) % args.val_every == 0 or epoch == args.epochs - 1:
            t1 = time.time()
            val = evaluate(baseline, routers, captured, val_loader, device)
            val_time = time.time() - t1
            entry["val"] = val
            entry["val_time_sec"] = val_time
            log(f"ep {epoch:02d}  train_bce={train_loss:.4f}  "
                f"({train_time:.0f}s)  |  val argmax_acc={val['overall_argmax_acc']:.2f}%  "
                f"node={val['per_block']['node']['argmax_acc']:.2f}%  "
                f"edge={val['per_block']['edge']['argmax_acc']:.2f}%  "
                f"(val {val_time:.0f}s)")
            for name in ["node", "edge"]:
                b = val["per_block"][name]
                ra = b["recall_argmax"]
                br = b["bin_recall_thr0.5"]
                bp = b["bin_precision_thr0.5"]
                log(f"    {name}: bce={b['bce_loss']:.4f}  L1={b['soft_l1']:.4f}  "
                    f"sig_sum={b['sig_sum_mean']:.3f}")
                log(f"      argmax recall: near={ra[0]:.1f}%  mid={ra[1]:.1f}%  far={ra[2]:.1f}%")
                log(f"      bin recall (thr=0.5):    near={br[0]:.1f}%  mid={br[1]:.1f}%  far={br[2]:.1f}%")
                log(f"      bin precision (thr=0.5): near={bp[0]:.1f}%  mid={bp[1]:.1f}%  far={bp[2]:.1f}%")

            if val["overall_argmax_acc"] > best_acc:
                best_acc = val["overall_argmax_acc"]
                torch.save({
                    "router_node": routers["node"].state_dict(),
                    "router_edge": routers["edge"].state_dict(),
                    "epoch": epoch,
                    "val": val,
                    "baseline_run": args.baseline_run,
                    "config": {
                        "ch_node": ch_node, "ch_edge": ch_edge, "n_experts": K,
                        "router_arch": "mlp3x3",
                        "loss": "sigmoid_bce_per_bin",
                        "bins": BINS,
                        "lr": args.lr, "batch_size": args.batch_size,
                    },
                }, os.path.join(args.out_dir, "best.pt"))
                log(f"    → saved best (argmax_acc={best_acc:.2f}%)")
        else:
            log(f"ep {epoch:02d}  train_bce={train_loss:.4f}  ({train_time:.0f}s)")

        history.append(entry)
        with open(os.path.join(args.out_dir, "history.json"), "w") as f:
            json.dump(history, f, indent=2)
        with open(os.path.join(args.out_dir, "log.txt"), "w") as f:
            f.write("\n".join(log_lines))

    # ---- Final save ----
    torch.save({
        "router_node": routers["node"].state_dict(),
        "router_edge": routers["edge"].state_dict(),
        "epoch": args.epochs - 1,
        "val": history[-1].get("val"),
        "baseline_run": args.baseline_run,
        "config": {"loss": "sigmoid_bce_per_bin", "bins": BINS},
    }, os.path.join(args.out_dir, "last.pt"))

    hook_node.remove(); hook_edge.remove()
    log(f"\nDone. Best val argmax_acc: {best_acc:.2f}%")
    with open(os.path.join(args.out_dir, "log.txt"), "w") as f:
        f.write("\n".join(log_lines))


if __name__ == "__main__":
    main()
