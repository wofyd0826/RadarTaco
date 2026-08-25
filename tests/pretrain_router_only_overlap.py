"""Router-only pretraining with OVERLAP bins + sigmoid + BCE loss.

Extends tests/pretrain_router_only_bce.py by changing the LABEL DEFINITION
(bin edges are now overlapping), while keeping the loss (sigmoid + BCE)
and everything else the same.

Overlap bins (default): [0, 25], [15, 55], [45, max_depth]
  Overlap zones:
    15-25 m: near ∩ mid   → both memberships = 1
    45-55 m: mid ∩ far    → both memberships = 1
  Purpose: a 22 m boundary token has target (1, 1, 0) — near AND mid are
  both legitimate. No argmax flip noise. This is a proper multi-label task
  and requires per-bin sigmoid + BCE (softmax cannot represent sum > 1).

Comparison target:
  - tests/pretrain_router_only.py (softmax + soft CE, discrete)  = 90.30% argmax
  - tests/pretrain_router_only_bce.py (sigmoid + BCE, discrete)  = ?
  - THIS SCRIPT (sigmoid + BCE, overlap)                         = ?

Compared side by side, BCE-discrete vs BCE-overlap isolates the effect of
the label definition (overlap) with the loss held constant. The BCE-discrete
vs softmax-CE isolates the loss with the label held constant.

Notes:
  - Argmax accuracy is still reported for continuity, but is a WEAKER
    metric here — for overlap targets, "argmax of the target" is
    ill-defined when two bins tie (which happens exactly for the boundary
    tokens overlap is meant to fix). More meaningful metrics: multi-label
    F1 at thr=0.5, per-bin L1, BCE loss.
  - Best.pt is selected by best mean per-bin F1 (not argmax acc), since
    F1 is the metric that reflects the overlap-target intent.
"""
import argparse
import json
import os
import sys
import time
from typing import Dict, List, Tuple

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


# Default overlap bins. Format: list of (lo, hi). LAST bin is open-ended
# upward (matches discrete `pixel >= edges[-1]` inclusion rule).
DEFAULT_OVERLAP_BINS: List[Tuple[float, float]] = [
    (0.0, 25.0),   # near:  0-25 m  (overlaps mid at 15-25)
    (15.0, 55.0),  # mid:  15-55 m  (overlaps near at 15-25, far at 45-55)
    (45.0, 100.0), # far: 45-100 m  (overlaps mid at 45-55, open-ended above)
]
K = 3
LABELS = ["near", "mid", "far"]


def compute_overlap_target(depth_gt_dense: torch.Tensor, tok_hw,
                            overlap_bins: List[Tuple[float, float]]) -> torch.Tensor:
    """Per-token OVERLAP membership target — per-bin [0,1] independently.

    For each pixel and bin k: memb_k(pixel) = 1 if pixel_depth is in the
    bin's overlap range, else 0. The last bin extends open-ended upward.
    Aggregate to token grid via adaptive_avg_pool2d → per-token fraction
    of pixels that fall in that bin.

    Returns (B, K, H_tok, W_tok). Per-token sum can be > 1 (pixels in
    overlap zones contribute to multiple bins).
    """
    ds = depth_gt_dense.squeeze(1)                         # (B, H, W)
    K_local = len(overlap_bins)
    B, H, W = ds.shape
    memb = torch.zeros(B, K_local, H, W,
                       device=ds.device, dtype=torch.float32)
    for k, (lo, hi) in enumerate(overlap_bins):
        is_last = (k == K_local - 1)
        if is_last:
            memb[:, k] = (ds >= lo).float()
        else:
            memb[:, k] = ((ds >= lo) & (ds < hi)).float()
    return F.adaptive_avg_pool2d(memb, tok_hw)


def make_router(ch: int, n_experts: int) -> nn.Module:
    """mlp3x3 (v4c) — K logit outputs interpreted per bin as sigmoid."""
    h1 = min(256, ch // 2)
    h2 = h1 // 2
    return nn.Sequential(
        nn.Conv2d(ch, h1, 3, padding=1), nn.GELU(),
        nn.Conv2d(h1, h2, 3, padding=1), nn.GELU(),
        nn.Conv2d(h2, n_experts, 1),
    )


@torch.no_grad()
def evaluate(baseline, routers, captured, loader, device,
             overlap_bins: List[Tuple[float, float]]) -> Dict:
    """Per-block metrics. See module docstring for metric semantics under
    overlap targets."""
    for r in routers.values():
        r.eval()
    correct = {n: 0 for n in routers}      # argmax match (weakened for overlap)
    total = {n: 0 for n in routers}
    per_bin_tp = {n: torch.zeros(K, dtype=torch.long) for n in routers}
    per_bin_fp = {n: torch.zeros(K, dtype=torch.long) for n in routers}
    per_bin_fn = {n: torch.zeros(K, dtype=torch.long) for n in routers}
    per_bin_tn = {n: torch.zeros(K, dtype=torch.long) for n in routers}
    bce_sum = {n: 0.0 for n in routers}
    l1_sum = {n: 0.0 for n in routers}
    l1_ct = {n: 0 for n in routers}
    sig_sum_accum = {n: 0.0 for n in routers}
    sig_sum_ct = {n: 0 for n in routers}
    target_sum_accum = {n: 0.0 for n in routers}  # avg Σ_k frac_k (may be > 1)
    n_batches = 0

    for batch in loader:
        rgb = batch["rgb_norm"].to(device, non_blocking=True)
        rp = batch["radar_points"].to(device, non_blocking=True)
        rm = batch["radar_mask"].to(device, non_blocking=True)
        dgd = batch["depth_gt_dense"].to(device, non_blocking=True)

        _ = baseline(rgb, rp, rm)
        for name, router in routers.items():
            feat = captured[name]
            logits = router(feat)
            sig = torch.sigmoid(logits)
            target = compute_overlap_target(dgd, logits.shape[-2:], overlap_bins)

            # argmax accuracy (weakened under overlap, but included for continuity)
            gt = target.argmax(dim=1)
            pred = logits.argmax(dim=1)
            correct[name] += (pred == gt).sum().item()
            total[name] += gt.numel()

            # multi-label per-bin (thr=0.5)
            for k in range(K):
                t_pos = target[:, k] > 0.5
                p_pos = sig[:, k] > 0.5
                per_bin_tp[name][k] += (t_pos & p_pos).sum().item()
                per_bin_fp[name][k] += (~t_pos & p_pos).sum().item()
                per_bin_fn[name][k] += (t_pos & ~p_pos).sum().item()
                per_bin_tn[name][k] += (~t_pos & ~p_pos).sum().item()

            bce_sum[name] += F.binary_cross_entropy_with_logits(
                logits, target, reduction="mean").item()

            l1_per_tok = (sig - target).abs().sum(dim=1)
            l1_sum[name] += float(l1_per_tok.sum().item())
            l1_ct[name] += int(l1_per_tok.numel())

            sig_sum = sig.sum(dim=1)
            sig_sum_accum[name] += float(sig_sum.sum().item())
            sig_sum_ct[name] += int(sig_sum.numel())

            target_sum_accum[name] += float(target.sum(dim=1).sum().item())
        n_batches += 1

    out = {"per_block": {}}
    tot_c = 0; tot_n = 0
    for name in routers:
        acc = 100.0 * correct[name] / max(total[name], 1)
        tp = per_bin_tp[name].float()
        fp = per_bin_fp[name].float()
        fn = per_bin_fn[name].float()
        bin_recall = (100.0 * tp / (tp + fn).clamp_min(1)).tolist()
        bin_precision = (100.0 * tp / (tp + fp).clamp_min(1)).tolist()
        bin_f1 = [2 * p * r / max(p + r, 1e-9)
                  for p, r in zip(bin_precision, bin_recall)]
        mean_f1 = float(np.mean(bin_f1))
        target_pos_pct = [100.0 * (tp[k].item() + fn[k].item())
                          / max(total[name], 1) for k in range(K)]
        out["per_block"][name] = {
            "argmax_acc": acc,
            "tokens": total[name],
            "bce_loss": bce_sum[name] / max(n_batches, 1),
            "soft_l1": l1_sum[name] / max(l1_ct[name], 1),
            "sig_sum_mean": sig_sum_accum[name] / max(sig_sum_ct[name], 1),
            "target_sum_mean": target_sum_accum[name] / max(sig_sum_ct[name], 1),
            "bin_recall_thr0.5": bin_recall,
            "bin_precision_thr0.5": bin_precision,
            "bin_f1_thr0.5": bin_f1,
            "mean_f1": mean_f1,
            "target_positive_pct": target_pos_pct,
        }
        tot_c += correct[name]; tot_n += total[name]
    out["overall_argmax_acc"] = 100.0 * tot_c / max(tot_n, 1)
    # Best-selection metric: mean F1 averaged across blocks
    out["overall_mean_f1"] = float(np.mean([
        out["per_block"][n]["mean_f1"] for n in routers
    ]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline-run",
                    default="output/shape_lidar_grad_shape_edge_res_fix")
    ap.add_argument("--ckpt", default="best.pt")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=12)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--val-every", type=int, default=1)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--out-dir", default="output/router_pretrain")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--overlap-bins",
                    default="0:25,15:55,45:100",
                    help="Comma-separated lo:hi per bin. Last bin is open-ended upward.")
    args = ap.parse_args()

    # Parse overlap bins
    overlap_bins = []
    for part in args.overlap_bins.split(","):
        lo_s, hi_s = part.split(":")
        overlap_bins.append((float(lo_s), float(hi_s)))
    assert len(overlap_bins) == K, f"Need exactly {K} bins, got {len(overlap_bins)}"

    args.out_dir = os.path.join(args.out_dir, args.tag)
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"Output dir: {args.out_dir}")
    print(f"Overlap bins: {overlap_bins}")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- baseline (frozen) ----
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

    l2_node_block = baseline.radar_fusion.node_blocks[2]
    l2_edge_block = baseline.radar_fusion.edge_blocks[2]
    # `_FusionBlock` (baseline) exposes `norm1.num_channels`. `MoEFusionBlock`
    # (v4c, etc.) does not — its channel dim lives in the router's first
    # Conv2d. Handle both so we can pretrain a router on top of either kind
    # of base model.
    def _get_l2_ch(block):
        if hasattr(block, "norm1"):
            return block.norm1.num_channels
        # MoE block: probe router's first conv
        r = block.router
        if isinstance(r, nn.Sequential):
            first_conv = r[0]
        else:
            first_conv = r
        return first_conv.in_channels
    ch_node = _get_l2_ch(l2_node_block)
    ch_edge = _get_l2_ch(l2_edge_block)
    print(f"  L2 node ch={ch_node}, edge ch={ch_edge}  "
          f"(node block type: {type(l2_node_block).__name__})")

    routers = {
        "node": make_router(ch_node, K).to(device),
        "edge": make_router(ch_edge, K).to(device),
    }
    n_router_params = sum(sum(p.numel() for p in r.parameters())
                          for r in routers.values())
    print(f"  New router params (trainable): {n_router_params:,}")

    captured: Dict[str, torch.Tensor] = {}
    def make_pre_hook(name):
        def _h(_m, args_):
            captured[name] = args_[0]
        return _h
    hook_node = l2_node_block.register_forward_pre_hook(make_pre_hook("node"))
    hook_edge = l2_edge_block.register_forward_pre_hook(make_pre_hook("edge"))

    # ---- data ----
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

    trainable = [p for r in routers.values() for p in r.parameters()]
    opt = torch.optim.Adam(trainable, lr=args.lr, weight_decay=args.weight_decay)

    log_lines = []
    history = []
    best_f1 = 0.0
    def log(msg):
        print(msg); log_lines.append(msg)

    log(f"Config: LOSS=BCE (overlap bins)  bins={overlap_bins}  "
        f"epochs={args.epochs}  batch_size={args.batch_size}  "
        f"lr={args.lr}  wd={args.weight_decay}")

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

            tgt_node = compute_overlap_target(dgd, logits_node.shape[-2:], overlap_bins)
            tgt_edge = compute_overlap_target(dgd, logits_edge.shape[-2:], overlap_bins)

            loss_node = F.binary_cross_entropy_with_logits(logits_node, tgt_node)
            loss_edge = F.binary_cross_entropy_with_logits(logits_edge, tgt_edge)
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
            val = evaluate(baseline, routers, captured, val_loader, device,
                           overlap_bins)
            val_time = time.time() - t1
            entry["val"] = val
            entry["val_time_sec"] = val_time
            log(f"ep {epoch:02d}  train_bce={train_loss:.4f}  "
                f"({train_time:.0f}s)  |  val mean_F1={val['overall_mean_f1']:.4f}  "
                f"argmax_acc={val['overall_argmax_acc']:.2f}%  (val {val_time:.0f}s)")
            for name in ["node", "edge"]:
                b = val["per_block"][name]
                br = b["bin_recall_thr0.5"]
                bp = b["bin_precision_thr0.5"]
                bf1 = b["bin_f1_thr0.5"]
                tp = b["target_positive_pct"]
                log(f"    {name}: bce={b['bce_loss']:.4f}  L1={b['soft_l1']:.4f}  "
                    f"sig_sum={b['sig_sum_mean']:.3f}  target_sum={b['target_sum_mean']:.3f}  "
                    f"F1_mean={b['mean_f1']:.4f}")
                log(f"      target pos%: near={tp[0]:.1f}%  mid={tp[1]:.1f}%  far={tp[2]:.1f}%")
                log(f"      recall (thr=0.5):    near={br[0]:.1f}%  mid={br[1]:.1f}%  far={br[2]:.1f}%")
                log(f"      precision (thr=0.5): near={bp[0]:.1f}%  mid={bp[1]:.1f}%  far={bp[2]:.1f}%")
                log(f"      F1 (thr=0.5):        near={bf1[0]:.2f}   mid={bf1[1]:.2f}   far={bf1[2]:.2f}")

            if val["overall_mean_f1"] > best_f1:
                best_f1 = val["overall_mean_f1"]
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
                        "overlap_bins": overlap_bins,
                        "lr": args.lr, "batch_size": args.batch_size,
                    },
                }, os.path.join(args.out_dir, "best.pt"))
                log(f"    → saved best (mean_F1={best_f1:.4f})")
        else:
            log(f"ep {epoch:02d}  train_bce={train_loss:.4f}  ({train_time:.0f}s)")

        history.append(entry)
        with open(os.path.join(args.out_dir, "history.json"), "w") as f:
            json.dump(history, f, indent=2)
        with open(os.path.join(args.out_dir, "log.txt"), "w") as f:
            f.write("\n".join(log_lines))

    torch.save({
        "router_node": routers["node"].state_dict(),
        "router_edge": routers["edge"].state_dict(),
        "epoch": args.epochs - 1,
        "val": history[-1].get("val"),
        "baseline_run": args.baseline_run,
        "config": {"loss": "sigmoid_bce_per_bin", "overlap_bins": overlap_bins},
    }, os.path.join(args.out_dir, "last.pt"))

    hook_node.remove(); hook_edge.remove()
    log(f"\nDone. Best val mean_F1: {best_f1:.4f}")
    with open(os.path.join(args.out_dir, "log.txt"), "w") as f:
        f.write("\n".join(log_lines))


if __name__ == "__main__":
    main()
