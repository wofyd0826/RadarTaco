"""Router-only pretraining on a frozen baseline encoder + fusion path.

Loads a baseline (single-fusion) RadarTaco model, freezes every parameter,
and trains only two NEW router heads (matching the v4c mlp3x3 architecture,
one for L2 node block, one for L2 edge block) via soft cross-entropy against
per-token pixel-bin fraction targets.

Purpose
-------
Diagnose the baseline encoder's routing potential without interference from
expert learning or depth loss. If router accuracy plateaus at ~90% (same as
v4c), we know feature quality is NOT the bottleneck — CE label noise is.
If it plateaus lower, encoder features themselves limit routing.

Setup
-----
Baseline model provides (frozen):
  image_encoder → aux_branch(opt) → radar_encoder → radar_fusion (L0, L1, L2)

Two new routers (trainable):
  router_node(feat_L2_node) → logits (B, K, 29, 50)
  router_edge(feat_L2_edge) → logits (B, K, 15, 25)

Loss: soft CE against per-token pixel-bin fractions (matches v4c training).

The L2 fusion block INPUTS (feat) are captured via forward_pre_hook so we
reuse the standard model.forward path without modification.

Outputs
-------
  <out_dir>/<tag>_best.pt         — best router state_dicts by val overall acc
  <out_dir>/<tag>_last.pt         — final router state_dicts
  <out_dir>/<tag>_log.txt         — training log
  <out_dir>/<tag>_history.json    — per-epoch metrics for plotting

Example
-------
  python tests/pretrain_router_only.py \
      --tag baseline_encoder \
      --epochs 50 --batch-size 8 --lr 1e-3
"""
import argparse, json, os, sys, time
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
    """Per-token pixel-bin fraction. Returns (B, K, H_tok, W_tok) float that
    sums to 1 along the K axis — the soft CE target."""
    edges = torch.tensor(bins, device=depth_gt_dense.device, dtype=torch.float32)
    ds = depth_gt_dense.squeeze(1)                                       # (B, H, W)
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
    """Same architecture as v4c's `moe_router_arch=mlp3x3`."""
    h1 = min(256, ch // 2)
    h2 = h1 // 2
    return nn.Sequential(
        nn.Conv2d(ch, h1, 3, padding=1), nn.GELU(),
        nn.Conv2d(h1, h2, 3, padding=1), nn.GELU(),
        nn.Conv2d(h2, n_experts, 1),
    )


@torch.no_grad()
def evaluate(baseline, routers, captured, loader, device) -> Dict:
    """Return dict of metrics: per-block hard accuracy + soft CE loss."""
    for r in routers.values():
        r.eval()
    correct = {name: 0 for name in routers}
    total = {name: 0 for name in routers}
    per_class_correct = {name: torch.zeros(K, dtype=torch.long) for name in routers}
    per_class_total = {name: torch.zeros(K, dtype=torch.long) for name in routers}
    soft_ce_sum = {name: 0.0 for name in routers}
    n_batches = 0

    for batch in loader:
        rgb = batch["rgb_norm"].to(device, non_blocking=True)
        rp = batch["radar_points"].to(device, non_blocking=True)
        rm = batch["radar_mask"].to(device, non_blocking=True)
        dgd = batch["depth_gt_dense"].to(device, non_blocking=True)

        _ = baseline(rgb, rp, rm)   # hooks populate `captured`
        for name, router in routers.items():
            feat = captured[name]
            logits = router(feat)
            frac = compute_frac(dgd, logits.shape[-2:], BINS)

            gt = frac.argmax(dim=1)                                       # (B, H, W)
            pred = logits.argmax(dim=1)
            correct[name] += (pred == gt).sum().item()
            total[name] += gt.numel()

            for k in range(K):
                mask_k = (gt == k)
                per_class_total[name][k] += mask_k.sum().item()
                per_class_correct[name][k] += (mask_k & (pred == k)).sum().item()

            soft_ce_sum[name] += F.cross_entropy(logits, frac).item()
        n_batches += 1

    out = {"per_block": {}}
    tot_c = 0; tot_n = 0
    for name in routers:
        acc = 100.0 * correct[name] / max(total[name], 1)
        rec = [100.0 * per_class_correct[name][k].item()
               / max(per_class_total[name][k].item(), 1) for k in range(K)]
        out["per_block"][name] = {
            "acc": acc,
            "recall": rec,
            "tokens": total[name],
            "soft_ce": soft_ce_sum[name] / max(n_batches, 1),
        }
        tot_c += correct[name]; tot_n += total[name]
    out["overall_acc"] = 100.0 * tot_c / max(tot_n, 1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline-run",
                    default="output/shape_lidar_grad_shape_edge_res_fix",
                    help="Baseline run dir with config.yaml + best.pt")
    ap.add_argument("--ckpt", default="best.pt")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--val-every", type=int, default=1,
                    help="Run val every N epochs")
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--out-dir", default="output/router_pretrain")
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()

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
    # `forward_pre_hook(module, args)` — args is tuple of positional args,
    # first is `feat`. We store a reference (baseline runs under no_grad,
    # so the tensor doesn't hold gradients from the baseline side).
    captured: Dict[str, torch.Tensor] = {}
    def make_pre_hook(name):
        def _h(_m, args):
            captured[name] = args[0]
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

    os.makedirs(args.out_dir, exist_ok=True)
    log_lines = []
    history = []
    best_acc = 0.0

    def log(msg):
        print(msg)
        log_lines.append(msg)

    log(f"Config: epochs={args.epochs}  batch_size={args.batch_size}  "
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
            # captured['node'/'edge'] are set by pre_hooks; detach so we
            # don't retain baseline's (empty) graph.
            feat_node = captured["node"].detach()
            feat_edge = captured["edge"].detach()

            logits_node = routers["node"](feat_node)
            logits_edge = routers["edge"](feat_edge)

            frac_node = compute_frac(dgd, logits_node.shape[-2:], BINS)
            frac_edge = compute_frac(dgd, logits_edge.shape[-2:], BINS)

            loss_node = F.cross_entropy(logits_node, frac_node)
            loss_edge = F.cross_entropy(logits_edge, frac_edge)
            loss = 0.5 * (loss_node + loss_edge)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            train_losses.append(loss.item())
            if (it + 1) % 100 == 0:
                pbar.set_postfix({"loss": f"{np.mean(train_losses[-100:]):.4f}"})
        train_time = time.time() - t0
        train_loss = float(np.mean(train_losses))

        # ---- Val ----
        entry = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_time_sec": train_time,
        }
        if (epoch + 1) % args.val_every == 0 or epoch == args.epochs - 1:
            t1 = time.time()
            val = evaluate(baseline, routers, captured, val_loader, device)
            val_time = time.time() - t1
            entry["val"] = val
            entry["val_time_sec"] = val_time

            log(f"ep {epoch:02d}  train_loss={train_loss:.4f}  "
                f"({train_time:.0f}s)  |  val overall_acc={val['overall_acc']:.2f}%  "
                f"node={val['per_block']['node']['acc']:.2f}%  "
                f"edge={val['per_block']['edge']['acc']:.2f}%  "
                f"(val {val_time:.0f}s)")
            for name in ["node", "edge"]:
                r = val["per_block"][name]["recall"]
                log(f"    {name} recall: near={r[0]:.1f}%  mid={r[1]:.1f}%  "
                    f"far={r[2]:.1f}%  soft_ce={val['per_block'][name]['soft_ce']:.4f}")

            if val["overall_acc"] > best_acc:
                best_acc = val["overall_acc"]
                torch.save({
                    "router_node": routers["node"].state_dict(),
                    "router_edge": routers["edge"].state_dict(),
                    "epoch": epoch,
                    "val": val,
                    "baseline_run": args.baseline_run,
                    "config": {
                        "ch_node": ch_node, "ch_edge": ch_edge, "n_experts": K,
                        "router_arch": "mlp3x3",
                        "bins": BINS,
                        "lr": args.lr, "batch_size": args.batch_size,
                    },
                }, os.path.join(args.out_dir, f"{args.tag}_best.pt"))
                log(f"    → saved best (acc={best_acc:.2f}%)")
        else:
            log(f"ep {epoch:02d}  train_loss={train_loss:.4f}  ({train_time:.0f}s)")

        history.append(entry)

        # Rolling save of history + log
        with open(os.path.join(args.out_dir, f"{args.tag}_history.json"), "w") as f:
            json.dump(history, f, indent=2)
        with open(os.path.join(args.out_dir, f"{args.tag}_log.txt"), "w") as f:
            f.write("\n".join(log_lines))

    # ---- Final save ----
    torch.save({
        "router_node": routers["node"].state_dict(),
        "router_edge": routers["edge"].state_dict(),
        "epoch": args.epochs - 1,
        "val": history[-1].get("val"),
        "baseline_run": args.baseline_run,
    }, os.path.join(args.out_dir, f"{args.tag}_last.pt"))

    hook_node.remove(); hook_edge.remove()
    log(f"\nDone. Best val overall_acc: {best_acc:.2f}%")
    log(f"Saved: {args.out_dir}/{args.tag}_{{best,last}}.pt")
    with open(os.path.join(args.out_dir, f"{args.tag}_log.txt"), "w") as f:
        f.write("\n".join(log_lines))


if __name__ == "__main__":
    main()
