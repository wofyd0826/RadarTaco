"""End-to-end MONO-DEPTH-to-bin classifier for router-routing diagnostic.

Twin of tests/pretrain_image_classifier.py — SAME architecture and training
recipe, but the input is the pre-computed relative-depth PNG (Marigold-style
mono-depth prior) instead of the RGB image.

Design decisions:
  • Backbone: ResNet18 (ImageNet init) via `ImageEncoder` — identical to the
    RGB variant so the ONLY difference is the input modality.
  • 1-channel rel_depth is repeated to 3 channels before the encoder, keeping
    the pretrained conv1 weights intact.
  • rel_depth is loaded raw (PNG / 1000 → [0, 1]); no ImageNet normalization
    applied (mono-depth statistics differ from natural images).

Purpose:
  Threshold-only baseline on rel_depth achieves 83.6% argmax accuracy vs the
  v4c learned router's 90.6%. This experiment asks: can a CNN pull more
  routing signal out of mono-depth alone, with per-image adaptation +
  spatial context? Answer tells us whether mono-depth is a promising router
  feature (to be combined with RGB in a follow-up) or an intrinsic dead-end.

Reference accuracy points (for comparison at the end of training):
    v4c learned router (RGB-based)  : 90.61%
    RGB image classifier (twin exp) : 90.30%
    Mono threshold-only fit         : 83.58%

Outputs (mirroring pretrain_image_classifier.py):
    output/router_pretrain/<tag>/{best,last}.pt, history.json, log.txt

Example:
    python tests/pretrain_mono_depth_classifier.py --tag mono_classifier
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

from src.dataset.nuscenes import NuScenesRadarDepthDataset  # noqa: E402
from src.model.image_encoder import ImageEncoder            # noqa: E402


BINS = [0.0, 20.0, 50.0, 100.0]
K = 3
LABELS = ["near", "mid", "far"]

# rel_depth PNG directory under `data_root` (from earlier diagnostic).
REL_DEPTH_DIR = "relative_depth"


def compute_frac(depth_gt_dense: torch.Tensor, tok_hw, bins) -> torch.Tensor:
    """Per-token pixel-bin fraction (verbatim from pretrain_image_classifier.py)."""
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


class MonoDepthBinClassifier(nn.Module):
    """ResNet18 (ImageNet init) + two 1x1 conv heads — takes 1-channel
    rel_depth (repeated to 3 channels) instead of RGB.

    Head structure identical to the RGB variant so any difference in val
    accuracy is attributable purely to the input modality.
    """
    def __init__(self, n_bins: int = K, pretrained: bool = True):
        super().__init__()
        self.encoder = ImageEncoder(pretrained=pretrained)
        ch_node = self.encoder.feat_channels[4]   # 512
        ch_edge = self.encoder.feat_channels[5]   # 512
        self.head_node = nn.Conv2d(ch_node, n_bins, kernel_size=1)
        self.head_edge = nn.Conv2d(ch_edge, n_bins, kernel_size=1)

    def forward(self, rel_depth: torch.Tensor):
        """rel_depth: (B, 1, H, W). Repeated to 3 channels for the pretrained
        RGB-shaped encoder. Returns dict with 'node' and 'edge' logits."""
        if rel_depth.shape[1] == 1:
            rel_depth = rel_depth.repeat(1, 3, 1, 1)
        feats = self.encoder(rel_depth)
        return {
            "node": self.head_node(feats[4]),
            "edge": self.head_edge(feats[5]),
        }


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: str) -> Dict:
    model.eval()
    correct = {"node": 0, "edge": 0}
    total = {"node": 0, "edge": 0}
    per_class_correct = {n: torch.zeros(K, dtype=torch.long) for n in ("node", "edge")}
    per_class_total = {n: torch.zeros(K, dtype=torch.long) for n in ("node", "edge")}
    soft_ce_sum = {"node": 0.0, "edge": 0.0}
    n_batches = 0

    for batch in loader:
        rd = batch["rel_depth"].to(device, non_blocking=True)
        dgd = batch["depth_gt_dense"].to(device, non_blocking=True)
        logits = model(rd)
        for name in ("node", "edge"):
            lg = logits[name]
            frac = compute_frac(dgd, lg.shape[-2:], BINS)
            gt = frac.argmax(dim=1)
            pred = lg.argmax(dim=1)
            correct[name] += (pred == gt).sum().item()
            total[name] += gt.numel()
            for k in range(K):
                mask_k = (gt == k)
                per_class_total[name][k] += mask_k.sum().item()
                per_class_correct[name][k] += (mask_k & (pred == k)).sum().item()
            soft_ce_sum[name] += F.cross_entropy(lg, frac).item()
        n_batches += 1

    out = {"per_block": {}}
    tot_c = 0
    tot_n = 0
    for name in ("node", "edge"):
        acc = 100.0 * correct[name] / max(total[name], 1)
        rec = [100.0 * per_class_correct[name][k].item()
               / max(per_class_total[name][k].item(), 1) for k in range(K)]
        out["per_block"][name] = {
            "acc": acc,
            "recall": rec,
            "tokens": total[name],
            "soft_ce": soft_ce_sum[name] / max(n_batches, 1),
        }
        tot_c += correct[name]
        tot_n += total[name]
    out["overall_acc"] = 100.0 * tot_c / max(tot_n, 1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline-run",
                    default="output/shape_lidar_grad_shape_edge_res_fix",
                    help="Only used to pick up dataset paths from its config.yaml.")
    ap.add_argument("--rel-depth-dir", default=REL_DEPTH_DIR,
                    help="Directory under data_root holding rel_depth PNGs.")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=12)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--val-every", type=int, default=1)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--out-dir", default="output/router_pretrain")
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()

    args.out_dir = os.path.join(args.out_dir, args.tag)
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"Output dir: {args.out_dir}")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    cfg_path = os.path.join(args.baseline_run, "config.yaml")
    cfg = OmegaConf.load(cfg_path)
    ds_root = cfg.dataset.data_root

    model = MonoDepthBinClassifier().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  MonoDepthBinClassifier params: {n_params:,}  ({n_params/1e6:.2f}M)")

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
            rel_depth_dir=args.rel_depth_dir,
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

    opt = torch.optim.Adam(model.parameters(),
                            lr=args.lr, weight_decay=args.weight_decay)

    log_lines = []
    history = []
    best_acc = 0.0

    def log(msg):
        print(msg)
        log_lines.append(msg)

    log(f"Config: epochs={args.epochs}  batch_size={args.batch_size}  "
        f"lr={args.lr}  wd={args.weight_decay}  input=rel_depth (repeated 3x)")

    for epoch in range(args.epochs):
        model.train()
        t0 = time.time()
        train_losses = []
        pbar = tqdm(train_loader, desc=f"ep{epoch:02d} train", leave=False)
        for it, batch in enumerate(pbar):
            rd = batch["rel_depth"].to(device, non_blocking=True)
            dgd = batch["depth_gt_dense"].to(device, non_blocking=True)
            logits = model(rd)
            loss_node = F.cross_entropy(
                logits["node"], compute_frac(dgd, logits["node"].shape[-2:], BINS))
            loss_edge = F.cross_entropy(
                logits["edge"], compute_frac(dgd, logits["edge"].shape[-2:], BINS))
            loss = 0.5 * (loss_node + loss_edge)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            train_losses.append(loss.item())
            if (it + 1) % 100 == 0:
                pbar.set_postfix({"loss": f"{np.mean(train_losses[-100:]):.4f}"})
        train_time = time.time() - t0
        train_loss = float(np.mean(train_losses))

        entry = {"epoch": epoch, "train_loss": train_loss,
                 "train_time_sec": train_time}
        if (epoch + 1) % args.val_every == 0 or epoch == args.epochs - 1:
            t1 = time.time()
            val = evaluate(model, val_loader, device)
            val_time = time.time() - t1
            entry["val"] = val
            entry["val_time_sec"] = val_time
            log(f"ep {epoch:02d}  train_loss={train_loss:.4f}  "
                f"({train_time:.0f}s)  |  val overall_acc={val['overall_acc']:.2f}%  "
                f"node={val['per_block']['node']['acc']:.2f}%  "
                f"edge={val['per_block']['edge']['acc']:.2f}%  "
                f"(val {val_time:.0f}s)")
            for name in ("node", "edge"):
                b = val["per_block"][name]
                r = b["recall"]
                log(f"    {name} recall: near={r[0]:.1f}%  mid={r[1]:.1f}%  "
                    f"far={r[2]:.1f}%  soft_ce={b['soft_ce']:.4f}")

            if val["overall_acc"] > best_acc:
                best_acc = val["overall_acc"]
                torch.save({
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "val": val,
                    "baseline_run": args.baseline_run,
                    "config": {
                        "bins": BINS, "n_experts": K,
                        "head_type": "conv1x1",
                        "backbone": "resnet18_imagenet",
                        "input": "rel_depth (1ch → 3ch repeat)",
                        "rel_depth_dir": args.rel_depth_dir,
                        "lr": args.lr, "batch_size": args.batch_size,
                        "epochs": args.epochs,
                    },
                }, os.path.join(args.out_dir, "best.pt"))
                log(f"    → saved best (acc={best_acc:.2f}%)")
        else:
            log(f"ep {epoch:02d}  train_loss={train_loss:.4f}  ({train_time:.0f}s)")

        history.append(entry)
        with open(os.path.join(args.out_dir, "history.json"), "w") as f:
            json.dump(history, f, indent=2)
        with open(os.path.join(args.out_dir, "log.txt"), "w") as f:
            f.write("\n".join(log_lines))

    torch.save({
        "model": model.state_dict(),
        "epoch": args.epochs - 1,
        "val": history[-1].get("val"),
        "baseline_run": args.baseline_run,
        "config": {
            "bins": BINS, "n_experts": K,
            "head_type": "conv1x1",
            "backbone": "resnet18_imagenet",
            "input": "rel_depth (1ch → 3ch repeat)",
            "rel_depth_dir": args.rel_depth_dir,
            "lr": args.lr, "batch_size": args.batch_size,
            "epochs": args.epochs,
        },
    }, os.path.join(args.out_dir, "last.pt"))
    log(f"\nDone. Best val overall_acc: {best_acc:.2f}%")
    with open(os.path.join(args.out_dir, "log.txt"), "w") as f:
        f.write("\n".join(log_lines))


if __name__ == "__main__":
    main()
