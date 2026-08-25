"""Router pretrain with frozen DINOv2 ViT-S/14 backbone + trainable adapter.

Design:
  RGB (900, 1600) ─resize─→ (896, 1568)
      │
      ▼  ImageNet normalize
  DINOv2 ViT-S/14 (frozen, bf16 autocast)
      │
      ▼  x_norm_patchtokens: (B, 7168, 384) → (B, 384, 64, 112)
  Adapter: 3 × ResBlock(384ch, GroupNorm-16, GELU, residual)
      │
  Head: Conv1x1 (384 → 3)
      │
      ▼
  Logits: (B, 3, 64, 112)   ← per-patch (14×14 pixel) depth-bin prediction

Target: pixel_fraction at (64, 112) from depth_gt_dense (soft CE).
Only adapter + head are trainable. DINOv2 stays frozen.

Grid: 64×112 = 7,168 tokens/image, each 14×14 pixel.
For reference the L2 baseline router uses 29×50 (1,450 tokens/img); L0
experiment used 113×200 (22,600 tokens). This sits between them.

Comparison points (v4c stage2 test set):
  L2 baseline_encoder pretrain:                 90.30%
  L2 v4c router (in-MoE):                       90.61%
  L2 mono classifier (frozen ResNet18):         86.55%
  L0 pretrain_router_only (feats[2] 128ch):     ~77%  (feature quality limited)
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


BINS = [0.0, 20.0, 50.0, 100.0]
K = 3
LABELS = ["near", "mid", "far"]

# Input size divisible by DINOv2 patch (14). 900×1600 → 896×1568.
INPUT_H = 896
INPUT_W = 1568
PATCH_H = INPUT_H // 14   # 64
PATCH_W = INPUT_W // 14   # 112

# ImageNet stats for DINOv2. rgb_norm from dataset is in [-1, 1]; convert.
_IN_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_IN_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def compute_frac(depth_gt_dense: torch.Tensor, tok_hw, bins) -> torch.Tensor:
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


def rgb_norm_to_dinov2(rgb_norm: torch.Tensor) -> torch.Tensor:
    """rgb_norm ∈ [-1, 1] → ImageNet-normalized for DINOv2, resized to
    (INPUT_H, INPUT_W) so the patch grid is exactly PATCH_H × PATCH_W."""
    # Denormalize [-1,1] → [0,1]
    rgb01 = (rgb_norm + 1.0) * 0.5
    mean = _IN_MEAN.to(rgb01.device, rgb01.dtype)
    std = _IN_STD.to(rgb01.device, rgb01.dtype)
    x = (rgb01 - mean) / std
    if x.shape[-2] != INPUT_H or x.shape[-1] != INPUT_W:
        x = F.interpolate(x, size=(INPUT_H, INPUT_W),
                           mode="bilinear", align_corners=False, antialias=True)
    return x


class ResBlock(nn.Module):
    def __init__(self, ch: int, gn_groups: int = 16):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1)
        self.norm1 = nn.GroupNorm(gn_groups, ch)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(gn_groups, ch)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r = self.act(self.norm1(self.conv1(x)))
        r = self.norm2(self.conv2(r))
        return self.act(r + x)


class DINOv2RouterHead(nn.Module):
    """3 × ResBlock adapter + Conv1x1 head. Operates on DINOv2 patch features."""
    def __init__(self, ch: int = 384, n_blocks: int = 3, n_bins: int = K):
        super().__init__()
        self.adapter = nn.Sequential(*[ResBlock(ch) for _ in range(n_blocks)])
        self.head = nn.Conv2d(ch, n_bins, kernel_size=1)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        return self.head(self.adapter(feat))


@torch.no_grad()
def dinov2_features(dinov2: nn.Module, rgb: torch.Tensor) -> torch.Tensor:
    """Run DINOv2 in bf16 autocast; return (B, 384, PATCH_H, PATCH_W) fp32."""
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        out = dinov2.forward_features(rgb)
    pt = out["x_norm_patchtokens"].float()          # (B, N, 384)
    B, N, D = pt.shape
    assert N == PATCH_H * PATCH_W, (N, PATCH_H, PATCH_W)
    feat = pt.reshape(B, PATCH_H, PATCH_W, D).permute(0, 3, 1, 2).contiguous()
    return feat                                     # (B, 384, 64, 112)


@torch.no_grad()
def evaluate(dinov2, router_head, loader, device) -> Dict:
    router_head.eval()
    correct = 0; total = 0
    per_class_correct = torch.zeros(K, dtype=torch.long)
    per_class_total = torch.zeros(K, dtype=torch.long)
    soft_ce_sum = 0.0
    n_batches = 0

    for batch in loader:
        rgb = batch["rgb_norm"].to(device, non_blocking=True)
        dgd = batch["depth_gt_dense"].to(device, non_blocking=True)
        x = rgb_norm_to_dinov2(rgb)
        feat = dinov2_features(dinov2, x)
        logits = router_head(feat)                  # (B, 3, 64, 112)

        frac = compute_frac(dgd, logits.shape[-2:], BINS)
        gt = frac.argmax(dim=1)
        pred = logits.argmax(dim=1)
        correct += (pred == gt).sum().item()
        total += gt.numel()
        for k in range(K):
            m = gt == k
            per_class_total[k] += m.sum().item()
            per_class_correct[k] += (m & (pred == k)).sum().item()
        soft_ce_sum += F.cross_entropy(logits, frac).item()
        n_batches += 1

    acc = 100.0 * correct / max(total, 1)
    rec = [100.0 * per_class_correct[k].item() / max(per_class_total[k].item(), 1)
           for k in range(K)]
    return {
        "overall_acc": acc,
        "recall": rec,
        "tokens": total,
        "soft_ce": soft_ce_sum / max(n_batches, 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline-run",
                    default="output/shape_lidar_grad_shape_edge_res_fix",
                    help="Only used for dataset paths from its config.yaml.")
    ap.add_argument("--dinov2-model", default="dinov2_vits14",
                    choices=["dinov2_vits14", "dinov2_vitb14",
                             "dinov2_vitl14"])
    ap.add_argument("--n-blocks", type=int, default=3,
                    help="Adapter ResBlock count")
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

    # ============ Frozen DINOv2 ============
    print(f"Loading DINOv2: {args.dinov2_model}")
    dinov2 = torch.hub.load(
        "facebookresearch/dinov2", args.dinov2_model,
        pretrained=True, source="github", verbose=False).to(device).eval()
    for p in dinov2.parameters():
        p.requires_grad = False
    print(f"  DINOv2 params (frozen): "
          f"{sum(p.numel() for p in dinov2.parameters()):,}")

    # ============ Trainable adapter + head ============
    ch = int(dinov2.embed_dim) if hasattr(dinov2, "embed_dim") else 384
    print(f"  DINOv2 embed dim: {ch}")
    head = DINOv2RouterHead(ch=ch, n_blocks=args.n_blocks).to(device)
    n_train = sum(p.numel() for p in head.parameters())
    print(f"  Adapter+head params (trainable): {n_train:,}  "
          f"({n_train/1e6:.2f}M)")

    # ============ Data ============
    cfg = OmegaConf.load(os.path.join(args.baseline_run, "config.yaml"))
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
    print(f"  Token grid: {PATCH_H}×{PATCH_W} = {PATCH_H*PATCH_W} tokens/image")

    opt = torch.optim.Adam(head.parameters(),
                            lr=args.lr, weight_decay=args.weight_decay)

    log_lines = []
    history = []
    best_acc = 0.0
    def log(msg):
        print(msg); log_lines.append(msg)

    log(f"Config: epochs={args.epochs}  batch_size={args.batch_size}  "
        f"lr={args.lr}  wd={args.weight_decay}  n_blocks={args.n_blocks}  "
        f"backbone={args.dinov2_model}  grid={PATCH_H}×{PATCH_W}")

    for epoch in range(args.epochs):
        head.train()
        t0 = time.time()
        train_losses = []
        pbar = tqdm(train_loader, desc=f"ep{epoch:02d} train", leave=False)
        for it, batch in enumerate(pbar):
            rgb = batch["rgb_norm"].to(device, non_blocking=True)
            dgd = batch["depth_gt_dense"].to(device, non_blocking=True)
            x = rgb_norm_to_dinov2(rgb)
            feat = dinov2_features(dinov2, x)         # (B, 384, 64, 112)

            logits = head(feat)                        # (B, 3, 64, 112)
            frac = compute_frac(dgd, logits.shape[-2:], BINS)
            loss = F.cross_entropy(logits, frac)
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
            val = evaluate(dinov2, head, val_loader, device)
            val_time = time.time() - t1
            entry["val"] = val
            entry["val_time_sec"] = val_time
            r = val["recall"]
            log(f"ep {epoch:02d}  train_loss={train_loss:.4f}  "
                f"({train_time:.0f}s)  |  val acc={val['overall_acc']:.2f}%  "
                f"(near={r[0]:.1f}%  mid={r[1]:.1f}%  far={r[2]:.1f}%  "
                f"soft_ce={val['soft_ce']:.4f})  (val {val_time:.0f}s)")

            if val["overall_acc"] > best_acc:
                best_acc = val["overall_acc"]
                torch.save({
                    "head": head.state_dict(),
                    "epoch": epoch,
                    "val": val,
                    "config": {
                        "backbone": args.dinov2_model,
                        "n_blocks": args.n_blocks,
                        "grid_hw": [PATCH_H, PATCH_W],
                        "input_hw": [INPUT_H, INPUT_W],
                        "bins": BINS,
                        "lr": args.lr, "batch_size": args.batch_size,
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
        "head": head.state_dict(),
        "epoch": args.epochs - 1,
        "val": history[-1].get("val"),
        "config": {"backbone": args.dinov2_model,
                    "n_blocks": args.n_blocks,
                    "grid_hw": [PATCH_H, PATCH_W]},
    }, os.path.join(args.out_dir, "last.pt"))
    log(f"\nDone. Best val overall_acc: {best_acc:.2f}%")


if __name__ == "__main__":
    main()
