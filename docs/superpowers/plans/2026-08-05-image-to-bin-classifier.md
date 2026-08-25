# Image-to-Bin Classifier Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement a single-file diagnostic training script `tests/pretrain_image_classifier.py` that trains an ImageNet-init ResNet18 end-to-end to classify per-token depth bins at both router grid resolutions (29×50 and 15×25) and reports val accuracy for comparison against the 90.3% baseline.

**Architecture:** One `nn.Module` (`BinClassifier`) that wraps `src.model.image_encoder.ImageEncoder` and adds two independent 1×1 conv heads consuming `feats[4]` and `feats[5]`. Training uses soft cross-entropy against `adaptive_avg_pool2d`-derived pixel-bin fractions — same target used by `pretrain_router_only.py` so the resulting val numbers are directly comparable to prior runs.

**Tech Stack:** PyTorch (Adam, `nn.CrossEntropyLoss`), `torch.utils.data.DataLoader`, existing `NuScenesRadarDepthDataset`, `OmegaConf` for reading baseline `config.yaml` (only to pick up dataset paths — no model config used), `tqdm` for progress.

Reference files:
- Spec: `docs/superpowers/specs/2026-08-05-image-to-bin-classifier-design.md`
- Sibling scripts to follow structurally: `tests/pretrain_router_only.py`, `tests/pretrain_router_only_radar.py`

---

## File Structure

**Create:**
- `tests/pretrain_image_classifier.py` — single self-contained script.

**Do NOT modify:**
- Any existing model / trainer / dataset / config file.
- Existing pretrain scripts.

Script sections (top-to-bottom order in the file):
1. Module docstring + imports + `ROOT` `sys.path` shim.
2. Constants (`BINS`, `K`, `LABELS`).
3. `compute_frac()` helper (copied verbatim from `pretrain_router_only.py`).
4. `BinClassifier(nn.Module)` — encoder + two heads.
5. `evaluate()` — @torch.no_grad, returns per-block metrics dict.
6. `main()` — argparse, out-dir setup, dataset/loader, optim, train loop with val + save.
7. `if __name__ == "__main__": main()`.

---

## Task 1: Script skeleton (docstring, imports, constants, helper)

**Files:**
- Create: `tests/pretrain_image_classifier.py`

- [ ] **Step 1: Write the skeleton with docstring, imports, constants, and `compute_frac` helper.**

```python
"""End-to-end image-to-bin classifier for router-routing diagnostic.

Trains an ImageNet-initialized ResNet18 (baseline's ImageEncoder) end-to-end
with soft cross-entropy against per-token pixel-bin fractions at BOTH router
grid resolutions (node: 29x50 from feats[4]; edge: 15x25 from feats[5]).

Purpose (see docs/superpowers/specs/2026-08-05-image-to-bin-classifier-design.md):
    Test whether the ~90.3% router-accuracy ceiling seen in
    output/router_pretrain/baseline_encoder/ is caused by the baseline
    encoder having been trained for depth-regression rather than
    bin-classification. Same architecture, different training objective.

Comparison target: baseline_encoder pretrain overall_acc = 90.30%.

Outputs (mirroring pretrain_router_only_radar.py layout):
    output/router_pretrain/<tag>/best.pt      # {"model", "epoch", "val", "config"}
    output/router_pretrain/<tag>/last.pt
    output/router_pretrain/<tag>/history.json
    output/router_pretrain/<tag>/log.txt

Example:
    python tests/pretrain_image_classifier.py --tag image_classifier
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


def compute_frac(depth_gt_dense: torch.Tensor, tok_hw, bins) -> torch.Tensor:
    """Per-token pixel-bin fraction. Returns (B, K, H_tok, W_tok) float that
    sums to 1 along the K axis — the soft CE target.

    Copied verbatim from tests/pretrain_router_only.py so the two scripts'
    numbers are directly comparable.
    """
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
```

- [ ] **Step 2: Verify the file parses.**

Run: `cd /workspace/RadarTaco && python -c "import ast; ast.parse(open('tests/pretrain_image_classifier.py').read()); print('OK')"`
Expected output: `OK`

- [ ] **Step 3: Verify imports resolve.**

Run:
```
cd /workspace/RadarTaco && python -c "
import sys; sys.path.insert(0, '.')
from tests.pretrain_image_classifier import compute_frac, BINS, K
import torch
d = torch.rand(2, 1, 90, 160) * 100
f = compute_frac(d, (29, 50), BINS)
assert f.shape == (2, 3, 29, 50), f.shape
assert torch.allclose(f.sum(dim=1), torch.ones(2, 29, 50), atol=1e-5)
print('compute_frac OK, shape=', tuple(f.shape))
"
```
Expected output: `compute_frac OK, shape= (2, 3, 29, 50)`

---

## Task 2: `BinClassifier` model

**Files:**
- Modify: `tests/pretrain_image_classifier.py` (append)

- [ ] **Step 1: Append the `BinClassifier` module.**

Append this to the end of `tests/pretrain_image_classifier.py`:

```python
class BinClassifier(nn.Module):
    """ResNet18 (via baseline's ImageEncoder) + two 1x1 conv heads.

    Heads are per-token linear classifiers — all representational work is done
    by the ResNet. `feats[4]` (29x50) → node head; `feats[5]` (15x25) → edge
    head. Both heads output logits over K=3 depth bins.
    """
    def __init__(self, n_bins: int = K, pretrained: bool = True):
        super().__init__()
        self.encoder = ImageEncoder(pretrained=pretrained)
        ch_node = self.encoder.feat_channels[4]   # 512
        ch_edge = self.encoder.feat_channels[5]   # 512
        self.head_node = nn.Conv2d(ch_node, n_bins, kernel_size=1)
        self.head_edge = nn.Conv2d(ch_edge, n_bins, kernel_size=1)

    def forward(self, rgb: torch.Tensor):
        """Returns dict with 'node' and 'edge' logit tensors."""
        feats = self.encoder(rgb)
        return {
            "node": self.head_node(feats[4]),   # (B, K, 29, 50)
            "edge": self.head_edge(feats[5]),   # (B, K, 15, 25)
        }
```

- [ ] **Step 2: Smoke-test model instantiation and forward shapes.**

Run:
```
CUDA_VISIBLE_DEVICES=3 python -c "
import sys; sys.path.insert(0, '/workspace/RadarTaco')
from tests.pretrain_image_classifier import BinClassifier
import torch
m = BinClassifier().cuda().train()
n = sum(p.numel() for p in m.parameters()) / 1e6
print(f'params: {n:.2f}M')
rgb = torch.randn(2, 3, 900, 1600).cuda()
out = m(rgb)
assert out['node'].shape == (2, 3, 29, 50), out['node'].shape
assert out['edge'].shape == (2, 3, 15, 25), out['edge'].shape
print('forward OK, node=', tuple(out['node'].shape), 'edge=', tuple(out['edge'].shape))
loss = (out['node'].mean() + out['edge'].mean())
loss.backward()
grads = [p.grad is not None and p.grad.abs().sum().item() > 0 for p in m.parameters()]
assert all(grads), 'some params received no gradient'
print(f'backward OK, {sum(grads)}/{len(grads)} params have gradients')
"
```
Expected output (params count may vary within ~0.1M):
```
params: 11.19M
forward OK, node= (2, 3, 29, 50) edge= (2, 3, 15, 25)
backward OK, <N>/<N> params have gradients
```

---

## Task 3: `evaluate()` function

**Files:**
- Modify: `tests/pretrain_image_classifier.py` (append)

- [ ] **Step 1: Append `evaluate()`.**

Append to the file:

```python
@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: str) -> Dict:
    """Compute per-block hard accuracy, per-class recall, and soft CE over
    the whole loader. Returns a dict shaped like pretrain_router_only.py's
    so history.json layouts stay comparable.
    """
    model.eval()
    correct = {"node": 0, "edge": 0}
    total = {"node": 0, "edge": 0}
    per_class_correct = {n: torch.zeros(K, dtype=torch.long) for n in ("node", "edge")}
    per_class_total = {n: torch.zeros(K, dtype=torch.long) for n in ("node", "edge")}
    soft_ce_sum = {"node": 0.0, "edge": 0.0}
    n_batches = 0

    for batch in loader:
        rgb = batch["rgb_norm"].to(device, non_blocking=True)
        dgd = batch["depth_gt_dense"].to(device, non_blocking=True)
        logits = model(rgb)
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
```

- [ ] **Step 2: Smoke-test `evaluate` returns the expected dict shape.**

Run:
```
CUDA_VISIBLE_DEVICES=3 python -c "
import sys; sys.path.insert(0, '/workspace/RadarTaco')
from tests.pretrain_image_classifier import BinClassifier, evaluate
import torch
from torch.utils.data import DataLoader, TensorDataset

# Build a tiny fake batch: (rgb_norm, depth_gt_dense).
rgb = torch.randn(6, 3, 900, 1600)
dgd = (torch.rand(6, 1, 900, 1600) * 100).float()

class FakeDS(torch.utils.data.Dataset):
    def __len__(self): return 6
    def __getitem__(self, i):
        return {'rgb_norm': rgb[i], 'depth_gt_dense': dgd[i]}

loader = DataLoader(FakeDS(), batch_size=3)
m = BinClassifier().cuda()
r = evaluate(m, loader, 'cuda')
assert 'overall_acc' in r and 'per_block' in r
for name in ('node', 'edge'):
    b = r['per_block'][name]
    assert set(b.keys()) == {'acc', 'recall', 'tokens', 'soft_ce'}, b.keys()
    assert len(b['recall']) == 3
print('evaluate shape OK, overall=', round(r['overall_acc'], 2),
      'node=', round(r['per_block']['node']['acc'], 2),
      'edge=', round(r['per_block']['edge']['acc'], 2))
"
```
Expected output: a single line like `evaluate shape OK, overall= <float> node= <float> edge= <float>` (values will be ~33% since the model is untrained on random inputs).

---

## Task 4: `main()` with argparse, output setup, data, and optimizer

**Files:**
- Modify: `tests/pretrain_image_classifier.py` (append)

- [ ] **Step 1: Append `main()` up through data/optim setup (no train loop yet).**

Append to the file:

```python
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline-run",
                    default="output/shape_lidar_grad_shape_edge_res_fix",
                    help="Only used to pick up dataset paths from its config.yaml.")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=12)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--val-every", type=int, default=1,
                    help="Run val every N epochs")
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

    # ============ Read dataset paths from baseline config (nothing else) ============
    cfg_path = os.path.join(args.baseline_run, "config.yaml")
    cfg = OmegaConf.load(cfg_path)
    ds_root = cfg.dataset.data_root

    # ============ Model ============
    model = BinClassifier().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  BinClassifier params: {n_params:,}  ({n_params/1e6:.2f}M)")

    # ============ Data ============
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
    opt = torch.optim.Adam(model.parameters(),
                            lr=args.lr, weight_decay=args.weight_decay)

    log_lines = []
    history = []
    best_acc = 0.0

    def log(msg):
        print(msg)
        log_lines.append(msg)

    log(f"Config: epochs={args.epochs}  batch_size={args.batch_size}  "
        f"lr={args.lr}  wd={args.weight_decay}")
```

Do NOT add `if __name__ == "__main__": main()` yet — the training loop is added in Task 5 before the guard.

- [ ] **Step 2: Verify main() runs its setup portion without errors.**

Because `main()` currently has no return and no train loop, we cannot execute it end-to-end yet. Instead, verify it at least parses and its argparse works with `--help`.

Run: `cd /workspace/RadarTaco && python -c "import ast; ast.parse(open('tests/pretrain_image_classifier.py').read()); print('OK')"`
Expected output: `OK`

Run: `cd /workspace/RadarTaco && python tests/pretrain_image_classifier.py --help`
Expected output: a `usage:` block listing all flags including `--tag`.

---

## Task 5: Train loop, val, and checkpoint saving

**Files:**
- Modify: `tests/pretrain_image_classifier.py` (append at the end of `main()`, then add `__main__` guard)

- [ ] **Step 1: Append the training loop, val, and save logic inside `main()`.**

Append this to `main()` after the `log(...)` config line:

```python
    for epoch in range(args.epochs):
        model.train()
        t0 = time.time()
        train_losses = []
        pbar = tqdm(train_loader, desc=f"ep{epoch:02d} train", leave=False)
        for it, batch in enumerate(pbar):
            rgb = batch["rgb_norm"].to(device, non_blocking=True)
            dgd = batch["depth_gt_dense"].to(device, non_blocking=True)
            logits = model(rgb)
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

    # ---- Final save ----
    torch.save({
        "model": model.state_dict(),
        "epoch": args.epochs - 1,
        "val": history[-1].get("val"),
        "baseline_run": args.baseline_run,
        "config": {
            "bins": BINS, "n_experts": K,
            "head_type": "conv1x1",
            "backbone": "resnet18_imagenet",
            "lr": args.lr, "batch_size": args.batch_size,
            "epochs": args.epochs,
        },
    }, os.path.join(args.out_dir, "last.pt"))
    log(f"\nDone. Best val overall_acc: {best_acc:.2f}%")
    with open(os.path.join(args.out_dir, "log.txt"), "w") as f:
        f.write("\n".join(log_lines))


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify the file still parses.**

Run: `cd /workspace/RadarTaco && python -c "import ast; ast.parse(open('tests/pretrain_image_classifier.py').read()); print('OK')"`
Expected: `OK`

---

## Task 6: End-to-end smoke test on real data (2 iters + 1 val batch)

**Files:**
- Modify (temporarily, then revert): none. This is a one-off command that patches the train loop through `--epochs 0` is not sufficient; use a shim inline.

- [ ] **Step 1: Run a wrapping smoke that instantiates everything, does 2 train iters and 1 val batch, and confirms shapes / no crash.**

Run:
```
CUDA_VISIBLE_DEVICES=3 timeout 240 python -c "
import sys, os, time, torch
sys.path.insert(0, '/workspace/RadarTaco')
os.chdir('/workspace/RadarTaco')

from omegaconf import OmegaConf
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tests.pretrain_image_classifier import (
    BinClassifier, evaluate, compute_frac, BINS, K)
from src.dataset.nuscenes import NuScenesRadarDepthDataset

device = 'cuda'
run = 'output/shape_lidar_grad_shape_edge_res_fix'
cfg = OmegaConf.load(f'{run}/config.yaml')
ds_root = cfg.dataset.data_root

model = BinClassifier().to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)

def make_ds(split, aug):
    return NuScenesRadarDepthDataset(
        data_root=ds_root,
        split_file=os.path.join(ds_root, 'splits', f'{split}.txt'),
        dense_gt_dir=cfg.dataset.dense_gt_dir,
        radar_3d_dir=cfg.dataset.radar_3d_dir,
        night_ids_file=cfg.dataset.night_ids_file,
        max_radar_points=int(cfg.dataset.max_radar_points),
        max_depth=float(cfg.dataset.max_depth),
        min_depth=float(cfg.dataset.min_depth),
        augmentation=aug)

train_loader = DataLoader(make_ds('train', True),
                          batch_size=4, shuffle=True, num_workers=2)
val_loader = DataLoader(make_ds('val', False),
                        batch_size=4, shuffle=False, num_workers=2)

model.train()
losses = []
for it, batch in enumerate(train_loader):
    if it >= 2: break
    rgb = batch['rgb_norm'].to(device)
    dgd = batch['depth_gt_dense'].to(device)
    t = time.time()
    out = model(rgb)
    ln = F.cross_entropy(out['node'], compute_frac(dgd, out['node'].shape[-2:], BINS))
    le = F.cross_entropy(out['edge'], compute_frac(dgd, out['edge'].shape[-2:], BINS))
    loss = 0.5 * (ln + le)
    opt.zero_grad(); loss.backward(); opt.step()
    torch.cuda.synchronize()
    print(f'train iter {it}: loss={loss.item():.4f}  ({(time.time()-t)*1000:.0f}ms)')
    losses.append(loss.item())

# One val batch through evaluate() to exercise its shape/reduction paths.
import itertools
class Head(torch.utils.data.Dataset):
    def __init__(self, base, n): self.base = base; self.n = n
    def __len__(self): return self.n
    def __getitem__(self, i): return self.base[i]
head = Head(val_loader.dataset, 4)
val_out = evaluate(model, DataLoader(head, batch_size=4), device)
print(f'val (4 samples): overall={val_out[\"overall_acc\"]:.2f}%  '
      f'node={val_out[\"per_block\"][\"node\"][\"acc\"]:.2f}%  '
      f'edge={val_out[\"per_block\"][\"edge\"][\"acc\"]:.2f}%')
print('smoke OK')
"
```

Expected output: two lines starting with `train iter 0: loss=…` / `train iter 1: loss=…` with losses roughly in `[0.7, 1.5]`, then a `val (4 samples): overall=… node=… edge=…` line with reasonable acc values (untrained-network values are typically 30–70%), and finally `smoke OK`.

If the smoke run fails, do NOT proceed. Look at the traceback, fix the underlying issue in `tests/pretrain_image_classifier.py`, and rerun the smoke.

---

## Task 7: Cleanup review + hand off

- [ ] **Step 1: Re-read the file for any obvious issues.**

Open `tests/pretrain_image_classifier.py` and verify:
- Imports are all used.
- No stray print/TODO/FIXME text.
- `main()` follows the order: argparse → out-dir mkdir → model → data → optim → train loop → final save.
- `if __name__ == "__main__": main()` is the last line.

Fix anything obvious inline. Do not add features beyond what this plan specified.

- [ ] **Step 2: Confirm the git diff is only the new file.**

Run: `cd /workspace/RadarTaco && git status --short`
Expected: `?? tests/pretrain_image_classifier.py` plus any pre-existing untracked/modified files that were already there before this task (do NOT modify any of those in this task).

- [ ] **Step 3: Report the run command for the user.**

Emit this line for the user (do not run it — the user will launch training themselves):

```
CUDA_VISIBLE_DEVICES=<gpu> python tests/pretrain_image_classifier.py --tag image_classifier
```

This concludes the plan. Do NOT launch the 50-epoch training run yourself.

---

## Self-review notes

Spec coverage check against `docs/superpowers/specs/2026-08-05-image-to-bin-classifier-design.md`:

- Architecture (RGB → ImageEncoder → dual 1×1 conv heads at feats[4]/feats[5]): Task 2.
- All-params trainable (no freezing): Task 2 (`BinClassifier.__init__` does not disable grad anywhere).
- Data (`NuScenesRadarDepthDataset`, train aug on / val aug off, uses only `rgb_norm` + `depth_gt_dense`): Task 4 (`make_ds`) and Task 5 / Task 3 (train / eval bodies).
- Soft-CE target via `compute_frac` at both grids: Task 1 (helper) + Task 5 (train loop) + Task 3 (evaluate).
- Optim (Adam, LR 1e-4, wd 1e-4, batch 12, 50 epochs): Task 4 defaults + Task 5 loop.
- Full val set every epoch: Task 4 (`val_loader` = full val_ds) + Task 5 (`val_every=1` default).
- Best-selection on overall_acc + save best.pt / last.pt / history.json / log.txt to `output/router_pretrain/<tag>/`: Task 5.
- `config` fields in checkpoints (`bins`, `head_type`, `backbone`, `lr`, `batch_size`, `epochs`): Task 5.
- Out-dir created before heavy init: Task 4 (`os.makedirs` immediately after argparse).
- Out of scope (radar input, MoE integration, hyperparameter sweeps): honored — no such code anywhere in the plan.

Placeholder scan: no TODO / TBD / "similar to" placeholders. All code blocks are literal.

Type / signature consistency:
- `BinClassifier.forward` returns `dict` with keys `'node'` and `'edge'` — used consistently in Task 3 (`evaluate`) and Task 5 (train loop).
- `compute_frac(depth_gt_dense, tok_hw, bins)` signature is the same at each call site.
- `evaluate(model, loader, device)` — called with the same three args in the smoke test (Task 6) and in `main()` (Task 5).
