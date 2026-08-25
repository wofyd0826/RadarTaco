# Image-to-Bin Classifier — Diagnostic + Candidate Router

Date: 2026-08-05
Author: brainstormed with user
Status: design approved, ready for implementation plan

## Purpose

Test whether the ~90.3% router-accuracy ceiling seen in `output/router_pretrain/baseline_encoder/` is caused by the baseline encoder having been trained for **depth regression** rather than **bin classification**. Do this by training the same architecture (ImageNet-initialized ResNet18) end-to-end on the bin-classification objective and comparing against the 90.3% baseline.

### Two-phase intent

1. **Phase A — Diagnostic (this spec).** Just answer: does a bin-obj-trained ResNet18 exceed 90.3%?
2. **Phase B — Candidate architecture (deferred).** Only if Phase A shows meaningful improvement, consider adopting this fine-tuned encoder as the routing backbone in a v5-style MoE run.

Phase B is out of scope for this spec — it would need its own design once Phase A results land.

### Result-interpretation table (fixed up-front so we do not rationalize later)

| Val overall acc | Interpretation | Follow-up |
|---|---|---|
| ≥ 92.0% | Bin-obj re-training gives materially better router features | Proceed to Phase B |
| 90.6% – 91.9% | Partial help; depth-obj re-training is a mild but not fatal handicap | Consider Phase B or label redesign |
| ≤ 90.5% | Within ~±0.3 of the 90.3% baseline — CE label ceiling confirmed | Abandon feature-side work; move to label redesign (overlap bins, soft-only training, bin subdivision) |

## Architecture

```
RGB (B, 3, 900, 1600)
    │
    ▼
ImageEncoder  (src.model.image_encoder.ImageEncoder — ResNet18, ImageNet init)
    │  All parameters trainable.
    │
    ├─→ feats[4]: (B, 512, 29, 50) ─→ Conv2d(512, 3, k=1) ─→ node_logits (B, 3, 29, 50)
    │
    └─→ feats[5]: (B, 512, 15, 25) ─→ Conv2d(512, 3, k=1) ─→ edge_logits (B, 3, 15, 25)
```

Design notes:
- `ImageEncoder` is the exact class the baseline uses, so any difference in results is attributable to **training objective**, not to encoder plumbing.
- Heads are single 1×1 convs (per-token linear classifiers). All representational work sits inside the ResNet — the head is projection only. Head params: `2 × (512 × 3 + 3) = 3,078`.
- Total trainable params: ~11 M (ResNet18) + ~3 K (heads).

## Data

- Dataset: `NuScenesRadarDepthDataset` (same as baseline pretrain), split files at `data_root/splits/{train,val}.txt`.
- Only these fields are actually used: `rgb_norm`, `depth_gt_dense`. Radar fields are read but ignored.
- Augmentation: reuse the baseline's train-time augmentation (`aug=True` for train, `aug=False` for val).

## Training target

Same as `pretrain_router_only.py`:

```python
edges = BINS = [0.0, 20.0, 50.0, 100.0]
pix_bin = bin_index(depth_gt_dense)          # per-pixel int in {0,1,2}
onehot  = one_hot(pix_bin, K=3)              # (B, 3, H, W)
frac    = adaptive_avg_pool2d(onehot, tok_hw)  # (B, 3, H_tok, W_tok), sums to 1 along K
loss    = F.cross_entropy(logits, frac)       # soft CE against pixel-bin fractions
```

Compute this at both grids:
- Node: `tok_hw = logits_node.shape[-2:]` → (29, 50)
- Edge: `tok_hw = logits_edge.shape[-2:]` → (15, 25)

Total loss: `0.5 * (loss_node + loss_edge)`.

## Optimization

- Optimizer: `Adam`
- LR: **1e-4** (uniform across encoder + heads — no discriminative LR)
- Weight decay: 1e-4
- Batch size: **12**
- Epochs: 50
- Grad clip: none (baseline pretrain does not use one; add only if instability appears)
- AMP: off (keep parity with pretrain_router_only.py; add later if needed)

## Validation

- Val set: **entire val split** (~6,008 samples) every epoch.
- Metrics per block (`node`, `edge`):
  - `acc` — hard-argmax accuracy of `pred` vs `gt = frac.argmax(dim=1)`
  - `recall[k]` for k ∈ {near, mid, far}
  - `soft_ce` — mean cross-entropy over the val loader
- Overall: token-weighted acc across both blocks.
- Best selection: highest `overall_acc`.

## Outputs

Following `pretrain_router_only_radar.py` layout:

```
output/router_pretrain/<tag>/
    best.pt         # {"model": encoder+heads state_dict, "epoch", "val", "config"}
    last.pt         # same shape, last epoch
    history.json    # per-epoch list of {epoch, train_loss, val: {...}}
    log.txt         # human-readable log
```

`best.pt["config"]` includes: `bins`, `head_type="conv1x1"`, `lr`, `batch_size`, `epochs`, backbone marker `"resnet18_imagenet"`.

## Script layout

New file: `tests/pretrain_image_classifier.py`, structured after `pretrain_router_only.py`:

- `compute_frac()` — copied verbatim from pretrain_router_only.py.
- `build_model()` — returns a `nn.Module` wrapping `ImageEncoder` + 2 heads.
- `evaluate(model, loader)` — no baseline forward, no hooks; just forwards `rgb_norm` and computes both grids' metrics.
- `main()` — argparse (`--tag`, `--epochs=50`, `--batch-size=12`, `--lr=1e-4`, `--weight-decay=1e-4`, `--num-workers=8`, `--out-dir=output/router_pretrain`), sets up out_dir up-front (mkdir before any heavy init), trains, val every epoch, saves best/last.

CLI:
```bash
CUDA_VISIBLE_DEVICES=<gpu> python tests/pretrain_image_classifier.py --tag image_classifier
```

## What's out of scope

- Radar input. This experiment is image-only on purpose — mixing radar back in reintroduces attribution confounds.
- Any downstream integration into the MoE model. Phase B is a separate spec.
- Any change to the baseline encoder, MoE code, or existing router-pretrain scripts.
- Hyperparameter sweeps. If the diagnostic result is inconclusive (~91%), we may add a follow-up run with LR / batch tweaks — but that would be a new decision, not part of this spec.

## Risk / known confounds

- **Overfit**: 50 epochs full-tune on ~30K samples with 11M params. Val monitoring every epoch will catch it; best.pt selection defends against it.
- **Warm-up**: LR 1e-4 with Adam is safe for a pretrained ResNet18. No warm-up scheduled; if the first epoch diverges we add one.
- **Val time**: full 6,008 samples per epoch × 50 epochs is expensive (~10 min val × 50 = 8+ hours val alone). Accepted by user; no subsample.
- **Numerical parity**: this uses the same `compute_frac` as pretrain_router_only.py, so val numbers are directly comparable to the 90.3% baseline.
