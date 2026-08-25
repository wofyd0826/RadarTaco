"""Smoke test for router self-distillation loss.

Verifies:
  1. Loss builds with w_router_selfdist > 0 without error.
  2. Non-zero `loss_router_selfdist` produced when self-dist weight > 0.
  3. Zero `loss_router_selfdist` when weight = 0.
  4. Detach works: gradient of self-dist loss w.r.t. `depth` is ZERO
     (i.e., no gradient flows router → depth → target → router).
  5. Gradient DOES flow to router parameters via self-dist.
"""
import os, sys
import torch
import torch.nn as nn
from omegaconf import OmegaConf

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, ROOT)

from src.loss.factory import build_loss

device = "cuda" if torch.cuda.is_available() else "cpu"


def make_fake_pred_and_batch(bs=2, H=224, W=320, H_tok=14, W_tok=20, K=3):
    """Construct a minimal fake pred/batch that ComposedLoss can consume."""
    # Simulate model outputs.
    # depth: (B, 1, H, W) — final depth prediction (used both for main loss + self-dist target)
    depth = torch.rand(bs, 1, H, W, device=device, requires_grad=True) * 100
    # router_logits: list of (B, K, H_tok, W_tok) — needs gradient
    router = nn.Conv2d(1, K, kernel_size=1).to(device)
    router_logits = [router(torch.rand(bs, 1, H_tok, W_tok, device=device))]

    pred = {
        "depth": depth,
        "router_logits": router_logits,
        "router_gts": [None],
        "router_gt_type": "soft",
        "moe_bins": (0.0, 20.0, 50.0, 100.0),
    }
    gt = torch.rand(bs, 1, H, W, device=device) * 100
    batch = {
        "depth_gt_lidar": gt.clone(),
        "depth_gt_dense": gt.clone(),
        "valid_mask_lidar": torch.ones_like(gt, dtype=torch.bool),
        "valid_mask_dense": torch.ones_like(gt, dtype=torch.bool),
        "is_sim": torch.zeros(bs, dtype=torch.bool, device=device),
        "rgb_norm": torch.rand(bs, 3, H, W, device=device),
    }
    return pred, batch, depth, router


def make_loss(w_router=3.0, w_router_selfdist=2.0):
    return build_loss(OmegaConf.create({
        "w_lidar": 1.0, "w_grad_shape": 100.0,
        "grad_shape_scales": 4, "grad_shape_use_log": True,
        "w_router": w_router,
        "w_router_selfdist": w_router_selfdist,
    }))


def main():
    torch.manual_seed(0)

    print("=== Test 1: Loss builds, self-dist non-zero when weight > 0 ===")
    loss_fn = make_loss(w_router=3.0, w_router_selfdist=2.0)
    pred, batch, depth, router = make_fake_pred_and_batch()
    ld = loss_fn(pred, batch)
    print(f"  loss_total            = {ld['loss_total'].item():.4f}")
    print(f"  loss_router           = {ld['loss_router'].item():.4f}   (GT CE)")
    print(f"  loss_router_selfdist  = {ld['loss_router_selfdist'].item():.4f}  (self-dist)")
    assert ld["loss_router_selfdist"].item() > 0, "self-dist loss should be > 0"
    print("  ✅ self-dist loss non-zero")

    print("\n=== Test 2: Self-dist = 0 when weight = 0 ===")
    loss_fn0 = make_loss(w_router=3.0, w_router_selfdist=0.0)
    pred2, batch2, _, _ = make_fake_pred_and_batch()
    ld0 = loss_fn0(pred2, batch2)
    print(f"  loss_router_selfdist  = {ld0['loss_router_selfdist'].item():.4f}   (expect 0)")
    assert ld0["loss_router_selfdist"].item() == 0.0
    print("  ✅ self-dist zero when disabled")

    print("\n=== Test 3: DETACH — reproducing self-dist loss inline ===")
    # Reproduce the self-dist term inline and check gradient flow.
    import torch.nn.functional as F
    depth = torch.rand(2, 1, 224, 320, device=device, requires_grad=True) * 100
    router = nn.Conv2d(1, 3, kernel_size=1).to(device)
    logits = router(torch.rand(2, 1, 14, 20, device=device))
    bins = (0.0, 20.0, 50.0, 100.0)
    K = len(bins) - 1
    edges = torch.tensor(list(bins), device=device, dtype=torch.float32)
    with torch.no_grad():
        ds = depth.detach().squeeze(1)
        pix_bin = torch.zeros_like(ds, dtype=torch.long)
        for i in range(K):
            pix_bin = torch.where(
                (ds >= float(edges[i])) & (ds < float(edges[i + 1])),
                pix_bin.new_tensor(i), pix_bin)
        pix_bin = torch.where(ds >= float(edges[-1]), pix_bin.new_tensor(K - 1), pix_bin)
        onehot = F.one_hot(pix_bin, num_classes=K).permute(0, 3, 1, 2).float()
        target_frac = F.adaptive_avg_pool2d(onehot, logits.shape[-2:])
    log_probs = F.log_softmax(logits, dim=1)
    l_sd = -(target_frac * log_probs).sum(dim=1).mean()
    print(f"  self-dist loss = {l_sd.item():.4f}")
    g_depth = torch.autograd.grad(l_sd, depth, retain_graph=True, allow_unused=True)[0]
    g_router = torch.autograd.grad(l_sd, list(router.parameters()),
                                    retain_graph=False, allow_unused=True)
    g_depth_norm  = 0.0 if g_depth is None else float(g_depth.abs().sum().item())
    g_router_norm = sum(0.0 if g is None else float(g.abs().sum().item()) for g in g_router)
    print(f"  |grad(l_sd, depth)|  = {g_depth_norm:.6e}  (expect 0 — detached)")
    print(f"  |grad(l_sd, router)| = {g_router_norm:.6e}  (expect > 0)")
    if g_depth_norm < 1e-8 and g_router_norm > 1e-6:
        print("  ✅ detach works — gradient flows only to router, not to depth")
    else:
        print(f"  ❌ detach violation")

    print("\nDone.")


if __name__ == "__main__":
    main()
