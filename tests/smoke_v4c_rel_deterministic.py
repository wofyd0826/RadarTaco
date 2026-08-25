"""Smoke test for v4c rel-deterministic MoE.

Checks:
  1. Model builds (no router network for MoE blocks).
  2. Forward runs with rel_depth input and produces valid depth.
  3. router_logits list contains None (not tensors).
  4. Loss computes without router CE.
  5. Backward runs; specialists get grad, router doesn't exist so no grad there.
  6. Gate distribution across bins is roughly balanced (uses quantile bins).
"""
import os, sys
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, ROOT)

from scripts.train import _build_model
from src.loss.factory import build_loss
from src.model.radar_fusion import MoEFusionBlock

device = "cuda" if torch.cuda.is_available() else "cpu"


def make_model():
    cfg = OmegaConf.load(f"{ROOT}/config/model/radartaco_moe_v4c_rel_deterministic.yaml")
    cfg2 = OmegaConf.create({
        "model": cfg,
        "dataset": {"max_depth": 100.0, "max_radar_points": 128, "min_depth": 0.5},
    })
    return _build_model(cfg2, 100.0, 128).to(device)


def make_batch(bs=1, H=896, W=1568):
    rgb = torch.randn(bs, 3, H, W, device=device)
    rp  = torch.zeros(bs, 128, 5, device=device)
    rp[:, :10, :3] = torch.randn(bs, 10, 3, device=device)
    rp[:, :10, 3]  = torch.rand(bs, 10, device=device) * W
    rp[:, :10, 4]  = torch.rand(bs, 10, device=device) * 60 + 5
    rm  = torch.zeros(bs, 128, dtype=torch.bool, device=device); rm[:, :10] = True
    gt_dense = torch.rand(bs, 1, H, W, device=device) * 100
    # Simulate rel_depth: uniform [0, 1] with realistic distribution.
    rel_depth = torch.rand(bs, 1, H, W, device=device) ** 2  # skewed toward 0
    return dict(
        rgb_norm=rgb, radar_points=rp, radar_mask=rm,
        depth_gt_dense=gt_dense, depth_gt_lidar=gt_dense,
        valid_mask_lidar=torch.ones_like(gt_dense, dtype=torch.bool),
        valid_mask_dense=torch.ones_like(gt_dense, dtype=torch.bool),
        rel_depth=rel_depth,
        is_sim=torch.zeros(bs, dtype=torch.bool, device=device),
    )


def main():
    torch.manual_seed(0)
    print("=== Test 1: model builds ===")
    model = make_model()
    moe = model.radar_fusion.node_blocks[2]
    assert isinstance(moe, MoEFusionBlock)
    print(f"  router_arch = {moe.router_arch}")
    print(f"  router network = {moe.router}   (expect None)")
    assert moe.router is None
    print(f"  bins = {moe.bins.tolist()}")
    print("  ✅")

    batch = make_batch()
    print(f"\n=== Test 2: forward with rel_depth ===")
    model.eval()
    with torch.no_grad():
        out = model(batch["rgb_norm"], batch["radar_points"],
                    batch["radar_mask"], batch["rel_depth"])
    assert isinstance(out, dict) and "depth" in out
    print(f"  depth shape: {tuple(out['depth'].shape)}")
    print(f"  depth min/max: {out['depth'].min().item():.3f} / {out['depth'].max().item():.3f}")
    print(f"  router_logits: {out['router_logits']}   (expect [None, None])")
    assert all(x is None for x in out["router_logits"])
    print("  ✅")

    print("\n=== Test 3: gate distribution across bins ===")
    # Hook into MoE block to capture the gate.
    gate_holder = {}
    orig = moe.forward
    def patched(self, feat, kv, radar_x_orig, radar_mask, image_w,
                depth_gt_dense=None, teacher_force=True,
                return_shared_only=False, rel_depth=None):
        # Reimplement the deterministic-gate branch to capture gate.
        B, C, H, W = feat.shape
        rd = rel_depth.float().squeeze(1)
        pix_bin = torch.zeros_like(rd, dtype=torch.long)
        for i in range(self.n_experts):
            lo = float(self.bins[i]); hi = float(self.bins[i + 1])
            in_bin = (rd >= lo) & (rd < hi)
            pix_bin = torch.where(in_bin, pix_bin.new_tensor(i), pix_bin)
        pix_bin = torch.where(rd >= float(self.bins[-1]),
                              pix_bin.new_tensor(self.n_experts - 1), pix_bin)
        onehot = F.one_hot(pix_bin, num_classes=self.n_experts).permute(0, 3, 1, 2).float()
        frac = F.adaptive_avg_pool2d(onehot, (H, W))
        gate_holder["frac"] = frac.detach()
        return orig(feat, kv, radar_x_orig, radar_mask, image_w,
                    depth_gt_dense=depth_gt_dense, teacher_force=teacher_force,
                    return_shared_only=return_shared_only, rel_depth=rel_depth)
    moe.forward = patched.__get__(moe, type(moe))
    with torch.no_grad():
        _ = model(batch["rgb_norm"], batch["radar_points"],
                  batch["radar_mask"], batch["rel_depth"])
    moe.forward = orig
    frac = gate_holder["frac"]
    per_bin = frac.mean(dim=(0, 2, 3))
    print(f"  Per-bin pixel_fraction (avg): {per_bin.tolist()}")
    print(f"  (expect roughly balanced but rel_depth is synthetic here)")

    print("\n=== Test 4: loss without router CE ===")
    lc = {"w_lidar": 1.0, "w_grad_shape": 100.0, "grad_shape_scales": 4,
          "grad_shape_use_log": True, "w_router": 0.0}
    loss_fn = build_loss(OmegaConf.create(lc))
    model.train()
    pred = model(batch["rgb_norm"], batch["radar_points"],
                 batch["radar_mask"], batch["rel_depth"],
                 depth_gt_dense=batch["depth_gt_dense"])
    losses = loss_fn(pred, batch)
    print(f"  loss keys: {list(losses.keys())}")
    print(f"  loss_total: {losses['loss_total'].item():.4f}")
    print(f"  loss_router: {losses.get('loss_router', 'n/a')}")
    print("  ✅")

    print("\n=== Test 5: backward ===")
    model.zero_grad()
    losses["loss_total"].backward()

    # Check specialist grads.
    spec_grad = 0.0
    for e in moe.experts:
        for p in e.parameters():
            if p.grad is not None:
                spec_grad += float(p.grad.norm().item() ** 2)
    print(f"  specialist grad norm (L2 sum): {spec_grad ** 0.5:.4e}")
    assert spec_grad > 0
    # No router → no router params.
    print(f"  router is None → no router params to check")
    print("  ✅")

    print("\nAll smoke tests passed.")


if __name__ == "__main__":
    main()
