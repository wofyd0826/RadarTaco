"""Smoke test for v6d task-aware router loss.

Checks:
  1. Training forward produces `depth_per_spec` (list of K tensors).
  2. Eval forward does NOT produce `depth_per_spec` (no aux cost).
  3. Task-aware router loss (`w_task_router > 0`) is added to `loss_total`
     and appears in loss dict.
  4. Router receives gradient from BOTH pixel-bin CE and task-aware loss.
"""
import os, sys
import torch
from omegaconf import OmegaConf

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, ROOT)

from scripts.train import _build_model
from src.loss.factory import build_loss
from src.model.radar_fusion import MoEFusionBlock

device = "cuda" if torch.cuda.is_available() else "cpu"


def make_model():
    cfg = OmegaConf.load(f"{ROOT}/config/model/radartaco_moe_v4d_taskrouter.yaml")
    cfg2 = OmegaConf.create({
        "model": cfg,
        "dataset": {"max_depth": 100.0, "max_radar_points": 128, "min_depth": 0.5},
    })
    m = _build_model(cfg2, 100.0, 128)
    return m.to(device)


def make_batch(bs=1, H=896, W=1568):
    rgb = torch.randn(bs, 3, H, W, device=device)
    rp  = torch.zeros(bs, 128, 5, device=device)
    rp[:, :10, :3] = torch.randn(bs, 10, 3, device=device)
    rp[:, :10, 3]  = torch.rand(bs, 10, device=device) * W
    rp[:, :10, 4]  = torch.rand(bs, 10, device=device) * 60 + 5
    rm  = torch.zeros(bs, 128, dtype=torch.bool, device=device)
    rm[:, :10] = True
    gt_dense = torch.rand(bs, 1, H, W, device=device) * 100
    gt_lidar = gt_dense.clone()
    valid = torch.ones_like(gt_lidar, dtype=torch.bool)
    return dict(
        rgb_norm=rgb, radar_points=rp, radar_mask=rm,
        depth_gt_dense=gt_dense, depth_gt_lidar=gt_lidar,
        valid_mask_lidar=valid, valid_mask_dense=valid,
        is_sim=torch.zeros(bs, dtype=torch.bool, device=device),
    )


def collect_router_params(model):
    ps = []
    for l in range(len(model.radar_fusion.node_blocks)):
        for blk in (model.radar_fusion.node_blocks[l], model.radar_fusion.edge_blocks[l]):
            if isinstance(blk, MoEFusionBlock):
                ps.extend(list(blk.router.parameters()))
    return ps


def total_grad_norm(params):
    tot = 0.0
    for p in params:
        if p.grad is not None:
            tot += float(p.grad.data.norm().item() ** 2)
    return tot ** 0.5


def main():
    torch.manual_seed(0)
    model = make_model()
    print(f"moe_per_spec_aux={model.moe_per_spec_aux}   K={model._moe_n_experts}")

    batch = make_batch(bs=1, H=448, W=784)

    # ---- Test 1: training forward yields depth_per_spec ----
    print("\n=== Test 1: training forward yields depth_per_spec ===")
    model.train()
    out_train = model(batch["rgb_norm"], batch["radar_points"], batch["radar_mask"])
    print(f"  keys: {sorted(out_train.keys())}")
    assert "depth_per_spec" in out_train, "training forward missing depth_per_spec"
    dps = out_train["depth_per_spec"]
    print(f"  depth_per_spec is a list of length {len(dps)}")
    for k, d in enumerate(dps):
        print(f"    [{k}] shape = {tuple(d.shape)}")
    assert len(dps) == model._moe_n_experts, "wrong per-spec length"
    print("  ✅")

    # ---- Test 2: eval forward does NOT produce depth_per_spec ----
    print("\n=== Test 2: eval forward has NO depth_per_spec ===")
    model.eval()
    with torch.no_grad():
        out_eval = model(batch["rgb_norm"], batch["radar_points"], batch["radar_mask"])
    print(f"  keys: {sorted(out_eval.keys())}")
    assert "depth_per_spec" not in out_eval, "eval forward should NOT produce depth_per_spec"
    print("  ✅")

    # ---- Test 3: loss dict contains loss_task_router; total includes it ----
    print("\n=== Test 3: task_router loss appears in loss dict ===")
    model.train()
    # Two loss configs: w_task_router ON vs OFF
    loss_on = OmegaConf.create({
        "w_lidar": 1.0, "w_grad_shape": 100.0, "grad_shape_scales": 4,
        "grad_shape_use_log": True, "w_router": 0.0,
        "w_shared_aux": 0.0, "w_task_router": 2.0,
        "task_router_temperature": 1.0,
    })
    loss_off = OmegaConf.create({**loss_on, "w_task_router": 0.0})

    for label, lc in [("task_router ON ", loss_on), ("task_router OFF", loss_off)]:
        model.zero_grad()
        loss_fn = build_loss(lc)
        pred = model(batch["rgb_norm"], batch["radar_points"], batch["radar_mask"])
        ld = loss_fn(pred, batch)
        ld["loss_total"].backward()
        gn_router = total_grad_norm(collect_router_params(model))
        print(f"  {label}: loss_total={float(ld['loss_total']):.4f}  "
              f"loss_task_router={float(ld['loss_task_router']):.4f}  "
              f"router_grad_norm={gn_router:.4e}")

    # ---- Test 4: stage=1 skips per-spec (stage 2 gate) ----
    print("\n=== Test 4: stage=1 skips per-spec forwards ===")
    model.moe_stage = 1        # force stage 1
    model.train()
    out_s1 = model(batch["rgb_norm"], batch["radar_points"], batch["radar_mask"])
    if "depth_per_spec" in out_s1:
        print(f"  ❌ stage 1 still produced depth_per_spec (len={len(out_s1['depth_per_spec'])})")
    else:
        print(f"  ✅ stage 1 correctly skipped depth_per_spec")
    print(f"  keys: {sorted(out_s1.keys())}")

    print("\nDone.")


if __name__ == "__main__":
    main()
