"""Smoke test for v6c "confidence_infer_disjoint" mode.

Checks:
  1. Training forward: `mixed` predicted depth does NOT depend on shared
     expert params — verify by zero-ing out shared params and confirming
     `depth` (main) is unchanged, while `depth_shared` (aux) changes.
  2. Backward: shared params receive gradient ONLY from aux loss.
     Verify by turning w_shared_aux ON vs OFF and comparing shared.grad
     norms — OFF should give near-zero grad on shared.
  3. Eval forward (self.training=False): `depth` DOES depend on shared
     (confidence gating combines both).
"""
import os, sys
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, ROOT)

from scripts.train import _build_model
from src.loss.factory import build_loss

device = "cuda" if torch.cuda.is_available() else "cpu"


def make_model():
    cfg = OmegaConf.load(f"{ROOT}/config/model/radartaco_moe_v6c_confshared.yaml")
    # Minimal dataset stub for _build_model.
    cfg2 = OmegaConf.create({
        "model": cfg,
        "dataset": {"max_depth": 100.0, "max_radar_points": 128, "min_depth": 0.5},
    })
    m = _build_model(cfg2, 100.0, 128)
    return m.to(device)


def make_batch(bs=2, H=896, W=1568):
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
        valid_mask_lidar=valid,
        valid_mask_dense=valid,
        is_sim=torch.zeros(bs, dtype=torch.bool, device=device),
    )


def collect_shared_params(model):
    """Return list of shared-expert parameters (across all MoE blocks)."""
    ps = []
    from src.model.radar_fusion import MoEFusionBlock
    for l in range(len(model.radar_fusion.node_blocks)):
        for blk in (model.radar_fusion.node_blocks[l], model.radar_fusion.edge_blocks[l]):
            if isinstance(blk, MoEFusionBlock) and blk.shared is not None:
                ps.extend(list(blk.shared.parameters()))
    return ps


def collect_spec_params(model):
    """Return list of specialist-expert parameters (across all MoE blocks)."""
    from src.model.radar_fusion import MoEFusionBlock
    ps = []
    for l in range(len(model.radar_fusion.node_blocks)):
        for blk in (model.radar_fusion.node_blocks[l], model.radar_fusion.edge_blocks[l]):
            if isinstance(blk, MoEFusionBlock):
                for e in blk.experts:
                    ps.extend(list(e.parameters()))
    return ps


def total_grad_norm(params):
    total = 0.0
    for p in params:
        if p.grad is not None:
            total += float(p.grad.data.norm().item() ** 2)
    return total ** 0.5


def main():
    torch.manual_seed(0)
    model = make_model()
    print(f"Model built. shared_gate_mode={model.radar_fusion.node_blocks[2].shared_gate_mode}")

    # Ensure moe_shared_aux is on (v6c depends on it).
    assert model.moe_shared_aux, "v6c requires moe_shared_aux=True"

    batch = make_batch(bs=2, H=224, W=320)  # small for speed; must be divisible by 16
    # Actual model needs H=896, W=1568 for image encoder shapes; skip if init would fail.
    # But minimal smoke — try with dataset-native size.
    batch = make_batch(bs=1, H=896, W=1568)

    # -------- Test 1: training forward, verify mixed ⊥ shared params --------
    print("\n=== Test 1: main head is independent of shared params (training) ===")
    model.train()
    with torch.no_grad():
        out1 = model(batch["rgb_norm"], batch["radar_points"], batch["radar_mask"])
        depth1        = out1["depth"].clone()
        depth_shared1 = out1["depth_shared"].clone()

    # Perturb shared params drastically.
    with torch.no_grad():
        for p in collect_shared_params(model):
            p.data.add_(torch.randn_like(p) * 0.5)

    with torch.no_grad():
        out2 = model(batch["rgb_norm"], batch["radar_points"], batch["radar_mask"])
        depth2        = out2["depth"].clone()
        depth_shared2 = out2["depth_shared"].clone()

    d_main   = (depth1 - depth2).abs().mean().item()
    d_shared = (depth_shared1 - depth_shared2).abs().mean().item()
    print(f"  |Δ depth (main)|       = {d_main:.6e}   (expect ≈ 0)")
    print(f"  |Δ depth_shared (aux)| = {d_shared:.6e}   (expect > 0)")
    if d_main < 1e-6 and d_shared > 1e-4:
        print("  ✅ main is disjoint from shared, aux depends on shared")
    else:
        print("  ❌ disjoint property violated")

    # Restore weights (rebuild).
    model = make_model()
    model.train()

    # -------- Test 2: gradients — shared trained ONLY via aux loss --------
    print("\n=== Test 2: shared grad requires aux loss ===")
    loss_cfg_aux_on  = {
        "w_lidar": 1.0, "w_grad_shape": 100.0, "grad_shape_scales": 4,
        "grad_shape_use_log": True, "w_router": 0.0, "w_shared_aux": 1.0,
    }
    loss_cfg_aux_off = {**loss_cfg_aux_on, "w_shared_aux": 0.0}

    for label, lc in [("aux ON", loss_cfg_aux_on), ("aux OFF", loss_cfg_aux_off)]:
        model.zero_grad()
        loss_fn = build_loss(OmegaConf.create(lc))
        pred = model(batch["rgb_norm"], batch["radar_points"], batch["radar_mask"])
        loss_dict = loss_fn(pred, batch)
        loss_dict["loss_total"].backward()
        gn_spec   = total_grad_norm(collect_spec_params(model))
        gn_shared = total_grad_norm(collect_shared_params(model))
        print(f"  {label:>10s}: spec_grad={gn_spec:.4e}   shared_grad={gn_shared:.4e}")

    # -------- Test 3: eval forward — depth DOES depend on shared --------
    print("\n=== Test 3: eval-mode depth DOES depend on shared ===")
    model = make_model()
    model.eval()
    with torch.no_grad():
        out1 = model(batch["rgb_norm"], batch["radar_points"], batch["radar_mask"])
        depth1_eval = out1["depth"].clone()
        assert "depth_shared" not in out1 or out1["depth_shared"] is None, \
            "aux path should NOT run in eval mode"
        print("  eval mode: no depth_shared in output ✅")
    with torch.no_grad():
        for p in collect_shared_params(model):
            p.data.add_(torch.randn_like(p) * 0.5)
        out2 = model(batch["rgb_norm"], batch["radar_points"], batch["radar_mask"])
        depth2_eval = out2["depth"].clone()
    d_eval = (depth1_eval - depth2_eval).abs().mean().item()
    print(f"  |Δ depth (eval)| = {d_eval:.6e}   (expect > 0 — infer combines both)")
    if d_eval > 1e-4:
        print("  ✅ eval-mode depth is confidence-mix of specs + shared")
    else:
        print("  ❌ eval-mode depth ignores shared — inference gating broken")

    print("\nDone.")


if __name__ == "__main__":
    main()
