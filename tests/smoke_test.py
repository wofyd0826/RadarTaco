"""Smoke test: forward + backward + 1-batch eval, no real data needed."""
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.evaluation.metrics import DepthEvaluator                            # noqa: E402
from src.loss.factory import ComposedLoss                                    # noqa: E402
from src.model.radartaco import RadarTaco                                    # noqa: E402


def _fake_batch(B: int, H: int, W: int, K: int, n_sim: int = 0):
    """Build a fake 6-channel radar batch: (front, left, up, x_pix, y_pix, depth)."""
    rgb = torch.rand(B, 3, H, W) * 2.0 - 1.0
    radar_pts = torch.zeros(B, K, 6)
    depth = torch.rand(B, K) * 75.0 + 5.0                   # depth ∈ [5, 80] m
    radar_pts[:, :, 0] = depth                              # front (= depth)
    radar_pts[:, :, 1] = (torch.rand(B, K) - 0.5) * 20.0    # left (±10 m lateral)
    radar_pts[:, :, 2] = (torch.rand(B, K) - 0.5) * 4.0     # up   (small; 3D radar weak elevation)
    radar_pts[:, :, 3] = torch.rand(B, K) * W               # x_pix
    radar_pts[:, :, 4] = torch.rand(B, K) * H               # y_pix
    radar_pts[:, :, 5] = depth                              # depth
    radar_mask = torch.ones(B, K, dtype=torch.bool)
    radar_mask[:, K // 2:] = False           # half padded → exercises masking

    depth = torch.rand(B, 1, H, W) * 80.0
    valid = torch.ones(B, 1, H, W, dtype=torch.bool)

    is_sim = torch.zeros(B, dtype=torch.bool)
    if n_sim > 0:
        is_sim[:n_sim] = True
    return {
        "rgb_norm": rgb,
        "radar_points": radar_pts,
        "radar_mask": radar_mask,
        "depth_gt_lidar": depth,
        "depth_gt_dense": depth,
        "valid_mask_lidar": valid,
        "valid_mask_dense": valid,
        "is_sim": is_sim,
        "is_night": torch.zeros(B, dtype=torch.bool),
    }


def _smoke_for_encoder(name: str, output_mode: str = "metric"):
    print(f"--- encoder: {name}  output_mode: {output_mode} ---")
    model = RadarTaco(
        radar_encoder_name=name,
        max_depth=100.0,
        max_radar_points=32,
        radar_channels=(32, 64, 128),     # reduced for CPU smoke speed
        attn_heads=4,
        pretrained_image_encoder=False,
        output_mode=output_mode,
        min_depth_clip=0.5,
    )
    n_p = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  params: {n_p:.2f}M")

    H, W, K = 64, 96, 32
    batch = _fake_batch(B=2, H=H, W=W, K=K, n_sim=1)
    pred = model(batch["rgb_norm"], batch["radar_points"], batch["radar_mask"])
    assert pred.shape == (2, 1, H, W), f"unexpected pred shape: {pred.shape}"
    assert torch.isfinite(pred).all(), "NaN/Inf in pred"
    print(f"  forward OK: pred shape {tuple(pred.shape)}, range [{pred.min():.2f}, {pred.max():.2f}]")

    loss_fn = ComposedLoss(lam=1.0, w_sim_grad=0.5, w_sim_smooth=1e-3)
    losses = loss_fn(pred, batch)
    losses["loss_total"].backward()
    has_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                   for p in model.parameters())
    assert has_grad, "no gradient flowed"
    print(f"  loss = {losses['loss_total'].item():.4f}  (lidar={losses['loss_lidar']:.4f}, "
          f"dense={losses['loss_dense']:.4f}, sim_l1={losses['loss_sim_l1']:.4f}, "
          f"sim_grad={losses['loss_sim_grad']:.4f})")

    # 1-sample eval
    eva = DepthEvaluator(min_depth=1e-3, max_depth=100.0)
    pn = pred[0, 0].detach().cpu().numpy()
    gn = batch["depth_gt_lidar"][0, 0].cpu().numpy()
    mn = batch["valid_mask_lidar"][0, 0].cpu().numpy().astype(bool)
    m = eva.evaluate_sample(pn, gn, mn, is_night=False)
    agg = eva.aggregate_metrics([m, m])
    assert "overall" in agg and "0-80m" in agg["overall"]
    assert "far" in agg and "50-80m" in agg["far"]
    assert "day" in agg, "missing day aggregate"
    print(f"  eval OK: 0-80m MAE={agg['overall']['0-80m']['mae']:.2f}  "
          f"far MAE={agg['far']['50-80m']['mae']:.2f}")


def main():
    torch.manual_seed(0)
    np.random.seed(0)
    _smoke_for_encoder("gnn", output_mode="metric")
    _smoke_for_encoder("mlp", output_mode="metric")
    _smoke_for_encoder("gnn", output_mode="inverse")
    print("\nALL SMOKE TESTS PASSED")


if __name__ == "__main__":
    main()
