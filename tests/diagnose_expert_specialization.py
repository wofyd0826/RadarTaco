"""Verify whether each MoE expert has actually specialised for its GT bin.

Method: force route ALL tokens through a single non-shared expert (via
router-output hook), then measure per-range mae on val. If specialisation
worked, the diagonal of the (forced_expert × depth_range) matrix should
show the LOWEST mae — i.e. expert_near best at near range, expert_mid
best at mid range, expert_far best at far range.

Also compares against normal (learned routing) as a reference.
"""
import argparse
import os
import sys
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from tqdm import tqdm

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, ROOT)

from src.dataset.nuscenes import NuScenesRadarDepthDataset  # noqa: E402
from src.model.radartaco import RadarTaco                    # noqa: E402
from src.model.radar_fusion import MoEFusionBlock            # noqa: E402


device = "cuda" if torch.cuda.is_available() else "cpu"


def load_model(run_dir):
    ck = os.path.join(run_dir, "best.pt")
    cfg = OmegaConf.load(os.path.join(run_dir, "config.yaml"))
    m = RadarTaco(
        radar_encoder_name=cfg.model.radar_encoder,
        max_depth=float(cfg.dataset.max_depth),
        max_radar_points=int(cfg.dataset.max_radar_points),
        k_neighbors=int(cfg.model.k_neighbors),
        a_l=tuple(cfg.model.a_l),
        radar_channels=tuple(cfg.model.radar_channels),
        attn_heads=int(cfg.model.attn_heads),
        pretrained_image_encoder=False,
        moe_at_l=tuple(cfg.model.get("moe_at_l") or ()),
        moe_n_experts=int(cfg.model.get("moe_n_experts", 3)),
        moe_use_shared=bool(cfg.model.get("moe_use_shared", True)),
        moe_top_k=(int(cfg.model.get("moe_top_k")) if cfg.model.get("moe_top_k") is not None else None),
        moe_stage=int(cfg.model.get("moe_stage", 2)),
        moe_bins=tuple(cfg.model.get("moe_bins", (0.0, 20.0, 50.0, 100.0))),
    )
    sd = torch.load(ck, map_location=device, weights_only=False)
    m.load_state_dict(sd["model"])
    return m.eval().to(device), cfg


def make_force_router_hook(expert_idx: int, K: int):
    """Return a Conv2d forward-hook that overrides the router output so
    softmax(output) becomes ~1.0 on `expert_idx` at every spatial pos."""
    def _hook(_mod, _inp, out):
        # out: (B, K, H, W)
        forced = torch.full_like(out, -1e4)
        forced[:, expert_idx] = 1e4
        return forced
    return _hook


def per_range_metrics(pred, gt, mask, ranges):
    """pred, gt: numpy (H, W). mask: bool (H, W). ranges: [(lo, hi), ...].
    Returns {range_name: (sum_abs_err, n_valid)}."""
    out = {}
    for (lo, hi) in ranges:
        m = mask & (gt >= lo) & (gt < hi) & (gt > 0)
        if m.any():
            err = np.abs(pred[m] - gt[m])
            out[f"[{lo},{hi})m"] = (float(err.sum()), int(m.sum()))
        else:
            out[f"[{lo},{hi})m"] = (0.0, 0)
    return out


def eval_setting(model, ds, indices, ranges, forced_expert_idx=None):
    """Return {range_name: mae} averaged over given indices under a
    specific forced-expert configuration (None = normal routing)."""
    hooks = []
    if forced_expert_idx is not None:
        K = model.moe_bins.__len__() - 1 if isinstance(model.moe_bins, list) else len(model.moe_bins) - 1
        for blk in (model.radar_fusion.node_blocks[2],
                    model.radar_fusion.edge_blocks[2]):
            if isinstance(blk, MoEFusionBlock):
                hooks.append(blk.router.register_forward_hook(
                    make_force_router_hook(forced_expert_idx, K)))

    acc = defaultdict(lambda: [0.0, 0])
    with torch.no_grad():
        for i in tqdm(indices, desc=f"expert={forced_expert_idx}"):
            s = ds[int(i)]
            rgb = s["rgb_norm"].unsqueeze(0).to(device)
            rp  = s["radar_points"].unsqueeze(0).to(device)
            rm  = s["radar_mask"].unsqueeze(0).to(device)
            out = model(rgb, rp, rm)
            pred = out["depth"] if isinstance(out, dict) else out
            pred_np = pred[0, 0].cpu().numpy()
            gt_np = s["depth_gt_lidar"][0].cpu().numpy()
            mask_np = s["valid_mask_lidar"][0].cpu().numpy().astype(bool)
            r = per_range_metrics(pred_np, gt_np, mask_np, ranges)
            for k, (s_err, n) in r.items():
                acc[k][0] += s_err
                acc[k][1] += n

    for h in hooks: h.remove()
    return {k: (v[0] / v[1] if v[1] > 0 else float("nan")) for k, v in acc.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="output/radartaco_moe_stage2_scratch/radartaco_moe_stage2_scratch")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--split", default="val")
    args = ap.parse_args()

    m, cfg = load_model(args.run)
    labels = ["near", "mid", "far"]
    ranges = [(0.0, 20.0), (20.0, 50.0), (50.0, 100.0)]

    split_file = getattr(cfg.dataset, f"split_{args.split}", None) or \
        os.path.join(cfg.dataset.data_root, "splits", f"{args.split}.txt")
    ds = NuScenesRadarDepthDataset(
        data_root=cfg.dataset.data_root, split_file=split_file,
        dense_gt_dir=cfg.dataset.dense_gt_dir,
        radar_3d_dir=cfg.dataset.radar_3d_dir,
        night_ids_file=cfg.dataset.night_ids_file,
        max_radar_points=int(cfg.dataset.max_radar_points),
        max_depth=float(cfg.dataset.max_depth),
        min_depth=float(cfg.dataset.min_depth),
        augmentation=False,
    )
    idxs = np.linspace(0, len(ds) - 1, args.n).astype(int).tolist()
    print(f"[loaded] {args.run}/best.pt  |  {args.n} samples from {args.split}")

    # ── Run 4 configurations ──
    results = {}
    print("\nNormal routing (learned)…")
    results["Normal (learned)"] = eval_setting(m, ds, idxs, ranges, forced_expert_idx=None)
    for k, lb in enumerate(labels):
        print(f"\nForce expert = {lb} only …")
        results[f"Force {lb}"] = eval_setting(m, ds, idxs, ranges, forced_expert_idx=k)

    # ── Report matrix ──
    range_names = [f"[{lo},{hi})m" for (lo, hi) in ranges]
    print("\n" + "=" * 70)
    print("MAE matrix (rows=router setting, cols=depth range on LiDAR GT)")
    print("=" * 70)
    print(f"{'Setting':<20}" + "  ".join(f"{r:>12}" for r in range_names))
    print("-" * 70)
    for setting, mae_by_range in results.items():
        row = f"{setting:<20}"
        for r in range_names:
            row += f"  {mae_by_range[r]:>12.4f}"
        print(row)

    print("\nExpected if specialization succeeded:")
    print("  'Force near' should have LOWEST mae in [0,20)m")
    print("  'Force mid'  should have LOWEST mae in [20,50)m")
    print("  'Force far'  should have LOWEST mae in [50,100)m")
    print()

    # Diagonal check
    print("=== Diagonal check ===")
    for k, (lb, r) in enumerate(zip(labels, range_names)):
        row_mae = {name: results[name][r] for name in results if name.startswith("Force")}
        best = min(row_mae, key=row_mae.get)
        target = f"Force {lb}"
        mark = "✅ 특화 확인" if best == target else "❌ 특화 실패"
        print(f"  {r}: 최저 mae expert = {best} (target={target})  {mark}")
        for name, v in sorted(row_mae.items(), key=lambda x: x[1]):
            arrow = " ← target" if name == target else ""
            print(f"      {name:<15} mae={v:.4f}{arrow}")


if __name__ == "__main__":
    main()
