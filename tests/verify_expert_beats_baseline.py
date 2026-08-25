"""각 expert 가 자기 담당 bin 에서 baseline 대비 더 좋은지 검증.

세 종류 mae 를 같은 val 서브셋 + 같은 [0,20)/[20,50)/[50,100) 구간에서 비교:
  (1) Baseline (single-model)               ─ 기준선
  (2) V2 MoE — Normal routing                ─ router 가 결정한 대로
  (3) V2 MoE — Force expert_X (X=near/mid/far)  ─ 그 expert 만 활성

의미:
  - Force near mae@[0,20) < Baseline mae@[0,20)  → near expert 가 자기 bin
    에서 baseline 을 이겼는가?  이게 이 실험이 답하려는 원래 질문.
  - 성립하면: MoE 는 원리적으로 유효, router 오차만 개선하면 됨.
  - 실패하면: MoE 자체가 baseline capacity 를 못 활용.
"""
import argparse
import os
import sys
from collections import defaultdict

import numpy as np
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, ROOT)

from src.dataset.nuscenes import NuScenesRadarDepthDataset  # noqa: E402
from src.model.radartaco import RadarTaco                    # noqa: E402
from src.model.radar_fusion import MoEFusionBlock            # noqa: E402
from scripts.train import _build_model                        # noqa: E402


device = "cuda" if torch.cuda.is_available() else "cpu"


def load_model_from_run(run_dir, ckpt_name="best.pt"):
    cfg = OmegaConf.load(os.path.join(run_dir, "config.yaml"))
    m = _build_model(cfg, float(cfg.dataset.max_depth), int(cfg.dataset.max_radar_points))
    sd = torch.load(os.path.join(run_dir, ckpt_name), map_location=device, weights_only=False)
    m.load_state_dict(sd["model"])
    print(f"  [loaded] {run_dir}/{ckpt_name}  epoch={sd.get('epoch','?')}")
    return m.eval().to(device), cfg


def make_force_router_hook(expert_idx, K):
    def _hook(_mod, _inp, out):
        forced = torch.full_like(out, -1e4)
        forced[:, expert_idx] = 1e4
        return forced
    return _hook


def per_range(pred, gt, mask, ranges):
    out = {}
    for (lo, hi) in ranges:
        m = mask & (gt >= lo) & (gt < hi) & (gt > 0)
        if m.any():
            err = np.abs(pred[m] - gt[m])
            out[f"[{lo},{hi})m"] = (float(err.sum()), int(m.sum()))
        else:
            out[f"[{lo},{hi})m"] = (0.0, 0)
    return out


def evaluate(model, ds, indices, ranges, force_idx=None, has_moe=False):
    hooks = []
    if has_moe and force_idx is not None:
        K = int(model.moe_bins.__len__() if isinstance(model.moe_bins, list) else len(model.moe_bins)) - 1
        for blk in (model.radar_fusion.node_blocks[2], model.radar_fusion.edge_blocks[2]):
            if isinstance(blk, MoEFusionBlock):
                hooks.append(blk.router.register_forward_hook(make_force_router_hook(force_idx, K)))

    acc = defaultdict(lambda: [0.0, 0])
    with torch.no_grad():
        for i in tqdm(indices, desc=f"force={force_idx}"):
            s = ds[int(i)]
            rgb = s["rgb_norm"].unsqueeze(0).to(device)
            rp = s["radar_points"].unsqueeze(0).to(device)
            rm = s["radar_mask"].unsqueeze(0).to(device)
            out = model(rgb, rp, rm)
            pred = out["depth"] if isinstance(out, dict) else out
            pred_np = pred[0, 0].cpu().numpy()
            gt_np = s["depth_gt_lidar"][0].cpu().numpy()
            mask_np = s["valid_mask_lidar"][0].cpu().numpy().astype(bool)
            r = per_range(pred_np, gt_np, mask_np, ranges)
            for k, (se, n) in r.items():
                acc[k][0] += se
                acc[k][1] += n
    for h in hooks: h.remove()
    return {k: (v[0] / v[1] if v[1] > 0 else float("nan")) for k, v in acc.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--split", default="val")
    ap.add_argument("--baseline", default="output/shape_lidar_grad_shape_edge_res")
    ap.add_argument("--moe_no_shared", default="output/radartaco_moe_v2/radartaco_moe_stage2_v2")
    ap.add_argument("--moe_shared",    default="output/radartaco_moe_shared_v2/radartaco_moe_stage2_shared_v2")
    ap.add_argument("--baseline_ckpt", default="best.pt")
    ap.add_argument("--moe_no_shared_ckpt", default="best.pt")
    ap.add_argument("--moe_shared_ckpt",    default="best.pt")
    args = ap.parse_args()

    labels = ["near", "mid", "far"]
    ranges = [(0.0, 20.0), (20.0, 50.0), (50.0, 100.0)]
    range_names = [f"[{lo},{hi})m" for (lo, hi) in ranges]

    # ── Load 3 models ──
    m_base, cfg_base = load_model_from_run(args.baseline, args.baseline_ckpt)
    m_no, cfg_no = load_model_from_run(args.moe_no_shared, args.moe_no_shared_ckpt)
    m_sh, cfg_sh = load_model_from_run(args.moe_shared, args.moe_shared_ckpt)

    # ── Same val subset ──
    split_file = getattr(cfg_base.dataset, f"split_{args.split}", None) or \
        os.path.join(cfg_base.dataset.data_root, "splits", f"{args.split}.txt")
    ds = NuScenesRadarDepthDataset(
        data_root=cfg_base.dataset.data_root, split_file=split_file,
        dense_gt_dir=cfg_base.dataset.dense_gt_dir,
        radar_3d_dir=cfg_base.dataset.radar_3d_dir,
        night_ids_file=cfg_base.dataset.night_ids_file,
        max_radar_points=int(cfg_base.dataset.max_radar_points),
        max_depth=float(cfg_base.dataset.max_depth),
        min_depth=float(cfg_base.dataset.min_depth),
        augmentation=False,
    )
    idxs = np.linspace(0, len(ds) - 1, args.n).astype(int).tolist()

    # ── Run each configuration ──
    results = {}
    print("\n[1/9] Baseline (single model)…")
    results["Baseline"] = evaluate(m_base, ds, idxs, ranges, has_moe=False)

    print("\n[2/9] MoE (no shared) — Normal routing…")
    results["MoE noSh Normal"] = evaluate(m_no, ds, idxs, ranges, has_moe=True, force_idx=None)
    for k, lb in enumerate(labels):
        print(f"\n[{3+k}/9] MoE (no shared) — Force {lb} …")
        results[f"MoE noSh Force-{lb}"] = evaluate(m_no, ds, idxs, ranges, has_moe=True, force_idx=k)

    print("\n[6/9] MoE (shared) — Normal routing…")
    results["MoE Sh Normal"] = evaluate(m_sh, ds, idxs, ranges, has_moe=True, force_idx=None)
    for k, lb in enumerate(labels):
        print(f"\n[{7+k}/9] MoE (shared) — Force {lb} …")
        results[f"MoE Sh Force-{lb}"] = evaluate(m_sh, ds, idxs, ranges, has_moe=True, force_idx=k)

    # ── Report ──
    print("\n" + "=" * 78)
    print("MAE table (rows=setting, cols=depth range on LiDAR GT, val subset)")
    print("=" * 78)
    print(f"{'Setting':<26}" + "  ".join(f"{r:>14}" for r in range_names))
    print("-" * 78)
    for name, mae_by_range in results.items():
        row = f"{name:<26}"
        for r in range_names:
            row += f"  {mae_by_range[r]:>14.4f}"
        print(row)

    # ── Head-to-head: does each expert beat baseline in its own bin? ──
    print("\n" + "=" * 78)
    print("Head-to-head: expert (자기 bin) vs Baseline (같은 bin)")
    print("=" * 78)
    for variant in ("noSh", "Sh"):
        print(f"\n--- MoE variant: {variant} ---")
        for k, (lb, r) in enumerate(zip(labels, range_names)):
            b_mae = results["Baseline"][r]
            f_mae = results[f"MoE {variant} Force-{lb}"][r]
            n_mae = results[f"MoE {variant} Normal"][r]
            print(f"  {lb} bin {r}:")
            print(f"    baseline mae:            {b_mae:.4f}")
            print(f"    Force {lb} expert only:   {f_mae:.4f}   Δ vs baseline = {f_mae - b_mae:+.4f}  ({'✓ 이김' if f_mae < b_mae else '✗ 짐'})")
            print(f"    Normal MoE routing:       {n_mae:.4f}   Δ vs baseline = {n_mae - b_mae:+.4f}")


if __name__ == "__main__":
    main()
