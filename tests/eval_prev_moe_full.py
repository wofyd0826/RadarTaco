"""Full analysis of PRIOR (non-warm-start) MoE stage-2 experiments:
   1. Normal routing per-bin MAE
   2. Force-expert per-bin MAE (each expert's ceiling in its own bin)
   3. Oracle routing per-bin MAE (per-token majority-vote GT)
   4. Router accuracy vs majority-vote GT

Targets:
   • no-shared v2:  output/radartaco_moe_v2/radartaco_moe_stage2_v2/best.pt
   • shared    v2:  output/radartaco_moe_shared_v2/radartaco_moe_stage2_shared_v2/best.pt
"""
import argparse, os, sys
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from tqdm import tqdm

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, ROOT)

from src.dataset.nuscenes import NuScenesRadarDepthDataset
from src.model.radar_fusion import MoEFusionBlock
from scripts.train import _build_model


device = "cuda" if torch.cuda.is_available() else "cpu"

BASELINE_MAE = {
    "[0.0,20.0)m": 0.7612,
    "[20.0,50.0)m": 3.6500,
    "[50.0,100.0)m": 8.7229,
}


def load_model(run_dir, ckpt):
    cfg = OmegaConf.load(os.path.join(run_dir, "config.yaml"))
    m = _build_model(cfg, float(cfg.dataset.max_depth), int(cfg.dataset.max_radar_points))
    sd = torch.load(os.path.join(run_dir, ckpt), map_location=device, weights_only=False)
    m.load_state_dict(sd["model"])
    print(f"  loaded {run_dir}/{ckpt}  epoch={sd.get('epoch','?')}")
    return m.eval().to(device)


def compute_token_gt(depth_gt_dense, tok_hw, bins):
    K = len(bins) - 1
    edges = torch.tensor(bins, device=depth_gt_dense.device, dtype=torch.float32)
    ds = depth_gt_dense.squeeze(1)
    pix_bin = torch.zeros_like(ds, dtype=torch.long)
    for i in range(K):
        pix_bin = torch.where((ds >= float(edges[i])) & (ds < float(edges[i+1])),
                              pix_bin.new_tensor(i), pix_bin)
    pix_bin = torch.where(ds >= float(edges[-1]),
                          pix_bin.new_tensor(K-1), pix_bin)
    onehot = F.one_hot(pix_bin, num_classes=K).permute(0, 3, 1, 2).float()
    frac = F.adaptive_avg_pool2d(onehot, tok_hw)
    return frac.argmax(dim=1)


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


def eval_with_hooks(model, ds, indices, ranges, hook_mode, bins=None, force_idx=None):
    """hook_mode: 'normal' | 'force' | 'oracle'"""
    moe_blocks = [blk for blk in
                  (model.radar_fusion.node_blocks[2], model.radar_fusion.edge_blocks[2])
                  if isinstance(blk, MoEFusionBlock)]

    current_gt = {}
    hooks = []

    if hook_mode == "force":
        def make_force(idx):
            def _h(_m, _i, out):
                forced = torch.full_like(out, -1e4)
                forced[:, idx] = 1e4
                return forced
            return _h
        for blk in moe_blocks:
            hooks.append(blk.router.register_forward_hook(make_force(force_idx)))
    elif hook_mode == "oracle":
        def make_oracle(blk):
            def _h(_m, _i, out):
                gt = current_gt.get(id(blk))
                if gt is None: return out
                forced = torch.full_like(out, -1e4)
                forced.scatter_(1, gt.unsqueeze(1), 1e4)
                return forced
            return _h
        for blk in moe_blocks:
            hooks.append(blk.router.register_forward_hook(make_oracle(blk)))

    acc = defaultdict(lambda: [0.0, 0])
    with torch.no_grad():
        for i in tqdm(indices, desc=hook_mode + (f":{force_idx}" if force_idx is not None else ""),
                      leave=False):
            s = ds[int(i)]
            rgb = s["rgb_norm"].unsqueeze(0).to(device)
            rp = s["radar_points"].unsqueeze(0).to(device)
            rm = s["radar_mask"].unsqueeze(0).to(device)

            if hook_mode == "oracle":
                dgd = s["depth_gt_dense"].unsqueeze(0).to(device)
                current_gt[id(moe_blocks[0])] = compute_token_gt(dgd, (29, 50), bins)
                current_gt[id(moe_blocks[1])] = compute_token_gt(dgd, (15, 25), bins)

            out = model(rgb, rp, rm)
            pred = out["depth"] if isinstance(out, dict) else out
            pred_np = pred[0, 0].cpu().numpy()
            gt_np = s["depth_gt_lidar"][0].cpu().numpy()
            mask_np = s["valid_mask_lidar"][0].cpu().numpy().astype(bool)
            r = per_range(pred_np, gt_np, mask_np, ranges)
            for k, (se, n) in r.items():
                acc[k][0] += se; acc[k][1] += n

    for h in hooks: h.remove()
    current_gt.clear()
    return {k: (v[0]/v[1] if v[1]>0 else float("nan")) for k, v in acc.items()}


def router_accuracy(model, ds, indices, bins):
    K = len(bins) - 1
    edges = torch.tensor(bins, device=device, dtype=torch.float32)
    confusion = torch.zeros(K, K, dtype=torch.long)

    with torch.no_grad():
        for i in tqdm(indices, desc="router-acc", leave=False):
            s = ds[int(i)]
            rgb = s["rgb_norm"].unsqueeze(0).to(device)
            rp = s["radar_points"].unsqueeze(0).to(device)
            rm = s["radar_mask"].unsqueeze(0).to(device)
            d_gt = s["depth_gt_dense"].unsqueeze(0).to(device)

            out = model(rgb, rp, rm)
            rl = out["router_logits"][0]
            token_pred = rl.softmax(dim=1).argmax(dim=1).squeeze(0)
            token_gt = compute_token_gt(d_gt, token_pred.shape, bins).squeeze(0)
            for gt in range(K):
                for pr in range(K):
                    confusion[gt, pr] += ((token_gt == gt) & (token_pred == pr)).sum().item()

    tot = confusion.sum().item()
    correct = confusion.diagonal().sum().item()
    return 100.0 * correct / max(tot, 1), confusion


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    args = ap.parse_args()

    ranges = [(0.0, 20.0), (20.0, 50.0), (50.0, 100.0)]
    range_names = [f"[{lo},{hi})m" for (lo, hi) in ranges]
    bins = [0.0, 20.0, 50.0, 100.0]
    labels = ["near", "mid", "far"]
    OUT = "output"

    runs = [
        ("no-shared v2 stage2", f"{OUT}/radartaco_moe_v2/radartaco_moe_stage2_v2",           "best.pt"),
        ("shared    v2 stage2", f"{OUT}/radartaco_moe_shared_v2/radartaco_moe_stage2_shared_v2", "best.pt"),
    ]

    cfg = OmegaConf.load(f"{OUT}/shape_lidar_grad_shape_edge_res/config.yaml")
    split_file = os.path.join(cfg.dataset.data_root, "splits", "val.txt")
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
    idxs = np.linspace(0, len(ds)-1, args.n).astype(int).tolist()

    all_results = {}
    for name, run_dir, ckpt in runs:
        print("\n" + "=" * 80); print(f"### {name}"); print("=" * 80)
        m = load_model(run_dir, ckpt)

        # Router accuracy
        print("\n-- Router accuracy --")
        acc, conf = router_accuracy(m, ds, idxs, bins)
        print(f"  Overall: {acc:.1f}%  (tokens={conf.sum().item():,})")
        print(f"  Confusion (rows=GT, cols=predicted):")
        print(f"  {'':6s}" + "  ".join(f"{l:>10s}" for l in labels))
        for gt in range(3):
            row = f"  {labels[gt]:6s}"
            for pr in range(3):
                row += f"  {conf[gt,pr].item():>10,}"
            print(row)
        for b in range(3):
            n_b = conf[b, :].sum().item()
            if n_b > 0:
                print(f"    {labels[b]} recall: {100*conf[b,b].item()/n_b:.1f}%")

        # Normal
        print("\n-- Normal routing --")
        res = {"Normal": eval_with_hooks(m, ds, idxs, ranges, "normal")}
        # Force each expert
        for k, lb in enumerate(labels):
            print(f"-- Force {lb} --")
            res[f"Force-{lb}"] = eval_with_hooks(m, ds, idxs, ranges, "force", force_idx=k)
        # Oracle
        print(f"-- Oracle routing --")
        res["Oracle"] = eval_with_hooks(m, ds, idxs, ranges, "oracle", bins=bins)
        all_results[name] = (res, acc, conf)

        del m; torch.cuda.empty_cache()

    # ── Report ──
    print("\n\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)
    for name, (res, acc, _) in all_results.items():
        print(f"\n### {name}  (router accuracy: {acc:.1f}%)")
        print(f"  {'Mode':<20}" + "  ".join(f"{r:>14}" for r in range_names))
        print(f"  {'Baseline @ e46':<20}" + "  ".join(f"{BASELINE_MAE[r]:>14.4f}" for r in range_names))
        for mode in ["Normal", "Oracle"] + [f"Force-{lb}" for lb in labels]:
            print(f"  {mode:<20}" + "  ".join(f"{res[mode][r]:>14.4f}" for r in range_names))

        print(f"\n  Head-to-head: Force-expert vs baseline (own bin):")
        for k, (lb, r) in enumerate(zip(labels, range_names)):
            b_mae = BASELINE_MAE[r]
            f_mae = res[f"Force-{lb}"][r]
            n_mae = res["Normal"][r]
            o_mae = res["Oracle"][r]
            print(f"    {lb} @ {r}: baseline={b_mae:.4f}  "
                  f"Force={f_mae:+.4f}({100*(f_mae-b_mae)/b_mae:+5.1f}%)  "
                  f"Oracle={o_mae:+.4f}({100*(o_mae-b_mae)/b_mae:+5.1f}%)  "
                  f"Normal={n_mae:+.4f}({100*(n_mae-b_mae)/b_mae:+5.1f}%)")


if __name__ == "__main__":
    main()
