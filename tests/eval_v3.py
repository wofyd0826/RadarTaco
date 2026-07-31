"""Evaluate v3 stage 1 + stage 2: Normal + Oracle + router accuracy.

No prediction clipping. Full test split by default.
Saves JSON + text summary to output/analysis/eval_v3.{json,txt}.
"""
import argparse, io, json, os, sys, time
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

BASELINE_MAE = {   # baseline @ e46 on FULL test split, no-clip
    "[0.0,20.0)m": 0.6974,
    "[20.0,50.0)m": 3.5083,
    "[50.0,100.0)m": 9.0317,
    "0-100m": 1.8840,
}
BINS = [0.0, 20.0, 50.0, 100.0]


def load_model(run_dir, ckpt="best.pt"):
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
    for cap in (80.0, 100.0):
        m = mask & (gt > 0) & (gt < cap)
        if m.any():
            err = np.abs(pred[m] - gt[m])
            out[f"0-{cap:g}m"] = (float(err.sum()), int(m.sum()))
        else:
            out[f"0-{cap:g}m"] = (0.0, 0)
    return out


def evaluate(model, ds, indices, ranges, mode, bins=None):
    """mode: 'normal' | 'oracle'.

    'normal': also computes router accuracy (from same forward — router
    logits emitted alongside depth pred).
    'oracle': overrides router via forward hook.
    Returns: (mae_dict, router_acc_or_None, confusion_or_None)
    """
    moe_blocks = [blk for blk in
                  (model.radar_fusion.node_blocks[2], model.radar_fusion.edge_blocks[2])
                  if isinstance(blk, MoEFusionBlock)]
    current_gt = {}
    hooks = []
    if mode == "oracle" and moe_blocks:
        def make_hook(blk):
            def _h(_m, _i, out):
                gt = current_gt.get(id(blk))
                if gt is None: return out
                forced = torch.full_like(out, -1e4)
                forced.scatter_(1, gt.unsqueeze(1), 1e4)
                return forced
            return _h
        for blk in moe_blocks:
            hooks.append(blk.router.register_forward_hook(make_hook(blk)))

    K = len(bins) - 1 if bins else 3
    acc = defaultdict(lambda: [0.0, 0])
    confusion = torch.zeros(K, K, dtype=torch.long) if mode == "normal" else None

    with torch.no_grad():
        for i in tqdm(indices, desc=mode, leave=False):
            s = ds[int(i)]
            rgb = s["rgb_norm"].unsqueeze(0).to(device)
            rp = s["radar_points"].unsqueeze(0).to(device)
            rm = s["radar_mask"].unsqueeze(0).to(device)
            dgd = s["depth_gt_dense"].unsqueeze(0).to(device)
            if mode == "oracle" and moe_blocks:
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

            # Router accuracy (only meaningful in normal mode — oracle hook
            # replaces router logits, so router prediction is not the model's
            # native prediction).
            if mode == "normal" and isinstance(out, dict) and out.get("router_logits"):
                rl = out["router_logits"][0]  # finest grid (L2 node)
                token_pred = rl.softmax(dim=1).argmax(dim=1).squeeze(0)
                token_gt = compute_token_gt(dgd, token_pred.shape, bins).squeeze(0)
                for gt_b in range(K):
                    for pr in range(K):
                        confusion[gt_b, pr] += ((token_gt == gt_b) & (token_pred == pr)).sum().item()

    for h in hooks: h.remove()
    current_gt.clear()
    mae = {k: (v[0]/v[1] if v[1]>0 else float("nan")) for k, v in acc.items()}
    if confusion is not None:
        tot = confusion.sum().item()
        router_acc = 100.0 * confusion.diagonal().sum().item() / max(tot, 1)
        return mae, router_acc, confusion
    return mae, None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--split", default="test")
    ap.add_argument("--out-dir", default="output/analysis",
                    help="Where to save eval_v3.json / eval_v3.txt")
    ap.add_argument("--tag", default="v3",
                    help="Filename tag: eval_<tag>.{json,txt}")
    ap.add_argument("--runs", nargs="+", default=None,
                    help="Override run list. Format: 'name:dir[:ckpt]' each. "
                         "Default: v3 stage1/2.")
    args = ap.parse_args()

    ranges = [(0.0, 20.0), (20.0, 50.0), (50.0, 100.0)]
    range_names = [f"[{lo},{hi})m" for (lo, hi) in ranges] + ["0-80m", "0-100m"]
    OUT = "output"

    if args.runs:
        runs = []
        for spec in args.runs:
            parts = spec.split(":")
            name, run_dir = parts[0], parts[1]
            ckpt = parts[2] if len(parts) > 2 else "best.pt"
            runs.append((name, run_dir, ckpt))
    else:
        runs = [
            ("v3 stage1 best", f"{OUT}/radartaco_moe_stage1_v3", "best.pt"),
            ("v3 stage2 best", f"{OUT}/radartaco_moe_stage2_v3", "best.pt"),
        ]

    base_cfg = OmegaConf.load(f"{OUT}/shape_lidar_grad_shape_edge_res/config.yaml")
    split_file = os.path.join(base_cfg.dataset.data_root, "splits", f"{args.split}.txt")
    ds = NuScenesRadarDepthDataset(
        data_root=base_cfg.dataset.data_root, split_file=split_file,
        dense_gt_dir=base_cfg.dataset.dense_gt_dir,
        radar_3d_dir=base_cfg.dataset.radar_3d_dir,
        night_ids_file=base_cfg.dataset.night_ids_file,
        max_radar_points=int(base_cfg.dataset.max_radar_points),
        max_depth=float(base_cfg.dataset.max_depth),
        min_depth=float(base_cfg.dataset.min_depth),
        augmentation=False,
    )
    n = args.n or len(ds)
    idxs = list(range(n)) if n == len(ds) else np.linspace(0, len(ds)-1, n).astype(int).tolist()
    print(f"{args.split} — {len(idxs):,}/{len(ds):,} samples\n")

    results = {}
    labels = ["near", "mid", "far"]
    for name, run_dir, ckpt in runs:
        print("=" * 78); print(f"### {name}"); print("=" * 78)
        m = load_model(run_dir, ckpt)
        t0 = time.time()
        # Single pass — computes Normal MAE + router accuracy together
        print("-- Normal + Router accuracy --")
        res_normal, router_acc, conf = evaluate(m, ds, idxs, ranges, "normal", bins=BINS)
        print(f"  Router: {router_acc:.1f}%  (tokens={conf.sum().item():,})")
        for b in range(3):
            n_b = conf[b, :].sum().item()
            if n_b > 0:
                print(f"    {labels[b]} recall: {100*conf[b,b].item()/n_b:.1f}%")
        print("-- Oracle --")
        res_oracle, _, _ = evaluate(m, ds, idxs, ranges, "oracle", bins=BINS)
        print(f"  ({time.time()-t0:.0f}s)")
        results[name] = {"Normal": res_normal, "Oracle": res_oracle,
                          "router_acc": router_acc, "confusion": conf.tolist()}
        del m; torch.cuda.empty_cache()

    # Build a text summary (also printed to stdout).
    buf = io.StringIO()
    def _p(s=""):
        print(s); buf.write(s + "\n")

    _p("\n" + "=" * 100)
    _p(f"MAE table ({args.split} — {len(idxs):,} samples, no-clip)")
    _p("=" * 100)
    _p(f"{'Setting':<28}" + "  ".join(f"{r:>12}" for r in range_names))
    _p(f"{'Baseline (cached)':<28}" +
       "  ".join(f"{BASELINE_MAE[r]:>12.4f}" if r in BASELINE_MAE
                 else f"{'—':>12}" for r in range_names))
    _p("-" * 100)
    for name, res in results.items():
        for mode in ["Normal", "Oracle"]:
            row = f"{name[:24]:<24} {mode:<4}"
            for r in range_names:
                row += f"  {res[mode][r]:>12.4f}"
            _p(row)

    _p("\n" + "=" * 100)
    _p("Δ vs Baseline (negative = better)")
    _p("=" * 100)
    base = BASELINE_MAE
    for name, res in results.items():
        for mode in ["Normal", "Oracle"]:
            row = f"{name[:24]:<24} {mode:<4}"
            for r in range_names:
                if r in base:
                    d = res[mode][r] - base[r]
                    pct = 100*d/base[r]
                    row += f"  {d:>+7.4f} ({pct:+6.1f}%)"
                else:
                    row += f"  {'—':>17}"
            _p(row)

    _p("\n" + "=" * 78)
    _p("Router accuracy summary")
    _p("=" * 78)
    for name, res in results.items():
        conf = torch.tensor(res["confusion"])
        recalls = []
        for b, lb in enumerate(labels):
            n_b = conf[b, :].sum().item()
            if n_b > 0:
                recalls.append(f"{lb} {100*conf[b,b].item()/n_b:.1f}%")
        _p(f"  {name}: overall {res['router_acc']:.1f}%   "
           f"({'   '.join(recalls)})   tokens={conf.sum().item():,}")

    # Persist: JSON (machine) + TXT (human). Matches
    # output/analysis/eval_full_test.json layout used by earlier experiments.
    os.makedirs(args.out_dir, exist_ok=True)
    payload = {
        "split": args.split,
        "n_samples": len(idxs),
        "results": {
            "Baseline @ e46": {"Normal": BASELINE_MAE},
            **{name: {
                "Normal": res["Normal"],
                "Oracle": res["Oracle"],
                "router_acc": res["router_acc"],
                "confusion": res["confusion"],
            } for name, res in results.items()},
        },
    }
    json_path = os.path.join(args.out_dir, f"eval_{args.tag}.json")
    txt_path  = os.path.join(args.out_dir, f"eval_{args.tag}.txt")
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)
    with open(txt_path, "w") as f:
        f.write(buf.getvalue())
    print(f"\nSaved: {json_path}\nSaved: {txt_path}")


if __name__ == "__main__":
    main()
