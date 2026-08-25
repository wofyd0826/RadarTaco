"""Transplant BCE-trained router into v4c model and evaluate.

Loads v4c model checkpoint, replaces its L2 node/edge router weights with
weights trained via BCE (either discrete or overlap), and evaluates:

  transplant-softmax  — BCE weights + v4c's original gate pipeline (softmax + top_k=2)
                        Semantic mismatch: BCE was trained expecting sigmoid,
                        here we feed its logits into softmax.
  transplant-sigmoid  — BCE weights + monkey-patched gate:
                        gate = sigmoid(logits) / sum(sigmoid),
                        then top_k=2 renormalization on that distribution.
                        Matches BCE training semantics.

Both transplant modes leave the experts UNTOUCHED (they were trained under
softmax-based routing). The point is to test whether BCE-trained routing
decisions transfer well to v4c's experts.

Reference (native v4c) MAE is HARDCODED from prior full-test eval
(output/analysis/eval_v4c.txt) so we don't re-run it here. Only the two
transplanted variants are actually evaluated.

Report: MAE table (per depth range) for each transplanted mode, with
Δ vs the native reference.

Usage:
  python tests/transplant_bce_router_v4c.py \\
      --v4c-run output/radartaco_moe_stage2_v4c \\
      --bce-ckpt output/router_pretrain/bce_overlap/best.pt \\
      --tag bce_overlap_transplant
"""
import argparse
import io
import os
import sys
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
from tqdm import tqdm

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, ROOT)

from src.dataset.nuscenes import NuScenesRadarDepthDataset
from src.model.radar_fusion import MoEFusionBlock
from scripts.train import _build_model


device = "cuda" if torch.cuda.is_available() else "cpu"


def load_v4c(run_dir, ckpt="best.pt"):
    cfg = OmegaConf.load(os.path.join(run_dir, "config.yaml"))
    m = _build_model(cfg, float(cfg.dataset.max_depth),
                     int(cfg.dataset.max_radar_points))
    sd = torch.load(os.path.join(run_dir, ckpt),
                    map_location=device, weights_only=False)
    m.load_state_dict(sd["model"])
    print(f"  loaded v4c {run_dir}/{ckpt}  epoch={sd.get('epoch','?')}")
    return m.eval().to(device), cfg


def load_bce_router(ckpt_path):
    sd = torch.load(ckpt_path, map_location=device, weights_only=False)
    print(f"  loaded BCE router {ckpt_path}  epoch={sd.get('epoch','?')}")
    print(f"    config: {sd.get('config', {})}")
    return sd


def transplant_routers(model, bce_sd):
    """Copy BCE-trained router weights into model's L2 node/edge routers."""
    # v4c's routers live at:
    #   model.radar_fusion.node_blocks[2].router
    #   model.radar_fusion.edge_blocks[2].router
    node_router = model.radar_fusion.node_blocks[2].router
    edge_router = model.radar_fusion.edge_blocks[2].router

    node_state = bce_sd["router_node"]
    edge_state = bce_sd["router_edge"]

    # Sanity: keys must match exactly
    m_node = node_router.state_dict()
    m_edge = edge_router.state_dict()
    assert set(m_node.keys()) == set(node_state.keys()), \
        f"node keys mismatch: model={list(m_node.keys())[:3]} bce={list(node_state.keys())[:3]}"
    assert set(m_edge.keys()) == set(edge_state.keys())

    # Shape check
    for k in m_node.keys():
        assert m_node[k].shape == node_state[k].shape, \
            f"node shape mismatch at {k}: {m_node[k].shape} vs {node_state[k].shape}"
    for k in m_edge.keys():
        assert m_edge[k].shape == edge_state[k].shape

    node_router.load_state_dict(node_state)
    edge_router.load_state_dict(edge_state)
    print(f"  Transplanted router weights into L2 node + edge routers.")


def install_sigmoid_renorm_hooks(model):
    """Post-hook on each L2 router: convert logits so that downstream
    softmax reproduces sigmoid+renorm distribution.

    If orig_logits produce sigmoid distribution s, and desired gate is
    s/sum(s), we want softmax(new_logits) = s/sum(s).
    Solution: new_logits = log(s/sum(s)) = log(s) - log(sum(s))
    Since softmax is shift-invariant: new_logits = log(s) works.
    """
    hooks = []
    for block_name, block in [
        ("L2_node", model.radar_fusion.node_blocks[2]),
        ("L2_edge", model.radar_fusion.edge_blocks[2]),
    ]:
        if not isinstance(block, MoEFusionBlock):
            continue
        def make_hook():
            def _h(_m, _i, out):
                # out: (B, K, H, W) — the router logits
                sig = torch.sigmoid(out)
                # log(sigmoid) — softmax(log(sig)) = sig / sum(sig)
                return torch.log(sig.clamp_min(1e-8))
            return _h
        h = block.router.register_forward_hook(make_hook())
        hooks.append(h)
    return hooks


def per_range(pred, gt, mask, ranges):
    """Return {range_name: (sum_abs_err, n_pix)} per depth-range bucket."""
    out = {}
    for name, (lo, hi) in ranges.items():
        m = mask & (gt >= lo) & (gt < hi) if name != "0-100m" else mask
        n = int(m.sum())
        if n == 0:
            out[name] = (0.0, 0)
        else:
            out[name] = (float(np.abs(pred[m] - gt[m]).sum()), n)
    return out


@torch.no_grad()
def eval_mode(model, ds, idxs, mode_label, ranges):
    """Evaluate under Normal routing. Returns per-range MAE dict."""
    model.eval()
    accs = defaultdict(lambda: [0.0, 0])
    for i in tqdm(idxs, desc=mode_label, leave=False):
        s = ds[int(i)]
        rgb = s["rgb_norm"].unsqueeze(0).to(device)
        rp = s["radar_points"].unsqueeze(0).to(device)
        rm = s["radar_mask"].unsqueeze(0).to(device)
        gt_np = s["depth_gt_lidar"][0].cpu().numpy()
        mask_np = s["valid_mask_lidar"][0].cpu().numpy().astype(bool)

        out = model(rgb, rp, rm)
        pred = (out["depth"] if isinstance(out, dict) else out)[0, 0].cpu().numpy()
        for k, (se, n) in per_range(pred, gt_np, mask_np, ranges).items():
            accs[k][0] += se; accs[k][1] += n

    return {k: (v[0]/v[1] if v[1] > 0 else float("nan")) for k, v in accs.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v4c-run", default="output/radartaco_moe_stage2_v4c")
    ap.add_argument("--v4c-ckpt", default="best.pt")
    ap.add_argument("--bce-ckpt", required=True,
                    help="Path to BCE router ckpt (e.g., output/router_pretrain/bce_overlap/best.pt)")
    ap.add_argument("--split", default="test")
    ap.add_argument("--n", type=int, default=None,
                    help="Sample count (default: full split)")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out-dir", default="output/analysis/transplant")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Dataset (paths from a fresh baseline config)
    base_cfg = OmegaConf.load(
        "output/shape_lidar_grad_shape_edge_res/config.yaml")
    split_file = os.path.join(base_cfg.dataset.data_root, "splits",
                              f"{args.split}.txt")
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
    idxs = list(range(n)) if n == len(ds) else \
           np.linspace(0, len(ds)-1, n).astype(int).tolist()
    print(f"{args.split} — {len(idxs):,}/{len(ds):,} samples")

    ranges = {
        "0-20m":   (0.0, 20.0),
        "20-50m":  (20.0, 50.0),
        "50-100m": (50.0, 100.0),
        "0-80m":   (0.0, 80.0),
        "0-100m":  (0.0, 100.0),
    }

    # Reference: hard-coded native v4c MAE from prior full-test eval
    # (output/analysis/eval_v4c.txt, "v4c stage2 Normal" row).
    # Skip re-running native to save ~20 min.
    mae_native = {
        "0-20m":   0.7402,
        "20-50m":  3.4357,
        "50-100m": 9.2949,
        "0-80m":   1.8246,
        "0-100m":  1.9124,
    }

    buf = io.StringIO()
    def _p(msg=""):
        print(msg); buf.write(msg + "\n")

    _p("=" * 96)
    _p(f"Router transplant experiment — {args.tag}")
    _p(f"  v4c: {args.v4c_run}/{args.v4c_ckpt}")
    _p(f"  BCE router: {args.bce_ckpt}")
    _p(f"  split: {args.split}, n_samples: {len(idxs):,}")
    _p(f"  native reference (hardcoded from eval_v4c.txt):")
    for k in ranges:
        _p(f"    {k:<10s}: {mae_native[k]:.4f}")
    _p("=" * 96)

    # ==== Pass 1: transplanted, softmax (naive) ====
    print("\n[1/2] Transplanted BCE router + v4c softmax pipeline (naive)")
    model, _ = load_v4c(args.v4c_run, args.v4c_ckpt)
    bce_sd = load_bce_router(args.bce_ckpt)
    transplant_routers(model, bce_sd)
    t0 = time.time()
    mae_soft = eval_mode(model, ds, idxs, "transplant-softmax", ranges)
    _p(f"\n-- Transplant + softmax (naive) — {time.time()-t0:.0f}s --")
    for k in ranges:
        delta = mae_soft[k] - mae_native[k]
        _p(f"  {k:<10s}: {mae_soft[k]:.4f}  (Δnative {delta:+.4f})")

    # ==== Pass 2: transplanted, sigmoid+renorm (semantic match) ====
    print("\n[2/2] Transplanted BCE router + sigmoid+renorm patch (semantic match)")
    hooks = install_sigmoid_renorm_hooks(model)
    t0 = time.time()
    try:
        mae_sig = eval_mode(model, ds, idxs, "transplant-sigmoid", ranges)
    finally:
        for h in hooks: h.remove()
    _p(f"\n-- Transplant + sigmoid+renorm — {time.time()-t0:.0f}s --")
    for k in ranges:
        delta = mae_sig[k] - mae_native[k]
        _p(f"  {k:<10s}: {mae_sig[k]:.4f}  (Δnative {delta:+.4f})")

    # ==== Summary table ====
    _p("\n" + "=" * 96)
    _p("SUMMARY (MAE, m)")
    _p("=" * 96)
    _p(f"  {'range':<10s}  {'native (ref)':>13s}  {'BCE+softmax':>12s}  "
       f"{'BCE+sigmoid':>12s}  {'best mode':>12s}")
    for k in ranges:
        best = min([('native', mae_native[k]),
                    ('softmax', mae_soft[k]),
                    ('sigmoid', mae_sig[k])], key=lambda x: x[1])
        _p(f"  {k:<10s}  {mae_native[k]:>13.4f}  {mae_soft[k]:>12.4f}  "
           f"{mae_sig[k]:>12.4f}  {best[0]:>12s}")

    _p("\nInterpretation:")
    _p("  - BCE+softmax Δ ≈ 0 vs native   → BCE-trained router behaves like v4c")
    _p("  - BCE+sigmoid better than BCE+softmax → sigmoid semantics matter")
    _p("  - Both worse than native → BCE routing doesn't help these experts")
    _p("    (experts were trained under softmax routing; need retraining)")

    out_txt = os.path.join(args.out_dir, f"{args.tag}.txt")
    with open(out_txt, "w") as f:
        f.write(buf.getvalue())
    print(f"\nSaved: {out_txt}")


if __name__ == "__main__":
    main()
