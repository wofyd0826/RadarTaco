"""Ceiling analysis — perfect soft routing on mixed tokens.

Uses the token's true pixel-bin fraction (from depth_gt_dense) as the
router gate, bypassing the learned router entirely for the chosen tokens.
Shows the MAE ceiling achievable if router capacity itself were perfect.

Four modes compared on the same model:

  normal            : self-routing (learned router, current behaviour)
  oracle            : hard one-hot GT gate for every token (existing)
  soft              : gate = pixel-bin fraction for every token
                        - pure tokens → near-one-hot (like oracle)
                        - mixed tokens → true soft mixture
  soft_mixed_oracle : mixed → soft gate, pure → hard-one-hot GT gate
  soft_mixed_normal : mixed → soft gate, pure → self-router
                        (isolates the impact of ONLY fixing mixed tokens)

Report gap:
  oracle − normal              : total router cost
  soft   − oracle              : incremental gain from soft over hard (all tokens)
  soft_mixed_normal − normal   : upper bound of "fix only mixed" strategy
  soft_mixed_oracle − oracle   : same as soft − oracle when pure already optimal

Bypasses top_k=1 gate collapse (gate=1) by rewriting the gate directly.

Usage
-----
  python tests/eval_perfect_soft.py \
      --run output/radartaco_moe_stage2_v3 \
      --tag v3_stage2 \
      --mixed-threshold 0.05
"""
import argparse, io, json, os, sys, time, types
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
BINS = [0.0, 20.0, 50.0, 100.0]
MODES = ["normal", "oracle", "soft", "soft_mixed_oracle", "soft_mixed_normal"]


def load_model(run_dir, ckpt="best.pt"):
    cfg = OmegaConf.load(os.path.join(run_dir, "config.yaml"))
    m = _build_model(cfg, float(cfg.dataset.max_depth), int(cfg.dataset.max_radar_points))
    sd = torch.load(os.path.join(run_dir, ckpt), map_location=device, weights_only=False)
    m.load_state_dict(sd["model"])
    print(f"  loaded {run_dir}/{ckpt}  epoch={sd.get('epoch','?')}")
    return m.eval().to(device)


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


class SoftRoutingHook:
    """Monkey-patches MoEFusionBlock.forward to support ceiling-analysis modes.

    The block's original forward is stashed and restored via attach/detach.
    Per-sample depth_gt_dense is placed in `holder[id(block)]` before the
    model forward call — the patched forward reads it from there.
    """

    def __init__(self, mode: str, mixed_threshold: float = 0.05):
        assert mode in MODES, f"mode must be one of {MODES}"
        self.mode = mode
        self.mixed_threshold = mixed_threshold
        self.holder = {}                # id(block) -> depth_gt_dense
        self.orig = {}                  # id(block) -> original forward

    def attach(self, blocks):
        for blk in blocks:
            self.orig[id(blk)] = blk.forward
            blk.forward = types.MethodType(self._make_forward(), blk)

    def detach(self, blocks):
        for blk in blocks:
            blk.forward = self.orig[id(blk)]
        self.holder.clear()
        self.orig.clear()

    def set_gt(self, blocks, dgd):
        for blk in blocks:
            self.holder[id(blk)] = dgd

    def _make_forward(self):
        mode = self.mode
        thr = self.mixed_threshold
        holder = self.holder

        def forward(self_blk, feat, kv, radar_x_orig, radar_mask, image_w,
                    depth_gt_dense=None, teacher_force=True):
            B, C, H, W = feat.shape
            logits = self_blk.router(feat)              # (B, K, H, W)

            dgd = holder.get(id(self_blk))
            frac = hard_gt = hard_gt_gate = is_mixed = None
            if dgd is not None:
                with torch.no_grad():
                    pix_bin = self_blk._depth_to_bin(dgd.float())
                    oh = F.one_hot(pix_bin, num_classes=self_blk.n_experts) \
                          .permute(0, 3, 1, 2).float()
                    frac = F.adaptive_avg_pool2d(oh, (H, W))         # (B, K, H, W)
                    hard_gt = frac.argmax(dim=1)                     # (B, H, W)
                    boundary = 1.0 - frac.max(dim=1).values          # (B, H, W)
                    is_mixed = (boundary > thr).unsqueeze(1).float() # (B, 1, H, W)
                hard_gt_gate = F.one_hot(hard_gt, num_classes=self_blk.n_experts) \
                                .permute(0, 3, 1, 2).to(logits.dtype)

            # Self-routed gate (top_k=1 collapse or full softmax)
            probs = F.softmax(logits, dim=1)
            if self_blk.top_k is not None and self_blk.top_k < self_blk.n_experts:
                _, idx = probs.topk(self_blk.top_k, dim=1)
                keep = torch.zeros_like(probs).scatter_(1, idx, 1.0)
                self_gate = probs * keep
                self_gate = self_gate / self_gate.sum(dim=1, keepdim=True).clamp_min(1e-8)
            else:
                self_gate = probs

            # Select gate by mode
            if mode == "normal":
                gate = self_gate
            elif mode == "oracle":
                gate = hard_gt_gate
            elif mode == "soft":
                gate = frac
            elif mode == "soft_mixed_oracle":
                gate = is_mixed * frac + (1.0 - is_mixed) * hard_gt_gate
            elif mode == "soft_mixed_normal":
                gate = is_mixed * frac + (1.0 - is_mixed) * self_gate
            else:
                raise ValueError(mode)

            expert_outs = torch.stack([
                e(feat, kv, radar_x_orig, radar_mask, image_w)
                for e in self_blk.experts
            ], dim=1)                                                 # (B, K, C, H, W)
            mixed = (gate.unsqueeze(2) * expert_outs).sum(dim=1)

            if self_blk.shared is not None:
                shared_delta = (self_blk.shared(feat, kv, radar_x_orig, radar_mask, image_w)
                                - feat)
                mixed = mixed + shared_delta

            return mixed, logits, hard_gt
        return forward


def evaluate(model, moe_blocks, ds, indices, ranges, mode, mixed_thr):
    hook = SoftRoutingHook(mode=mode, mixed_threshold=mixed_thr)
    hook.attach(moe_blocks)
    acc = defaultdict(lambda: [0.0, 0])
    try:
        with torch.no_grad():
            for i in tqdm(indices, desc=mode, leave=False):
                s = ds[int(i)]
                rgb = s["rgb_norm"].unsqueeze(0).to(device)
                rp  = s["radar_points"].unsqueeze(0).to(device)
                rm  = s["radar_mask"].unsqueeze(0).to(device)
                dgd = s["depth_gt_dense"].unsqueeze(0).to(device)

                hook.set_gt(moe_blocks, dgd)
                out = model(rgb, rp, rm)
                pred = out["depth"] if isinstance(out, dict) else out
                pred_np = pred[0, 0].cpu().numpy()
                gt_np   = s["depth_gt_lidar"][0].cpu().numpy()
                mask_np = s["valid_mask_lidar"][0].cpu().numpy().astype(bool)

                r = per_range(pred_np, gt_np, mask_np, ranges)
                for k, (se, n) in r.items():
                    acc[k][0] += se; acc[k][1] += n
    finally:
        hook.detach(moe_blocks)

    return {k: (v[0] / v[1] if v[1] > 0 else float("nan"))
            for k, v in acc.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="Model run_dir")
    ap.add_argument("--tag", required=True, help="e.g. v3_stage2 — used in filenames")
    ap.add_argument("--ckpt", default="best.pt")
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--split", default="test")
    ap.add_argument("--mixed-threshold", type=float, default=0.05,
                    help="Boundary score threshold for 'mixed' (>thr). "
                         "0.00 = every non-pure token counts as mixed; "
                         "0.05 = single-bin-dominant tokens (max ≥ 95%%) excluded.")
    ap.add_argument("--modes", nargs="+", default=["soft"],
                    choices=MODES, help="Which modes to run (default: soft only)")
    ap.add_argument("--out-dir", default="output/analysis/perfect_soft")
    args = ap.parse_args()

    ranges = [(0.0, 20.0), (20.0, 50.0), (50.0, 100.0)]
    range_names = [f"[{lo},{hi})m" for (lo, hi) in ranges] + ["0-80m", "0-100m"]

    OUT = "output"
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
    print(f"{args.split} — {len(idxs):,}/{len(ds):,} samples")
    print(f"tag = {args.tag}, run = {args.run}, mixed_threshold = {args.mixed_threshold}")

    model = load_model(args.run, args.ckpt)
    moe_blocks = [blk for blk in
                  (model.radar_fusion.node_blocks[2], model.radar_fusion.edge_blocks[2])
                  if isinstance(blk, MoEFusionBlock)]
    assert moe_blocks, "No MoE blocks found — this script requires an MoE model."
    print(f"  {len(moe_blocks)} MoE block(s) at L2")

    results = {}
    for mode in args.modes:
        t0 = time.time()
        print(f"\n== {mode} ==")
        mae = evaluate(model, moe_blocks, ds, idxs, ranges, mode, args.mixed_threshold)
        print(f"  0-100m MAE = {mae['0-100m']:.4f}  ({time.time()-t0:.0f}s)")
        results[mode] = mae

    # Build summary text
    buf = io.StringIO()
    def _p(msg=""):
        print(msg); buf.write(msg + "\n")

    _p("\n" + "=" * 110)
    _p(f"Perfect-soft ceiling analysis — {args.tag} "
       f"({args.split}, n={len(idxs):,}, mixed_thr={args.mixed_threshold})")
    _p("=" * 110)
    _p(f"{'Mode':<24}" + "  ".join(f"{r:>12}" for r in range_names))
    _p("-" * 110)
    for mode in args.modes:
        row = f"{mode:<24}"
        for r in range_names:
            row += f"  {results[mode][r]:>12.4f}"
        _p(row)

    if "normal" in results and "oracle" in results:
        _p("\n" + "=" * 110)
        _p("Δ vs normal (negative = better)")
        _p("=" * 110)
        base = results["normal"]
        for mode in args.modes:
            if mode == "normal": continue
            row = f"{mode:<24}"
            for r in range_names:
                d = results[mode][r] - base[r]
                pct = 100 * d / base[r] if base[r] > 0 else 0
                row += f"  {d:>+7.4f} ({pct:+6.1f}%)"
            _p(row)

    if "oracle" in results and "soft" in results:
        _p("\n" + "=" * 60)
        _p("Δ soft vs oracle  (impact of using soft on ALL tokens)")
        _p("=" * 60)
        for r in range_names:
            d = results["soft"][r] - results["oracle"][r]
            _p(f"  {r:<14}: {d:>+7.4f}")

    if "normal" in results and "soft_mixed_normal" in results:
        _p("\n" + "=" * 60)
        _p("Δ soft_mixed_normal vs normal  (fix ONLY mixed tokens)")
        _p("=" * 60)
        for r in range_names:
            d = results["soft_mixed_normal"][r] - results["normal"][r]
            pct = 100 * d / results["normal"][r] if results["normal"][r] > 0 else 0
            _p(f"  {r:<14}: {d:>+7.4f}  ({pct:+6.1f}%)")

    os.makedirs(args.out_dir, exist_ok=True)
    json_path = os.path.join(args.out_dir, f"perfect_soft_{args.tag}.json")
    txt_path  = os.path.join(args.out_dir, f"perfect_soft_{args.tag}.txt")
    with open(json_path, "w") as f:
        json.dump({
            "split": args.split, "n_samples": len(idxs),
            "run": args.run, "ckpt": args.ckpt,
            "mixed_threshold": args.mixed_threshold,
            "results": results,
        }, f, indent=2)
    with open(txt_path, "w") as f:
        f.write(buf.getvalue())
    print(f"\nSaved: {json_path}\nSaved: {txt_path}")


if __name__ == "__main__":
    main()
