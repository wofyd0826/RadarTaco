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


def compute_token_gt(depth_gt_dense, tok_hw, bins, return_type: str = "hard"):
    """Per-token routing GT derived from `depth_gt_dense`.

    Args:
        return_type: "hard" → (B, H, W) int64 majority-vote bin index.
                     "soft" → (B, K, H, W) float pixel-bin FRACTION (sums to 1
                                 along K); matches MoEFusionBlock's soft
                                 `router_gt_type` regime.
                     "both" → (frac_soft, argmax_hard) tuple — useful when
                              the caller needs both formats without recompute.
    """
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
    frac = F.adaptive_avg_pool2d(onehot, tok_hw)                # (B, K, H, W)
    if return_type == "soft":
        return frac
    if return_type == "both":
        return frac, frac.argmax(dim=1)
    return frac.argmax(dim=1)                                   # hard (default)


def per_range(pred, gt, mask, ranges):
    """Return {range_name: (sum_l1, sum_sq, count)} for MAE + RMSE aggregation."""
    out = {}
    for (lo, hi) in ranges:
        m = mask & (gt >= lo) & (gt < hi) & (gt > 0)
        if m.any():
            diff = pred[m] - gt[m]
            out[f"[{lo},{hi})m"] = (float(np.abs(diff).sum()),
                                    float((diff * diff).sum()),
                                    int(m.sum()))
        else:
            out[f"[{lo},{hi})m"] = (0.0, 0.0, 0)
    for cap in (80.0, 100.0):
        m = mask & (gt > 0) & (gt < cap)
        if m.any():
            diff = pred[m] - gt[m]
            out[f"0-{cap:g}m"] = (float(np.abs(diff).sum()),
                                  float((diff * diff).sum()),
                                  int(m.sum()))
        else:
            out[f"0-{cap:g}m"] = (0.0, 0.0, 0)
    return out


def _collect_moe_blocks(model):
    """Enumerate MoE blocks in `router_logits` order (node then edge per level)."""
    out = []
    for l in range(len(model.radar_fusion.node_blocks)):
        nb = model.radar_fusion.node_blocks[l]
        eb = model.radar_fusion.edge_blocks[l]
        if isinstance(nb, MoEFusionBlock): out.append((f"L{l}_node", nb))
        if isinstance(eb, MoEFusionBlock): out.append((f"L{l}_edge", eb))
    return out


def evaluate_all_modes(model, ds, indices, ranges, bins=None, spec_only=False):
    """Unified per-sample evaluation of Normal + OracleHard + OracleSoft.

    Speedups over calling `evaluate()` three times:
      • Dataset loaded ONCE per sample (was 3×).
      • MoE block enumeration + hook creation done ONCE (was 3×).
      • OracleSoft is SKIPPED entirely for models whose MoE blocks all have
        `top_k == 1`, because the soft-frac gate collapses to hard via the
        top-1 gate mechanism (result is identical to OracleHard, verified
        empirically on v3_fix). OracleSoft is then filled in by copying
        OracleHard so downstream code stays uniform.
      • Router stats (hard acc + soft_l1 per MoE block) computed as a
        side-effect of the Normal pass — no extra forward.

    Encoder / radar / L0-L1 fusion still re-run for each oracle pass — a
    deeper cache would need model-forward surgery.

    Returns: (
        results: {"Normal": {"mae": range_dict, "rmse": range_dict},
                  "OracleHard": {...}, "OracleSoft": {...}},
        router_stats: dict from Normal pass (see below),
        skipped_soft: bool — True if OracleSoft was copied from OracleHard,
        timing: {mode: {"total_s": float, "per_sample_s": float,
                        "n_forwards": int}}  — GPU-synchronized per-mode
                                               forward-only wall time.
    )

    router_stats layout:
      {'blocks': {block_name: {'confusion': KxK, 'accuracy', 'recall',
                                'soft_l1', 'tokens'}},
       'overall': {'accuracy', 'tokens', 'soft_l1'}}
    """
    moe_blocks = _collect_moe_blocks(model)

    # Optionally strip shared expert and disable confidence gating so the
    # forward is "pure specialist mix" — same as v4c's inference regime.
    # Useful for isolating specialist quality in models trained with a
    # shared expert (v5_shared, v6a, v6b, v6c). Restored in `finally`.
    saved_shared = {}
    saved_gate_mode = {}
    if spec_only:
        for name, blk in moe_blocks:
            saved_shared[name] = blk.shared
            saved_gate_mode[name] = blk.shared_gate_mode
            blk.shared = None                  # skip shared code path entirely
            blk.shared_gate_mode = "always_on" # disable α scaling of gate

    # Detect if OracleSoft can be safely skipped (top_k == 1 for all MoE blocks).
    skip_soft = bool(moe_blocks) and all(
        (blk.top_k is not None and blk.top_k == 1) for _, blk in moe_blocks
    )
    modes_to_run = ["Normal", "OracleHard"] + ([] if skip_soft else ["OracleSoft"])

    # Single hook per router; mode flag decides behaviour on each forward.
    dgd_holder = {"dgd": None, "mode": None}  # mode ∈ {None, 'hard', 'soft'}
    hooks = []
    if moe_blocks:
        def _hook(_m, _i, out):
            mode = dgd_holder["mode"]
            dgd = dgd_holder["dgd"]
            if mode is None or dgd is None:
                return out                                   # Normal pass — passthrough
            if mode == "soft":
                frac = compute_token_gt(dgd, out.shape[-2:], bins,
                                        return_type="soft")  # (B, K, H, W)
                return torch.log(frac.clamp_min(1e-8))
            tok_gt = compute_token_gt(dgd, out.shape[-2:], bins,
                                      return_type="hard")    # (B, H, W)
            forced = torch.full_like(out, -1e4)
            forced.scatter_(1, tok_gt.unsqueeze(1), 1e4)
            return forced
        for _, blk in moe_blocks:
            hooks.append(blk.router.register_forward_hook(_hook))

    K = len(bins) - 1 if bins else 3
    # (sum_l1, sum_sq, count) accumulators per mode → MAE + RMSE.
    accs = {m: defaultdict(lambda: [0.0, 0.0, 0]) for m in modes_to_run}
    # Per-mode GPU-synchronized forward wall time.
    timing = {m: {"total_s": 0.0, "n_forwards": 0} for m in modes_to_run}
    # Router stats (only for Normal pass).
    confusion = {name: torch.zeros(K, K, dtype=torch.long) for name, _ in moe_blocks}
    soft_l1 = {name: [0.0, 0] for name, _ in moe_blocks}

    def _timed_forward(mode_name):
        """Run one model forward, GPU-synchronized on both sides for accurate
        wall-time. Returns the model output."""
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = model(rgb, rp, rm)
        if device == "cuda":
            torch.cuda.synchronize()
        timing[mode_name]["total_s"] += (time.perf_counter() - t0)
        timing[mode_name]["n_forwards"] += 1
        return out

    try:
        with torch.no_grad():
            for i in tqdm(indices, desc="all modes", leave=False):
                s = ds[int(i)]
                rgb = s["rgb_norm"].unsqueeze(0).to(device)
                rp  = s["radar_points"].unsqueeze(0).to(device)
                rm  = s["radar_mask"].unsqueeze(0).to(device)
                dgd = s["depth_gt_dense"].unsqueeze(0).to(device)
                gt_np = s["depth_gt_lidar"][0].cpu().numpy()
                mask_np = s["valid_mask_lidar"][0].cpu().numpy().astype(bool)
                dgd_holder["dgd"] = dgd

                # ---- Pass 1: Normal (no gate override) ----
                dgd_holder["mode"] = None
                out = _timed_forward("Normal")
                pred = (out["depth"] if isinstance(out, dict) else out)[0, 0].cpu().numpy()
                for k, (sl1, ssq, n) in per_range(pred, gt_np, mask_np, ranges).items():
                    a = accs["Normal"][k]
                    a[0] += sl1; a[1] += ssq; a[2] += n

                # Router stats side-effect of Normal pass (real router logits).
                if isinstance(out, dict) and out.get("router_logits"):
                    for (name, _blk), rl in zip(moe_blocks, out["router_logits"]):
                        probs = rl.softmax(dim=1)
                        token_pred = probs.argmax(dim=1).squeeze(0)
                        frac, token_gt_hard = compute_token_gt(
                            dgd, token_pred.shape, bins, return_type="both")
                        token_gt = token_gt_hard.squeeze(0)
                        for gt_b in range(K):
                            for pr in range(K):
                                confusion[name][gt_b, pr] += (
                                    (token_gt == gt_b) & (token_pred == pr)).sum().item()
                        l1_per_token = (probs - frac).abs().sum(dim=1)
                        soft_l1[name][0] += float(l1_per_token.sum().item())
                        soft_l1[name][1] += int(l1_per_token.numel())

                # ---- Pass 2: OracleHard ----
                dgd_holder["mode"] = "hard"
                out = _timed_forward("OracleHard")
                pred = (out["depth"] if isinstance(out, dict) else out)[0, 0].cpu().numpy()
                for k, (sl1, ssq, n) in per_range(pred, gt_np, mask_np, ranges).items():
                    a = accs["OracleHard"][k]
                    a[0] += sl1; a[1] += ssq; a[2] += n

                # ---- Pass 3: OracleSoft (skipped when top_k == 1 everywhere) ----
                if not skip_soft:
                    dgd_holder["mode"] = "soft"
                    out = _timed_forward("OracleSoft")
                    pred = (out["depth"] if isinstance(out, dict) else out)[0, 0].cpu().numpy()
                    for k, (sl1, ssq, n) in per_range(pred, gt_np, mask_np, ranges).items():
                        a = accs["OracleSoft"][k]
                        a[0] += sl1; a[1] += ssq; a[2] += n
    finally:
        for h in hooks: h.remove()
        dgd_holder.clear()
        # Restore any temporarily-disabled shared experts / gating modes.
        if spec_only:
            for name, blk in moe_blocks:
                blk.shared = saved_shared[name]
                blk.shared_gate_mode = saved_gate_mode[name]

    # Per-mode MAE + RMSE.
    results = {}
    for m in modes_to_run:
        mae_d, rmse_d = {}, {}
        for k, (sl1, ssq, n) in accs[m].items():
            mae_d[k]  = (sl1 / n)              if n > 0 else float("nan")
            rmse_d[k] = float(np.sqrt(ssq / n)) if n > 0 else float("nan")
        results[m] = {"mae": mae_d, "rmse": rmse_d}
    # Copy OracleSoft ≡ OracleHard when skipped so downstream code is uniform.
    if skip_soft:
        results["OracleSoft"] = {"mae":  dict(results["OracleHard"]["mae"]),
                                 "rmse": dict(results["OracleHard"]["rmse"])}
        timing["OracleSoft"] = dict(timing["OracleHard"])

    # Finalize timing (per-sample = total / n_forwards).
    for m, t in timing.items():
        t["per_sample_s"] = (t["total_s"] / t["n_forwards"]) if t["n_forwards"] else float("nan")

    # Router stats aggregation (same as before).
    router_stats = None
    if moe_blocks:
        blocks = {}
        total_correct = total_tokens = 0
        total_soft_l1_sum = 0.0
        total_soft_l1_n = 0
        for name, C in confusion.items():
            tot = int(C.sum().item())
            corr = int(C.diagonal().sum().item())
            recall = []
            for b in range(K):
                nb = int(C[b, :].sum().item())
                recall.append(100.0 * int(C[b, b].item()) / nb if nb > 0 else float("nan"))
            s_sum, s_n = soft_l1[name]
            blocks[name] = {
                "confusion": C.tolist(),
                "accuracy": 100.0 * corr / max(tot, 1),
                "recall": recall,
                "soft_l1": (s_sum / s_n) if s_n > 0 else None,
                "tokens": tot,
            }
            total_correct += corr; total_tokens += tot
            total_soft_l1_sum += s_sum; total_soft_l1_n += s_n
        router_stats = {
            "blocks": blocks,
            "overall": {
                "accuracy": 100.0 * total_correct / max(total_tokens, 1),
                "tokens": total_tokens,
                "soft_l1": (total_soft_l1_sum / total_soft_l1_n
                            if total_soft_l1_n > 0 else None),
            },
        }
    return results, router_stats, skip_soft, timing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--split", default="test")
    ap.add_argument("--out-dir", default="output/analysis",
                    help="Where to save eval_v3.json / eval_v3.txt")
    ap.add_argument("--tag", default="v3",
                    help="Filename tag: eval_<tag>.{json,txt}")
    ap.add_argument("--spec-only", action="store_true",
                    help="Disable shared expert and confidence gating for "
                         "eval. Measures pure specialist mix (v4c-style "
                         "inference) even on models trained with shared "
                         "(v5_shared / v6a / v6b / v6c).")
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
        # Unified per-sample loop: Normal + OracleHard (+ OracleSoft unless
        # skipped for top_k=1 models). Saves a full dataset iteration and
        # can save an extra oracle pass when the soft/hard gates collapse.
        all_res, router_stats, skipped_soft, timing = evaluate_all_modes(
            m, ds, idxs, ranges, bins=BINS, spec_only=args.spec_only)
        if router_stats:
            ov = router_stats["overall"]
            print(f"  Router overall: acc={ov['accuracy']:.2f}%  "
                  f"soft_l1={ov['soft_l1']:.4f}  (tokens={ov['tokens']:,})")
            for bname, bstats in router_stats["blocks"].items():
                rec_str = "  ".join(f"{labels[b]} {bstats['recall'][b]:.1f}%"
                                    for b in range(3))
                print(f"    {bname:<10} acc={bstats['accuracy']:5.2f}%  "
                      f"soft_l1={bstats['soft_l1']:.4f}  "
                      f"tokens={bstats['tokens']:>10,}  ({rec_str})")
        # Per-mode forward-only inference time (GPU-sync).
        for mode_name, tinfo in timing.items():
            print(f"  Inference time [{mode_name:>10}]: "
                  f"{tinfo['per_sample_s']*1000:.2f} ms/sample  "
                  f"(total {tinfo['total_s']:.1f}s, "
                  f"{tinfo['n_forwards']:,} forwards)")
        if skipped_soft:
            print("  (OracleSoft skipped — top_k=1 → identical to OracleHard)")
        print(f"  ({time.time()-t0:.0f}s total)")

        results[name] = {
            "Normal": all_res["Normal"],
            "OracleHard": all_res["OracleHard"],
            "OracleSoft": all_res["OracleSoft"],
            "router_stats": router_stats,
            "oracle_soft_skipped": skipped_soft,
            "timing": timing,
        }
        del m; torch.cuda.empty_cache()

    # Build a text summary (also printed to stdout).
    buf = io.StringIO()
    def _p(s=""):
        print(s); buf.write(s + "\n")

    MODES = ["Normal", "OracleHard", "OracleSoft"]

    _p("\n" + "=" * 100)
    _p(f"MAE table ({args.split} — {len(idxs):,} samples, no-clip)")
    _p("=" * 100)
    _p(f"{'Setting':<28}" + "  ".join(f"{r:>12}" for r in range_names))
    _p(f"{'Baseline (cached)':<28}" +
       "  ".join(f"{BASELINE_MAE[r]:>12.4f}" if r in BASELINE_MAE
                 else f"{'—':>12}" for r in range_names))
    _p("-" * 100)
    for name, res in results.items():
        for mode in MODES:
            row = f"{name[:22]:<22} {mode:<11}"
            for r in range_names:
                row += f"  {res[mode]['mae'][r]:>12.4f}"
            _p(row)

    _p("\n" + "=" * 100)
    _p(f"RMSE table ({args.split} — {len(idxs):,} samples, no-clip)")
    _p("=" * 100)
    _p(f"{'Setting':<28}" + "  ".join(f"{r:>12}" for r in range_names))
    _p("-" * 100)
    for name, res in results.items():
        for mode in MODES:
            row = f"{name[:22]:<22} {mode:<11}"
            for r in range_names:
                row += f"  {res[mode]['rmse'][r]:>12.4f}"
            _p(row)

    _p("\n" + "=" * 100)
    _p("Δ vs Baseline MAE (negative = better)")
    _p("=" * 100)
    base = BASELINE_MAE
    for name, res in results.items():
        for mode in MODES:
            row = f"{name[:22]:<22} {mode:<11}"
            for r in range_names:
                if r in base:
                    d = res[mode]["mae"][r] - base[r]
                    pct = 100*d/base[r]
                    row += f"  {d:>+7.4f} ({pct:+6.1f}%)"
                else:
                    row += f"  {'—':>17}"
            _p(row)

    _p("\n" + "=" * 100)
    _p("Inference time — GPU-sync forward-only, ms per sample")
    _p("=" * 100)
    _p(f"{'Setting':<28}" + "  ".join(f"{m:>14}" for m in MODES))
    _p("-" * 100)
    for name, res in results.items():
        row = f"{name[:22]:<28}"
        for mode in MODES:
            t = res["timing"].get(mode, {})
            ms = t.get("per_sample_s", float("nan")) * 1000
            row += f"  {ms:>14.2f}"
        _p(row)

    _p("\n" + "=" * 110)
    _p("Router accuracy summary — per MoE block + weighted overall")
    _p("=" * 110)
    for name, res in results.items():
        rs = res.get("router_stats")
        if not rs:
            _p(f"  {name}: (no router stats — model has no MoE?)")
            continue
        _p(f"  {name}:")
        for bname, bstats in rs["blocks"].items():
            rec_str = "  ".join(f"{labels[b]} {bstats['recall'][b]:5.1f}%"
                                for b in range(3))
            _p(f"    {bname:<10} acc={bstats['accuracy']:6.2f}%  "
               f"soft_l1={bstats['soft_l1']:.4f}  "
               f"tokens={bstats['tokens']:>10,}  ({rec_str})")
        _p(f"    {'OVERALL':<10} acc={rs['overall']['accuracy']:6.2f}%  "
           f"soft_l1={rs['overall']['soft_l1']:.4f}  "
           f"tokens={rs['overall']['tokens']:>10,}  (token-weighted)")

    # Persist: JSON (machine) + TXT (human). Matches
    # output/analysis/eval_full_test.json layout used by earlier experiments.
    os.makedirs(args.out_dir, exist_ok=True)
    payload = {
        "split": args.split,
        "n_samples": len(idxs),
        "results": {
            "Baseline @ e46": {"Normal": {"mae": BASELINE_MAE, "rmse": {}}},
            **{name: {
                "Normal": res["Normal"],
                "OracleHard": res["OracleHard"],
                "OracleSoft": res["OracleSoft"],
                "oracle_soft_skipped": res["oracle_soft_skipped"],
                "router_stats": res["router_stats"],
                "timing": res["timing"],
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
