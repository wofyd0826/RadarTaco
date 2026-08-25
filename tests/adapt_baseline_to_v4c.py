"""Adapt a full-dim RadarTaco baseline checkpoint → v4c 3-expert MoE (halved).

Baseline (`shape_lidar_grad_shape_edge_res_fix`) has a SINGLE L2 fusion block
at full expert channel `C_full = 512`. v4c splits L2 into 3 halved experts
at `C_ex = C_full * moe_expert_ch_ratio = 256`.

This script rewrites the baseline ckpt so it can be loaded into v4c via
`load_weights_from` with `strict=False`:

  1. All non-L2 params (image encoder, radar encoder, L0/L1 fusion, decoder)
     are copied VERBATIM — keys match v4c 1:1.
  2. Baseline's L2 fusion params (`node_blocks.2.*`, `edge_blocks.2.*`) are
     CLONED into each of the 3 experts (`experts.{0,1,2}.block.*`) with
     dim-axis TRUNCATION to the halved expert channel.
  3. Non-Q dims that hit the KV side (`norm_kv`, `k_proj/v_proj` input,
     `out_proj` input via residual) are NOT halved — copied directly.
  4. `proj_in`, `proj_out`, `router.*`, `bins` are LEFT ABSENT — v4c will
     random-init them (Kaiming/normal per PyTorch default) at model build.

Truncation is deterministic ("first C_ex rows/cols") so all 3 experts start
IDENTICAL. Divergence comes from the stage-1 teacher-force gate: each
expert only sees its own bin's tokens, so gradients differ from step 1.

Usage:
    python tests/adapt_baseline_to_v4c.py \
        --baseline output/shape_lidar_grad_shape_edge_res_fix/best.pt \
        --out      output/baseline_adapted_v4c.pt

Then in an experiment config:
    training:
      load_weights_from: output/baseline_adapted_v4c.pt
"""
import argparse
import os
import sys

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, ROOT)

C_FULL = 512
C_EX = 256
N_EXPERTS = 3
L2_PREFIXES = ["radar_fusion.node_blocks.2.", "radar_fusion.edge_blocks.2."]


# ------------------------------------------------------------------ truncation

def _truncate(baseline_param: torch.Tensor, key: str) -> torch.Tensor:
    """Truncate a baseline L2 param to the expert's halved channel dim.

    Rules by suffix:
      norm1, norm2        (D,)            → (C_ex,)      = t[:C_ex]
      norm_kv             (D_kv,)         → unchanged    (KV not halved)
      attn.q_proj.weight  (D, D)          → (C_ex, C_ex) = t[:C_ex, :C_ex]
      attn.q_proj.bias    (D,)            → (C_ex,)      = t[:C_ex]
      attn.k_proj.weight  (D, D_kv)       → (C_ex, D_kv) = t[:C_ex, :]   (KV in unchanged)
      attn.k_proj.bias    (D,)            → (C_ex,)      = t[:C_ex]
      attn.v_proj.*       — same as k_proj
      attn.out_proj.weight(D, D)          → (C_ex, C_ex) = t[:C_ex, :C_ex]
      attn.out_proj.bias  (D,)            → (C_ex,)
      attn.k_null / v_null(1, 1, D)       → (1, 1, C_ex) = t[..., :C_ex]
      mlp.0.weight        (2D, D, 1, 1)   → (2*C_ex, C_ex, 1, 1) = t[:2*C_ex, :C_ex]
      mlp.0.bias          (2D,)           → (2*C_ex,)
      mlp.2.weight        (D, 2D, 1, 1)   → (C_ex, 2*C_ex, 1, 1) = t[:C_ex, :2*C_ex]
      mlp.2.bias          (D,)            → (C_ex,)
    """
    t = baseline_param

    if key.endswith("norm_kv.weight") or key.endswith("norm_kv.bias"):
        return t.clone()

    if key.endswith("norm1.weight") or key.endswith("norm1.bias") \
            or key.endswith("norm2.weight") or key.endswith("norm2.bias"):
        return t[:C_EX].clone()

    if key.endswith(".attn.q_proj.weight"):
        return t[:C_EX, :C_EX].clone()
    if key.endswith(".attn.q_proj.bias"):
        return t[:C_EX].clone()

    if key.endswith(".attn.k_proj.weight") or key.endswith(".attn.v_proj.weight"):
        # (D_out=D, D_in=D_kv) — halve out only.
        return t[:C_EX, :].clone()
    if key.endswith(".attn.k_proj.bias") or key.endswith(".attn.v_proj.bias"):
        return t[:C_EX].clone()

    if key.endswith(".attn.out_proj.weight"):
        return t[:C_EX, :C_EX].clone()
    if key.endswith(".attn.out_proj.bias"):
        return t[:C_EX].clone()

    if key.endswith(".attn.k_null") or key.endswith(".attn.v_null"):
        return t[..., :C_EX].clone()

    if key.endswith(".mlp.0.weight"):
        return t[:2 * C_EX, :C_EX, :, :].clone()
    if key.endswith(".mlp.0.bias"):
        return t[:2 * C_EX].clone()

    if key.endswith(".mlp.2.weight"):
        return t[:C_EX, :2 * C_EX, :, :].clone()
    if key.endswith(".mlp.2.bias"):
        return t[:C_EX].clone()

    raise ValueError(f"Unhandled L2 baseline param: {key}")


# ------------------------------------------------------------------- adaptation

def adapt_state_dict(baseline_sd: dict) -> dict:
    """Return an adapted state_dict for v4c.

    Non-L2 params: copied verbatim.
    L2 params: expanded into 3 expert clones with truncation.
    proj_in/proj_out/router/bins: OMITTED (random init on load).
    """
    out = {}
    for k, v in baseline_sd.items():
        is_l2 = any(k.startswith(p) for p in L2_PREFIXES)
        if not is_l2:
            out[k] = v.clone()
            continue
        # L2 — expand into experts.
        for pre in L2_PREFIXES:
            if k.startswith(pre):
                suffix = k[len(pre):]        # e.g. "attn.k_proj.weight"
                for i in range(N_EXPERTS):
                    new_key = f"{pre}experts.{i}.block.{suffix}"
                    out[new_key] = _truncate(v, k)
                break
    return out


# --------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline",
                    default="output/shape_lidar_grad_shape_edge_res_fix/best.pt")
    ap.add_argument("--out",
                    default="output/baseline_adapted_v4c.pt")
    args = ap.parse_args()

    print(f"Loading baseline: {args.baseline}")
    ckpt = torch.load(args.baseline, map_location="cpu")
    assert "model" in ckpt, "expected trainer-style ckpt with 'model' key"
    baseline_sd = ckpt["model"]
    print(f"  baseline params: {len(baseline_sd)}")

    # L2 param counts sanity.
    n_l2 = sum(1 for k in baseline_sd if any(k.startswith(p) for p in L2_PREFIXES))
    print(f"  baseline L2 params: {n_l2} (will each expand to {N_EXPERTS} experts)")

    print("Adapting...")
    adapted_sd = adapt_state_dict(baseline_sd)
    print(f"  adapted params: {len(adapted_sd)}")

    # Verify: build v4c model and try loading.
    print("\nVerifying against v4c model...")
    from hydra import compose, initialize_config_dir
    CFG_DIR = os.path.abspath("config")
    with initialize_config_dir(config_dir=CFG_DIR, version_base=None):
        cfg = compose(config_name="config",
                      overrides=["+experiment=radartaco_moe_stage1_v4c"])
    from scripts.train import _build_model
    model = _build_model(cfg,
                         max_depth=100.0,
                         max_radar_points=int(cfg.dataset.max_radar_points))
    mkeys = set(model.state_dict().keys())
    akeys = set(adapted_sd.keys())
    matched = mkeys & akeys
    missing = mkeys - akeys     # will be random-init
    unexpected = akeys - mkeys  # would be dropped (should be 0 if adapter is correct)
    print(f"  v4c params: {len(mkeys)}   adapted params: {len(akeys)}")
    print(f"  matched: {len(matched)}   random-init (v4c-only): {len(missing)}   "
          f"unexpected: {len(unexpected)}")

    if unexpected:
        print("  UNEXPECTED keys (bug — should be 0):")
        for k in sorted(unexpected)[:10]:
            print(f"    {k}")
        raise SystemExit(1)

    # Categorize random-init.
    from collections import Counter
    def _cat(k):
        if k.endswith("proj_in.weight") or k.endswith("proj_in.bias"): return "proj_in"
        if k.endswith("proj_out.weight") or k.endswith("proj_out.bias"): return "proj_out"
        if ".router." in k: return "router"
        if k.endswith(".bins"): return "bins (buffer)"
        return "other"
    print("  random-init categories:", dict(Counter(_cat(k) for k in missing)))

    # Actually do strict=False load to confirm no shape errors.
    result = model.load_state_dict(adapted_sd, strict=False)
    print(f"  strict=False load OK: missing={len(result.missing_keys)}, "
          f"unexpected={len(result.unexpected_keys)}")

    # Save.
    out_ckpt = {"model": adapted_sd}
    torch.save(out_ckpt, args.out)
    print(f"\nSaved adapted ckpt: {args.out}")


if __name__ == "__main__":
    main()
