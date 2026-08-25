"""Build a warm-start checkpoint for the depth-specialised MoE model.

Takes one BASELINE checkpoint (single-fusion `_FusionBlock` per level)
and N=3 SPECIALIST checkpoints (loss-mask trained on GT bins
[0,20)/[20,50)/[50,100)m) and produces an MoE checkpoint whose:

    node_blocks[2].experts[0]  ← S_near.node_blocks[2]
    node_blocks[2].experts[1]  ← S_mid.node_blocks[2]
    node_blocks[2].experts[2]  ← S_far.node_blocks[2]
    edge_blocks[2].experts[0]  ← S_near.edge_blocks[2]
    edge_blocks[2].experts[1]  ← S_mid.edge_blocks[2]
    edge_blocks[2].experts[2]  ← S_far.edge_blocks[2]
    all other params            ← baseline (image encoder, radar encoder,
                                    node/edge_blocks[0,1], depth decoder)
    routers                     ← random init (kept as-is)

Note: shared expert is no longer used (`use_shared=false` in the MoE
config). The baseline's fusion block weights only initialise the non-MoE
levels; the MoE level's shared slot doesn't exist to fill.

The output is a single .pt file with keys {"model": state_dict} suitable
for `training.load_weights_from` in the stage-1 experiment overlay.

Usage:
    python scripts/warm_start_moe.py \
        --baseline output/shape_lidar_grad_shape_edge_res/best.pt \
        --near     output/shape_edge_res_bin_near/best.pt \
        --mid      output/shape_edge_res_bin_mid/best.pt \
        --far      output/shape_edge_res_bin_far/best.pt \
        --out      output/radartaco_moe_warm/init.pt
"""
import argparse
import os
import sys

import torch
from omegaconf import OmegaConf

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, ROOT)

from src.model.radartaco import RadarTaco  # noqa: E402


def _load(ckpt_path):
    """Load a state_dict from a .pt checkpoint (train/eval format)."""
    d = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    return d["model"] if "model" in d else d


def _build_moe_model(cfg_path):
    """Instantiate a RadarTaco with MoE=(2,) matching the training configs."""
    saved = OmegaConf.load(cfg_path)
    return RadarTaco(
        radar_encoder_name=saved.model.radar_encoder,
        max_depth=float(saved.dataset.max_depth),
        max_radar_points=int(saved.dataset.max_radar_points),
        k_neighbors=int(saved.model.k_neighbors),
        a_l=tuple(saved.model.a_l),
        radar_channels=tuple(saved.model.radar_channels),
        attn_heads=int(saved.model.attn_heads),
        mlp_hidden=int(saved.model.get("mlp_hidden", 128)),
        pretrained_image_encoder=False,
        output_mode=str(saved.model.get("output_mode", "metric")),
        min_depth_clip=float(saved.model.get("min_depth_clip", 0.5)),
        multi_scale=bool(saved.model.get("multi_scale", False)),
        multi_scale_levels=tuple(saved.model.get("multi_scale_levels", (2, 4, 8, 16))),
        use_aux_branch=bool(saved.model.get("use_aux_branch", False)),
        # MoE — must match target training config
        moe_at_l=(2,),
        moe_n_experts=3,
        moe_use_shared=False,
        moe_top_k=1,
        moe_stage=1,
        moe_bins=(0.0, 20.0, 50.0, 100.0),
    )


# The MoE blocks we are populating.
_MOE_PATHS = ["node_blocks.2", "edge_blocks.2"]


def _rewrite_source_key(k, level_key, dest_role):
    """Map a source-checkpoint key that starts with `radar_fusion.<level_key>.<sub>`
    to the MoE-model key `radar_fusion.<level_key>.<dest_role>.<sub>`.
    Returns None if the key is unrelated to this fusion block.
    """
    prefix = f"radar_fusion.{level_key}."
    if not k.startswith(prefix):
        return None
    tail = k[len(prefix):]
    return f"radar_fusion.{level_key}.{dest_role}.{tail}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True,
                    help="ckpt of shape_lidar_grad_shape_edge_res "
                         "(single-fusion, provides all non-MoE weights + shared expert)")
    ap.add_argument("--near", required=True)
    ap.add_argument("--mid",  required=True)
    ap.add_argument("--far",  required=True)
    ap.add_argument("--out",  required=True)
    ap.add_argument("--cfg",  default=None,
                    help="config.yaml adjacent to the baseline ckpt "
                         "(defaults to <baseline_dir>/config.yaml)")
    args = ap.parse_args()

    cfg_path = args.cfg or os.path.join(os.path.dirname(args.baseline), "config.yaml")
    print(f"[cfg] {cfg_path}")

    moe = _build_moe_model(cfg_path)
    dst = moe.state_dict()

    baseline_sd = _load(args.baseline)
    near_sd = _load(args.near)
    mid_sd  = _load(args.mid)
    far_sd  = _load(args.far)

    n_copied = {"baseline_bulk": 0,
                "expert_near": 0, "expert_mid": 0, "expert_far": 0,
                "skipped_router": 0}

    # 1) Copy every baseline key that also exists verbatim in dst.
    #    This handles image_encoder, radar_encoder, node_blocks[0,1],
    #    edge_blocks[0,1], decoder, aux_branch (if present).
    for k, v in baseline_sd.items():
        if k in dst and dst[k].shape == v.shape:
            dst[k] = v.clone()
            n_copied["baseline_bulk"] += 1

    # 2) For each MoE level, remap each specialist onto experts[i]. No
    #    shared expert to fill (use_shared=false).
    for lp in _MOE_PATHS:
        for role_name, sd, expert_idx in (
            ("expert_near", near_sd, 0),
            ("expert_mid",  mid_sd,  1),
            ("expert_far",  far_sd,  2),
        ):
            dest_role = f"experts.{expert_idx}"
            for k, v in sd.items():
                nk = _rewrite_source_key(k, lp, dest_role)
                if nk is not None and nk in dst and dst[nk].shape == v.shape:
                    dst[nk] = v.clone()
                    n_copied[role_name] += 1

    # 3) Sanity: router params are left at random init (not in any source).
    router_keys = [k for k in dst if ".router." in k]
    n_copied["skipped_router"] = len(router_keys)

    load_result = moe.load_state_dict(dst, strict=True)
    print("[load] load_state_dict result:", load_result)
    print("[copy] statistics:")
    for k, v in n_copied.items():
        print(f"    {k:>18}  {v:>6}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save({"model": dst, "warm_start_from": {
        "baseline": args.baseline, "near": args.near, "mid": args.mid, "far": args.far,
    }}, args.out)
    print(f"[out] saved: {args.out}")


if __name__ == "__main__":
    main()
