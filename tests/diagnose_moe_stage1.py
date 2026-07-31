"""Diagnose Stage-1 expert specialisation & router behavior.

Checks per MoE block (node_blocks[2], edge_blocks[2]):
  1. Weight divergence between experts (are they DIFFERENT from each other?).
  2. Weight divergence between each expert and the SHARED expert.
  3. Router: probability distribution on val samples (uniform vs peaked?).
  4. Router accuracy vs GT bin (per-sample argmax) — should be > chance (33%).
  5. Actual expert usage: which expert would be picked at test-time on val?
"""
import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, ROOT)

from src.dataset.nuscenes import NuScenesRadarDepthDataset  # noqa: E402
from src.model.radartaco import RadarTaco                    # noqa: E402


def load_model(run_dir, device):
    ck = os.path.join(run_dir, "best.pt")
    cfg = OmegaConf.load(os.path.join(run_dir, "config.yaml"))
    m = RadarTaco(
        radar_encoder_name=cfg.model.radar_encoder,
        max_depth=float(cfg.dataset.max_depth),
        max_radar_points=int(cfg.dataset.max_radar_points),
        k_neighbors=int(cfg.model.k_neighbors),
        a_l=tuple(cfg.model.a_l), radar_channels=tuple(cfg.model.radar_channels),
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


def flat_params(module):
    """Concatenate all parameters of a module into one flat tensor."""
    return torch.cat([p.detach().flatten().cpu() for p in module.parameters()])


def rel_diff(a, b):
    """Relative L2 difference: ‖a-b‖ / ‖(‖a‖+‖b‖)/2‖."""
    da = (a - b).norm().item()
    scale = 0.5 * (a.norm().item() + b.norm().item()) + 1e-12
    return da / scale


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="output/radartaco_moe_stage1_scratch/radartaco_moe_stage1_scratch")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--split", default="val")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    m, cfg = load_model(args.run, device)
    print(f"[loaded] {args.run}/best.pt")
    print(f"[cfg] moe_stage={cfg.model.get('moe_stage')} moe_at_l={cfg.model.get('moe_at_l')}")

    # ------------------------------------------------------------ (1)(2)
    print("\n=== Expert weight divergence ===")
    for block_name in ("node_blocks.2", "edge_blocks.2"):
        parent = m.radar_fusion
        # Access node_blocks[2] via getattr chain
        for part in block_name.split("."):
            parent = parent[int(part)] if part.isdigit() else getattr(parent, part)
        blk = parent
        experts = blk.experts       # ModuleList of 3
        shared  = blk.shared
        vecs = [flat_params(e) for e in experts]
        vs = flat_params(shared)
        labels = ["near(e0)", "mid(e1)", "far(e2)"]
        print(f"\n[{block_name}]  #experts={len(experts)}  shared={shared is not None}")
        print(f"  pairwise rel-diff (0 = identical, ~1 = very different):")
        print(f"    {'':11}", " ".join(f"{l:>10}" for l in labels + ["shared"]))
        for i, li in enumerate(labels):
            row = f"    {li:11}"
            for j, _ in enumerate(labels):
                row += f" {rel_diff(vecs[i], vecs[j]):>10.3f}"
            row += f" {rel_diff(vecs[i], vs):>10.3f}"
            print(row)

    # ------------------------------------------------------- (3)(4)(5)
    print("\n=== Router behavior on val set ===")
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
    idxs = np.linspace(0, len(ds) - 1, args.n).astype(int)
    all_gt = []
    all_pred_node = []    # (N, 3) softmax probs
    all_pred_edge = []

    with torch.no_grad():
        for i in idxs:
            s = ds[int(i)]
            rgb = s["rgb_norm"].unsqueeze(0).to(device)
            rp  = s["radar_points"].unsqueeze(0).to(device)
            rm  = s["radar_mask"].unsqueeze(0).to(device)
            dgt = s["depth_gt_dense"].unsqueeze(0).to(device)
            # Force MoE stage=2 behavior (self-routing) — router uses image feat
            m.moe_stage = 2
            out = m(rgb, rp, rm)
            probs = [F.softmax(l, dim=-1)[0].cpu().numpy() for l in out["router_logits"]]
            all_pred_node.append(probs[0])
            all_pred_edge.append(probs[1])
            # GT bin
            gt_bin = m._compute_router_gt(dgt, m.moe_bins)[0].item()
            all_gt.append(gt_bin)

    all_gt   = np.array(all_gt)
    pred_n   = np.stack(all_pred_node, axis=0)   # (N, 3)
    pred_e   = np.stack(all_pred_edge, axis=0)

    labels = ["near", "mid", "far"]
    for name, probs in (("node_blocks[2]", pred_n), ("edge_blocks[2]", pred_e)):
        argm = probs.argmax(axis=-1)
        acc  = float((argm == all_gt).mean())
        ent  = -(probs * np.log(probs + 1e-12)).sum(-1).mean()
        print(f"\n[{name}] router")
        print(f"  mean prob per expert:  " + "  ".join(
            f"{l}={probs.mean(0)[k]:.3f}" for k, l in enumerate(labels)))
        print(f"  argmax dist:            " + "  ".join(
            f"{l}={100*(argm==k).mean():.1f}%" for k, l in enumerate(labels)))
        print(f"  entropy/sample (max = {np.log(3):.3f}): {ent:.3f}   "
              f"({'peaked' if ent < 0.7 else 'uniform-ish' if ent > 1.0 else 'moderate'})")
        print(f"  ★ accuracy vs GT bin:   {100*acc:.1f}%   (chance = 33.3%)")

    # GT bin distribution
    print(f"\n[GT bin distribution on {args.split} subset ({args.n})]")
    for k, l in enumerate(labels):
        print(f"  {l}: {100*(all_gt==k).mean():.1f}%")


if __name__ == "__main__":
    main()
