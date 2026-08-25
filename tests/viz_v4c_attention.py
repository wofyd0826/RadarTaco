"""Visualize v4c's routing + radar-cross-attention on sample images.

For a handful of test samples, save PNG grids showing:
  1. RGB + radar point overlay
  2. GT depth (colorized)
  3. Router bin prediction (color per token: red=near, green=mid, blue=far)
  4. GT bin per token (majority-vote of dense depth)
  5. Router confidence α (heatmap)
  6. Correct/misroute per token overlay (red = wrong, green = right)
  7. Per-bin router probability (3 heatmaps: near, mid, far)
  8. Radar cross-attention map (average over heads) from the L=2 MoE block
       — shows how strongly each image token attends to radar overall.

Usage:
  python tests/viz_v4c_attention.py --n 6 --split test \
      --run-dir output/radartaco_moe_stage2_v4c
"""
import argparse, os, sys
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from PIL import Image

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, ROOT)

from src.dataset.nuscenes import NuScenesRadarDepthDataset
from src.model.radar_fusion import MoEFusionBlock, RadarCenteredAttention
from src.evaluation.viz import (build_grid, colorize_depth,
                                overlay_radar_points, rgb_to_uint8)
from scripts.train import _build_model

device = "cuda" if torch.cuda.is_available() else "cpu"


def load_model(run_dir, ckpt="best.pt"):
    cfg = OmegaConf.load(os.path.join(run_dir, "config.yaml"))
    m = _build_model(cfg, float(cfg.dataset.max_depth), int(cfg.dataset.max_radar_points))
    sd = torch.load(os.path.join(run_dir, ckpt), map_location=device, weights_only=False)
    m.load_state_dict(sd["model"])
    return m.eval().to(device), cfg


def _collect_moe_blocks(model):
    out = []
    for l in range(len(model.radar_fusion.node_blocks)):
        nb = model.radar_fusion.node_blocks[l]
        eb = model.radar_fusion.edge_blocks[l]
        if isinstance(nb, MoEFusionBlock): out.append((f"L{l}_node", nb))
        if isinstance(eb, MoEFusionBlock): out.append((f"L{l}_edge", eb))
    return out


def _collect_attention_modules(model):
    """Find RadarCenteredAttention modules inside every MoE block's experts."""
    mods = []
    for name, blk in _collect_moe_blocks(model):
        for e_i, expert in enumerate(blk.experts):
            for sub_name, sm in expert.named_modules():
                if isinstance(sm, RadarCenteredAttention):
                    mods.append((f"{name}_expert{e_i}_{sub_name or 'attn'}", sm))
    return mods


def compute_token_gt(depth_gt_dense, tok_hw, bins):
    """Return (B, H, W) int64 majority-vote bin index."""
    K = len(bins) - 1
    edges = torch.tensor(list(bins), device=depth_gt_dense.device, dtype=torch.float32)
    ds = depth_gt_dense.squeeze(1)
    pix_bin = torch.zeros_like(ds, dtype=torch.long)
    for i in range(K):
        pix_bin = torch.where((ds >= float(edges[i])) & (ds < float(edges[i + 1])),
                              pix_bin.new_tensor(i), pix_bin)
    pix_bin = torch.where(ds >= float(edges[-1]),
                          pix_bin.new_tensor(K - 1), pix_bin)
    onehot = F.one_hot(pix_bin, num_classes=K).permute(0, 3, 1, 2).float()
    frac = F.adaptive_avg_pool2d(onehot, tok_hw)
    return frac.argmax(dim=1)


def bin_to_rgb(bin_map: np.ndarray, K: int = 3) -> np.ndarray:
    """(H, W) int in [0,K) → (H, W, 3) uint8 RGB. Fixed palette."""
    palette = np.array([
        [220,  50,  50],   # near = red
        [ 60, 200,  60],   # mid  = green
        [ 60,  90, 220],   # far  = blue
    ], dtype=np.uint8)
    palette = palette[:K]
    return palette[bin_map]


def confidence_to_heatmap(alpha: np.ndarray) -> np.ndarray:
    """(H, W) in [0, 1] → (H, W, 3) uint8 grayscale for now (viridis if we had matplotlib)."""
    # Use turbo-like: R,G,B channels of a simple mapping.
    a = np.clip(alpha, 0, 1)
    r = (255 * a).astype(np.uint8)
    g = (255 * (1 - abs(2 * a - 1))).astype(np.uint8)
    b = (255 * (1 - a)).astype(np.uint8)
    return np.stack([r, g, b], axis=-1)


def correctness_overlay(pred_bin, gt_bin) -> np.ndarray:
    """(H, W) → (H, W, 3): green=correct, red=wrong."""
    correct = (pred_bin == gt_bin)
    out = np.zeros((*pred_bin.shape, 3), dtype=np.uint8)
    out[correct]  = [60, 200, 60]
    out[~correct] = [220, 50, 50]
    return out


def upsample_pixel(arr: np.ndarray, target_hw) -> np.ndarray:
    """Nearest-upsample (H, W, ...) to (H_t, W_t, ...)."""
    H_t, W_t = target_hw
    H, W = arr.shape[:2]
    ry = np.linspace(0, H - 1, H_t).astype(int)
    rx = np.linspace(0, W - 1, W_t).astype(int)
    return arr[ry][:, rx]


def probs_to_heatmap(probs: np.ndarray) -> np.ndarray:
    """(H, W) in [0,1] → (H, W, 3) uint8 grayscale mapped to red gradient."""
    p = np.clip(probs, 0, 1)
    r = (255 * p).astype(np.uint8)
    g = (255 * (1 - p) * 0.4).astype(np.uint8)
    b = (255 * (1 - p) * 0.4).astype(np.uint8)
    return np.stack([r, g, b], axis=-1)


def attn_to_heatmap(attn: np.ndarray) -> np.ndarray:
    """(H, W) attention weight sum → (H, W, 3) heatmap (turbo-ish)."""
    a = attn - attn.min()
    a = a / max(a.max(), 1e-8)
    return confidence_to_heatmap(a)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default="output/radartaco_moe_stage2_v4c")
    ap.add_argument("--ckpt", default="best.pt")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--split", default="test")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    tag = os.path.basename(args.run_dir.rstrip("/"))
    out_dir = args.out_dir or f"output/analysis/viz_attention_{tag}"
    os.makedirs(out_dir, exist_ok=True)

    print(f"Loading model from {args.run_dir}/{args.ckpt}")
    model, cfg = load_model(args.run_dir, args.ckpt)
    router_bins = list(cfg.model.moe_bins)
    K = len(router_bins) - 1
    print(f"  moe_bins = {router_bins}   K = {K}")

    split_file = os.path.join(cfg.dataset.data_root, "splits", f"{args.split}.txt")
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

    moe_blocks = _collect_moe_blocks(model)
    attn_mods  = _collect_attention_modules(model)
    print(f"MoE blocks: {[nm for nm, _ in moe_blocks]}")
    print(f"Attention modules: {len(attn_mods)}")

    # Enable attention recording on all modules.
    for _, sm in attn_mods:
        sm.record_attention = True

    n = min(args.n, len(ds))
    idxs = np.linspace(0, len(ds) - 1, n).astype(int).tolist()
    print(f"Visualizing {n} samples: indices {idxs}")

    with torch.no_grad():
        for i in idxs:
            s = ds[int(i)]
            rgb = s["rgb_norm"].unsqueeze(0).to(device)
            rp  = s["radar_points"].unsqueeze(0).to(device)
            rm  = s["radar_mask"].unsqueeze(0).to(device)
            dgd = s["depth_gt_dense"].unsqueeze(0).to(device)

            out = model(rgb, rp, rm)
            depth_pred = out["depth"][0, 0].cpu().numpy() if isinstance(out, dict) \
                         else out[0, 0].cpu().numpy()
            router_logits = out["router_logits"]     # list of (B, K, H, W)

            rgb_np = rgb_to_uint8(rgb[0].cpu().numpy())      # (H, W, 3)
            H_img, W_img = rgb_np.shape[:2]
            gt_np  = s["depth_gt_dense"][0].cpu().numpy()    # (H, W)

            # Radar overlay
            radar_pts_np = rp[0].cpu().numpy()
            radar_msk_np = rm[0].cpu().numpy()
            rgb_with_radar = overlay_radar_points(
                rgb_np.copy(),
                radar_pts_np,
                radar_msk_np.astype(bool),
            )

            gt_depth_viz = colorize_depth(gt_np, vmin=0.5, vmax=80.0)
            pred_depth_viz = colorize_depth(depth_pred, vmin=0.5, vmax=80.0)

            # Per-block visualizations.
            panels_per_block = {}
            for (blk_name, blk), rl in zip(moe_blocks, router_logits):
                _, K_, H_tok, W_tok = rl.shape
                probs = F.softmax(rl, dim=1)[0].cpu().numpy()          # (K, H, W)
                # Post top_k=2 gate (matches inference).
                if blk.top_k is not None and blk.top_k < K:
                    p_t = torch.from_numpy(probs)[None]
                    _, top_idx = p_t.topk(blk.top_k, dim=1)
                    keep = torch.zeros_like(p_t).scatter_(1, top_idx, 1.0)
                    gate = (p_t * keep)
                    gate = gate / gate.sum(dim=1, keepdim=True).clamp_min(1e-8)
                    gate = gate[0].numpy()
                else:
                    gate = probs
                max_gate = gate.max(axis=0)                              # (H, W)
                alpha = ((K_ * max_gate - 1.0) / max(K_ - 1, 1)).clip(0, 1)

                pred_bin = probs.argmax(axis=0)                          # (H, W)
                gt_bin = compute_token_gt(dgd, (H_tok, W_tok),
                                          router_bins)[0].cpu().numpy()

                # Upsample all token-level maps to image res for the grid.
                pred_bin_rgb = bin_to_rgb(pred_bin, K=K_)
                gt_bin_rgb   = bin_to_rgb(gt_bin, K=K_)
                alpha_rgb    = confidence_to_heatmap(alpha)
                correct_rgb  = correctness_overlay(pred_bin, gt_bin)
                bin_probs_rgbs = [probs_to_heatmap(probs[k]) for k in range(K_)]

                pred_bin_up = upsample_pixel(pred_bin_rgb, (H_img, W_img))
                gt_bin_up   = upsample_pixel(gt_bin_rgb, (H_img, W_img))
                alpha_up    = upsample_pixel(alpha_rgb, (H_img, W_img))
                correct_up  = upsample_pixel(correct_rgb, (H_img, W_img))
                bin_prob_ups = [upsample_pixel(bp, (H_img, W_img))
                                for bp in bin_probs_rgbs]

                panels_per_block[blk_name] = {
                    "pred_bin": pred_bin_up,
                    "gt_bin":   gt_bin_up,
                    "alpha":    alpha_up,
                    "correct":  correct_up,
                    "bin_probs": bin_prob_ups,   # list of K arrays
                }

            # Radar cross-attention: sum over radar-key axis (K+1, last=null token)
            # → per-image-token "attention mass to radar" scalar.
            attn_panels = []
            for a_name, sm in attn_mods:
                if sm.last_attn is None:
                    continue
                a = sm.last_attn[0]              # (heads, H_tok, W_tok, K+1)
                # Sum over heads and over radar keys (exclude null token at last idx).
                a_np = a[..., :-1].sum(dim=-1).mean(dim=0).numpy()  # (H_tok, W_tok)
                attn_up = upsample_pixel(attn_to_heatmap(a_np), (H_img, W_img))
                attn_panels.append((a_name, attn_up))

            # Compose grid.
            #   Row 1: RGB+radar | GT depth | Pred depth
            #   Row 2 per block: pred_bin | gt_bin | alpha | correct | 3 bin probs
            #   Row 3: attention panels (one per attn module)
            rows = []
            rows.append([rgb_with_radar, gt_depth_viz, pred_depth_viz])
            for blk_name, p in panels_per_block.items():
                rows.append([
                    p["pred_bin"], p["gt_bin"], p["alpha"], p["correct"],
                    *p["bin_probs"],
                ])
            # Attention: chunk 4 per row.
            if attn_panels:
                for start in range(0, len(attn_panels), 4):
                    rows.append([img for _, img in attn_panels[start:start+4]])

            # build_grid expects flat list + ncols. Save each row as separate PNG
            # for clarity, plus a combined vertical stack.
            # Simpler: pad each row to same #cols by adding blanks, then vstack.
            max_cols = max(len(r) for r in rows)
            def pad_row(row, target):
                if len(row) == target:
                    return row
                blank = np.zeros_like(row[0])
                return row + [blank] * (target - len(row))
            rows_padded = [pad_row(r, max_cols) for r in rows]
            row_imgs = []
            for r in rows_padded:
                row_imgs.append(np.concatenate(r, axis=1))
            grid = np.concatenate(row_imgs, axis=0)

            out_path = os.path.join(out_dir, f"sample_{i:04d}.png")
            Image.fromarray(grid).save(out_path)
            print(f"  saved {out_path}   shape={grid.shape}")

    # Legend text (for reader reference).
    legend_txt = f"""Grid layout (per sample):
  Row 1: RGB+radar (red dots)  |  GT depth (turbo)  |  Pred depth
  For each MoE block (in order {[nm for nm, _ in moe_blocks]}):
    pred_bin (red=near, green=mid, blue=far)
    gt_bin (same palette)
    α confidence (red=high, blue=low)
    correct/misroute (green=right, red=wrong)
    bin probability heatmaps (near / mid / far, red gradient)
  Last rows: radar cross-attention per expert (image tokens' total attention mass to radar)

MoE bin edges: {router_bins}
"""
    with open(os.path.join(out_dir, "LEGEND.txt"), "w") as f:
        f.write(legend_txt)
    print(f"\nSaved legend → {out_dir}/LEGEND.txt")


if __name__ == "__main__":
    main()
