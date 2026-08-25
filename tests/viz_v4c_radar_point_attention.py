"""Visualize where a specific radar point is attended by image tokens.

For each sample, pick N radar points (spread across depth), then for each
selected point produce a heatmap of attention weight FROM every image token
TO that point (averaged over heads).

Grid per sample (rows × cols = 1 + n_experts × n_points):
  Row 1: for each radar point, RGB with THAT point highlighted (yellow disk)
         and depth label overlay.
  Rows 2..: for each MoE block's expert, attention heatmap from image tokens
            to that specific radar point.

Usage:
  python tests/viz_v4c_radar_point_attention.py --run-dir output/radartaco_moe_stage2_v4c \
      --sample-idx 6007 --n-points 6
"""
import argparse, os, sys
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from PIL import Image, ImageDraw

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, ROOT)

from src.dataset.nuscenes import NuScenesRadarDepthDataset
from src.model.radar_fusion import MoEFusionBlock, RadarCenteredAttention
from src.evaluation.viz import colorize_depth, rgb_to_uint8
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


def _collect_attention_modules(model, level_filter=None):
    """Return [(short_name, module)] for every RadarCenteredAttention in the
    radar_fusion pyramid — non-MoE levels (L=0, L=1) and MoE experts (L=2).

    level_filter: iterable of int (e.g. {0, 1}) or None (all levels).
    """
    mods = []
    for name, m in model.named_modules():
        if not isinstance(m, RadarCenteredAttention):
            continue
        # Derive level index from path like 'radar_fusion.node_blocks.0.attn'
        parts = name.split(".")
        level = None
        for i, p in enumerate(parts):
            if p.endswith("_blocks") and i + 1 < len(parts):
                try:
                    level = int(parts[i + 1])
                except ValueError:
                    pass
                break
        if level is None:
            continue
        if level_filter is not None and level not in level_filter:
            continue
        # Short display name from the full module path.
        short = (name
                  .replace("radar_fusion.", "")
                  .replace("_blocks.", "")
                  .replace(".experts.", "_expert")
                  .replace(".block.attn", "_attn")
                  .replace(".attn", "_attn"))
        # Prefix with L<level>
        short = f"L{level}_{short}"
        mods.append((short, m))
    return mods


def upsample(arr: np.ndarray, target_hw) -> np.ndarray:
    """Nearest upsample (H, W) or (H, W, 3) to target."""
    H_t, W_t = target_hw
    H, W = arr.shape[:2]
    ry = np.linspace(0, H - 1, H_t).astype(int)
    rx = np.linspace(0, W - 1, W_t).astype(int)
    return arr[ry][:, rx]


def turbo(x: np.ndarray) -> np.ndarray:
    """(H, W) in [0, 1] → (H, W, 3) turbo-ish colormap."""
    a = np.clip(x, 0, 1)
    r = np.clip((255 * (2 * a - 0.4)), 0, 255).astype(np.uint8)
    g = np.clip((255 * (1 - abs(2 * a - 1))), 0, 255).astype(np.uint8)
    b = np.clip((255 * (1 - 1.6 * a)), 0, 255).astype(np.uint8)
    return np.stack([r, g, b], axis=-1)


def mark_radar_point(rgb: np.ndarray, x_px: float, y_px: float,
                     radius: int = 12, color=(255, 255, 0),
                     text: str = None) -> np.ndarray:
    img = Image.fromarray(rgb.copy())
    dr = ImageDraw.Draw(img)
    x, y = int(x_px), int(y_px)
    dr.ellipse([x - radius, y - radius, x + radius, y + radius],
               outline=color, width=3)
    dr.ellipse([x - 3, y - 3, x + 3, y + 3], fill=color)
    if text:
        dr.text((x + radius + 2, y - 8), text, fill=color)
    return np.array(img)


def select_radar_points(radar_points_np: np.ndarray, mask_np: np.ndarray,
                        n: int = 6):
    """Return indices of N radar points spread across depth range."""
    valid_idx = np.where(mask_np)[0]
    if len(valid_idx) == 0:
        return []
    depths = radar_points_np[valid_idx, -1]              # (M,) depth channel
    order = np.argsort(depths)
    # Pick n evenly spaced by depth rank.
    picks = np.linspace(0, len(order) - 1, n).astype(int)
    return valid_idx[order[picks]].tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default="output/radartaco_moe_stage2_v4c")
    ap.add_argument("--ckpt", default="best.pt")
    ap.add_argument("--sample-idx", type=int, nargs="+", default=[6007],
                    help="One or more test-split sample indices to analyze.")
    ap.add_argument("--n-points", type=int, default=6,
                    help="How many radar points to visualize per sample.")
    ap.add_argument("--levels", type=int, nargs="+", default=[0, 1, 2],
                    help="Which pyramid levels to record attention at "
                         "(default: all — 0, 1, 2).")
    ap.add_argument("--split", default="test")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--x-pix-ch", type=int, default=3,
                    help="Index in radar_points of image-plane x (default 3).")
    ap.add_argument("--y-pix-ch", type=int, default=4,
                    help="Index in radar_points of image-plane y (default 4).")
    args = ap.parse_args()

    tag = os.path.basename(args.run_dir.rstrip("/"))
    out_dir = args.out_dir or f"output/analysis/viz_radar_point_attn_{tag}"
    os.makedirs(out_dir, exist_ok=True)

    print(f"Loading model from {args.run_dir}/{args.ckpt}")
    model, cfg = load_model(args.run_dir, args.ckpt)

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
    attn_mods = _collect_attention_modules(model, level_filter=set(args.levels))
    print(f"  found {len(attn_mods)} attention modules (levels={args.levels}):")
    for nm, sm in attn_mods:
        sm.record_attention = True
        print(f"    {nm}")

    # Patch RadarCenteredAttention.forward so `last_attn` is stored ALSO in the
    # sparse case (sel_idx != None). MoE experts always run sparse, so the
    # upstream implementation only records for the non-MoE fusion path.
    _orig_forward = RadarCenteredAttention.forward
    def _forward_recording(self, feat, kv, radar_x_orig, radar_mask, image_w,
                            sel_idx=None, sel_valid=None, q_tok=None, hw=None):
        # We only patch when recording is on; else fall back to fast path.
        if not self.record_attention:
            return _orig_forward(self, feat, kv, radar_x_orig, radar_mask,
                                  image_w, sel_idx, sel_valid, q_tok, hw)
        # Reproduce the manual softmax branch but always save last_attn,
        # scattering sparse attention back to a full (H, W) grid when needed.
        if q_tok is None:
            B, C, H, W = feat.shape
            device = feat.device
        else:
            B, _, C = q_tok.shape
            H, W = hw
            device = q_tok.device
        K = kv.shape[1]
        scale = W / float(image_w)
        x_p = radar_x_orig * scale
        a_pix = max(self.a_l * scale, 1.0)
        col = torch.arange(W, device=device).float()
        col_mask = (col[None, :, None] - x_p[:, None, :]).abs() < a_pix
        col_mask = col_mask & radar_mask[:, None, :]
        if q_tok is not None:
            xq = q_tok
            col_of = sel_idx % W
        elif sel_idx is None:
            xq = feat.permute(0, 2, 3, 1).reshape(B, H * W, C)
            col_of = (torch.arange(H * W, device=device) % W)[None].expand(B, -1)
        else:
            x = feat.permute(0, 2, 3, 1).reshape(B, H * W, C)
            xq = torch.gather(x, 1, sel_idx[..., None].expand(-1, -1, C))
            col_of = sel_idx % W
        N = xq.shape[1]
        q = self.q_proj(xq).view(B, N, self.heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(kv).view(B, K, self.heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(kv).view(B, K, self.heads, self.head_dim).transpose(1, 2)
        k_null = self.k_null.view(1, 1, self.heads, self.head_dim).transpose(1, 2)
        v_null = self.v_null.view(1, 1, self.heads, self.head_dim).transpose(1, 2)
        k = torch.cat([k, k_null.expand(B, -1, -1, -1).to(k.dtype)], dim=2)
        v = torch.cat([v, v_null.expand(B, -1, -1, -1).to(v.dtype)], dim=2)
        attn_keep = torch.gather(col_mask, 1, col_of[..., None].expand(-1, -1, K))
        any_valid_pix = attn_keep.any(dim=-1, keepdim=True)
        bias_radar = torch.zeros_like(attn_keep, dtype=q.dtype)
        bias_radar = bias_radar.masked_fill(~attn_keep, float("-inf"))
        bias_null = torch.zeros(B, N, 1, dtype=q.dtype, device=device)
        attn_bias = torch.cat([bias_radar, bias_null], dim=-1).unsqueeze(1)

        scores = (q @ k.transpose(-1, -2)) * self.scale
        scores = scores + attn_bias
        attn = torch.softmax(scores, dim=-1)                              # (B, h, N, K+1)
        out = attn @ v

        # Always record — scatter sparse attention back to (B, heads, H, W, K+1)
        # so viz code can treat both cases uniformly.
        if sel_idx is None:
            full_attn = attn.detach().reshape(B, self.heads, H, W, K + 1).cpu()
        else:
            HW = H * W
            full = torch.zeros(B, self.heads, HW, K + 1, device=attn.device,
                                dtype=attn.dtype)
            idx = sel_idx[:, None, :, None].expand(-1, self.heads, -1, K + 1)
            full.scatter_(2, idx, attn)
            full_attn = full.reshape(B, self.heads, H, W, K + 1).detach().cpu()
        self.last_attn = full_attn
        self.last_hw = (H, W)

        out = out.transpose(1, 2).reshape(B, N, C)
        out = self.out_proj(out)
        out = out * any_valid_pix.to(out.dtype)
        if sel_idx is None and q_tok is None:
            return out.transpose(1, 2).reshape(B, C, H, W)
        return out * sel_valid[..., None].to(out.dtype)
    RadarCenteredAttention.forward = _forward_recording

    for idx in args.sample_idx:
        print(f"\n--- Sample idx {idx} of {len(ds)} ---")
        process_sample(idx, ds, model, attn_mods, args, out_dir)


def process_sample(idx, ds, model, attn_mods, args, out_dir):
    s = ds[int(idx)]
    rgb  = s["rgb_norm"].unsqueeze(0).to(device)
    rp   = s["radar_points"].unsqueeze(0).to(device)
    rm   = s["radar_mask"].unsqueeze(0).to(device)

    # Clear stale last_attn from previous samples so we don't accidentally viz
    # a module that didn't fire on the current sample.
    for _, sm in attn_mods:
        sm.last_attn = None

    with torch.no_grad():
        _ = model(rgb, rp, rm)

    rgb_np = rgb_to_uint8(rgb[0].cpu().numpy())              # (H_img, W_img, 3)
    H_img, W_img = rgb_np.shape[:2]
    rp_np = rp[0].cpu().numpy()
    rm_np = rm[0].cpu().numpy().astype(bool)

    # Inspect radar layout so user knows what channels are used.
    print(f"  radar_points shape (N, F) = {rp_np.shape}")
    print(f"  valid radar points        = {int(rm_np.sum())}")
    # Depth is last channel by convention in nuscenes.py pipeline.
    depth_ch = rp_np.shape[-1] - 1
    print(f"  using x_pix ch={args.x_pix_ch}, y_pix ch={args.y_pix_ch}, depth ch={depth_ch}")

    picks = select_radar_points(rp_np, rm_np, n=args.n_points)
    if not picks:
        print("  no valid radar points; aborting"); return
    print(f"  selected radar indices: {picks}   depths: "
          f"{[f'{rp_np[i, depth_ch]:.1f}m' for i in picks]}")

    # Build the point-highlight strip (row 1).
    top_row = []
    for k in picks:
        x_px = rp_np[k, args.x_pix_ch]
        y_px = rp_np[k, args.y_pix_ch]
        depth = rp_np[k, depth_ch]
        marked = mark_radar_point(rgb_np, x_px, y_px, radius=18,
                                  color=(255, 255, 0),
                                  text=f"{depth:.0f}m")
        top_row.append(marked)

    # For each expert's attention module, build one row across selected points.
    attn_rows = []
    for a_name, sm in attn_mods:
        if sm.last_attn is None:
            print(f"    skip {a_name}: no attention recorded (expert didn't run)")
            continue
        attn = sm.last_attn[0]                                # (heads, H_tok, W_tok, K+1)
        heads, H_tok, W_tok, K1 = attn.shape
        # Radar-point axis includes null-token at last position; skip it.
        K = K1 - 1
        attn_np = attn.numpy()                                # CPU tensor already
        # Sanity check: our selected radar indices must be < K.
        row = []
        for k in picks:
            k_use = int(k)
            if k_use >= K:
                # Radar point wasn't in the truncated set the model saw; blank.
                row.append(np.zeros_like(top_row[0]))
                continue
            # Average over heads.
            a_map = attn_np[:, :, :, k_use].mean(axis=0)      # (H_tok, W_tok)
            # Normalize within [0, 1] for viz (each map own scale).
            a_map = a_map - a_map.min()
            if a_map.max() > 1e-12:
                a_map = a_map / a_map.max()
            heat = turbo(a_map)                                # (H_tok, W_tok, 3)
            heat_up = upsample(heat, (H_img, W_img))
            # Overlay marker again on the heatmap so we can see WHERE the point is.
            heat_marked = mark_radar_point(
                heat_up, rp_np[k_use, args.x_pix_ch], rp_np[k_use, args.y_pix_ch],
                radius=18, color=(255, 255, 255),
                text=f"{rp_np[k_use, depth_ch]:.0f}m",
            )
            row.append(heat_marked)
        attn_rows.append((a_name, row))

    # Compose grid: row 1 (marked RGB), then one row per expert.
    row_imgs = [np.concatenate(top_row, axis=1)]
    label_col_w = 200  # for text label column on the left; skipped for simplicity
    for a_name, row in attn_rows:
        row_img = np.concatenate(row, axis=1)
        row_imgs.append(row_img)

    grid = np.concatenate(row_imgs, axis=0)
    out_path = os.path.join(out_dir, f"sample_{idx:04d}_radar_point_attention.png")
    Image.fromarray(grid).save(out_path)
    print(f"\nSaved {out_path}   shape={grid.shape}")

    # Also save a legend describing rows.
    legend = (
        f"Sample idx: {idx}\n"
        f"Radar-point indices (top row markers, left→right): {picks}\n"
        f"Radar depths (m): {[f'{rp_np[i, depth_ch]:.1f}' for i in picks]}\n\n"
        f"Row 1: RGB with each selected radar point marked (yellow circle + depth label).\n"
        f"Rows 2..: for each MoE expert with recorded attention, heatmap showing\n"
        f"          attention weight FROM every image token TO that specific radar point.\n"
        f"          Bright red = strong attention; dark = weak.\n"
        f"          White marker in each heatmap = the radar point's location on the image plane.\n\n"
        f"Row labels (top to bottom, after row 1):\n"
    )
    for i, (a_name, _) in enumerate(attn_rows):
        legend += f"  Row {i + 2}: {a_name}\n"
    with open(os.path.join(out_dir, f"sample_{idx:04d}_LEGEND.txt"), "w") as f:
        f.write(legend)
    print(f"Saved legend → {out_dir}/sample_{idx:04d}_LEGEND.txt")


if __name__ == "__main__":
    main()
