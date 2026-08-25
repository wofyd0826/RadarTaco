"""Dump per-token (probs, pixel_fraction, MAE, position) for v4c stage2.

For every token in the L2 node grid (29 × 50), across every test sample in
input file order, save:
  sample_idx      — 0..(n_samples-1), index into sample_ids array
  token_row       — 0..28
  token_col       — 0..49
  probs           — (3,) router softmax under Normal routing
  frac            — (3,) target pixel_fraction (aggregated from dense GT bin)
  mae             — per-token mean |pred - gt_lidar| over valid LiDAR pixels
                    (NaN when the token has 0 valid LiDAR pixels)
  depth           — per-token mean of dense-GT depth (not LiDAR) at the token
  valid_pix_count — per-token count of valid LiDAR pixels
  argmax_p        — router argmax (0=near, 1=mid, 2=far)
  argmax_f        — pixel_fraction argmax

Also saves alongside:
  sample_ids      — (n_samples,) object array of nuScenes sample_id strings
  grid_hw         — (2,) grid resolution [H_tok, W_tok] for traceback
  image_hw        — (2,) image resolution [H_img, W_img]
  bins            — (K+1,) bin edges used for pixel_fraction
  meta            — dict with run path, ckpt, split

Traceback: for token (row, col) in sample s:
  H_img, W_img = image_hw
  H_tok, W_tok = grid_hw
  y0, y1 = row * H_img // H_tok, (row+1) * H_img // H_tok
  x0, x1 = col * W_img // W_tok, (col+1) * W_img // W_tok
  dense_gt[y0:y1, x0:x1]  # source pixels for this token
"""
import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from tqdm import tqdm

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, ROOT)

from src.dataset.nuscenes import NuScenesRadarDepthDataset   # noqa: E402
from src.model.radar_fusion import MoEFusionBlock            # noqa: E402
from scripts.train import _build_model                       # noqa: E402


device = "cuda" if torch.cuda.is_available() else "cpu"
BINS = [0.0, 20.0, 50.0, 100.0]
K = 3


def load_model(run_dir, ckpt="best.pt"):
    cfg = OmegaConf.load(os.path.join(run_dir, "config.yaml"))
    m = _build_model(cfg, float(cfg.dataset.max_depth),
                     int(cfg.dataset.max_radar_points))
    sd = torch.load(os.path.join(run_dir, ckpt),
                    map_location=device, weights_only=False)
    m.load_state_dict(sd["model"])
    print(f"  loaded {run_dir}/{ckpt}  epoch={sd.get('epoch','?')}")
    return m.eval().to(device)


def pixel_fraction(dgd, tok_hw):
    edges = torch.tensor(BINS, device=dgd.device, dtype=torch.float32)
    ds = dgd.squeeze(1)
    pix_bin = torch.zeros_like(ds, dtype=torch.long)
    for i in range(K):
        pix_bin = torch.where((ds >= float(edges[i])) & (ds < float(edges[i+1])),
                              pix_bin.new_tensor(i), pix_bin)
    pix_bin = torch.where(ds >= float(edges[-1]),
                          pix_bin.new_tensor(K-1), pix_bin)
    onehot = F.one_hot(pix_bin, num_classes=K).permute(0, 3, 1, 2).float()
    return F.adaptive_avg_pool2d(onehot, tok_hw)


def per_token_err(pred, gt, valid_mask, tok_hw):
    err = (pred - gt).abs() * valid_mask
    err_pooled = F.adaptive_avg_pool2d(err, tok_hw).squeeze(1)
    valid_pool = F.adaptive_avg_pool2d(valid_mask, tok_hw).squeeze(1)
    out = err_pooled / valid_pool.clamp_min(1e-8)
    return torch.where(valid_pool > 0, out, torch.full_like(out, float("nan")))


def per_token_valid_count(mask_lidar, tok_hw):
    cnt = F.adaptive_avg_pool2d(mask_lidar, tok_hw).squeeze(1) * \
           float(mask_lidar.shape[-2] * mask_lidar.shape[-1]) / \
           float(tok_hw[0] * tok_hw[1])
    return cnt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="output/radartaco_moe_stage2_v4c")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--ckpt", default="best.pt")
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--split", default="test")
    ap.add_argument("--out-dir", default="output/analysis/router_cost")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

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

    m = load_model(args.run, args.ckpt)

    # Determine node L2 block for router logits reference
    moe_blocks = [(f"L{l}_node", m.radar_fusion.node_blocks[l])
                  for l in range(len(m.radar_fusion.node_blocks))
                  if isinstance(m.radar_fusion.node_blocks[l], MoEFusionBlock)]
    assert moe_blocks, "No node MoE blocks found."

    # Storage — we know per-sample counts once we've seen the first sample.
    sample_ids = []
    all_probs = []
    all_frac = []
    all_mae = []
    all_depth = []
    all_vc = []
    all_sample_idx = []
    all_row = []
    all_col = []
    tok_hw_final = None
    image_hw_final = None

    t0 = time.time()
    with torch.no_grad():
        for local_idx, i in enumerate(tqdm(idxs, desc="dump")):
            s = ds[int(i)]
            sample_ids.append(str(s["sample_id"]))
            rgb = s["rgb_norm"].unsqueeze(0).to(device)
            rp  = s["radar_points"].unsqueeze(0).to(device)
            rm  = s["radar_mask"].unsqueeze(0).to(device)
            dgd = s["depth_gt_dense"].unsqueeze(0).to(device)
            gt_lidar = s["depth_gt_lidar"].unsqueeze(0).to(device)
            mask_lidar = s["valid_mask_lidar"].unsqueeze(0).to(device).float()

            out = m(rgb, rp, rm)
            depth_pred = out["depth"] if isinstance(out, dict) else out
            rl = out["router_logits"][0]                     # (1, K, H, W) node L2
            tok_hw = tuple(rl.shape[-2:])
            if tok_hw_final is None:
                tok_hw_final = tok_hw
                image_hw_final = tuple(rgb.shape[-2:])
            else:
                assert tok_hw == tok_hw_final, "Grid size changed between samples"

            probs = rl.softmax(dim=1)[0].cpu().numpy()       # (K, H, W)
            frac = pixel_fraction(dgd, tok_hw)[0].cpu().numpy()
            mae = per_token_err(depth_pred, gt_lidar, mask_lidar, tok_hw)[0].cpu().numpy()
            vc = per_token_valid_count(mask_lidar, tok_hw)[0].cpu().numpy()
            depth_mean = F.adaptive_avg_pool2d(dgd, tok_hw).squeeze(1)[0].cpu().numpy()

            H_tok, W_tok = tok_hw
            row_grid, col_grid = np.meshgrid(np.arange(H_tok), np.arange(W_tok),
                                              indexing="ij")

            all_probs.append(probs.reshape(K, -1).T.astype(np.float32))  # (H*W, K)
            all_frac.append(frac.reshape(K, -1).T.astype(np.float32))
            all_mae.append(mae.ravel().astype(np.float32))
            all_depth.append(depth_mean.ravel().astype(np.float32))
            all_vc.append(vc.ravel().astype(np.float32))
            all_sample_idx.append(
                np.full(H_tok * W_tok, local_idx, dtype=np.int32))
            all_row.append(row_grid.ravel().astype(np.int16))
            all_col.append(col_grid.ravel().astype(np.int16))

    print(f"  ({time.time()-t0:.0f}s forward)")

    probs = np.concatenate(all_probs, axis=0)           # (N, 3)
    frac = np.concatenate(all_frac, axis=0)             # (N, 3)
    mae = np.concatenate(all_mae, axis=0)               # (N,)
    depth = np.concatenate(all_depth, axis=0)
    vc = np.concatenate(all_vc, axis=0)
    sample_idx = np.concatenate(all_sample_idx, axis=0)
    row = np.concatenate(all_row, axis=0)
    col = np.concatenate(all_col, axis=0)
    argmax_p = probs.argmax(axis=1).astype(np.uint8)
    argmax_f = frac.argmax(axis=1).astype(np.uint8)
    print(f"  Tokens: {len(mae):,}  samples: {len(sample_ids):,}  "
          f"grid: {tok_hw_final}  image: {image_hw_final}")

    npz_path = os.path.join(args.out_dir, f"{args.tag}.npz")
    np.savez_compressed(
        npz_path,
        sample_idx=sample_idx,
        token_row=row,
        token_col=col,
        probs=probs,
        frac=frac,
        mae=mae,
        depth=depth,
        valid_pix_count=vc,
        argmax_p=argmax_p,
        argmax_f=argmax_f,
        sample_ids=np.array(sample_ids, dtype=object),
        grid_hw=np.array(tok_hw_final, dtype=np.int32),
        image_hw=np.array(image_hw_final, dtype=np.int32),
        bins=np.array(BINS, dtype=np.float32),
        meta_run=args.run,
        meta_ckpt=args.ckpt,
        meta_split=args.split,
    )
    print(f"  Saved: {npz_path}")


if __name__ == "__main__":
    main()
