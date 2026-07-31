"""DA-v2 + MoGe ROE refinement on nuScenes test split, three modes side-by-side:
  - global : single (scale, shift) per image (current baseline)
  - grid   : K×K regular grid of cells, per-cell ROE → bilinear-interp (scale, shift) field
  - random : MoGe-style multi-level random patches, density-aware ROE → gaussian-RBF blend

DA-v2 is run once per image; all three refinements share the disparity output.

Outputs (uint16 ×256 m PNG):
  <out_global>/<sample_id>.png
  <out_grid>/<sample_id>.png
  <out_random>/<sample_id>.png
"""
import argparse
import os
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

_DA_ROOT = "/workspace/Depth-Anything-V2"
_MOGE_ROOT = "/workspace/MoGe"
for p in (_DA_ROOT, _MOGE_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

from depth_anything_v2.dpt import DepthAnythingV2          # noqa: E402
from moge.utils.alignment import align_depth_affine        # noqa: E402

DA_V2_CONFIGS = {
    "vits": dict(encoder="vits", features=64,  out_channels=[48, 96, 192, 384]),
    "vitb": dict(encoder="vitb", features=128, out_channels=[96, 192, 384, 768]),
    "vitl": dict(encoder="vitl", features=256, out_channels=[256, 512, 1024, 1024]),
}


def load_da_v2(encoder, device):
    from huggingface_hub import hf_hub_download
    repo = {"vits": "depth-anything/Depth-Anything-V2-Small",
            "vitb": "depth-anything/Depth-Anything-V2-Base",
            "vitl": "depth-anything/Depth-Anything-V2-Large"}[encoder]
    ckpt = hf_hub_download(repo_id=repo, filename=f"depth_anything_v2_{encoder}.pth")
    m = DepthAnythingV2(**DA_V2_CONFIGS[encoder])
    # Load weights directly into target device to avoid the 3-5GB CPU
    # spike that pickle deserialize would otherwise produce — important
    # under cgroup memory pressure (we hit oom_kill 8 here on shared box).
    state = torch.load(ckpt, map_location=device, weights_only=True)
    m.to(device)
    m.load_state_dict(state)
    del state
    return m.eval()


# ----------------------------------------------------------------- modes --

_ROE_MAX_ANCHORS = 2000
_ROE_RNG = np.random.default_rng(0)

def _roe_batch(src_np_list, tgt_np_list, eps, trunc):
    """Batched ROE. Pads ragged anchor counts with weight=0.
    Returns (scales, shifts) as numpy arrays of length B.

    MoGe's align_depth_affine internally materializes an (N x N) tensor
    of pairwise differences (alignment.py line 194-195). For N=50k this
    is ~10 GB per side -> SIGKILL on shared boxes. Subsample to at most
    _ROE_MAX_ANCHORS per item; 5k is plenty for robust scale-shift."""
    src_np_list_ = []
    tgt_np_list_ = []
    for s, t in zip(src_np_list, tgt_np_list):
        if len(s) > _ROE_MAX_ANCHORS:
            sel = _ROE_RNG.choice(len(s), size=_ROE_MAX_ANCHORS, replace=False)
            s = s[sel]
            t = t[sel]
        src_np_list_.append(s)
        tgt_np_list_.append(t)
    src_np_list, tgt_np_list = src_np_list_, tgt_np_list_

    n_b = len(src_np_list)
    max_N = max(len(s) for s in src_np_list)
    src = torch.zeros(n_b, max_N, dtype=torch.float32)
    tgt = torch.zeros(n_b, max_N, dtype=torch.float32)
    w   = torch.zeros(n_b, max_N, dtype=torch.float32)
    for k, (s, t) in enumerate(zip(src_np_list, tgt_np_list)):
        n = len(s)
        s_t = torch.from_numpy(s.astype(np.float32))
        t_t = torch.from_numpy(t.astype(np.float32))
        src[k, :n] = s_t
        tgt[k, :n] = t_t
        w[k, :n] = 1.0 / torch.clamp_min(t_t, eps)
    with torch.no_grad():
        scale, shift = align_depth_affine(src, tgt, w, trunc=float(trunc))
    return scale.numpy(), shift.numpy()


def fit_global(pred_disp, gt_depth, valid, trunc, eps):
    if int(valid.sum()) < 50:
        return None, None
    src = pred_disp[valid].astype(np.float32)
    tgt = (1.0 / gt_depth[valid].astype(np.float32))
    s, sh = _roe_batch([src], [tgt], eps, trunc)
    return float(s[0]), float(sh[0])


def fit_grid(pred_disp, gt_depth, valid, K, trunc, eps, min_inliers, fallback):
    H, W = pred_disp.shape
    s_grid = np.full((K, K), fallback[0], dtype=np.float32)
    sh_grid = np.full((K, K), fallback[1], dtype=np.float32)
    ys = np.linspace(0, H, K + 1).astype(int)
    xs = np.linspace(0, W, K + 1).astype(int)
    src_list, tgt_list, ij_list = [], [], []
    for i in range(K):
        for j in range(K):
            v = valid[ys[i]:ys[i+1], xs[j]:xs[j+1]]
            if int(v.sum()) < min_inliers:
                continue
            src_list.append(pred_disp[ys[i]:ys[i+1], xs[j]:xs[j+1]][v])
            tgt_list.append(1.0 / np.clip(gt_depth[ys[i]:ys[i+1], xs[j]:xs[j+1]][v], eps, None))
            ij_list.append((i, j))
    if src_list:
        s_b, sh_b = _roe_batch(src_list, tgt_list, eps, trunc)
        for k, (i, j) in enumerate(ij_list):
            s_grid[i, j] = s_b[k]
            sh_grid[i, j] = sh_b[k]
    # Bilinear K×K → H×W (align_corners=True so corner cells map to image corners)
    s_field = F.interpolate(
        torch.from_numpy(s_grid)[None, None], size=(H, W),
        mode="bilinear", align_corners=True,
    )[0, 0].numpy()
    sh_field = F.interpolate(
        torch.from_numpy(sh_grid)[None, None], size=(H, W),
        mode="bilinear", align_corners=True,
    )[0, 0].numpy()
    return s_field, sh_field, int(len(ij_list))


def fit_random(pred_disp, gt_depth, valid, levels_and_n, trunc, eps,
               min_inliers, fallback, rng, blend_ds=8, sigma_ratio=0.7):
    """MoGe-style multi-scale random patches, gaussian-RBF blended fields."""
    H, W = pred_disp.shape
    D = float(np.sqrt(H * H + W * W))
    valid_idx = np.where(valid)
    n_valid = len(valid_idx[0])
    if n_valid == 0:
        s_field = np.full((H, W), fallback[0], dtype=np.float32)
        sh_field = np.full((H, W), fallback[1], dtype=np.float32)
        return s_field, sh_field, 0

    centers_y, centers_x, scales, shifts, radii = [], [], [], [], []
    for level, num_p in levels_and_n:
        radius_2d = int(np.ceil(0.5 / level * D))
        sel = rng.integers(0, n_valid, size=num_p)
        cy = valid_idx[0][sel]
        cx = valid_idx[1][sel]
        src_list, tgt_list, meta = [], [], []
        for k in range(num_p):
            y0, y1 = max(0, cy[k] - radius_2d), min(H, cy[k] + radius_2d + 1)
            x0, x1 = max(0, cx[k] - radius_2d), min(W, cx[k] + radius_2d + 1)
            v = valid[y0:y1, x0:x1]
            if int(v.sum()) < min_inliers:
                continue
            src_list.append(pred_disp[y0:y1, x0:x1][v])
            tgt_list.append(1.0 / np.clip(gt_depth[y0:y1, x0:x1][v], eps, None))
            meta.append((int(cy[k]), int(cx[k]), radius_2d))
        if not src_list:
            continue
        s_b, sh_b = _roe_batch(src_list, tgt_list, eps, trunc)
        for k, (py, px, r) in enumerate(meta):
            centers_y.append(py)
            centers_x.append(px)
            scales.append(float(s_b[k]))
            shifts.append(float(sh_b[k]))
            radii.append(r)

    if not centers_y:
        s_field = np.full((H, W), fallback[0], dtype=np.float32)
        sh_field = np.full((H, W), fallback[1], dtype=np.float32)
        return s_field, sh_field, 0

    cy_arr = np.array(centers_y, dtype=np.float32)
    cx_arr = np.array(centers_x, dtype=np.float32)
    s_arr = np.array(scales, dtype=np.float32)
    sh_arr = np.array(shifts, dtype=np.float32)
    sigma = np.array(radii, dtype=np.float32) * sigma_ratio

    # Compute RBF on a downsampled grid → upsample (faster, smooth field anyway)
    DS = max(1, int(blend_ds))
    H_d = max(2, H // DS)
    W_d = max(2, W // DS)
    ys_d = (np.linspace(0, H - 1, H_d)).astype(np.float32)
    xs_d = (np.linspace(0, W - 1, W_d)).astype(np.float32)
    yy, xx = np.meshgrid(ys_d, xs_d, indexing="ij")
    dy = yy[..., None] - cy_arr[None, None, :]
    dx = xx[..., None] - cx_arr[None, None, :]
    d2 = dy * dy + dx * dx
    w = np.exp(-d2 / (2.0 * (sigma[None, None, :] ** 2) + 1e-8))
    w_sum = w.sum(axis=-1)
    # Where weight sum is too small (no nearby patch), fallback
    s_field_d = np.where(
        w_sum > 1e-6,
        (w * s_arr[None, None, :]).sum(axis=-1) / np.maximum(w_sum, 1e-8),
        fallback[0],
    ).astype(np.float32)
    sh_field_d = np.where(
        w_sum > 1e-6,
        (w * sh_arr[None, None, :]).sum(axis=-1) / np.maximum(w_sum, 1e-8),
        fallback[1],
    ).astype(np.float32)
    s_field = F.interpolate(
        torch.from_numpy(s_field_d)[None, None], size=(H, W),
        mode="bilinear", align_corners=True,
    )[0, 0].numpy()
    sh_field = F.interpolate(
        torch.from_numpy(sh_field_d)[None, None], size=(H, W),
        mode="bilinear", align_corners=True,
    )[0, 0].numpy()
    return s_field, sh_field, len(centers_y)


def disp_field_to_depth_png(scale_field, shift_field, pred_disp,
                             disp_floor, max_m, scale_unit):
    refined_disp = scale_field * pred_disp + shift_field
    refined_depth = 1.0 / np.clip(refined_disp, disp_floor, None)
    refined_depth = np.where(np.isfinite(refined_depth), refined_depth, 0.0)
    refined_depth = np.clip(refined_depth, 0.0, max_m)
    q = (refined_depth * scale_unit).round().clip(0, 65535).astype(np.uint16)
    return q


# ----------------------------------------------------------------- main --

@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split", required=True)
    p.add_argument("--data_root", required=True)
    p.add_argument("--out_global", default=None)
    p.add_argument("--out_grid",   default=None)
    p.add_argument("--out_random", default=None,
                   help="omit to skip the random-patch mode entirely.")
    p.add_argument("--gt_dir", default="depth_lidar")
    p.add_argument("--encoder", default="vitl", choices=list(DA_V2_CONFIGS))
    p.add_argument("--input_size", type=int, default=518)
    p.add_argument("--min_depth", type=float, default=0.5)
    p.add_argument("--max_depth_m", type=float, default=None)
    p.add_argument("--trunc", type=float, default=1.0)
    p.add_argument("--eps", type=float, default=1e-6)
    p.add_argument("--scale_unit", type=float, default=256.0)
    p.add_argument("--fp16", action="store_true", default=True)
    p.add_argument("--no_fp16", dest="fp16", action="store_false")
    p.add_argument("--skip_existing", action="store_true", default=True)
    p.add_argument("--no_skip_existing", dest="skip_existing", action="store_false")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--shard", type=str, default=None,
                   help="for parallel workers on the same split. Format "
                        "'i/N' means this worker processes lines with "
                        "0-indexed row % N == i. e.g. '0/4', '1/4', ...")
    p.add_argument("--image_subdir", default="image",
                   help="When split lines have no \\t<image_path>, fall back "
                        "to <data_root>/<image_subdir>/<sid>.png "
                        "(default: 'image').")
    # grid params
    p.add_argument("--grid_K", type=int, default=8)
    p.add_argument("--grid_min_inliers", type=int, default=30)
    # random params (MoGe-style multi-scale)
    p.add_argument("--random_levels", type=int, nargs="+", default=[4, 16])
    p.add_argument("--random_num_patches", type=int, nargs="+", default=[16, 64])
    p.add_argument("--random_min_inliers", type=int, default=20)
    p.add_argument("--random_blend_ds", type=int, default=8)
    p.add_argument("--random_sigma_ratio", type=float, default=0.7)
    args = p.parse_args()
    assert len(args.random_levels) == len(args.random_num_patches), \
        "--random_levels and --random_num_patches must have same length"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_da_v2(args.encoder, device)
    # NOTE: xformers memory-efficient attention requires fp16/bf16 on CUDA.
    # We .half() the model AND manually cast inputs to keep them in sync.
    if args.fp16:
        model.half()
    max_m = args.max_depth_m if args.max_depth_m is not None else 65535.0 / args.scale_unit
    disp_floor = 1.0 / max_m
    rng = np.random.default_rng(args.seed)

    MODES = [m for m in ("global", "grid", "random")
             if getattr(args, f"out_{m}") is not None]
    assert MODES, "at least one of --out_global / --out_grid / --out_random must be set"

    items = list(_iter_split(args.split, data_root=args.data_root,
                             image_subdir=args.image_subdir))
    n_total = len(items)
    shard_label = ""
    if args.shard:
        i_shard, n_shards = (int(x) for x in args.shard.split("/"))
        assert 0 <= i_shard < n_shards, f"invalid shard '{args.shard}'"
        items = [it for k, it in enumerate(items) if k % n_shards == i_shard]
        shard_label = f"  [shard {i_shard}/{n_shards}]"
    if args.limit:
        items = items[:args.limit]
    for m in MODES:
        os.makedirs(getattr(args, f"out_{m}"), exist_ok=True)

    print(f"[multi_refine]{shard_label} {len(items)}/{n_total} samples  "
          f"encoder={args.encoder}  trunc={args.trunc}  modes={MODES}")
    if "global" in MODES:
        print(f"  → global = {args.out_global}")
    if "grid" in MODES:
        print(f"  → grid   = {args.out_grid}  (K={args.grid_K})")
    if "random" in MODES:
        print(f"  → random = {args.out_random}  (levels={args.random_levels}  N={args.random_num_patches})")

    saved = {m: 0 for m in MODES}
    skipped = 0
    missing = 0

    for sid, image_path in tqdm(items):
        out_paths = {m: os.path.join(getattr(args, f"out_{m}"), f"{sid}.png")
                     for m in MODES}
        for d in out_paths.values():
            os.makedirs(os.path.dirname(d), exist_ok=True)
        todo = [m for m in MODES
                if (not args.skip_existing) or (not os.path.exists(out_paths[m]))]
        if not todo:
            skipped += 1
            continue

        gt_path = os.path.join(args.data_root, args.gt_dir, f"{sid}.png")
        if not image_path or not os.path.exists(image_path) or not os.path.exists(gt_path):
            missing += 1
            continue
        bgr = cv2.imread(image_path)
        if bgr is None:
            missing += 1
            continue
        H, W = bgr.shape[:2]

        # Replicate model.infer_image() but with manual dtype/device control
        # so the input matches the (possibly fp16) model weights & device.
        image, (img_h, img_w) = model.image2tensor(bgr, input_size=args.input_size)
        image = image.to(device)
        if args.fp16:
            image = image.half()
        depth_t = model.forward(image)
        depth_t = F.interpolate(depth_t[:, None], (img_h, img_w),
                                mode="bilinear", align_corners=True)[0, 0]
        pred_disp = depth_t.float().cpu().numpy().astype(np.float32)
        if pred_disp.shape != (H, W):
            raise RuntimeError(f"DA-v2 shape mismatch: {pred_disp.shape} vs {(H,W)}")

        gt = np.asarray(Image.open(gt_path), dtype=np.float32) / args.scale_unit
        valid = (gt > args.min_depth) & (gt < max_m) & np.isfinite(gt) & np.isfinite(pred_disp)

        # 1) Global
        a, b = fit_global(pred_disp, gt, valid, trunc=args.trunc, eps=args.eps)
        if a is None:
            # Not enough anchors → skip this sample
            continue

        if "global" in todo:
            q = disp_field_to_depth_png(
                np.full_like(pred_disp, a), np.full_like(pred_disp, b),
                pred_disp, disp_floor, max_m, args.scale_unit,
            )
            Image.fromarray(q, mode="I;16").save(out_paths["global"])
            saved["global"] += 1

        # 2) Grid
        if "grid" in todo:
            s_field, sh_field, _ = fit_grid(
                pred_disp, gt, valid, K=args.grid_K, trunc=args.trunc, eps=args.eps,
                min_inliers=args.grid_min_inliers, fallback=(a, b),
            )
            q = disp_field_to_depth_png(s_field, sh_field, pred_disp,
                                         disp_floor, max_m, args.scale_unit)
            Image.fromarray(q, mode="I;16").save(out_paths["grid"])
            saved["grid"] += 1

        # 3) Random multi-scale
        if "random" in todo:
            s_field, sh_field, _ = fit_random(
                pred_disp, gt, valid,
                levels_and_n=list(zip(args.random_levels, args.random_num_patches)),
                trunc=args.trunc, eps=args.eps,
                min_inliers=args.random_min_inliers, fallback=(a, b),
                rng=rng, blend_ds=args.random_blend_ds,
                sigma_ratio=args.random_sigma_ratio,
            )
            q = disp_field_to_depth_png(s_field, sh_field, pred_disp,
                                         disp_floor, max_m, args.scale_unit)
            Image.fromarray(q, mode="I;16").save(out_paths["random"])
            saved["random"] += 1

    print(f"[done] saved: {saved}  skipped(all-existing)={skipped}  missing={missing}")


def _iter_split(split, data_root=None, image_subdir="image"):
    """Yield (sample_id, image_path). If the split line is sid-only
    (no \\t<image_path>), fall back to <data_root>/<image_subdir>/<sid>.png
    — useful for datasets like ZJU-4DRadarCam where split files are
    plain sid lists."""
    with open(split) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) > 1:
                img = parts[1]
            elif data_root is not None:
                img = os.path.join(data_root, image_subdir, parts[0] + ".png")
            else:
                img = None
            yield parts[0], img


if __name__ == "__main__":
    main()
