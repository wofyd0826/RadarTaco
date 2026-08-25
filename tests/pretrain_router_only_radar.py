"""Router-only pretraining with RADAR signal injected into the router input.

Extension of tests/pretrain_router_only.py. Same frozen-baseline setup, but
each router sees `image_feat` PLUS a physically-motivated radar encoding at
the router grid resolution.

Radar encoding (10 channels per router token) — respects radar's physical
limits (no vertical resolution, noisy, sparse):

  Column stats (row-broadcast, function of u only):
    ch 0  has_hit_col           binary
    ch 1  depth_median_col      /max_depth
    ch 2  depth_min_col         /max_depth
    ch 3  depth_max_col         /max_depth
    ch 4  depth_std_col         /max_depth   (noise proxy)
    ch 5  log(count_col+1)      /log(K_max+1)

  Strip stats — bounded vertical extent (nominal 2m object height, fy-scaled):
    ch 6  in_strip              binary
    ch 7  strip_depth           /max_depth   (min over overlapping strips)
    ch 8  v_offset_norm         in [0, 1]    (0 = strip bottom, 1 = top)

  Column KNN fallback (ensures every token has some signal):
    ch 9  col_dist_to_nearest_hit  /W_router

Ablation modes:
  --radar-mode none     baseline (image feat only) — same as pretrain_router_only.py
  --radar-mode naive    5-ch point-only concat (no P1/P2/KNN — for negative control)
  --radar-mode col      P1 only (6ch column stats), concat with image
  --radar-mode col_strip  P1+P2 (9ch), concat
  --radar-mode full     P1+P2+KNN (10ch) + radar_embedder + gated fusion (recommended)

Example
-------
  python tests/pretrain_router_only_radar.py --tag full_radar \\
      --radar-mode full --epochs 50 --batch-size 8 --lr 1e-3
"""
import argparse, json, math, os, sys, time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, ROOT)

from src.dataset.nuscenes import NuScenesRadarDepthDataset
from src.dataset.intrinsics import (CH_XPIX, CH_YPIX, CH_DEPTH,
                                    NUSCENES_CAM_FRONT_INTRINSIC)
from scripts.train import _build_model


BINS = [0.0, 20.0, 50.0, 100.0]
K = 3
LABELS = ["near", "mid", "far"]

# nuScenes CAM_FRONT — used for strip pixel-height computation.
FY_NOMINAL = NUSCENES_CAM_FRONT_INTRINSIC["fy"]
H_NOMINAL_M = 2.0            # nominal object vertical extent (m)


def compute_frac(depth_gt_dense: torch.Tensor, tok_hw, bins) -> torch.Tensor:
    edges = torch.tensor(bins, device=depth_gt_dense.device, dtype=torch.float32)
    ds = depth_gt_dense.squeeze(1)
    pix_bin = torch.zeros_like(ds, dtype=torch.long)
    for i in range(K):
        pix_bin = torch.where(
            (ds >= float(edges[i])) & (ds < float(edges[i+1])),
            pix_bin.new_tensor(i), pix_bin)
    pix_bin = torch.where(ds >= float(edges[-1]),
                          pix_bin.new_tensor(K-1), pix_bin)
    onehot = F.one_hot(pix_bin, num_classes=K).permute(0, 3, 1, 2).float()
    return F.adaptive_avg_pool2d(onehot, tok_hw)


# ============================================================================
# Radar-to-token encoding
# ============================================================================

def _naive_point_map(radar_points: torch.Tensor, radar_mask: torch.Tensor,
                     H_tok: int, W_tok: int, H_img: int, W_img: int,
                     max_depth: float, k_max: int) -> torch.Tensor:
    """Old-school point-only encoding (5ch: has, min, max, mean, log_count).
    Negative control for the ablation."""
    B, K_dim, _ = radar_points.shape
    device = radar_points.device
    out = torch.zeros(B, 5, H_tok, W_tok, device=device, dtype=torch.float32)
    log_norm = math.log(k_max + 1.0)
    for b in range(B):
        m = radar_mask[b].bool()
        if not m.any():
            continue
        x_pix = radar_points[b, m, CH_XPIX]
        y_pix = radar_points[b, m, CH_YPIX]
        d = radar_points[b, m, CH_DEPTH]
        valid = (d > 0) & (x_pix >= 0) & (x_pix < W_img) & (y_pix >= 0) & (y_pix < H_img)
        if not valid.any():
            continue
        x_pix = x_pix[valid]; y_pix = y_pix[valid]; d = d[valid]
        u = torch.clamp((x_pix / W_img * W_tok).long(), 0, W_tok - 1)
        v = torch.clamp((y_pix / H_img * H_tok).long(), 0, H_tok - 1)
        idx = (v * W_tok + u).long()
        flat_has = torch.zeros(H_tok * W_tok, device=device)
        flat_min = torch.full((H_tok * W_tok,), float("inf"), device=device)
        flat_max = torch.full((H_tok * W_tok,), float("-inf"), device=device)
        flat_sum = torch.zeros(H_tok * W_tok, device=device)
        flat_cnt = torch.zeros(H_tok * W_tok, device=device)
        flat_has.scatter_(0, idx, torch.ones_like(d))
        flat_min.scatter_reduce_(0, idx, d, reduce="amin", include_self=True)
        flat_max.scatter_reduce_(0, idx, d, reduce="amax", include_self=True)
        flat_sum.scatter_add_(0, idx, d)
        flat_cnt.scatter_add_(0, idx, torch.ones_like(d))
        flat_min = torch.where(flat_cnt > 0, flat_min, torch.zeros_like(flat_min))
        flat_max = torch.where(flat_cnt > 0, flat_max, torch.zeros_like(flat_max))
        flat_mean = torch.where(flat_cnt > 0, flat_sum / flat_cnt.clamp_min(1.0),
                                 torch.zeros_like(flat_sum))
        out[b, 0] = flat_has.view(H_tok, W_tok)
        out[b, 1] = flat_min.view(H_tok, W_tok) / max_depth
        out[b, 2] = flat_max.view(H_tok, W_tok) / max_depth
        out[b, 3] = flat_mean.view(H_tok, W_tok) / max_depth
        out[b, 4] = torch.log1p(flat_cnt).view(H_tok, W_tok) / log_norm
    return out


def radar_to_token_map(
    radar_points: torch.Tensor,        # (B, K, 6)
    radar_mask: torch.Tensor,          # (B, K) bool
    H_tok: int, W_tok: int,
    H_img: int, W_img: int,
    max_depth: float,
    k_max: int,
    include_strip: bool = True,
    include_knn: bool = True,
) -> torch.Tensor:
    """Radar → per-token feature map at (H_tok, W_tok).

    Fully vectorized (no per-hit Python loops):
      • Column stats via scatter_reduce over flat (b, u) indices.
      • Strip stats via broadcasting hits × rows into a (N_hit, H_tok) mask,
        then scatter_reduce (amin) into flat (b, r, u) indices.
      • KNN via pairwise |Δu| across all (batch, hit) pairs, masked min.

    Physically-motivated: column aggregation respects "no vertical resolution",
    strip uses bounded fy-scaled height, KNN in u-axis only fills empty columns.
    """
    B, Kmax, _ = radar_points.shape
    device = radar_points.device
    log_norm = math.log(k_max + 1.0)

    # Flatten valid radar hits across the batch, remember which sample each belongs to.
    m = radar_mask.bool()
    d_all = radar_points[..., CH_DEPTH]
    x_all = radar_points[..., CH_XPIX]
    y_all = radar_points[..., CH_YPIX]
    valid = m & (d_all > 0) & (x_all >= 0) & (x_all < W_img) \
              & (y_all >= 0) & (y_all < H_img)
    b_idx, k_idx = valid.nonzero(as_tuple=True)                # (N,)
    N = b_idx.numel()

    ch_col = torch.zeros(B, 6, W_tok, device=device, dtype=torch.float32)
    strip_channels = torch.zeros(B, 3, H_tok, W_tok, device=device, dtype=torch.float32)
    knn_channel = torch.zeros(B, 1, H_tok, W_tok, device=device, dtype=torch.float32)

    if N == 0:
        # No radar anywhere → col_dist saturates to 1.0
        if include_knn:
            knn_channel.fill_(1.0)
        col_bcast = ch_col.unsqueeze(2).expand(B, 6, H_tok, W_tok)
        parts = [col_bcast]
        if include_strip:
            parts.append(strip_channels)
        if include_knn:
            parts.append(knn_channel)
        return torch.cat(parts, dim=1)

    d = d_all[b_idx, k_idx]
    x_pix = x_all[b_idx, k_idx]
    y_pix = y_all[b_idx, k_idx]
    u = torch.clamp((x_pix / W_img * W_tok).long(), 0, W_tok - 1)     # (N,)
    v_hit_row = (y_pix / H_img * H_tok)                                # (N,) float

    # ---- Column stats via scatter_reduce over (b, u) flat indices ----
    col_flat_idx = (b_idx * W_tok + u)                                 # (N,)
    n_flat_col = B * W_tok
    flat_has = torch.zeros(n_flat_col, device=device)
    flat_min = torch.full((n_flat_col,), float("inf"), device=device)
    flat_max = torch.full((n_flat_col,), float("-inf"), device=device)
    flat_sum = torch.zeros(n_flat_col, device=device)
    flat_sq = torch.zeros(n_flat_col, device=device)
    flat_cnt = torch.zeros(n_flat_col, device=device)
    flat_has.scatter_(0, col_flat_idx, torch.ones_like(d))
    flat_min.scatter_reduce_(0, col_flat_idx, d, reduce="amin", include_self=True)
    flat_max.scatter_reduce_(0, col_flat_idx, d, reduce="amax", include_self=True)
    flat_sum.scatter_add_(0, col_flat_idx, d)
    flat_sq.scatter_add_(0, col_flat_idx, d * d)
    flat_cnt.scatter_add_(0, col_flat_idx, torch.ones_like(d))
    has = (flat_cnt > 0).float()
    flat_min = torch.where(flat_cnt > 0, flat_min, torch.zeros_like(flat_min))
    flat_max = torch.where(flat_cnt > 0, flat_max, torch.zeros_like(flat_max))
    flat_mean = torch.where(flat_cnt > 0, flat_sum / flat_cnt.clamp_min(1.0),
                             torch.zeros_like(flat_sum))
    var = torch.where(flat_cnt > 0,
                      (flat_sq / flat_cnt.clamp_min(1.0)) - flat_mean * flat_mean,
                      torch.zeros_like(flat_sq))
    std = var.clamp_min(0.0).sqrt()
    # Approx median with mean — mean is cheap-and-batched. Median would need
    # per-column sort. For sparse radar (usually 1-3 hits/col) the two agree.
    ch_col[:, 0] = has.view(B, W_tok)
    ch_col[:, 1] = flat_mean.view(B, W_tok) / max_depth                # "median" ≈ mean
    ch_col[:, 2] = flat_min.view(B, W_tok) / max_depth
    ch_col[:, 3] = flat_max.view(B, W_tok) / max_depth
    ch_col[:, 4] = std.view(B, W_tok) / max_depth
    ch_col[:, 5] = torch.log1p(flat_cnt).view(B, W_tok) / log_norm

    # ---- Strip stats — vectorized over (N_hits × H_tok) ----
    if include_strip:
        strip_h_pix = H_NOMINAL_M * FY_NOMINAL / d.clamp_min(1e-3)     # (N,)
        v_top_row = (v_hit_row - strip_h_pix / H_img * H_tok).clamp_min(0.0)
        v_hit_row_c = v_hit_row.clamp(0.0, H_tok - 1e-6)
        rows = torch.arange(H_tok, device=device, dtype=torch.float32) # (H,)
        # (N, H): does row r fall within hit i's strip?
        in_hit = (rows.view(1, -1) >= v_top_row.view(-1, 1)) & \
                 (rows.view(1, -1) <= v_hit_row_c.view(-1, 1))          # (N, H)
        # Expand to (N, H): each hit contributes to H rows. Flat idx = b*H*W + r*W + u.
        # Only keep entries where in_hit=True.
        n_pairs = in_hit.sum()
        if n_pairs > 0:
            hit_i, row_r = in_hit.nonzero(as_tuple=True)                # (M,)
            b_at = b_idx[hit_i]
            u_at = u[hit_i]
            d_at = d[hit_i]
            v_hit_at = v_hit_row_c[hit_i]
            v_top_at = v_top_row[hit_i]
            row_f = row_r.float()
            v_off_at = ((v_hit_at - row_f) /
                        (v_hit_at - v_top_at).clamp_min(1e-3)).clamp(0.0, 1.0)
            # Occlusion: keep the MIN depth per (b, r, u) via scatter_reduce amin.
            strip_flat = (b_at * H_tok * W_tok + row_r * W_tok + u_at)  # (M,)
            n_flat_strip = B * H_tok * W_tok
            depth_flat = torch.full((n_flat_strip,), float("inf"), device=device)
            depth_flat.scatter_reduce_(0, strip_flat, d_at,
                                        reduce="amin", include_self=True)
            in_strip = torch.isfinite(depth_flat)
            depth_out = torch.where(in_strip, depth_flat,
                                     torch.zeros_like(depth_flat))
            # For v_offset: pick the offset from the winning hit (=argmin of d_at at
            # that flat idx). Argmin per bucket isn't a native op, so we approximate
            # by taking the offset of the closest-depth hit via a second reduce:
            # gate d_at against the winner and keep the corresponding v_off.
            #
            # Simple approximation: mean v_offset over overlapping hits. In practice
            # overlap is rare and this is a diagnostic signal — mean is fine.
            voff_sum = torch.zeros(n_flat_strip, device=device)
            voff_cnt = torch.zeros(n_flat_strip, device=device)
            voff_sum.scatter_add_(0, strip_flat, v_off_at)
            voff_cnt.scatter_add_(0, strip_flat, torch.ones_like(v_off_at))
            voff_out = torch.where(voff_cnt > 0, voff_sum / voff_cnt.clamp_min(1.0),
                                    torch.zeros_like(voff_sum))
            strip_channels[:, 0] = in_strip.view(B, H_tok, W_tok).float()
            strip_channels[:, 1] = (depth_out / max_depth).view(B, H_tok, W_tok)
            strip_channels[:, 2] = voff_out.view(B, H_tok, W_tok)

    # ---- KNN in u-axis: per-sample pairwise |u_grid - u_hit| min ----
    if include_knn:
        all_u = torch.arange(W_tok, device=device, dtype=torch.float32)  # (W,)
        # For each batch entry, compute (W_tok, N_b) then min over N_b.
        # Rather than looping over B, do (B, W, N_max) via scatter of hit u's per sample.
        # Simplest fast route: for each sample b, gather its hits; use segmented min.
        # Given B is small (batch=8-16), a light Python loop over B is fine here.
        for b in range(B):
            sel = (b_idx == b)
            if not sel.any():
                knn_channel[b, 0].fill_(1.0)
                continue
            hits_u = u[sel].float()                                     # (Nb,)
            dist = (all_u.view(-1, 1) - hits_u.view(1, -1)).abs().min(dim=1).values
            knn_channel[b, 0] = (dist / W_tok).view(1, W_tok).expand(H_tok, W_tok)

    # Broadcast column stats to rows
    col_bcast = ch_col.unsqueeze(2).expand(B, 6, H_tok, W_tok)
    parts = [col_bcast]
    if include_strip:
        parts.append(strip_channels)
    if include_knn:
        parts.append(knn_channel)
    return torch.cat(parts, dim=1)


# ============================================================================
# Router head variants
# ============================================================================

def make_router(ch_in: int, n_experts: int) -> nn.Module:
    """mlp3x3, matching v4c."""
    h1 = min(256, ch_in // 2)
    h2 = h1 // 2
    return nn.Sequential(
        nn.Conv2d(ch_in, h1, 3, padding=1), nn.GELU(),
        nn.Conv2d(h1, h2, 3, padding=1), nn.GELU(),
        nn.Conv2d(h2, n_experts, 1),
    )


class RadarEmbedder(nn.Module):
    """10ch raw radar → 128ch dense embedding."""
    def __init__(self, in_ch: int, out_ch: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 64, 1), nn.GELU(),
            nn.Conv2d(64, out_ch, 3, padding=1), nn.GELU(),
            nn.Conv2d(out_ch, out_ch, 1),
        )
    def forward(self, x): return self.net(x)


class GatedRouter(nn.Module):
    """Router with gated image/radar fusion.

    Path:
      img_proj    = Conv1x1(img_ch → mid)
      radar_embed = RadarEmbedder(radar_ch → mid)
      gate        = σ(Conv1x1(concat([img_proj, radar_embed]) → mid))
      fused       = img_proj + gate * radar_embed
      logits      = mlp3x3(fused → K)
    """
    def __init__(self, img_ch: int, radar_ch: int, n_experts: int,
                 mid: int = 128):
        super().__init__()
        self.img_proj = nn.Conv2d(img_ch, mid, 1)
        self.radar_embed = RadarEmbedder(radar_ch, mid)
        self.gate = nn.Sequential(
            nn.Conv2d(2 * mid, mid, 1), nn.Sigmoid(),
        )
        self.head = make_router(mid, n_experts)

    def forward(self, img_feat: torch.Tensor, radar_map: torch.Tensor):
        ip = self.img_proj(img_feat)
        re = self.radar_embed(radar_map)
        g = self.gate(torch.cat([ip, re], dim=1))
        fused = ip + g * re
        return self.head(fused), g

    @torch.no_grad()
    def mean_gate(self, img_feat, radar_map) -> float:
        _, g = self.forward(img_feat, radar_map)
        return float(g.mean())


class ConcatRouter(nn.Module):
    """Simple concat router (used by naive / col / col_strip modes).
    No embedder, no gate — image and raw radar channels concatenated then
    fed to the mlp3x3."""
    def __init__(self, img_ch: int, radar_ch: int, n_experts: int):
        super().__init__()
        self.head = make_router(img_ch + radar_ch, n_experts)

    def forward(self, img_feat: torch.Tensor, radar_map: torch.Tensor):
        return self.head(torch.cat([img_feat, radar_map], dim=1)), None


def build_routers(mode: str, ch_node: int, ch_edge: int):
    """Return dict {'node': Module, 'edge': Module} and radar-channel budget."""
    if mode == "none":
        return {"node": make_router(ch_node, K),
                "edge": make_router(ch_edge, K)}, 0, False, False, False
    if mode == "naive":
        rch = 5
        return {"node": ConcatRouter(ch_node, rch, K),
                "edge": ConcatRouter(ch_edge, rch, K)}, rch, False, False, False
    if mode == "col":
        rch = 6
        return {"node": ConcatRouter(ch_node, rch, K),
                "edge": ConcatRouter(ch_edge, rch, K)}, rch, True, False, False
    if mode == "col_strip":
        rch = 9
        return {"node": ConcatRouter(ch_node, rch, K),
                "edge": ConcatRouter(ch_edge, rch, K)}, rch, True, True, False
    if mode == "full":
        rch = 10
        return {"node": GatedRouter(ch_node, rch, K),
                "edge": GatedRouter(ch_edge, rch, K)}, rch, True, True, True
    raise ValueError(f"Unknown radar mode: {mode}")


def build_radar_map(mode: str, radar_points, radar_mask, H_tok, W_tok,
                    H_img, W_img, max_depth, k_max,
                    use_strip: bool, use_knn: bool) -> Optional[torch.Tensor]:
    """Return (B, C, H_tok, W_tok) or None (mode='none')."""
    if mode == "none":
        return None
    if mode == "naive":
        return _naive_point_map(radar_points, radar_mask,
                                H_tok, W_tok, H_img, W_img, max_depth, k_max)
    return radar_to_token_map(radar_points, radar_mask,
                              H_tok, W_tok, H_img, W_img, max_depth, k_max,
                              include_strip=use_strip, include_knn=use_knn)


# ============================================================================
# Eval
# ============================================================================

@torch.no_grad()
def evaluate(baseline, routers, captured, loader, device,
             mode: str, use_strip: bool, use_knn: bool,
             max_depth: float, k_max: int) -> Dict:
    """Metrics: per-block accuracy, per-class recall, soft CE, AND per-token
    split by has_radar_column (Ch 0 == 1 vs Ch 0 == 0)."""
    for r in routers.values():
        r.eval()
    stats = {name: {"correct": 0, "total": 0,
                    "correct_hit": 0, "total_hit": 0,
                    "correct_nohit": 0, "total_nohit": 0,
                    "pcc": torch.zeros(K, dtype=torch.long),
                    "pct": torch.zeros(K, dtype=torch.long),
                    "soft_ce_sum": 0.0,
                    "gate_sum": 0.0, "gate_n": 0}
             for name in routers}
    n_batches = 0

    for batch in loader:
        rgb = batch["rgb_norm"].to(device, non_blocking=True)
        rp = batch["radar_points"].to(device, non_blocking=True)
        rm = batch["radar_mask"].to(device, non_blocking=True)
        dgd = batch["depth_gt_dense"].to(device, non_blocking=True)
        H_img, W_img = rgb.shape[-2:]
        _ = baseline(rgb, rp, rm)
        for name, router in routers.items():
            feat = captured[name]
            H_tok, W_tok = feat.shape[-2:]
            radar_map = build_radar_map(mode, rp, rm, H_tok, W_tok,
                                         H_img, W_img, max_depth, k_max,
                                         use_strip, use_knn)
            if radar_map is None:
                logits = router(feat)
                gate_mean = None
            else:
                out = router(feat, radar_map) if isinstance(router, (GatedRouter, ConcatRouter)) \
                      else router(torch.cat([feat, radar_map], dim=1))
                if isinstance(out, tuple):
                    logits, g = out
                    gate_mean = None if g is None else float(g.mean())
                else:
                    logits = out
                    gate_mean = None

            frac = compute_frac(dgd, logits.shape[-2:], BINS)
            gt = frac.argmax(dim=1)
            pred = logits.argmax(dim=1)
            s = stats[name]
            s["correct"] += (pred == gt).sum().item()
            s["total"] += gt.numel()
            for k in range(K):
                mk = (gt == k)
                s["pct"][k] += mk.sum().item()
                s["pcc"][k] += (mk & (pred == k)).sum().item()
            s["soft_ce_sum"] += F.cross_entropy(logits, frac).item()
            if gate_mean is not None:
                s["gate_sum"] += gate_mean; s["gate_n"] += 1

            # has_radar_column split (based on the radar map's ch 0 or, for
            # naive mode, ch 0 which is per-token has_radar).
            if radar_map is not None:
                col_hit = (radar_map[:, 0] > 0.5)               # (B, H, W)
                correct_mask = (pred == gt)
                s["correct_hit"] += (correct_mask & col_hit).sum().item()
                s["total_hit"] += col_hit.sum().item()
                s["correct_nohit"] += (correct_mask & ~col_hit).sum().item()
                s["total_nohit"] += (~col_hit).sum().item()
        n_batches += 1

    out = {"per_block": {}}
    tot_c = 0; tot_n = 0
    for name in routers:
        s = stats[name]
        acc = 100.0 * s["correct"] / max(s["total"], 1)
        rec = [100.0 * s["pcc"][k].item() / max(s["pct"][k].item(), 1)
               for k in range(K)]
        block = {
            "acc": acc,
            "recall": rec,
            "tokens": s["total"],
            "soft_ce": s["soft_ce_sum"] / max(n_batches, 1),
        }
        if s["total_hit"] + s["total_nohit"] > 0:
            block["acc_hit_col"] = 100.0 * s["correct_hit"] / max(s["total_hit"], 1)
            block["acc_nohit_col"] = 100.0 * s["correct_nohit"] / max(s["total_nohit"], 1)
            block["tokens_hit"] = s["total_hit"]
            block["tokens_nohit"] = s["total_nohit"]
        if s["gate_n"] > 0:
            block["gate_mean"] = s["gate_sum"] / s["gate_n"]
        out["per_block"][name] = block
        tot_c += s["correct"]; tot_n += s["total"]
    out["overall_acc"] = 100.0 * tot_c / max(tot_n, 1)
    return out


# ============================================================================
# Main
# ============================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline-run",
                    default="output/shape_lidar_grad_shape_edge_res_fix")
    ap.add_argument("--ckpt", default="best.pt")
    ap.add_argument("--radar-mode", default="full",
                    choices=["none", "naive", "col", "col_strip", "full"])
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--val-every", type=int, default=1)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--out-dir", default="output/router_pretrain")
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()

    # Per-tag subdirectory. Create it up-front so any early failure leaves a
    # visible artifact (rather than silently going nowhere).
    args.out_dir = os.path.join(args.out_dir, args.tag)
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"Output dir: {args.out_dir}")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- Load frozen baseline ----
    cfg_path = os.path.join(args.baseline_run, "config.yaml")
    cfg = OmegaConf.load(cfg_path)
    max_depth = float(cfg.dataset.max_depth)
    k_max = int(cfg.dataset.max_radar_points)
    print(f"Loading baseline from {args.baseline_run}")
    baseline = _build_model(cfg, max_depth, k_max)
    sd = torch.load(os.path.join(args.baseline_run, args.ckpt),
                    map_location=device, weights_only=False)
    baseline.load_state_dict(sd["model"])
    baseline = baseline.to(device).eval()
    for p in baseline.parameters():
        p.requires_grad = False

    l2_node_block = baseline.radar_fusion.node_blocks[2]
    l2_edge_block = baseline.radar_fusion.edge_blocks[2]
    ch_node = l2_node_block.norm1.num_channels
    ch_edge = l2_edge_block.norm1.num_channels

    # ---- Build routers per mode ----
    (routers_dict, rch, use_col, use_strip, use_knn) = \
        build_routers(args.radar_mode, ch_node, ch_edge)
    routers = {k: v.to(device) for k, v in routers_dict.items()}
    n_router_params = sum(sum(p.numel() for p in r.parameters())
                          for r in routers.values())
    print(f"  L2 node ch={ch_node}, edge ch={ch_edge}")
    print(f"  Radar mode: {args.radar_mode}  (radar_ch={rch}, "
          f"use_strip={use_strip}, use_knn={use_knn})")
    print(f"  Trainable router params: {n_router_params:,}")

    # ---- Hooks on L2 block inputs ----
    captured: Dict[str, torch.Tensor] = {}
    def make_pre_hook(name):
        def _h(_m, args_):
            captured[name] = args_[0]
        return _h
    hook_node = l2_node_block.register_forward_pre_hook(make_pre_hook("node"))
    hook_edge = l2_edge_block.register_forward_pre_hook(make_pre_hook("edge"))

    # ---- Data ----
    ds_root = cfg.dataset.data_root
    def make_ds(split, aug):
        return NuScenesRadarDepthDataset(
            data_root=ds_root,
            split_file=os.path.join(ds_root, "splits", f"{split}.txt"),
            dense_gt_dir=cfg.dataset.dense_gt_dir,
            radar_3d_dir=cfg.dataset.radar_3d_dir,
            night_ids_file=cfg.dataset.night_ids_file,
            max_radar_points=k_max,
            max_depth=max_depth,
            min_depth=float(cfg.dataset.min_depth),
            augmentation=aug,
        )
    train_ds = make_ds("train", aug=True)
    val_ds = make_ds("val", aug=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True,
                              persistent_workers=(args.num_workers > 0))
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=max(2, args.num_workers // 2),
                            pin_memory=True,
                            persistent_workers=(args.num_workers > 0))
    print(f"  Train samples: {len(train_ds):,}   Val samples: {len(val_ds):,}")

    # ---- Optim ----
    trainable = [p for r in routers.values() for p in r.parameters()]
    opt = torch.optim.Adam(trainable, lr=args.lr, weight_decay=args.weight_decay)

    log_lines = []
    history = []
    best_acc = 0.0

    def log(msg):
        print(msg); log_lines.append(msg)

    log(f"Config: mode={args.radar_mode}  epochs={args.epochs}  "
        f"batch_size={args.batch_size}  lr={args.lr}  wd={args.weight_decay}")

    for epoch in range(args.epochs):
        for r in routers.values(): r.train()
        t0 = time.time()
        train_losses = []
        pbar = tqdm(train_loader, desc=f"ep{epoch:02d} train", leave=False)
        for it, batch in enumerate(pbar):
            rgb = batch["rgb_norm"].to(device, non_blocking=True)
            rp = batch["radar_points"].to(device, non_blocking=True)
            rm = batch["radar_mask"].to(device, non_blocking=True)
            dgd = batch["depth_gt_dense"].to(device, non_blocking=True)
            H_img, W_img = rgb.shape[-2:]

            with torch.no_grad():
                _ = baseline(rgb, rp, rm)
            feat_node = captured["node"].detach()
            feat_edge = captured["edge"].detach()

            def _run(name, feat):
                H_tok, W_tok = feat.shape[-2:]
                r = routers[name]
                radar_map = build_radar_map(args.radar_mode, rp, rm,
                                             H_tok, W_tok, H_img, W_img,
                                             max_depth, k_max,
                                             use_strip, use_knn)
                if radar_map is None:
                    return r(feat)
                out = r(feat, radar_map)
                return out[0] if isinstance(out, tuple) else out

            logits_node = _run("node", feat_node)
            logits_edge = _run("edge", feat_edge)

            frac_node = compute_frac(dgd, logits_node.shape[-2:], BINS)
            frac_edge = compute_frac(dgd, logits_edge.shape[-2:], BINS)
            loss = 0.5 * (F.cross_entropy(logits_node, frac_node)
                          + F.cross_entropy(logits_edge, frac_edge))

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            train_losses.append(loss.item())
            if (it + 1) % 100 == 0:
                pbar.set_postfix({"loss": f"{np.mean(train_losses[-100:]):.4f}"})
        train_time = time.time() - t0
        train_loss = float(np.mean(train_losses))

        entry = {"epoch": epoch, "train_loss": train_loss,
                 "train_time_sec": train_time}
        if (epoch + 1) % args.val_every == 0 or epoch == args.epochs - 1:
            t1 = time.time()
            val = evaluate(baseline, routers, captured, val_loader, device,
                           args.radar_mode, use_strip, use_knn, max_depth, k_max)
            val_time = time.time() - t1
            entry["val"] = val
            entry["val_time_sec"] = val_time

            log(f"ep {epoch:02d}  train_loss={train_loss:.4f}  "
                f"({train_time:.0f}s)  |  val overall_acc={val['overall_acc']:.2f}%  "
                f"node={val['per_block']['node']['acc']:.2f}%  "
                f"edge={val['per_block']['edge']['acc']:.2f}%  "
                f"(val {val_time:.0f}s)")
            for name in ["node", "edge"]:
                b = val["per_block"][name]
                r = b["recall"]
                extra = ""
                if "acc_hit_col" in b:
                    extra += (f"  hit={b['acc_hit_col']:.2f}% "
                              f"({b['tokens_hit']/1e6:.2f}M)  "
                              f"nohit={b['acc_nohit_col']:.2f}% "
                              f"({b['tokens_nohit']/1e6:.2f}M)")
                if "gate_mean" in b:
                    extra += f"  gate={b['gate_mean']:.3f}"
                log(f"    {name} recall: near={r[0]:.1f}%  mid={r[1]:.1f}%  "
                    f"far={r[2]:.1f}%  soft_ce={b['soft_ce']:.4f}{extra}")

            if val["overall_acc"] > best_acc:
                best_acc = val["overall_acc"]
                torch.save({
                    "router_node": routers["node"].state_dict(),
                    "router_edge": routers["edge"].state_dict(),
                    "epoch": epoch,
                    "val": val,
                    "baseline_run": args.baseline_run,
                    "config": {
                        "ch_node": ch_node, "ch_edge": ch_edge, "n_experts": K,
                        "radar_mode": args.radar_mode, "radar_ch": rch,
                        "bins": BINS, "lr": args.lr,
                        "batch_size": args.batch_size,
                    },
                }, os.path.join(args.out_dir, "best.pt"))
                log(f"    → saved best (acc={best_acc:.2f}%)")
        else:
            log(f"ep {epoch:02d}  train_loss={train_loss:.4f}  ({train_time:.0f}s)")

        history.append(entry)
        with open(os.path.join(args.out_dir, "history.json"), "w") as f:
            json.dump(history, f, indent=2)
        with open(os.path.join(args.out_dir, "log.txt"), "w") as f:
            f.write("\n".join(log_lines))

    torch.save({
        "router_node": routers["node"].state_dict(),
        "router_edge": routers["edge"].state_dict(),
        "epoch": args.epochs - 1,
        "val": history[-1].get("val"),
        "baseline_run": args.baseline_run,
        "config": {"radar_mode": args.radar_mode, "radar_ch": rch},
    }, os.path.join(args.out_dir, "last.pt"))

    hook_node.remove(); hook_edge.remove()
    log(f"\nDone. Best val overall_acc: {best_acc:.2f}%")
    with open(os.path.join(args.out_dir, "log.txt"), "w") as f:
        f.write("\n".join(log_lines))


if __name__ == "__main__":
    main()
