"""Pyramid Radar fusion with Radar-centered Flash Attention (paper §3.2).

Per layer l ∈ {1,2,3}:
    block A: fuse N_l with image feature F_{2l-1}  (kv_dim = C_l)
    block B: fuse E_l with image feature F_{2l}    (kv_dim = K_max)

Each block runs a Radar-centered cross-attention transformer block:
    queries  = image pixels at level l
    keys / v = radar features (N_l rows / E_l rows)
    mask     = pixels and radar points within |x_pix - x_radar| < a_l

PyTorch's `scaled_dot_product_attention` dispatches to FlashAttention(2) /
memory-efficient kernels when available (paper refs [7, 8]).
"""
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class RadarCenteredAttention(nn.Module):
    """Multi-head Radar-centered cross-attention with horizontal-window mask.

    Attention recording (for visualization):
        Set `module.record_attention = True` on any instance to switch from
        the fast SDPA path to a manual softmax that caches the resulting
        attention weights on `self.last_attn` (B, heads, H, W, K) and the
        keep mask on `self.last_keep` (B, H, W, K). Toggle off for training
        / fast inference. Memory is proportional to B·heads·H·W·K, so use
        small input resolutions and/or deeper layers for large K.
    """

    record_attention: bool = False
    last_attn: torch.Tensor | None = None
    last_keep: torch.Tensor | None = None
    last_hw: tuple | None = None

    def __init__(self, ch: int, kv_dim: int, heads: int, a_l: float) -> None:
        super().__init__()
        # Allow heads that don't divide ch by reducing heads to a divisor.
        while ch % heads != 0 and heads > 1:
            heads -= 1
        self.ch = ch
        self.heads = heads
        self.head_dim = ch // heads
        self.scale = self.head_dim ** -0.5
        self.a_l = a_l
        self.q_proj = nn.Linear(ch, ch)
        self.k_proj = nn.Linear(kv_dim, ch)
        self.v_proj = nn.Linear(kv_dim, ch)
        self.out_proj = nn.Linear(ch, ch)

    def forward(
        self,
        feat: torch.Tensor,            # (B, C_l, H_l, W_l)
        kv: torch.Tensor,              # (B, K, kv_dim)
        radar_x_orig: torch.Tensor,    # (B, K) horizontal coord in input-image space
        radar_mask: torch.Tensor,      # (B, K) bool
        image_w: int,                  # actual input-image width (per forward call)
    ) -> torch.Tensor:
        B, C, H, W = feat.shape
        K = kv.shape[1]
        device = feat.device

        # Build (B, W, K) horizontal-window mask, broadcast across rows.
        scale = W / float(image_w)
        x_p = radar_x_orig * scale                                    # (B, K)
        a_pix = self.a_l * scale
        col = torch.arange(W, device=device).float()                  # (W,)
        col_mask = (col[None, :, None] - x_p[:, None, :]).abs() < a_pix
        col_mask = col_mask & radar_mask[:, None, :]                  # zero out padded keys

        # Queries from image pixels.
        x = feat.permute(0, 2, 3, 1).reshape(B, H * W, C)
        q = self.q_proj(x).view(B, H * W, self.heads, self.head_dim).transpose(1, 2)   # (B,h,HW,d)
        k = self.k_proj(kv).view(B, K, self.heads, self.head_dim).transpose(1, 2)      # (B,h,K,d)
        v = self.v_proj(kv).view(B, K, self.heads, self.head_dim).transpose(1, 2)

        # Build (B, HW, K) keep mask (expand col_mask along H rows).
        attn_keep = col_mask[:, None, :, :].expand(-1, H, -1, -1).reshape(B, H * W, K)
        any_valid_pix = attn_keep.any(dim=-1, keepdim=True)
        # Safe rows so SDPA doesn't NaN on fully-masked rows; we zero output later.
        safe_keep = torch.where(any_valid_pix.expand_as(attn_keep),
                                attn_keep, torch.ones_like(attn_keep))
        attn_bias = torch.zeros_like(safe_keep, dtype=q.dtype)
        attn_bias = attn_bias.masked_fill(~safe_keep, float("-inf")).unsqueeze(1)

        if self.record_attention:
            # Manual path: compute & cache softmax(QK^T / √d) for visualization.
            scores = (q @ k.transpose(-1, -2)) * self.scale            # (B, h, HW, K)
            scores = scores + attn_bias                                 # broadcast head dim
            attn = torch.softmax(scores, dim=-1)
            out = attn @ v                                              # (B, h, HW, d)
            self.last_attn = attn.detach().reshape(B, self.heads, H, W, K).cpu()
            self.last_keep = attn_keep.detach().reshape(B, H, W, K).cpu()
            self.last_hw = (H, W)
        else:
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_bias)
        out = out.transpose(1, 2).reshape(B, H * W, C)
        out = self.out_proj(out)
        # Pixels with no valid radar in window → zero contribution (residual passthrough).
        out = out * any_valid_pix.to(out.dtype)
        return out.transpose(1, 2).reshape(B, C, H, W)


class _FusionBlock(nn.Module):
    """Pre-LN transformer block: RadarCenteredAttention + 1×1 MLP, both residual."""

    def __init__(self, ch: int, kv_dim: int, heads: int, a_l: float, mlp_ratio: float = 2.0) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(1, ch)
        self.attn = RadarCenteredAttention(ch, kv_dim, heads, a_l)
        self.norm2 = nn.GroupNorm(1, ch)
        hidden = int(ch * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Conv2d(ch, hidden, 1),
            nn.GELU(),
            nn.Conv2d(hidden, ch, 1),
        )

    def forward(self, feat, kv, radar_x_orig, radar_mask, image_w):
        feat = feat + self.attn(self.norm1(feat), kv, radar_x_orig, radar_mask, image_w)
        feat = feat + self.mlp(self.norm2(feat))
        return feat


class PyramidRadarFusion(nn.Module):
    """Hierarchical fusion module across L=3 layers and 6 image scales."""

    def __init__(
        self,
        radar_channels: Tuple[int, int, int] = (64, 128, 512),
        img_channels: Tuple[int, ...] = (64, 64, 128, 256, 512, 512),
        a_l: Tuple[float, float, float] = (48.0, 32.0, 16.0),
        max_radar_points: int = 128,
        heads: int = 4,
    ) -> None:
        super().__init__()
        assert len(img_channels) == 2 * len(radar_channels), \
            f"img_channels must have 2L entries (got {len(img_channels)})"
        self.L = len(radar_channels)
        # block A (N_l ↔ F_{2l-1}): kv_dim = C_l
        self.node_blocks = nn.ModuleList([
            _FusionBlock(ch=img_channels[2 * l],
                         kv_dim=radar_channels[l],
                         heads=heads, a_l=a_l[l])
            for l in range(self.L)
        ])
        # block B (E_l ↔ F_{2l}): kv_dim = K_max
        self.edge_blocks = nn.ModuleList([
            _FusionBlock(ch=img_channels[2 * l + 1],
                         kv_dim=max_radar_points,
                         heads=heads, a_l=a_l[l])
            for l in range(self.L)
        ])

    def forward(
        self,
        feats: List[torch.Tensor],
        N_list: List[torch.Tensor],
        E_list: List[torch.Tensor],
        radar_points: torch.Tensor,
        radar_mask: torch.Tensor,
        image_w: int,
    ) -> List[torch.Tensor]:
        # Channel 3 = x_pix (image-plane horizontal) for the radar-centered
        # attention's horizontal-window mask. The 3D camera coords (channels
        # 0..2) are already consumed upstream by the radar encoder.
        radar_x = radar_points[:, :, 3]
        out = list(feats)
        for l in range(self.L):
            f_odd = feats[2 * l]
            f_even = feats[2 * l + 1]
            out[2 * l] = self.node_blocks[l](f_odd, N_list[l], radar_x, radar_mask, image_w)
            out[2 * l + 1] = self.edge_blocks[l](f_even, E_list[l], radar_x, radar_mask, image_w)
        return out
