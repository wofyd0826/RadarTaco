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
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _pad_token_index(sel: torch.Tensor):
    """(B, T) bool → ((B, N) long token indices, (B, N) bool padding mask).

    N is the largest per-sample selection count, so every sample is packed into
    the same dense layout and the attention stays a single batched call instead
    of a Python loop over the batch.
    """
    B, T = sel.shape
    counts = sel.sum(dim=-1)
    n_max = int(counts.max().item())
    idx = torch.zeros(B, max(n_max, 1), dtype=torch.long, device=sel.device)
    valid = torch.zeros_like(idx, dtype=torch.bool)
    if n_max == 0:
        return idx, valid
    b_i, t_i = sel.nonzero(as_tuple=True)
    rank = (sel.cumsum(dim=-1) - 1)[b_i, t_i]        # position within its row
    idx[b_i, rank] = t_i
    valid[b_i, rank] = True
    return idx, valid


def _masked_group_norm(tok: torch.Tensor, valid: torch.Tensor,
                       gn: nn.GroupNorm) -> torch.Tensor:
    """`GroupNorm(1, C)` for a padded token list, ignoring padded slots.

    `GroupNorm(1, C)` on (B, C, H, W) normalises over C*H*W per sample, i.e.
    over every token and channel. Doing the same over the selected tokens makes
    the expert normalise across its own routed set. With every token selected
    the statistics are identical to the dense module, so the sparse path stays
    exact in that case.
    """
    m = valid[..., None].to(tok.dtype)                       # (B, N, 1)
    n = (valid.sum(dim=-1, dtype=tok.dtype) * tok.shape[-1]).clamp_min(1.0)
    n = n[:, None, None]
    x = tok * m
    mean = x.sum(dim=(1, 2), keepdim=True) / n
    var = (((tok - mean) * m) ** 2).sum(dim=(1, 2), keepdim=True) / n
    out = (tok - mean) * torch.rsqrt(var + gn.eps)
    return (out * gn.weight + gn.bias) * m


def _scatter_tokens(tok: torch.Tensor, idx: torch.Tensor,
                    valid: torch.Tensor, T: int) -> torch.Tensor:
    """(B, N, C) token values → (B, T, C), zeros where nothing was selected.

    Padded slots are routed to a scratch row at index T so they cannot race
    with a genuine write to token 0.
    """
    B, _, C = tok.shape
    idx_safe = torch.where(valid, idx, torch.full_like(idx, T))
    buf = torch.zeros(B, T + 1, C, dtype=tok.dtype, device=tok.device)
    buf.scatter_(1, idx_safe[..., None].expand(-1, -1, C), tok)
    return buf[:, :T]


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
        # "No radar applies to me" token. The softmax normalises over the radar
        # axis, so without it every pixel inside the window must spend a full
        # unit of attention mass on some radar point — even though only ~10% of
        # in-window pixels actually match one in depth. The null key gives the
        # remaining 90% somewhere to put that mass; `v_null` starts at zero so
        # attending to it initially contributes no message.
        self.k_null = nn.Parameter(torch.zeros(1, 1, ch))
        self.v_null = nn.Parameter(torch.zeros(1, 1, ch))
        nn.init.normal_(self.k_null, std=0.02)

    def forward(
        self,
        feat: torch.Tensor,            # (B, C_l, H_l, W_l)
        kv: torch.Tensor,              # (B, K, kv_dim)
        radar_x_orig: torch.Tensor,    # (B, K) horizontal coord in input-image space
        radar_mask: torch.Tensor,      # (B, K) bool
        image_w: int,                  # actual input-image width (per forward call)
        sel_idx: Optional[torch.Tensor] = None,    # (B, N) flat token indices
        sel_valid: Optional[torch.Tensor] = None,  # (B, N) bool — padding of sel_idx
        q_tok: Optional[torch.Tensor] = None,      # (B, N, C) pre-normalised queries
        hw: Optional[Tuple[int, int]] = None,      # feature size when q_tok is used
    ) -> torch.Tensor:
        """Returns (B, C, H, W) normally, or (B, N, C) when `sel_idx` is given.

        `sel_idx` restricts the queries to a subset of pixels — used by
        `MoEFusionBlock` so an expert only pays for the tokens routed to it.
        Queries are independent, so the sparse path is exact for those tokens.
        """
        if q_tok is None:
            B, C, H, W = feat.shape
            device = feat.device
        else:
            B, _, C = q_tok.shape
            H, W = hw
            device = q_tok.device
        K = kv.shape[1]

        # Build (B, W, K) horizontal-window mask, broadcast across rows.
        scale = W / float(image_w)
        x_p = radar_x_orig * scale                                    # (B, K)
        # `a_l` is in input-image pixels, so at deep levels it shrinks below one
        # feature column (0.5 at f5, 0.25 at f6) and half the radar points stop
        # matching any column at all. Floor it at one column so every point
        # keeps at least its own column.
        a_pix = max(self.a_l * scale, 1.0)
        col = torch.arange(W, device=device).float()                  # (W,)
        col_mask = (col[None, :, None] - x_p[:, None, :]).abs() < a_pix
        col_mask = col_mask & radar_mask[:, None, :]                  # zero out padded keys

        # Queries from image pixels — all of them, or just the routed subset.
        if q_tok is not None:
            xq = q_tok
            col_of = sel_idx % W                                          # (B, N)
        elif sel_idx is None:
            xq = feat.permute(0, 2, 3, 1).reshape(B, H * W, C)
            col_of = (torch.arange(H * W, device=device) % W)[None].expand(B, -1)
        else:
            x = feat.permute(0, 2, 3, 1).reshape(B, H * W, C)
            xq = torch.gather(x, 1, sel_idx[..., None].expand(-1, -1, C))
            col_of = sel_idx % W
        N = xq.shape[1]
        q = self.q_proj(xq).view(B, N, self.heads, self.head_dim).transpose(1, 2)      # (B,h,N,d)
        k = self.k_proj(kv).view(B, K, self.heads, self.head_dim).transpose(1, 2)      # (B,h,K,d)
        v = self.v_proj(kv).view(B, K, self.heads, self.head_dim).transpose(1, 2)

        # Append the null key/value — always visible, never masked.
        k_null = self.k_null.view(1, 1, self.heads, self.head_dim).transpose(1, 2)
        v_null = self.v_null.view(1, 1, self.heads, self.head_dim).transpose(1, 2)
        k = torch.cat([k, k_null.expand(B, -1, -1, -1).to(k.dtype)], dim=2)   # (B,h,K+1,d)
        v = torch.cat([v, v_null.expand(B, -1, -1, -1).to(v.dtype)], dim=2)

        # Per-token keep mask: each token inherits its column's mask.
        attn_keep = torch.gather(col_mask, 1, col_of[..., None].expand(-1, -1, K))  # (B,N,K)
        any_valid_pix = attn_keep.any(dim=-1, keepdim=True)
        # The null column is always open, so no row is fully -inf and softmax
        # cannot NaN — the previous `safe_keep` workaround is unnecessary.
        bias_radar = torch.zeros_like(attn_keep, dtype=q.dtype)
        bias_radar = bias_radar.masked_fill(~attn_keep, float("-inf"))
        bias_null = torch.zeros(B, N, 1, dtype=q.dtype, device=device)
        attn_bias = torch.cat([bias_radar, bias_null], dim=-1).unsqueeze(1)

        if self.record_attention:
            # Manual path: compute & cache softmax(QK^T / √d) for visualization.
            # Cached maps have K+1 columns; the last one is the null token.
            scores = (q @ k.transpose(-1, -2)) * self.scale            # (B, h, N, K+1)
            scores = scores + attn_bias                                 # broadcast head dim
            attn = torch.softmax(scores, dim=-1)
            out = attn @ v                                              # (B, h, N, d)
            if sel_idx is None:                     # recording only makes sense dense
                self.last_attn = attn.detach().reshape(B, self.heads, H, W, K + 1).cpu()
                self.last_keep = attn_keep.detach().reshape(B, H, W, K).cpu()
                self.last_hw = (H, W)
        else:
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_bias)
        out = out.transpose(1, 2).reshape(B, N, C)
        out = self.out_proj(out)
        # Pixels with no valid radar in window → zero contribution (residual passthrough).
        out = out * any_valid_pix.to(out.dtype)
        if sel_idx is None and q_tok is None:
            return out.transpose(1, 2).reshape(B, C, H, W)
        # Padded slots must not carry a message back to the scatter.
        return out * sel_valid[..., None].to(out.dtype)


class _FusionBlock(nn.Module):
    """Pre-LN transformer block: RadarCenteredAttention + 1×1 MLP, both residual."""

    def __init__(self, ch: int, kv_dim: int, heads: int, a_l: float, mlp_ratio: float = 2.0) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(1, ch)
        # The query stream is normalised, so the context has to be too
        # (Perceiver-style). Raw `E_l` rows are softmax outputs with elements
        # around 1/K ≈ 0.009 — two orders below the normalised queries — which
        # let `v_proj`'s bias dominate the value vectors and made them nearly
        # identical across radar points.
        self.norm_kv = nn.LayerNorm(kv_dim)
        self.attn = RadarCenteredAttention(ch, kv_dim, heads, a_l)
        self.norm2 = nn.GroupNorm(1, ch)
        hidden = int(ch * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Conv2d(ch, hidden, 1),
            nn.GELU(),
            nn.Conv2d(hidden, ch, 1),
        )

    def forward(self, feat, kv, radar_x_orig, radar_mask, image_w, sel=None):
        """`sel`: optional (B, H, W) bool. When given the whole block — norms,
        attention and MLP — runs only on those tokens, and the returned tensor
        equals `feat` outside them, i.e. the block's delta is confined to the
        selected positions.

        The norms use `_masked_group_norm`, so an expert normalises over the
        token set routed to it. With every token selected that reduces to the
        dense `GroupNorm(1, C)`, keeping the two paths consistent.
        """
        kv = self.norm_kv(kv)
        if sel is None:
            feat = feat + self.attn(self.norm1(feat), kv,
                                    radar_x_orig, radar_mask, image_w)
            feat = feat + self.mlp(self.norm2(feat))
            return feat

        B, C, H, W = feat.shape
        flat = feat.permute(0, 2, 3, 1).reshape(B, H * W, C)
        idx, valid = _pad_token_index(sel.reshape(B, H * W))
        xt = torch.gather(flat, 1, idx[..., None].expand(-1, -1, C))       # (B, N, C)

        n1 = _masked_group_norm(xt, valid, self.norm1)
        h = xt + self.attn(None, kv, radar_x_orig, radar_mask, image_w,
                           sel_idx=idx, sel_valid=valid, q_tok=n1, hw=(H, W))
        n2 = _masked_group_norm(h, valid, self.norm2)
        h = h + self.mlp(n2.transpose(1, 2).unsqueeze(-1)).squeeze(-1).transpose(1, 2)

        delta = (h - xt) * valid[..., None].to(h.dtype)
        return feat + _scatter_tokens(delta, idx, valid, H * W).transpose(1, 2).reshape(B, C, H, W)


class _ReducedExpertBlock(nn.Module):
    """Halved-capacity expert: I/O projections around an internal `_FusionBlock`.

    Preserves the two invariants the outer `MoEFusionBlock` relies on:
      • output = feat + delta  (so gated summation gives the correct residual)
      • sel-masked path       (dispatch to selected tokens only)

    Internal channel dim is `int(ch * expert_ch_ratio)`. Compute + parameters
    of the inner block scale ~ratio²; the 1×1 I/O projections add
    `2 · ch · ch_r` params. With ch=512, ratio=0.5 the wrapper cost is
    ~0.26M vs the internal block's ~0.52M — total ~0.79M per expert against
    ~2.1M for a full `_FusionBlock`. Two halved experts (~1.57M active) come
    in slightly under one full expert (~2.1M active); we treat this as
    "approximately matched" for top_k=1 → top_k=2 comparisons.
    """

    def __init__(self, ch: int, kv_dim: int, heads: int, a_l: float,
                 mlp_ratio: float = 2.0, expert_ch_ratio: float = 0.5) -> None:
        super().__init__()
        ch_r = max(1, int(ch * expert_ch_ratio))
        self.proj_in = nn.Conv2d(ch, ch_r, 1)
        self.proj_out = nn.Conv2d(ch_r, ch, 1)
        self.block = _FusionBlock(ch_r, kv_dim, heads, a_l, mlp_ratio=mlp_ratio)

    def forward(self, feat, kv, radar_x_orig, radar_mask, image_w, sel=None):
        # Compute delta at reduced dim, project it back, add to full-dim feat.
        # `sel` masking is applied inside `_FusionBlock` (reduced dim), and
        # again after `proj_out` in case proj_out's bias would otherwise leak
        # non-zero delta at unselected positions.
        red = self.proj_in(feat)                                    # (B, ch_r, H, W)
        red_out = self.block(red, kv, radar_x_orig, radar_mask, image_w, sel=sel)
        delta_r = red_out - red                                     # (B, ch_r, H, W)
        delta = self.proj_out(delta_r)                              # (B, ch,   H, W)
        if sel is not None:
            delta = delta * sel.unsqueeze(1).to(delta.dtype)
        return feat + delta


class _DPTFusionBlock(nn.Module):
    """DPT-style RefineNet fusion block used by `DPTRouter`.

    Takes a coming-up-from-below feature and (optionally) a same-resolution
    skip from a shallower encoder layer, refines both with residual conv
    units, and 1x1-projects the sum. Mirrors `DepthBin.head_dpt.FusionBlock`
    exactly so behaviour is comparable.
    """

    def __init__(self, channels: int, has_skip: bool = True) -> None:
        super().__init__()
        self.unit1 = _DPTResidualConvUnit(channels) if has_skip else None
        self.unit2 = _DPTResidualConvUnit(channels)
        self.out_conv = nn.Conv2d(channels, channels, 1)

    def forward(self, x: torch.Tensor,
                skip: Optional[torch.Tensor] = None) -> torch.Tensor:
        if skip is not None:
            assert self.unit1 is not None, \
                "this _DPTFusionBlock was built without a skip path"
            # size=skip.shape[-2:] avoids the 1-px offset that scale_factor=2
            # introduces at odd resolutions.
            x = F.interpolate(x, size=skip.shape[-2:],
                              mode="bilinear", align_corners=False)
            x = x + self.unit1(skip)
        x = self.unit2(x)
        return self.out_conv(x)


class _DPTResidualConvUnit(nn.Module):
    """3x3 → BN → ReLU → 3x3 → BN residual unit (DPT RefineNet standard)."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(x)
        out = self.bn1(self.conv1(out))
        out = self.relu(out)
        out = self.bn2(self.conv2(out))
        return out + x


class DPTRouter(nn.Module):
    """Multi-scale DPT-style router for MoE gate.

    Takes several image_encoder feature maps at DIFFERENT native resolutions
    (e.g., F2..F5 = 1/4, 1/8, 1/16, 1/32 for ResNet-18) and reassembles
    them into a common `fusion_ch`-channel representation. RefineNet-style
    bottom-up fusion (deepest first, progressively adding shallower skips)
    yields a rich per-token feature which a shallow classifier turns into K
    expert logits at the token grid.

    Compared to the single-scale `mlp3x3` router, this exposes the routing
    decision to both fine-grained cues (near-field texture from F2/F3) and
    scene-level structure (horizon/sky from F5) — the exact discriminative
    signals that the depth-bin partition ([0,20], [20,50], [50,100]) hinges
    on.

    Args:
        in_channels: channel counts of the reassembly source features,
                     ordered SHALLOW → DEEP (e.g., [64, 128, 256, 512]).
        n_experts:   number of gate outputs K.
        token_hw_ref: which source index defines the OUTPUT token grid.
                     By default the deepest layer (last index). All source
                     features are interpolated to this feature's spatial
                     size before entering the fusion cascade.
        fusion_ch:   internal channel dim used across projections + fusion
                     units. 128 keeps total DPT router cost close to the
                     mlp3x3 baseline (~1.5M params) while adding multi-
                     scale integration.
    """

    def __init__(self, in_channels: List[int], n_experts: int,
                 token_hw_ref: int = -1, fusion_ch: int = 128) -> None:
        super().__init__()
        assert len(in_channels) >= 2, \
            f"DPTRouter needs >=2 source layers, got {len(in_channels)}"
        self.n_sources = len(in_channels)
        self.token_hw_ref = (token_hw_ref
                             if token_hw_ref >= 0
                             else self.n_sources + token_hw_ref)
        self.projections = nn.ModuleList([
            nn.Conv2d(c, fusion_ch, 1) for c in in_channels
        ])
        # `n_sources` fusion blocks; fusions[0] handles the deepest source
        # alone (no skip); subsequent ones add progressively shallower skips.
        self.fusions = nn.ModuleList([
            _DPTFusionBlock(fusion_ch, has_skip=(i > 0))
            for i in range(self.n_sources)
        ])
        self.classifier = nn.Sequential(
            nn.Conv2d(fusion_ch, fusion_ch // 2, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(fusion_ch // 2, n_experts, 1),
        )

    def forward(self, feats: List[torch.Tensor],
                token_hw: Optional[Tuple[int, int]] = None) -> torch.Tensor:
        """Args:
            feats:    list of source features (shallow → deep) matching
                      `in_channels` order.
            token_hw: target token grid. If None, uses the resolution of
                      `feats[self.token_hw_ref]`. Overridable so a caller
                      can force a specific grid (e.g., the node/edge block's
                      own H×W) independent of source layer choice.
        Returns:
            (B, n_experts, H_tok, W_tok) logits.
        """
        assert len(feats) == self.n_sources, \
            f"DPTRouter expected {self.n_sources} feats, got {len(feats)}"
        if token_hw is None:
            token_hw = feats[self.token_hw_ref].shape[-2:]

        # Project + resize each source to the token grid.
        scaled = []
        for proj, f in zip(self.projections, feats):
            p = proj(f)
            if p.shape[-2:] != token_hw:
                p = F.interpolate(p, size=token_hw,
                                  mode="bilinear", align_corners=False)
            scaled.append(p)

        # Bottom-up fusion (deepest → shallowest). fusions[0] takes deepest
        # alone; fusions[i>0] adds scaled[-1-i] as the next-shallower skip.
        x = self.fusions[0](scaled[-1])
        for i in range(1, self.n_sources):
            x = self.fusions[i](x, scaled[-1 - i])
        return self.classifier(x)


class MoEFusionBlock(nn.Module):
    """Depth-specialised MoE variant of `_FusionBlock` — PER-TOKEN routing.

    Structure (DriveMoE analogue, adapted to depth's per-pixel nature):
      • 1 shared expert — always active, provides baseline fusion at every
        spatial position.
      • K non-shared experts (default K=3, near/mid/far) — mixed per token
        by a router that produces a spatial gate map (B, K, H, W).

    Router (image-only, PER-TOKEN):
      • Input: image feature at this scale, shape (B, C, H, W).
      • Head: 1×1 Conv2d(C → K). Output: (B, K, H, W) logits, softmaxed
        along K to produce a per-position gate.
      • Rationale: depth is a per-pixel property (a single frame contains
        near+mid+far pixels), so routing must also be per-pixel. Empirical
        argmax-per-sample was severely imbalanced (91.8/8.1/0.1) — see
        tests/check_moe_gt_routing.py; per-token GT restores balance to
        the natural pixel distribution (~51/30/18).

    Two-stage training (DriveMoE §2.5):
      stage=1: teacher-forcing. `depth_gt_dense: (B, 1, H_full, W_full)`
               is bucketized PER PIXEL by `self.bins`, then each token's
               GT is the MAJORITY-VOTE bin over its receptive field
               (~1000 pixels for F_4, ~3800 for F_5). This avoids the
               averaging artifact where near+far bimodal tokens would
               land in 'mid' — see tests/verify_per_token_routing.py.
               Gate = one_hot(gt) per token; router logits are exposed
               for CE supervision.
      stage=2: self-routing. Gate = softmax(logits) along K, per token;
               optional sparsification to top_k experts per token.

    Output: gated non-shared mixture + shared-expert output (always added).
    """

    def __init__(
        self,
        ch: int,
        kv_dim: int,
        heads: int,
        a_l: float,
        mlp_ratio: float = 2.0,
        n_experts: int = 3,
        use_shared: bool = True,
        top_k: Optional[int] = None,
        bins: Tuple[float, ...] = (0.0, 20.0, 50.0, 100.0),
        router_arch: str = "conv1x1",
        expert_ch_ratio: float = 1.0,
        router_gt_type: str = "hard",
        overlap_bins: Optional[Tuple[Tuple[float, float], ...]] = None,
        shared_gate_mode: str = "always_on",
        # DPT-style multi-scale router options. Ignored unless
        # `router_arch == "dpt"`.
        #   router_dpt_source_channels — list of source layer channel counts
        #       (SHALLOW → DEEP). Must match what the caller passes in via
        #       `router_multi_scale_feats` at forward time.
        #   router_dpt_fusion_ch — internal channel dim inside DPTRouter.
        router_dpt_source_channels: Optional[List[int]] = None,
        router_dpt_fusion_ch: int = 128,
        # Two-stage (pre-fusion) MoE options. When `pre_fusion_enabled` is
        # True an EXTRA `_FusionBlock` runs upstream of the router+experts,
        # producing a radar-informed feature that the router (and optionally
        # the experts) consumes instead of the raw image feat.
        #   pre_fusion_enabled       — main switch.
        #   pre_fusion_feed_experts  — Option A when True (experts operate
        #                              on the pre-fused feature, residual
        #                              taken against it). Option B when
        #                              False (router-only pre-fusion; experts
        #                              still see the raw feat).
        #   pre_fusion_ch_ratio      — capacity scaling for the pre-fusion
        #                              block. 1.0 = full ch (~2M params for
        #                              L=2). 0.5 = halved (matches expert).
        #   pre_fusion_router_detach — when True, the pre-fused feature seen
        #                              by the router is detached, so router
        #                              CE gradients never touch the pre-
        #                              fusion weights. Useful when you want
        #                              pre-fusion trained ONLY by downstream
        #                              task loss (Option A) — has no effect
        #                              on Option B, where routing loss is the
        #                              only gradient source anyway.
        pre_fusion_enabled: bool = False,
        pre_fusion_feed_experts: bool = False,
        pre_fusion_ch_ratio: float = 1.0,
        pre_fusion_router_detach: bool = False,
    ) -> None:
        super().__init__()
        self.n_experts = int(n_experts)
        self.top_k = int(top_k) if top_k is not None else None
        # Shared-expert gating mode:
        #   "always_on"        — shared applied unconditionally on top of the
        #                        specialist mixture (v5 behavior, default).
        #   "confidence"       — router uncertainty α ∈ [0, 1] splits weight
        #                        between specialists (α) and shared (1 − α).
        #                        Applied in BOTH training and inference
        #                        (conservation; v6a). Shared receives gradient
        #                        only from uncertain tokens.
        #   "confidence_infer" — training uses always-on (specialists at full
        #                        gate, shared at full weight → shared learns
        #                        as a generalist on ALL tokens); inference
        #                        switches to conservation (spec × α, shared
        #                        1 − α). v6b — shared trained broadly but
        #                        acts as confidence-gated fallback at eval.
        #   "confidence_infer_disjoint"
        #                      — training path: `mixed` = specialists only
        #                        (shared_weight ← 0), and `mixed_shared_only`
        #                        = feat + shared_delta (via `return_shared_only`).
        #                        Combined with `moe_shared_aux=True` this
        #                        yields TWO disjoint depth heads trained by
        #                        separate task losses: main (spec-only) and
        #                        aux (shared-only). Inference switches to
        #                        conservation (spec × α, shared 1 − α) — the
        #                        two paths are then MIXED. v6c.
        self.shared_gate_mode = str(shared_gate_mode)
        assert self.shared_gate_mode in ("always_on", "confidence",
                                          "confidence_infer",
                                          "confidence_infer_disjoint"), \
            self.shared_gate_mode
        # Expert capacity scaling. `expert_ch_ratio < 1.0` swaps each
        # `_FusionBlock` for a `_ReducedExpertBlock` that runs at reduced
        # internal channel dim — used to hold total ACTIVE params roughly
        # constant when switching top_k=1 → top_k=2.
        self.expert_ch_ratio = float(expert_ch_ratio)
        # Router GT format for teacher-forcing + CE loss:
        #   "hard"    → majority-vote bin index (B, H, W) int64 — softmax + CE
        #   "soft"    → per-token pixel-bin fraction (B, K, H, W) float, sum=1
        #               — softmax + soft CE
        #   "overlap" → per-token OVERLAP membership (B, K, H, W) float, per-bin
        #               [0,1] independent (sum can be > 1). Requires `overlap_bins`
        #               [(lo0, hi0), (lo1, hi1), ...] with last bin open-ended.
        #               Trained with SIGMOID + BCE per bin. Self-routing gate is
        #               sigmoid + renormalize.
        self.router_gt_type = str(router_gt_type)
        assert self.router_gt_type in ("hard", "soft", "overlap"), self.router_gt_type
        if self.router_gt_type == "overlap":
            assert overlap_bins is not None, \
                "overlap_bins must be provided when router_gt_type='overlap'"
            assert len(overlap_bins) == int(n_experts), \
                f"overlap_bins must have {n_experts} entries, got {len(overlap_bins)}"

        def _mk_expert() -> nn.Module:
            if self.expert_ch_ratio < 1.0:
                return _ReducedExpertBlock(
                    ch, kv_dim, heads, a_l,
                    mlp_ratio=mlp_ratio,
                    expert_ch_ratio=self.expert_ch_ratio,
                )
            return _FusionBlock(ch, kv_dim, heads, a_l, mlp_ratio=mlp_ratio)

        self.experts = nn.ModuleList([_mk_expert() for _ in range(self.n_experts)])
        self.shared = _mk_expert() if use_shared else None
        # Image-only PER-TOKEN router. Output shape (B, K, H, W).
        # - "conv1x1"  : single 1×1 linear classifier (legacy, minimal).
        # - "mlp3x3"   : 3×3 conv → GELU → 3×3 conv → GELU → 1×1 (Option C,
        #                spatial context + non-linearity, ~1.5M params).
        self.router_arch = router_arch
        if router_arch == "conv1x1":
            self.router = nn.Conv2d(ch, self.n_experts, kernel_size=1)
        elif router_arch == "mlp3x3":
            h1 = min(256, ch // 2)
            h2 = h1 // 2
            self.router = nn.Sequential(
                nn.Conv2d(ch, h1, kernel_size=3, padding=1),
                nn.GELU(),
                nn.Conv2d(h1, h2, kernel_size=3, padding=1),
                nn.GELU(),
                nn.Conv2d(h2, self.n_experts, kernel_size=1),
            )
        elif router_arch == "deterministic_rel_depth":
            # NO learned router network. Gate is computed at forward time
            # directly from the precomputed rel_depth (Marigold, [0, 1]) by
            # binning each pixel against `self.bins` and pooling to the
            # token grid (pixel_fraction). Router CE loss should be OFF
            # (nothing to train). Requires `rel_depth` argument to forward().
            self.router = None
        elif router_arch == "dpt":
            # DPT-style multi-scale router. Consumes several image_encoder
            # feature maps (via `router_multi_scale_feats` at forward time)
            # and RefineNet-fuses them into K logits at the block's token
            # grid. Requires `router_dpt_source_channels` to be passed at
            # construction so we can pre-build the projection weights.
            assert router_dpt_source_channels is not None, (
                "router_arch='dpt' requires router_dpt_source_channels "
                "(list of source-layer channel counts, shallow → deep)."
            )
            self.router = DPTRouter(
                in_channels=list(router_dpt_source_channels),
                n_experts=self.n_experts,
                fusion_ch=int(router_dpt_fusion_ch),
            )
        else:
            raise ValueError(f"Unknown router_arch: {router_arch!r}")
        # Bin edges for teacher-forcing GT (len K+1 → K bins). Registered
        # as a buffer so it moves with .to(device) automatically.
        # Optional pre-fusion block (see `pre_fusion_*` kwargs). Runs one
        # generic radar-image fusion BEFORE the router so routing sees a
        # radar-informed feature. Full-cap when `pre_fusion_ch_ratio == 1.0`
        # (identical shape to `_FusionBlock`), halved otherwise (wraps the
        # inner block with `_ReducedExpertBlock`'s proj_in / proj_out).
        self.pre_fusion_enabled = bool(pre_fusion_enabled)
        self.pre_fusion_feed_experts = bool(pre_fusion_feed_experts)
        self.pre_fusion_router_detach = bool(pre_fusion_router_detach)
        if self.pre_fusion_enabled:
            r = float(pre_fusion_ch_ratio)
            if r >= 1.0:
                self.pre_fusion = _FusionBlock(
                    ch, kv_dim, heads, a_l, mlp_ratio=mlp_ratio)
            else:
                self.pre_fusion = _ReducedExpertBlock(
                    ch, kv_dim, heads, a_l,
                    mlp_ratio=mlp_ratio, expert_ch_ratio=r)
        else:
            self.pre_fusion = None
        assert len(bins) == self.n_experts + 1, \
            f"bins must have n_experts+1 edges (got {len(bins)}, need {self.n_experts + 1})"
        self.register_buffer("bins", torch.tensor(bins, dtype=torch.float32))
        if self.router_gt_type == "overlap":
            # Store overlap ranges as (K, 2) float buffer: [[lo0, hi0], ...].
            # Last row's hi is treated as open-ended (>=) at target computation.
            self.register_buffer(
                "overlap_bins",
                torch.tensor(overlap_bins, dtype=torch.float32),
            )

    def _depth_to_bin(self, d: torch.Tensor) -> torch.Tensor:
        """(B, 1, H, W) metric depth → (B, H, W) int64 bin indices.

        Partition per pixel into [bins[i], bins[i+1]) for i ∈ 0..K-1;
        pixels ≥ bins[-1] (== max_depth) land in the last bin. Sky/invalid
        cells that dense_gt fills with max_depth (100 m) fall into 'far'.
        """
        B, _, H, W = d.shape
        K = self.n_experts
        ds = d.squeeze(1)                               # (B, H, W)
        gt = torch.zeros(B, H, W, dtype=torch.long, device=d.device)
        for i in range(K):
            lo = self.bins[i].item()
            hi = self.bins[i + 1].item()
            in_bin = (ds >= lo) & (ds < hi)
            gt = torch.where(in_bin, gt.new_tensor(i, dtype=torch.long), gt)
        gt = torch.where(ds >= self.bins[-1].item(),
                         gt.new_tensor(K - 1, dtype=torch.long), gt)
        return gt

    def forward(
        self,
        feat: torch.Tensor,
        kv: torch.Tensor,
        radar_x_orig: torch.Tensor,
        radar_mask: torch.Tensor,
        image_w: int,
        depth_gt_dense: Optional[torch.Tensor] = None,
        teacher_force: bool = True,
        return_shared_only: bool = False,
        rel_depth: Optional[torch.Tensor] = None,
        force_expert_idx: Optional[int] = None,
        router_multi_scale_feats: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            depth_gt_dense: (B, 1, H_full, W_full) full-res dense depth GT.
                            If given, `router_gt` is ALWAYS computed and
                            returned (used by CE loss). Absent → no GT
                            available, self-route only.
            teacher_force:  Controls which distribution drives `gate`.
                            True  → gate = one_hot(router_gt) (stage 1 teacher-forcing)
                            False → gate = softmax(logits) [+top_k] (self-route)
                            Note: router_gt is still returned when
                            `depth_gt_dense` is present, regardless of this
                            flag — enables CE-anchor training during
                            stage-2 self-routing.
            rel_depth:      (B, 1, H_full, W_full) precomputed monocular relative
                            depth (Marigold, [0, 1]). REQUIRED when
                            `router_arch == "deterministic_rel_depth"`.
            router_multi_scale_feats:
                            List of image_encoder feature maps (shallow → deep),
                            used ONLY when `router_arch == "dpt"`. Must match the
                            `router_dpt_source_channels` supplied at construction.
                            All are reassembled to this block's token grid H×W
                            inside `DPTRouter` regardless of their native scales.
        Returns:
            fused          : (B, C, H, W)
            router_logits  : (B, K, H, W) or None (None for deterministic).
            router_gt      : (B, H, W) int64 or None. Present whenever
                             `depth_gt_dense` was supplied.
        """
        B, C, H, W = feat.shape

        # ── Optional pre-fusion (two-stage MoE) ──────────────────────────────
        # When enabled, run one generic radar-image fusion FIRST. Two knobs:
        #   • pre_fusion_feed_experts=True  (Option A): rebind `feat` to the
        #     pre-fused tensor so ALL downstream residuals / expert outputs
        #     are anchored on radar-informed features. Output shape is
        #     unchanged; the "baseline" state before any expert delta is
        #     already fused.
        #   • pre_fusion_feed_experts=False (Option B): experts still see the
        #     raw feat; only the router receives the pre-fused feature. In
        #     this mode the pre-fusion block gets gradient ONLY via router
        #     loss (unless detach flag is set, which disables it entirely —
        #     don't combine detach=True with Option B).
        # router_input tracks what the router sees regardless of the target;
        # detach flag controls gradient flow from router loss back into the
        # pre-fusion weights.
        router_input = feat
        if self.pre_fusion is not None:
            feat_prefused = self.pre_fusion(feat, kv, radar_x_orig,
                                            radar_mask, image_w)
            router_input = (feat_prefused.detach()
                            if self.pre_fusion_router_detach else feat_prefused)
            if self.pre_fusion_feed_experts:
                feat = feat_prefused

        # ── Deterministic rel_depth routing ──────────────────────────────────
        # No learned router. Compute gate directly from rel_depth: bin each
        # pixel against self.bins (defined on rel_depth ∈ [0, 1]) and pool the
        # one-hot map to the token grid → per-token pixel_fraction. This IS
        # the gate (soft) — top_k trims + renormalizes as usual. Skips
        # depth_gt_dense / teacher_force / self-routing entirely.
        if self.router_arch == "deterministic_rel_depth":
            assert rel_depth is not None, \
                "deterministic_rel_depth requires rel_depth to be passed to forward"
            with torch.no_grad():
                rd = rel_depth.float().squeeze(1)                # (B, H_full, W_full)
                pix_bin = torch.zeros_like(rd, dtype=torch.long)
                for i in range(self.n_experts):
                    lo = float(self.bins[i]); hi = float(self.bins[i + 1])
                    in_bin = (rd >= lo) & (rd < hi)
                    pix_bin = torch.where(in_bin, pix_bin.new_tensor(i), pix_bin)
                pix_bin = torch.where(rd >= float(self.bins[-1]),
                                      pix_bin.new_tensor(self.n_experts - 1), pix_bin)
                onehot = F.one_hot(pix_bin, num_classes=self.n_experts) \
                          .permute(0, 3, 1, 2).float()
                frac = F.adaptive_avg_pool2d(onehot, (H, W))     # (B, K, H, W)
            gate = frac.to(feat.dtype)
            if self.top_k is not None and self.top_k < self.n_experts:
                _, idx = gate.topk(self.top_k, dim=1)
                keep = torch.zeros_like(gate).scatter_(1, idx, 1.0)
                gate = gate * keep
                gate = gate / gate.sum(dim=1, keepdim=True).clamp_min(1e-8)
            # Route through the same specialist+shared mixing code below by
            # jumping past the router-logits/teacher-force branches.
            logits = None
            router_gt = None
            # Confidence-based mixing may still apply (shared_gate_mode).
            shared_weight = None
            if self.shared is not None and self.shared_gate_mode != "always_on":
                max_gate = gate.max(dim=1, keepdim=True).values
                K = self.n_experts
                alpha = ((K * max_gate - 1.0) / (K - 1)).clamp(0.0, 1.0)
                if self.shared_gate_mode == "confidence":
                    gate = gate * alpha
                    shared_weight = 1.0 - alpha
                elif self.shared_gate_mode == "confidence_infer":
                    if not self.training:
                        gate = gate * alpha
                        shared_weight = 1.0 - alpha
                elif self.shared_gate_mode == "confidence_infer_disjoint":
                    if not self.training:
                        gate = gate * alpha
                        shared_weight = 1.0 - alpha
                    else:
                        shared_weight = torch.zeros_like(alpha)
            mixed = feat
            for e_i, expert in enumerate(self.experts):
                sel = gate[:, e_i] > 0
                if not bool(sel.any()):
                    continue
                dense = bool(sel.all())
                out_e = expert(feat, kv, radar_x_orig, radar_mask, image_w,
                               sel=None if dense else sel)
                mixed = mixed + gate[:, e_i:e_i + 1] * (out_e - feat)
            mixed_shared_only: Optional[torch.Tensor] = None
            if self.shared is not None:
                shared_delta = (self.shared(feat, kv, radar_x_orig, radar_mask, image_w)
                                - feat)
                if shared_weight is None:
                    mixed = mixed + shared_delta
                else:
                    mixed = mixed + shared_weight * shared_delta
                if return_shared_only:
                    mixed_shared_only = feat + shared_delta
            if return_shared_only:
                return mixed, logits, router_gt, mixed_shared_only
            return mixed, logits, router_gt
        # ──────────────────────────────────────────────────────────────────────

        # DPT router: consume multi-scale image_encoder feats, reassemble to
        # this block's token grid (H, W), classify to K logits. Falls back to
        # the single-scale router when router_arch != "dpt" — behaviour is
        # bit-identical to previous code path for non-DPT archs.
        # Skip router entirely when force_expert_idx is set — the returned
        # logits are unused by the per-spec caller and DPT would otherwise
        # require the extra multi-scale feats to be threaded through
        # `forward_per_spec` for nothing.
        if force_expert_idx is not None:
            logits = None
        elif self.router_arch == "dpt":
            assert router_multi_scale_feats is not None, (
                "router_arch='dpt' requires router_multi_scale_feats to be "
                "passed to MoEFusionBlock.forward."
            )
            logits = self.router(router_multi_scale_feats, token_hw=(H, W))
        else:
            logits = self.router(router_input)                      # (B, K, H, W)

        # Per-specialist "what-if" forward — force one-hot gate on a single
        # expert, skip shared entirely. Used by RadarTaco to produce
        # depth_per_spec[k] during training for the task-aware router loss
        # (see moe_per_spec_aux + w_task_router). router_gt not computed
        # (this path does not train the router directly).
        if force_expert_idx is not None:
            # gate_forced allocated only for the expert dispatch; router was
            # skipped for this path so `logits` is None and unused downstream.
            mixed = feat
            expert = self.experts[force_expert_idx]
            out_e = expert(feat, kv, radar_x_orig, radar_mask, image_w, sel=None)
            mixed = mixed + (out_e - feat)
            # shared skipped (per-spec must reflect specialist-alone behavior).
            return mixed, logits, None

        router_gt: Optional[torch.Tensor] = None
        if depth_gt_dense is not None:
            with torch.no_grad():
                if self.router_gt_type == "overlap":
                    # Per-pixel independent membership in each overlap bin,
                    # then pool → per-token fraction ∈ [0,1] per bin (sum can be > 1).
                    ds = depth_gt_dense.float().squeeze(1)             # (B, H_full, W_full)
                    memb = torch.zeros(B, self.n_experts,
                                       ds.shape[-2], ds.shape[-1],
                                       device=ds.device, dtype=torch.float32)
                    for k in range(self.n_experts):
                        lo = float(self.overlap_bins[k, 0])
                        hi = float(self.overlap_bins[k, 1])
                        if k == self.n_experts - 1:
                            memb[:, k] = (ds >= lo).float()             # last bin open-ended
                        else:
                            memb[:, k] = ((ds >= lo) & (ds < hi)).float()
                    router_gt = F.adaptive_avg_pool2d(memb, (H, W))     # (B, K, H, W)
                else:
                    # Per-token GT as pixel-bin FRACTION; emit depends on gt_type:
                    #   "hard" → argmax → (B, H, W) int64  (majority vote)
                    #   "soft" → frac itself → (B, K, H, W) float
                    pix_bin = self._depth_to_bin(depth_gt_dense.float())
                    onehot = F.one_hot(pix_bin, num_classes=self.n_experts) \
                              .permute(0, 3, 1, 2).float()
                    frac = F.adaptive_avg_pool2d(onehot, (H, W))
                    if self.router_gt_type == "soft":
                        router_gt = frac
                    else:
                        router_gt = frac.argmax(dim=1)

        if teacher_force and router_gt is not None:
            # Stage-1 teacher-forcing: gate follows the GT distribution.
            #   hard   → one-hot(int)
            #   soft   → frac (already sums to 1)
            #   overlap → renormalize (sum > 1 possible) so gate remains a
            #             convex combination for expert mixing
            if router_gt.dtype == torch.long:
                gate = F.one_hot(router_gt, num_classes=self.n_experts) \
                        .permute(0, 3, 1, 2).to(logits.dtype)
            elif self.router_gt_type == "overlap":
                s = router_gt.sum(dim=1, keepdim=True).clamp_min(1e-8)
                gate = (router_gt / s).to(logits.dtype)
            else:
                gate = router_gt.to(logits.dtype)
        else:
            # Self-routing: gate from router's own logits.
            #   overlap → sigmoid per bin (independent) → renormalize
            #   else    → softmax across bins
            if self.router_gt_type == "overlap":
                probs = torch.sigmoid(logits)
                probs = probs / probs.sum(dim=1, keepdim=True).clamp_min(1e-8)
            else:
                probs = F.softmax(logits, dim=1)
            if self.top_k is not None and self.top_k < self.n_experts:
                _, idx = probs.topk(self.top_k, dim=1)
                keep = torch.zeros_like(probs).scatter_(1, idx, 1.0)
                gate = probs * keep
                gate = gate / gate.sum(dim=1, keepdim=True).clamp_min(1e-8)
            else:
                gate = probs

        # Confidence-based mixing:
        #   α = (K · max(gate) − 1) / (K − 1) ∈ [0, 1]. α → 1 for one-hot /
        #   fully confident routing, α → 0 for uniform / uninformative.
        # `gate` here is post-top_k (self-routing) or the teacher-force target;
        # its max already reflects the routing distribution actually mixing.
        shared_weight = None    # None ⇒ apply shared at full weight
        if self.shared is not None and self.shared_gate_mode != "always_on":
            max_gate = gate.max(dim=1, keepdim=True).values          # (B, 1, H, W)
            K = self.n_experts
            alpha = ((K * max_gate - 1.0) / (K - 1)).clamp(0.0, 1.0)  # (B, 1, H, W)
            if self.shared_gate_mode == "confidence":
                # v6a: conservation applied in BOTH training and inference.
                #   spec ← spec · α, shared_w ← 1 − α  ⇒ total = 1.
                gate = gate * alpha
                shared_weight = 1.0 - alpha
            elif self.shared_gate_mode == "confidence_infer":
                # v6b: training keeps v5 behavior (spec full, shared full),
                # so shared receives full gradient on EVERY token and learns
                # a generalist path. At inference, switch to conservation
                # so shared acts as a fallback when the router is uncertain.
                if not self.training:
                    gate = gate * alpha
                    shared_weight = 1.0 - alpha
                # else: leave gate as-is and shared_weight=None (full weight)
            elif self.shared_gate_mode == "confidence_infer_disjoint":
                # v6c: training DISJOINTS the two heads — `mixed` gets ONLY
                # specialists (shared_weight ← 0), and the aux path
                # (`mixed_shared_only` = feat + shared_delta) is decoded
                # separately by the caller. Combined with `moe_shared_aux`
                # this trains specialists via the main task loss and shared
                # via the aux task loss, WITHOUT their gradients entangling
                # through a shared mixed prediction. At inference, switch to
                # conservation so the two disjointly-trained paths are then
                # combined by α.
                if not self.training:
                    gate = gate * alpha
                    shared_weight = 1.0 - alpha
                else:
                    # Zero-weight shared in `mixed` (main head = specialists
                    # only). Note: shared_delta is STILL computed below so
                    # that `mixed_shared_only` (aux head) is available.
                    shared_weight = torch.zeros_like(alpha)

        # Each expert runs ONLY on the tokens routed to it. Because Σ_k gate[k]
        # = 1 per token, the mixture is
        #    Σ_k gate[k] · (feat + delta_k) = feat + Σ_k gate[k] · delta_k
        # so the residual is applied exactly once and tokens with gate[k] == 0
        # contribute nothing — computing delta_k there was pure waste. With
        # top-1 routing the measured occupancy is 48/32/21%, i.e. the dense
        # version paid 3x for the MoE blocks. An expert nobody selects is
        # skipped entirely. With a dense gate (top_k=None) every token selects
        # every expert and this degrades gracefully to the old behaviour.
        mixed = feat
        for e_i, expert in enumerate(self.experts):
            sel = gate[:, e_i] > 0                               # (B, H, W)
            if not bool(sel.any()):
                continue
            dense = bool(sel.all())
            out_e = expert(feat, kv, radar_x_orig, radar_mask, image_w,
                           sel=None if dense else sel)
            mixed = mixed + gate[:, e_i:e_i + 1] * (out_e - feat)

        mixed_shared_only: Optional[torch.Tensor] = None
        if self.shared is not None:
            # Shared expert also returns feat + delta_shared. Naively adding
            # would double-count `feat` (2·feat + Σ deltas), breaking the
            # magnitude the downstream decoder expects. Extract only the
            # delta and add once. `shared_weight` = None ⇒ full weight;
            # otherwise multiplies the delta per-token.
            shared_delta = (self.shared(feat, kv, radar_x_orig, radar_mask, image_w)
                            - feat)
            if shared_weight is None:
                mixed = mixed + shared_delta
            else:
                mixed = mixed + shared_weight * shared_delta
            # Auxiliary path: shared-only fused feature (no specialists).
            # Used by RadarTaco to produce a second depth prediction and
            # by ComposedLoss to compute an aux task loss that trains the
            # shared expert as a standalone generalist predictor.
            if return_shared_only:
                mixed_shared_only = feat + shared_delta
        if return_shared_only:
            return mixed, logits, router_gt, mixed_shared_only
        return mixed, logits, router_gt


class PyramidRadarFusion(nn.Module):
    """Hierarchical fusion module across L=3 layers and 6 image scales."""

    def __init__(
        self,
        radar_channels: Tuple[int, int, int] = (64, 128, 512),
        img_channels: Tuple[int, ...] = (64, 64, 128, 256, 512, 512),
        a_l: Tuple[float, float, float] = (48.0, 32.0, 16.0),
        max_radar_points: int = 128,
        heads: int = 4,
        # MoE options — enable per-level. `moe_at_l` is an iterable of level
        # indices (0..L-1) at which BOTH the node and edge fusion blocks
        # are replaced by `MoEFusionBlock`. Empty → identical to baseline.
        moe_at_l: Optional[Tuple[int, ...]] = None,
        moe_n_experts: int = 3,
        moe_use_shared: bool = True,
        moe_top_k: Optional[int] = None,
        moe_bins: Tuple[float, ...] = (0.0, 20.0, 50.0, 100.0),
        moe_router_arch: str = "conv1x1",
        moe_expert_ch_ratio: float = 1.0,
        moe_router_gt_type: str = "hard",
        moe_overlap_bins: Optional[Tuple[Tuple[float, float], ...]] = None,
        moe_shared_gate_mode: str = "always_on",
        # DPT-style multi-scale router options (ignored unless
        # `moe_router_arch == "dpt"`).
        #   moe_router_dpt_source_layers — indices into the image_encoder's
        #       feat list (0..5 for F1..F6) that the router should consume,
        #       ordered shallow → deep. Default (2, 3, 4, 5) uses F3..F6.
        #   moe_router_dpt_fusion_ch — internal DPTRouter channel dim.
        moe_router_dpt_source_layers: Optional[Tuple[int, ...]] = None,
        moe_router_dpt_fusion_ch: int = 128,
        # Two-stage pre-fusion MoE options (see MoEFusionBlock for details).
        moe_pre_fusion_enabled: bool = False,
        moe_pre_fusion_feed_experts: bool = False,
        moe_pre_fusion_ch_ratio: float = 1.0,
        moe_pre_fusion_router_detach: bool = False,
    ) -> None:
        super().__init__()
        assert len(img_channels) == 2 * len(radar_channels), \
            f"img_channels must have 2L entries (got {len(img_channels)})"
        self.L = len(radar_channels)
        self.moe_at_l = set(moe_at_l or ())
        self.moe_router_gt_type = str(moe_router_gt_type)
        # DPT source layer indices — normalised & channel counts stored so
        # the MoE forward can pick the right feats and callers don't need
        # to know the encoder layout.
        self.moe_router_arch = str(moe_router_arch)
        self.moe_router_dpt_source_layers: Tuple[int, ...] = tuple(
            moe_router_dpt_source_layers or (2, 3, 4, 5))
        dpt_source_channels: Optional[List[int]] = None
        if self.moe_router_arch == "dpt":
            for i in self.moe_router_dpt_source_layers:
                assert 0 <= i < len(img_channels), (
                    f"moe_router_dpt_source_layers index {i} out of range "
                    f"(image encoder has {len(img_channels)} outputs)")
            dpt_source_channels = [img_channels[i]
                                    for i in self.moe_router_dpt_source_layers]

        def _mk_block(ch, kv_dim, a):
            if _mk_block._current_l in self.moe_at_l:
                return MoEFusionBlock(
                    ch=ch, kv_dim=kv_dim, heads=heads, a_l=a,
                    n_experts=moe_n_experts,
                    use_shared=moe_use_shared,
                    top_k=moe_top_k,
                    bins=moe_bins,
                    router_arch=moe_router_arch,
                    expert_ch_ratio=moe_expert_ch_ratio,
                    router_gt_type=moe_router_gt_type,
                    overlap_bins=moe_overlap_bins,
                    shared_gate_mode=moe_shared_gate_mode,
                    router_dpt_source_channels=dpt_source_channels,
                    router_dpt_fusion_ch=int(moe_router_dpt_fusion_ch),
                    pre_fusion_enabled=bool(moe_pre_fusion_enabled),
                    pre_fusion_feed_experts=bool(moe_pre_fusion_feed_experts),
                    pre_fusion_ch_ratio=float(moe_pre_fusion_ch_ratio),
                    pre_fusion_router_detach=bool(moe_pre_fusion_router_detach),
                )
            return _FusionBlock(ch=ch, kv_dim=kv_dim, heads=heads, a_l=a)

        # `_mk_block` closes over `_current_l`; set it per level below.
        _mk_block._current_l = 0
        node_blocks = []
        edge_blocks = []
        for l in range(self.L):
            _mk_block._current_l = l
            node_blocks.append(_mk_block(img_channels[2 * l], radar_channels[l], a_l[l]))
            edge_blocks.append(_mk_block(img_channels[2 * l + 1], max_radar_points, a_l[l]))
        self.node_blocks = nn.ModuleList(node_blocks)
        self.edge_blocks = nn.ModuleList(edge_blocks)

    def forward(
        self,
        feats: List[torch.Tensor],
        N_list: List[torch.Tensor],
        E_list: List[torch.Tensor],
        radar_points: torch.Tensor,
        radar_mask: torch.Tensor,
        image_w: int,
        depth_gt_dense: Optional[torch.Tensor] = None,
        teacher_force: bool = True,
        return_shared_only: bool = False,
        rel_depth: Optional[torch.Tensor] = None,
    ):
        """Forward.

        Args:
            rel_depth: (B, 1, H_full, W_full) monocular relative depth.
                Required when any MoE block uses `deterministic_rel_depth`
                routing. Passed through to each MoE block unchanged.
            depth_gt_dense: full-res dense depth GT (B, 1, H_full, W_full).
                If given, each MoE block returns per-token router_gt (for
                CE loss). Pass None for eval / when GT unavailable.
            teacher_force: bool. Controls MoE block's gate mode.
                True  → gate = one_hot(router_gt) (stage-1 teacher-forcing)
                False → gate = softmax(logits) [+top-k] (self-routing)
                router_gt is still returned when depth_gt_dense present.
            return_shared_only: if True, additionally return a second
                `fused_shared` feature list in which every MoE block's
                output is replaced by `feat + shared_delta` (the shared
                expert acting alone). Non-MoE levels are identical to
                `fused`. Used by RadarTaco.forward to produce a
                shared-only depth prediction that the aux task loss
                supervises — trains the shared expert as a standalone
                generalist. Adds one shared-expert forward per MoE block.

        Returns:
            fused (List[Tensor], len 2L): scale-wise fused features.
            router_logits, router_gts: see below.
            fused_shared (List[Tensor]): only present when
                `return_shared_only=True`. Same shape as `fused` but with
                MoE levels replaced by shared-only mixes.
        """
        # Channel 3 = x_pix (image-plane horizontal) for the radar-centered
        # attention's horizontal-window mask. The 3D camera coords (channels
        # 0..2) are already consumed upstream by the radar encoder.
        radar_x = radar_points[:, :, 3]
        out = list(feats)
        out_shared = list(feats) if return_shared_only else None
        router_logits: List[torch.Tensor] = []
        router_gts: List[Optional[torch.Tensor]] = []
        # For DPT router, pre-slice the requested image encoder outputs once
        # so both node and edge blocks (and all pyramid levels) see the same
        # source features. None otherwise so non-DPT MoE blocks ignore it.
        dpt_ms: Optional[List[torch.Tensor]] = None
        if self.moe_router_arch == "dpt":
            dpt_ms = [feats[i] for i in self.moe_router_dpt_source_layers]
        for l in range(self.L):
            f_odd = feats[2 * l]
            f_even = feats[2 * l + 1]
            nb, eb = self.node_blocks[l], self.edge_blocks[l]
            if isinstance(nb, MoEFusionBlock):
                if return_shared_only:
                    x, log_n, gt_n, x_shared = nb(
                        f_odd, N_list[l], radar_x, radar_mask, image_w,
                        depth_gt_dense=depth_gt_dense,
                        teacher_force=teacher_force,
                        return_shared_only=True,
                        rel_depth=rel_depth,
                        router_multi_scale_feats=dpt_ms)
                    y, log_e, gt_e, y_shared = eb(
                        f_even, E_list[l], radar_x, radar_mask, image_w,
                        depth_gt_dense=depth_gt_dense,
                        teacher_force=teacher_force,
                        return_shared_only=True,
                        rel_depth=rel_depth,
                        router_multi_scale_feats=dpt_ms)
                    # Fallback if shared expert is disabled — reuse main.
                    out_shared[2 * l] = x_shared if x_shared is not None else x
                    out_shared[2 * l + 1] = y_shared if y_shared is not None else y
                else:
                    x, log_n, gt_n = nb(
                        f_odd, N_list[l], radar_x, radar_mask, image_w,
                        depth_gt_dense=depth_gt_dense,
                        teacher_force=teacher_force,
                        rel_depth=rel_depth,
                        router_multi_scale_feats=dpt_ms)
                    y, log_e, gt_e = eb(
                        f_even, E_list[l], radar_x, radar_mask, image_w,
                        depth_gt_dense=depth_gt_dense,
                        teacher_force=teacher_force,
                        rel_depth=rel_depth,
                        router_multi_scale_feats=dpt_ms)
                router_logits.append(log_n)
                router_logits.append(log_e)
                router_gts.append(gt_n)
                router_gts.append(gt_e)
            else:
                x = nb(f_odd, N_list[l], radar_x, radar_mask, image_w)
                y = eb(f_even, E_list[l], radar_x, radar_mask, image_w)
                if return_shared_only:
                    # Non-MoE level: shared-only path is identical to main.
                    out_shared[2 * l] = x
                    out_shared[2 * l + 1] = y
            out[2 * l] = x
            out[2 * l + 1] = y
        if return_shared_only:
            return out, router_logits, router_gts, out_shared
        return out, router_logits, router_gts

    def forward_per_spec(
        self,
        feats: List[torch.Tensor],
        N_list: List[torch.Tensor],
        E_list: List[torch.Tensor],
        radar_points: torch.Tensor,
        radar_mask: torch.Tensor,
        image_w: int,
        spec_idx: int,
        main_out: List[torch.Tensor],
    ) -> List[torch.Tensor]:
        """Recompute ONLY the MoE-level outputs with the gate forced to
        one-hot on `spec_idx` (shared disabled). Non-MoE levels are copied
        verbatim from `main_out` (the main-forward's `fused`) — saves
        recomputing L=0/1 fusion K times per training step.

        Returns:
            List[Tensor] of length 2L, matching `main_out`'s layout, with
            MoE-level positions replaced by per-spec forward outputs.
        """
        radar_x = radar_points[:, :, 3]
        out = list(main_out)   # copy; will overwrite MoE positions
        for l in range(self.L):
            nb, eb = self.node_blocks[l], self.edge_blocks[l]
            if not isinstance(nb, MoEFusionBlock):
                continue   # non-MoE level: keep cached output
            f_odd = feats[2 * l]
            f_even = feats[2 * l + 1]
            x, _, _ = nb(
                f_odd, N_list[l], radar_x, radar_mask, image_w,
                depth_gt_dense=None, teacher_force=False,
                force_expert_idx=spec_idx)
            y, _, _ = eb(
                f_even, E_list[l], radar_x, radar_mask, image_w,
                depth_gt_dense=None, teacher_force=False,
                force_expert_idx=spec_idx)
            out[2 * l] = x
            out[2 * l + 1] = y
        return out
