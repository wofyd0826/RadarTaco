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
    ) -> None:
        super().__init__()
        self.n_experts = int(n_experts)
        self.top_k = int(top_k) if top_k is not None else None
        self.experts = nn.ModuleList([
            _FusionBlock(ch, kv_dim, heads, a_l, mlp_ratio=mlp_ratio)
            for _ in range(self.n_experts)
        ])
        self.shared = (_FusionBlock(ch, kv_dim, heads, a_l, mlp_ratio=mlp_ratio)
                       if use_shared else None)
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
        else:
            raise ValueError(f"Unknown router_arch: {router_arch!r}")
        # Bin edges for teacher-forcing GT (len K+1 → K bins). Registered
        # as a buffer so it moves with .to(device) automatically.
        assert len(bins) == self.n_experts + 1, \
            f"bins must have n_experts+1 edges (got {len(bins)}, need {self.n_experts + 1})"
        self.register_buffer("bins", torch.tensor(bins, dtype=torch.float32))

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
        Returns:
            fused          : (B, C, H, W)
            router_logits  : (B, K, H, W) — per-token raw logits.
            router_gt      : (B, H, W) int64 or None. Present whenever
                             `depth_gt_dense` was supplied.
        """
        B, C, H, W = feat.shape
        logits = self.router(feat)                              # (B, K, H, W)

        router_gt: Optional[torch.Tensor] = None
        if depth_gt_dense is not None:
            with torch.no_grad():
                # Per-token GT via MAJORITY VOTE of per-pixel bins over the
                # token's receptive field. Always compute when GT available.
                pix_bin = self._depth_to_bin(depth_gt_dense.float())
                onehot = F.one_hot(pix_bin, num_classes=self.n_experts) \
                          .permute(0, 3, 1, 2).float()
                frac = F.adaptive_avg_pool2d(onehot, (H, W))
                router_gt = frac.argmax(dim=1)

        if teacher_force and router_gt is not None:
            # Stage-1 teacher-forcing: gate is fixed to GT bin.
            gate = F.one_hot(router_gt, num_classes=self.n_experts) \
                    .permute(0, 3, 1, 2).to(logits.dtype)
        else:
            # Self-routing: gate from router's own softmax (+ optional top-k).
            probs = F.softmax(logits, dim=1)
            if self.top_k is not None and self.top_k < self.n_experts:
                _, idx = probs.topk(self.top_k, dim=1)
                keep = torch.zeros_like(probs).scatter_(1, idx, 1.0)
                gate = probs * keep
                gate = gate / gate.sum(dim=1, keepdim=True).clamp_min(1e-8)
            else:
                gate = probs

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

        if self.shared is not None:
            # Shared expert also returns feat + delta_shared. Naively adding
            # would double-count `feat` (2·feat + Σ deltas), breaking the
            # magnitude the downstream decoder expects. Extract only the
            # delta and add once.
            shared_delta = (self.shared(feat, kv, radar_x_orig, radar_mask, image_w)
                            - feat)
            mixed = mixed + shared_delta
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
    ) -> None:
        super().__init__()
        assert len(img_channels) == 2 * len(radar_channels), \
            f"img_channels must have 2L entries (got {len(img_channels)})"
        self.L = len(radar_channels)
        self.moe_at_l = set(moe_at_l or ())

        def _mk_block(ch, kv_dim, a):
            if _mk_block._current_l in self.moe_at_l:
                return MoEFusionBlock(
                    ch=ch, kv_dim=kv_dim, heads=heads, a_l=a,
                    n_experts=moe_n_experts,
                    use_shared=moe_use_shared,
                    top_k=moe_top_k,
                    bins=moe_bins,
                    router_arch=moe_router_arch,
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
    ):
        """Forward.

        Args:
            depth_gt_dense: full-res dense depth GT (B, 1, H_full, W_full).
                If given, each MoE block returns per-token router_gt (for
                CE loss). Pass None for eval / when GT unavailable.
            teacher_force: bool. Controls MoE block's gate mode.
                True  → gate = one_hot(router_gt) (stage-1 teacher-forcing)
                False → gate = softmax(logits) [+top-k] (self-routing)
                router_gt is still returned when depth_gt_dense present.

        Returns:
            fused (List[Tensor], len 2L): scale-wise fused features.
            router_logits (List[Tensor]): one entry per MoE block (node
                then edge, per MoE level, in order). Empty if no MoE.
                Each entry is (B, K, H_block, W_block).
            router_gts (List[Optional[Tensor]]): matching per-token GT
                labels (B, H_block, W_block) int64, or None per block when
                depth_gt_dense was not provided.
        """
        # Channel 3 = x_pix (image-plane horizontal) for the radar-centered
        # attention's horizontal-window mask. The 3D camera coords (channels
        # 0..2) are already consumed upstream by the radar encoder.
        radar_x = radar_points[:, :, 3]
        out = list(feats)
        router_logits: List[torch.Tensor] = []
        router_gts: List[Optional[torch.Tensor]] = []
        for l in range(self.L):
            f_odd = feats[2 * l]
            f_even = feats[2 * l + 1]
            nb, eb = self.node_blocks[l], self.edge_blocks[l]
            if isinstance(nb, MoEFusionBlock):
                x, log_n, gt_n = nb(f_odd, N_list[l], radar_x, radar_mask, image_w,
                                    depth_gt_dense=depth_gt_dense,
                                    teacher_force=teacher_force)
                y, log_e, gt_e = eb(f_even, E_list[l], radar_x, radar_mask, image_w,
                                    depth_gt_dense=depth_gt_dense,
                                    teacher_force=teacher_force)
                router_logits.append(log_n)
                router_logits.append(log_e)
                router_gts.append(gt_n)
                router_gts.append(gt_e)
            else:
                x = nb(f_odd, N_list[l], radar_x, radar_mask, image_w)
                y = eb(f_even, E_list[l], radar_x, radar_mask, image_w)
            out[2 * l] = x
            out[2 * l + 1] = y
        return out, router_logits, router_gts
