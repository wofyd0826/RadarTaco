"""U-Net style depth decoder (paper Supp §8.2).

Consumes [F'_1..F'_6] (high-res to low-res strides 1/2..1/64) and outputs
metric depth at full input resolution.

Two output parametrizations are supported:

  output_mode = "metric"  (paper-faithful default)
      d = max_depth * sigmoid(head)
      Sigmoid saturates at the upper end → far-range gradient is small.
      Most common, easiest to reason about, matches the paper.

  output_mode = "inverse"  (Option B: long-range-friendly)
      inv_d = inv_min + sigmoid(head) * (inv_max - inv_min)
      d     = 1.0 / inv_d                                  (returned to caller)
      Sigmoid output is in *inverse-depth* space which compresses far
      distances and expands near distances → no saturation problem at
      long range. The conversion to metric `d` happens inside this
      decoder so downstream losses remain in metric space (paper Eq. 5
      stays valid). This combination — inverse parametrization +
      metric loss — keeps long-range gradients alive while still using
      a stable output representation.

Both modes return depth in meters in the range
  [min_depth_clip, max_depth] (inverse mode) or
  [0,              max_depth] (metric mode).

Multi-scale auxiliary heads (deep supervision):

  multi_scale = True
      A small (3×3 conv → ELU → 1×1 conv) head is added at each requested
      decoder scale (1/2, 1/4, 1/8, 1/16 by default — `multi_scale_levels`).
      During TRAINING the decoder returns a dict
        {"depth": d_full, "depth_s2": d_1_2, "depth_s4": d_1_4, ...}
      so the loss can apply per-scale L1 (typically against the dense GT
      after bilinear upsampling). During EVAL the aux heads are skipped
      and the decoder returns just the final-scale depth tensor — same
      interface as before, no caller-side branching needed.
      The aux heads use the same output_mode/min_depth_clip parametrization
      as the final head, so depth ranges are consistent across scales.
"""
from typing import Dict, List, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class _UpConv(nn.Module):
    """Upsample (bilinear, to target size) → 3×3 conv → BN → ELU."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ELU(inplace=True)

    def forward(self, x: torch.Tensor, size=None) -> torch.Tensor:
        if size is None:
            x = F.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=False)
        else:
            x = F.interpolate(x, size=size, mode="bilinear", align_corners=False)
        return self.act(self.bn(self.conv(x)))


class _FuseBlock(nn.Module):
    """3×3 conv → BN → ELU (consumes already-concatenated [up, skip])."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ELU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


def _make_aux_head(in_ch: int) -> nn.Sequential:
    """Lightweight aux head: 3×3 conv → ELU → 1×1 conv → 1ch logit."""
    hidden = max(in_ch // 2, 16)
    return nn.Sequential(
        nn.Conv2d(in_ch, hidden, 3, padding=1),
        nn.ELU(inplace=True),
        nn.Conv2d(hidden, 1, 1),
    )


# Map "scale level" (denominator of input-image stride) → which fuseN block's
# output to tap for that scale. After up{N}+fuseN at scale 1/(2^(N-? )) ...
# Concretely:
#   fuse5 → 1/32, fuse4 → 1/16, fuse3 → 1/8, fuse2 → 1/4, fuse1 → 1/2
_LEVEL_TO_FUSE = {32: "fuse5", 16: "fuse4", 8: "fuse3", 4: "fuse2", 2: "fuse1"}
# Channel index in `decoder_channels` corresponding to each fuse output
_LEVEL_TO_CH_IDX = {32: 4, 16: 3, 8: 2, 4: 1, 2: 0}


class DepthDecoder(nn.Module):
    def __init__(
        self,
        feat_channels: Tuple[int, ...] = (64, 64, 128, 256, 512, 512),
        decoder_channels: Tuple[int, ...] = (64, 64, 128, 128, 256, 256),
        max_depth: float = 100.0,
        output_mode: str = "metric",
        min_depth_clip: float = 0.5,
        multi_scale: bool = False,
        multi_scale_levels: Tuple[int, ...] = (2, 4, 8, 16),
    ) -> None:
        super().__init__()
        assert len(feat_channels) == 6 and len(decoder_channels) == 6
        assert output_mode in ("metric", "inverse"), output_mode
        for s in multi_scale_levels:
            if s not in _LEVEL_TO_FUSE:
                raise ValueError(f"multi_scale_levels entry {s} not in "
                                 f"{sorted(_LEVEL_TO_FUSE)}")
        self.max_depth = max_depth
        self.output_mode = output_mode
        self.min_depth_clip = float(min_depth_clip)
        self.multi_scale = bool(multi_scale)
        self.multi_scale_levels: Tuple[int, ...] = tuple(multi_scale_levels)
        self._inv_min = 1.0 / float(max_depth)
        self._inv_max = 1.0 / max(self.min_depth_clip, 1e-3)
        c1, c2, c3, c4, c5, c6 = decoder_channels
        f1, f2, f3, f4, f5, f6 = feat_channels

        self.proj6 = nn.Sequential(
            nn.Conv2d(f6, c6, 3, padding=1, bias=False),
            nn.BatchNorm2d(c6), nn.ELU(inplace=True),
        )
        self.up5 = _UpConv(c6, c5)
        self.fuse5 = _FuseBlock(c5 + f5, c5)
        self.up4 = _UpConv(c5, c4)
        self.fuse4 = _FuseBlock(c4 + f4, c4)
        self.up3 = _UpConv(c4, c3)
        self.fuse3 = _FuseBlock(c3 + f3, c3)
        self.up2 = _UpConv(c3, c2)
        self.fuse2 = _FuseBlock(c2 + f2, c2)
        self.up1 = _UpConv(c2, c1)
        self.fuse1 = _FuseBlock(c1 + f1, c1)
        self.up0 = _UpConv(c1, c1 // 2)
        self.head = nn.Sequential(
            nn.Conv2d(c1 // 2, c1 // 2, 3, padding=1),
            nn.ELU(inplace=True),
            nn.Conv2d(c1 // 2, 1, 1),
        )

        # Auxiliary heads — built only when enabled. Each head taps the
        # output of the matching fuseN block (see _LEVEL_TO_FUSE).
        if self.multi_scale:
            self.aux_heads = nn.ModuleDict({
                f"s{s}": _make_aux_head(decoder_channels[_LEVEL_TO_CH_IDX[s]])
                for s in self.multi_scale_levels
            })

    # ------------------------------------------------------------ helpers --
    def _head_to_depth(self, head_out: torch.Tensor) -> torch.Tensor:
        """Apply the configured output parametrization to a raw head logit."""
        if self.output_mode == "metric":
            return self.max_depth * torch.sigmoid(head_out)
        # inverse mode (cast to fp32 — see module docstring)
        sig = torch.sigmoid(head_out.float())
        inv_d = self._inv_min + sig * (self._inv_max - self._inv_min)
        return 1.0 / inv_d.clamp(min=self._inv_min, max=self._inv_max)

    # ------------------------------------------------------------ forward --
    def forward(
        self,
        feats: List[torch.Tensor],
        out_hw=None,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        f1, f2, f3, f4, f5, f6 = feats
        x = self.proj6(f6)
        x = self.up5(x, size=f5.shape[-2:]); x = self.fuse5(torch.cat([x, f5], dim=1))
        s32 = x
        x = self.up4(x, size=f4.shape[-2:]); x = self.fuse4(torch.cat([x, f4], dim=1))
        s16 = x
        x = self.up3(x, size=f3.shape[-2:]); x = self.fuse3(torch.cat([x, f3], dim=1))
        s8 = x
        x = self.up2(x, size=f2.shape[-2:]); x = self.fuse2(torch.cat([x, f2], dim=1))
        s4 = x
        x = self.up1(x, size=f1.shape[-2:]); x = self.fuse1(torch.cat([x, f1], dim=1))
        s2 = x
        if out_hw is None:
            out_hw = (f1.shape[-2] * 2, f1.shape[-1] * 2)
        x = self.up0(x, size=out_hw)
        depth = self._head_to_depth(self.head(x))

        # Aux heads only fire during training (eval/inference returns the
        # plain tensor — same interface as the single-scale decoder).
        if not self.multi_scale or not self.training:
            return depth

        feats_by_level = {32: s32, 16: s16, 8: s8, 4: s4, 2: s2}
        out: Dict[str, torch.Tensor] = {"depth": depth}
        for s in self.multi_scale_levels:
            feat = feats_by_level[s]
            head_logit = self.aux_heads[f"s{s}"](feat)
            out[f"depth_s{s}"] = self._head_to_depth(head_logit)
        return out
