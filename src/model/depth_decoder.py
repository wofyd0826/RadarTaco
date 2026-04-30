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
"""
from typing import List, Tuple

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


class DepthDecoder(nn.Module):
    def __init__(
        self,
        feat_channels: Tuple[int, ...] = (64, 64, 128, 256, 512, 512),
        decoder_channels: Tuple[int, ...] = (64, 64, 128, 128, 256, 256),
        max_depth: float = 100.0,
        output_mode: str = "metric",                     # 'metric' or 'inverse'
        min_depth_clip: float = 0.5,                     # used by 'inverse' mode only
    ) -> None:
        super().__init__()
        assert len(feat_channels) == 6 and len(decoder_channels) == 6
        assert output_mode in ("metric", "inverse"), output_mode
        self.max_depth = max_depth
        self.output_mode = output_mode
        self.min_depth_clip = float(min_depth_clip)
        # Pre-compute inverse-depth bounds for 'inverse' mode. sigmoid=0 →
        # inv_min → d=max_depth (far). sigmoid=1 → inv_max → d=min_depth_clip.
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

    def forward(self, feats: List[torch.Tensor], out_hw=None) -> torch.Tensor:
        f1, f2, f3, f4, f5, f6 = feats
        x = self.proj6(f6)
        x = self.up5(x, size=f5.shape[-2:]); x = self.fuse5(torch.cat([x, f5], dim=1))
        x = self.up4(x, size=f4.shape[-2:]); x = self.fuse4(torch.cat([x, f4], dim=1))
        x = self.up3(x, size=f3.shape[-2:]); x = self.fuse3(torch.cat([x, f3], dim=1))
        x = self.up2(x, size=f2.shape[-2:]); x = self.fuse2(torch.cat([x, f2], dim=1))
        x = self.up1(x, size=f1.shape[-2:]); x = self.fuse1(torch.cat([x, f1], dim=1))
        if out_hw is None:
            out_hw = (f1.shape[-2] * 2, f1.shape[-1] * 2)
        x = self.up0(x, size=out_hw)
        head_out = self.head(x)
        if self.output_mode == "metric":
            depth = self.max_depth * torch.sigmoid(head_out)
        else:  # inverse
            # Cast to fp32 for the reciprocal — fp16 underflow is real
            # when inv_d ~ 1e-2. Returned `depth` will be fp32, the loss
            # functions promote naturally.
            sig = torch.sigmoid(head_out.float())
            inv_d = self._inv_min + sig * (self._inv_max - self._inv_min)
            depth = 1.0 / inv_d.clamp(min=self._inv_min, max=self._inv_max)
        return depth
