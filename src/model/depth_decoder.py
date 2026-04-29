"""U-Net style depth decoder (paper Supp §8.2).

Consumes [F'_1..F'_6] (high-res to low-res strides 1/2..1/64) and outputs
metric depth at full input resolution. Final head: sigmoid * max_depth.
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
        max_depth: float = 80.0,
    ) -> None:
        super().__init__()
        assert len(feat_channels) == 6 and len(decoder_channels) == 6
        self.max_depth = max_depth
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
        depth = self.max_depth * torch.sigmoid(self.head(x))
        return depth
