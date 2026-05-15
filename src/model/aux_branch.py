"""Auxiliary input branch for plug-in mode (TacoDepth §3.3, Supp §8.3).

Takes initial relative depth `D*` (1 channel, output of a frozen mono
depth predictor like DPT-Hybrid / MiDaS / Depth-Anything-v2) and produces
multi-scale features at the SAME spatial scales and channel widths as
the image encoder pyramid `{F1..F6}`. Each scale's image feature is
fused with its aux feature via channel concat + 1×1 conv.

Supp §8.3 verbatim:
    "convolution extracts features from initial depth, which are then
    fused with RGB features by concatenation and convolution. Other
    steps are identical to the independent mode."

When `rel_depth` is None or all zeros, the branch still runs (the bias
terms of the convs produce a small constant feature) but the fusion
conv learns to suppress its contribution — this is the §3.4 "the other
half adopts zero as input" path. So during training the branch sees
both real D* and zero, learning a single set of weights that handles
both inference modes.
"""
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class _AuxStage(nn.Module):
    """Stride-2 Conv → BN → ReLU."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 2) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class AuxiliaryDepthBranch(nn.Module):
    """Multi-scale aux feature extractor + per-scale concat-conv fusion.

    Input: rel_depth (B, 1, H, W) — typically normalized to [0, 1] or
    inverse-depth from a mono predictor. Output: list of fused feature
    maps at the 6 image-encoder scales (1/2, 1/4, 1/8, 1/16, 1/32, 1/64).
    """

    def __init__(self, feat_channels: Tuple[int, ...]) -> None:
        super().__init__()
        assert len(feat_channels) == 6, \
            f"expected 6 image encoder scales, got {len(feat_channels)}"
        c = feat_channels
        # Aux pyramid: 1 → c1 (1/2) → c2 (1/4) → c3 (1/8) → c4 (1/16)
        #            → c5 (1/32) → c6 (1/64). Stride-2 conv at every stage.
        self.stem = _AuxStage(1, c[0], stride=2)          # 1/2
        self.stage2 = _AuxStage(c[0], c[1], stride=2)     # 1/4
        self.stage3 = _AuxStage(c[1], c[2], stride=2)     # 1/8
        self.stage4 = _AuxStage(c[2], c[3], stride=2)     # 1/16
        self.stage5 = _AuxStage(c[3], c[4], stride=2)     # 1/32
        self.stage6 = _AuxStage(c[4], c[5], stride=2)     # 1/64
        # Fusion: F_l_new = Conv1x1(Concat(F_l, aux_l))
        self.fuse = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(2 * ch, ch, 1, bias=False),
                nn.BatchNorm2d(ch),
                nn.ReLU(inplace=True),
            ) for ch in c
        ])

    def _aux_pyramid(self, rel_depth: torch.Tensor) -> List[torch.Tensor]:
        a1 = self.stem(rel_depth)
        a2 = self.stage2(a1)
        a3 = self.stage3(a2)
        a4 = self.stage4(a3)
        a5 = self.stage5(a4)
        a6 = self.stage6(a5)
        return [a1, a2, a3, a4, a5, a6]

    def forward(
        self,
        rel_depth: torch.Tensor,            # (B, 1, H, W)
        image_feats: List[torch.Tensor],    # 6 levels from ImageEncoder
    ) -> List[torch.Tensor]:
        aux_feats = self._aux_pyramid(rel_depth)
        out: List[torch.Tensor] = []
        for i, (img_f, aux_f) in enumerate(zip(image_feats, aux_feats)):
            # Align spatial sizes — rare 1-pixel offsets can happen when
            # H/W aren't powers of 2 (e.g. 900×1600 nuScenes).
            if aux_f.shape[-2:] != img_f.shape[-2:]:
                aux_f = F.interpolate(aux_f, size=img_f.shape[-2:],
                                      mode="bilinear", align_corners=False)
            out.append(self.fuse[i](torch.cat([img_f, aux_f], dim=1)))
        return out
