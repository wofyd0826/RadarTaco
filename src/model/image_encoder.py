"""ResNet-18 multi-scale image encoder.

Produces 6 feature levels {F1..F6} for the L=3 pyramid fusion module
(paper §3.1: image_encoder outputs {F1..F2L}). 5 levels come from the
standard ResNet stages (stem, layer1..layer4); a sixth level is added via
an extra strided block on top of layer4.

Output channels: (64, 64, 128, 256, 512, 512)
Output strides:  (1/2, 1/4, 1/8, 1/16, 1/32, 1/64)
"""
from typing import List, Tuple

import torch
import torch.nn as nn
import torchvision.models as tvm


class _ExtraDownBlock(nn.Module):
    """Stride-2 BasicBlock-style block producing the F6 (1/64) level."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, stride=2, bias=False),
            nn.BatchNorm2d(out_ch),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + identity)


class ImageEncoder(nn.Module):
    feat_channels: Tuple[int, ...] = (64, 64, 128, 256, 512, 512)

    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        weights = tvm.ResNet18_Weights.DEFAULT if pretrained else None
        backbone = tvm.resnet18(weights=weights)
        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)  # 1/2, 64
        self.maxpool = backbone.maxpool                                          # 1/2 → 1/4
        self.layer1 = backbone.layer1                                            # 1/4, 64
        self.layer2 = backbone.layer2                                            # 1/8, 128
        self.layer3 = backbone.layer3                                            # 1/16, 256
        self.layer4 = backbone.layer4                                            # 1/32, 512
        self.extra = _ExtraDownBlock(512, 512)                                   # 1/64, 512

    def forward(self, rgb: torch.Tensor) -> List[torch.Tensor]:
        f1 = self.stem(rgb)
        x = self.maxpool(f1)
        f2 = self.layer1(x)
        f3 = self.layer2(f2)
        f4 = self.layer3(f3)
        f5 = self.layer4(f4)
        f6 = self.extra(f5)
        return [f1, f2, f3, f4, f5, f6]
