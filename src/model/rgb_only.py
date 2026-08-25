"""RGB-only depth U-Net (Singh et al.-style baseline, paper Table 5 row 1).

ResNet-18 encoder + the same U-Net decoder used by RadarTaco. No radar
encoder, no fusion. Forward signature matches RadarTaco so the existing
trainer (which always passes `radar_points`, `radar_mask`) works unchanged
— the radar tensors are simply ignored.

Use this as a controlled "RGB Baseline" run in the same training/eval
pipeline as the radar model to isolate "evaluation environment leniency"
from "real architectural improvement."
"""
from typing import Tuple

import torch
import torch.nn as nn

from .image_encoder import ImageEncoder
from .depth_decoder import DepthDecoder


class RGBOnlyDepth(nn.Module):
    def __init__(
        self,
        max_depth: float = 100.0,
        pretrained_image_encoder: bool = True,
        output_mode: str = "metric",
        min_depth_clip: float = 0.5,
        multi_scale: bool = False,
        multi_scale_levels: Tuple[int, ...] = (2, 4, 8, 16),
    ) -> None:
        super().__init__()
        self.image_encoder = ImageEncoder(pretrained=pretrained_image_encoder)
        self.depth_decoder = DepthDecoder(
            feat_channels=self.image_encoder.feat_channels,
            max_depth=max_depth,
            output_mode=output_mode,
            min_depth_clip=min_depth_clip,
            multi_scale=multi_scale,
            multi_scale_levels=tuple(multi_scale_levels),
        )

    def forward(
        self,
        rgb: torch.Tensor,
        radar_points: torch.Tensor = None,    # ignored
        radar_mask: torch.Tensor = None,      # ignored
    ) -> torch.Tensor:
        feats = self.image_encoder(rgb)
        return self.depth_decoder(feats, out_hw=rgb.shape[-2:])
