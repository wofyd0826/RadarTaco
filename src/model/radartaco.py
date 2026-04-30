"""Top-level RadarTaco model (paper §3.1) — independent inference mode."""
from typing import Tuple

import torch
import torch.nn as nn

from .image_encoder import ImageEncoder
from .radar_encoder import build_radar_encoder
from .radar_fusion import PyramidRadarFusion
from .depth_decoder import DepthDecoder


class RadarTaco(nn.Module):
    """ResNet18 + (MLP|GNN) radar encoder + Pyramid Radar-centered fusion + U-Net decoder."""

    def __init__(
        self,
        radar_encoder_name: str = "gnn",
        max_depth: float = 100.0,
        max_radar_points: int = 128,
        k_neighbors: int = 20,
        a_l: Tuple[float, float, float] = (48.0, 32.0, 16.0),
        radar_channels: Tuple[int, int, int] = (64, 128, 512),
        attn_heads: int = 4,
        mlp_hidden: int = 128,
        pretrained_image_encoder: bool = True,
    ) -> None:
        super().__init__()
        self.image_encoder = ImageEncoder(pretrained=pretrained_image_encoder)
        self.radar_encoder = build_radar_encoder(
            name=radar_encoder_name,
            channels=radar_channels,
            k_neighbors=k_neighbors,
            mlp_hidden=mlp_hidden,
        )
        self.radar_fusion = PyramidRadarFusion(
            radar_channels=radar_channels,
            img_channels=self.image_encoder.feat_channels,
            a_l=a_l,
            max_radar_points=max_radar_points,
            heads=attn_heads,
        )
        self.depth_decoder = DepthDecoder(
            feat_channels=self.image_encoder.feat_channels,
            max_depth=max_depth,
        )

    def forward(
        self,
        rgb: torch.Tensor,             # (B, 3, H, W) in [-1, 1]
        radar_points: torch.Tensor,    # (B, K, 3) image-space (x_pix, y_pix, depth)
        radar_mask: torch.Tensor,      # (B, K) bool
    ) -> torch.Tensor:
        image_w = rgb.shape[-1]
        feats = self.image_encoder(rgb)
        N_list, E_list = self.radar_encoder(radar_points, radar_mask)
        fused = self.radar_fusion(feats, N_list, E_list,
                                  radar_points, radar_mask, image_w=image_w)
        return self.depth_decoder(fused, out_hw=rgb.shape[-2:])
