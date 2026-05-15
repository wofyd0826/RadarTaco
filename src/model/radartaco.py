"""Top-level RadarTaco model (paper §3.1) — independent inference mode."""
from typing import Tuple

import torch
import torch.nn as nn

from .image_encoder import ImageEncoder
from .radar_encoder import build_radar_encoder
from .radar_fusion import PyramidRadarFusion
from .depth_decoder import DepthDecoder
from .aux_branch import AuxiliaryDepthBranch


class RadarTaco(nn.Module):
    """ResNet18 + (MLP|GNN) radar encoder + Pyramid Radar-centered fusion + U-Net decoder.

    Optional plug-in mode (paper §3.3 + Supp §8.3): set `use_aux_branch=True`
    to add an auxiliary input branch that fuses an initial relative depth
    map `D*` with image features at every pyramid scale. When `use_aux_branch`
    is True, the model's `forward` accepts an extra `rel_depth` argument.
    During training, half the samples are typically given real `D*` (from a
    frozen mono depth predictor like DPT-Hybrid) and the other half are
    given zeros — this is paper §3.4's "two equal portions" recipe and lets
    a single model serve both independent and plug-in inference modes.
    """

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
        output_mode: str = "metric",                  # 'metric' or 'inverse'
        min_depth_clip: float = 0.5,                  # used by 'inverse' mode
        multi_scale: bool = False,                    # deep supervision aux heads
        multi_scale_levels: Tuple[int, ...] = (2, 4, 8, 16),
        use_aux_branch: bool = False,                 # paper §3.3 plug-in branch
    ) -> None:
        super().__init__()
        self.image_encoder = ImageEncoder(pretrained=pretrained_image_encoder)
        self.use_aux_branch = bool(use_aux_branch)
        if self.use_aux_branch:
            self.aux_branch = AuxiliaryDepthBranch(self.image_encoder.feat_channels)
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
            output_mode=output_mode,
            min_depth_clip=min_depth_clip,
            multi_scale=multi_scale,
            multi_scale_levels=tuple(multi_scale_levels),
        )

    def forward(
        self,
        rgb: torch.Tensor,             # (B, 3, H, W) in [-1, 1]
        radar_points: torch.Tensor,    # (B, K, 3) image-space (x_pix, y_pix, depth)
        radar_mask: torch.Tensor,      # (B, K) bool
        rel_depth: torch.Tensor = None,  # (B, 1, H, W) optional initial relative depth
    ) -> torch.Tensor:
        image_w = rgb.shape[-1]
        feats = self.image_encoder(rgb)
        if self.use_aux_branch:
            # Independent mode is realized by passing zeros — paper §3.4
            # trains both modes by stochastically zeroing rel_depth.
            if rel_depth is None:
                rel_depth = torch.zeros(rgb.shape[0], 1, rgb.shape[2], rgb.shape[3],
                                        device=rgb.device, dtype=rgb.dtype)
            feats = self.aux_branch(rel_depth, feats)
        N_list, E_list = self.radar_encoder(radar_points, radar_mask)
        fused = self.radar_fusion(feats, N_list, E_list,
                                  radar_points, radar_mask, image_w=image_w)
        return self.depth_decoder(fused, out_hw=rgb.shape[-2:])
