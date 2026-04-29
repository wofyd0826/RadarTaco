"""Factory for swappable radar encoders.

Both encoders expose the same interface:
    encoder(radar_points, radar_mask) -> (N_list, E_list)
        N_list: list[(B, K, C_l)] of length L
        E_list: list[(B, K, K)]   of length L

so the downstream pyramid fusion module is encoder-agnostic.
"""
from typing import Tuple

import torch.nn as nn

from .radar_encoder_gnn import GnnRadarEncoder
from .radar_encoder_mlp import MlpRadarEncoder


def build_radar_encoder(
    name: str,
    channels: Tuple[int, int, int] = (64, 128, 512),
    k_neighbors: int = 20,
    mlp_hidden: int = 128,
) -> nn.Module:
    name = name.lower()
    if name == "gnn":
        return GnnRadarEncoder(channels=channels, k_neighbors=k_neighbors)
    if name == "mlp":
        return MlpRadarEncoder(channels=channels, hidden=mlp_hidden)
    raise ValueError(f"unknown radar encoder: {name!r} (expected 'gnn' or 'mlp')")
