"""MLP radar encoder (paper §3.1 ablation baseline).

Per-point MLP producing pseudo-(N_l, E_l) outputs at the same channel widths
as the GNN encoder, so the pyramid fusion module can consume both
interchangeably without architecture changes.

Since an MLP has no notion of "edges", we synthesize a soft adjacency from
the node features by softmax(N_l · N_l^T / √C_l). This keeps the fusion
module's edge-block (`E_l ↔ F_{2l}`) well-defined while remaining
faithful to the original paper's "MLP-only" baseline (paper §3.1: "extract
features in 32,256 dims from coordinates using an MLP").
"""
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class MlpRadarEncoder(nn.Module):
    """Per-point MLP encoder + softmax(QK^T) pseudo-adjacency.

    Produces L outputs (N_l, E_l) at increasing channel widths matching
    the GNN encoder so the rest of the pipeline is encoder-agnostic.
    """

    def __init__(
        self,
        channels: Tuple[int, int, int] = (64, 128, 512),
        hidden: int = 128,
    ) -> None:
        super().__init__()
        self.channels = channels
        self.in_proj = nn.Sequential(
            nn.Linear(3, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )
        self.heads = nn.ModuleList(
            [nn.Sequential(nn.Linear(hidden, c), nn.GELU(), nn.Linear(c, c))
             for c in channels]
        )
        self.attn_temp = nn.Parameter(torch.zeros(len(channels)))

    def forward(
        self,
        radar_points: torch.Tensor,    # (B, K, 6) hybrid layout
        radar_mask: torch.Tensor,      # (B, K) bool
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        # Use ego-frame 3D coords (front, left, up) — physically consistent
        # meter units for per-point feature extraction. The image-pixel
        # channels are reserved for the radar-centered attention mask.
        coords3d = radar_points[:, :, :3]
        x = coords3d * radar_mask[:, :, None].float()
        h = self.in_proj(x)                                                        # (B,K,hidden)

        N_list: List[torch.Tensor] = []
        E_list: List[torch.Tensor] = []
        for i, head in enumerate(self.heads):
            n = head(h)                                                            # (B,K,C_l)
            n = n * radar_mask[:, :, None].float()
            scale = (n.size(-1) ** -0.5) * torch.exp(self.attn_temp[i])
            scores = torch.matmul(n, n.transpose(-1, -2)) * scale                  # (B,K,K)
            scores = scores.masked_fill(~radar_mask[:, None, :], float("-inf"))
            adj = torch.softmax(scores, dim=-1)
            adj = torch.nan_to_num(adj, nan=0.0) * radar_mask[:, :, None].float()
            N_list.append(n)
            E_list.append(adj)
        return N_list, E_list
