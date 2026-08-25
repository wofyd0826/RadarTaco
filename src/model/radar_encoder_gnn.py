"""Graph-based radar structure extractor (paper §3.1).

DGCNN edge-conv stack + PCA-GM Gconv aggregation, L=3 layers. Each layer
emits a node feature `N_l: (B, K, C_l)` and a soft adjacency edge feature
`E_l: (B, K, K)` consumed by the pyramid fusion module.

Adapted from /workspace/RGM/models/{dgcnn,gconv}.py and
/workspace/dgcnn/pytorch/model.py.
"""
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def masked_knn(coords: torch.Tensor, valid: torch.Tensor, k: int) -> torch.Tensor:
    """Padding-aware kNN. Invalid keys get -inf so they're never selected.

    When a frame holds fewer valid points than `k`, `topk` fills the remainder
    from the -inf (padded) entries, injecting phantom neighbours at the origin.
    `k_eff` is therefore clamped to the smallest valid count in the batch.
    nuScenes never triggers this (min 49 points/frame) but sparser sensors do.
    """
    inner = -2 * torch.matmul(coords.transpose(2, 1).contiguous(), coords)        # (B,K,K)
    xx = (coords ** 2).sum(dim=1, keepdim=True)
    pairwise = -xx - inner - xx.transpose(2, 1).contiguous()                       # neg dist²
    pairwise = pairwise.masked_fill(~valid[:, None, :], float("-inf"))
    n_min = int(valid.sum(dim=-1).min().item())
    k_eff = max(1, min(k, pairwise.size(-1), n_min))
    return pairwise.topk(k=k_eff, dim=-1)[1]                                       # (B,K,k)


def gather_graph_feature(coords: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """Build (B, 6, K, k) edge feature [neighbor−self, self] from coords."""
    B, _, K = coords.shape
    k = idx.shape[-1]
    base = torch.arange(0, B, device=coords.device).view(-1, 1, 1) * K
    flat = (idx + base).view(-1)
    coords_t = coords.transpose(2, 1).contiguous()                                 # (B,K,3)
    nbr = coords_t.view(B * K, -1)[flat, :].view(B, K, k, 3)
    self_ = coords_t.view(B, K, 1, 3).expand(-1, -1, k, -1)
    feat = torch.cat([nbr - self_, self_], dim=-1)                                 # (B,K,k,6)
    return feat.permute(0, 3, 1, 2).contiguous()                                   # (B,6,K,k)


def gather_neighbor_features(feat: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """Gather per-neighbor feature [nbr−self, self] from prior-layer node features."""
    B, C, K = feat.shape
    k = idx.shape[-1]
    base = torch.arange(0, B, device=feat.device).view(-1, 1, 1) * K
    flat = (idx + base).view(-1)
    feat_t = feat.transpose(2, 1).contiguous()
    nbr = feat_t.view(B * K, C)[flat, :].view(B, K, k, C)
    self_ = feat_t.view(B, K, 1, C).expand(-1, -1, k, -1)
    out = torch.cat([nbr - self_, self_], dim=-1)                                  # (B,K,k,2C)
    return out.permute(0, 3, 1, 2).contiguous()


class _NodeGenerator(nn.Module):
    """DGCNN-style two-stage edge-conv → max-pool → 1×1 proj."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        c1 = max(out_ch // 2, 32)
        c2 = out_ch
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, c1, 1, bias=False),
            nn.BatchNorm2d(c1),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(c1, c2, 1, bias=False),
            nn.BatchNorm2d(c2),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.proj = nn.Sequential(
            nn.Conv1d(c1 + c2, out_ch, 1, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, edge_feat: torch.Tensor) -> torch.Tensor:
        x = self.conv1(edge_feat)
        x1 = x.max(dim=-1)[0]
        x = self.conv2(x)
        x2 = x.max(dim=-1)[0]
        return self.proj(torch.cat([x1, x2], dim=1))


class _EdgeGenerator(nn.Module):
    """Soft adjacency via softmax(QK^T / √d)."""

    def __init__(self, ch: int, attn_dim: int = None) -> None:
        super().__init__()
        d = attn_dim or ch
        self.q_proj = nn.Linear(ch, d)
        self.k_proj = nn.Linear(ch, d)
        self.scale = d ** -0.5

    def forward(self, node: torch.Tensor, valid: torch.Tensor,
                knn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = node.transpose(1, 2).contiguous()
        q = self.q_proj(x)
        k = self.k_proj(x)
        scores = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        # Geometric prior: restrict each row's candidates to its kNN. Without
        # it the softmax runs over every valid point and collapses to a
        # query-independent (rank-1) matrix — measured 98.6% first-singular-
        # value energy on E_2, i.e. every point's "relation profile" is the
        # same vector, which makes the fusion edge blocks' keys
        # indistinguishable. Restricting candidates makes rank-1 impossible
        # because each row sees a different candidate set.
        if knn_mask is not None:
            scores = scores.masked_fill(~knn_mask, float("-inf"))
        scores = scores.masked_fill(~valid[:, None, :], float("-inf"))
        adj = torch.softmax(scores, dim=-1)
        adj = torch.nan_to_num(adj, nan=0.0)
        adj = adj * valid[:, :, None].float()
        return adj


class _Gconv(nn.Module):
    """PCA-GM graph convolution: x = norm(A) · ReLU(W_a x) + ReLU(W_u x)."""

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.a_fc = nn.Linear(in_features, out_features)
        self.u_fc = nn.Linear(in_features, out_features)

    def forward(self, A: torch.Tensor, x: torch.Tensor, norm: bool = True) -> torch.Tensor:
        if norm:
            A = F.normalize(A, p=1, dim=-2)
        return torch.bmm(A, F.relu(self.a_fc(x))) + F.relu(self.u_fc(x))


class _GraphRadarLayer(nn.Module):
    """One layer = NodeGen + EdgeGen + Gconv aggregation."""

    def __init__(self, in_ch: int, out_ch: int, k_neighbors: int, is_first: bool) -> None:
        super().__init__()
        self.k = k_neighbors
        self.is_first = is_first
        node_in = 6 if is_first else (2 * in_ch)
        self.node_gen = _NodeGenerator(node_in, out_ch)
        self.edge_gen = _EdgeGenerator(out_ch)
        self.gconv = _Gconv(out_ch, out_ch)
        self.skip = nn.Linear(in_ch, out_ch) if (not is_first and in_ch != out_ch) else None

    def forward(self, coords, node_in, knn_idx, valid, knn_mask=None):
        if self.is_first:
            edge_feat = gather_graph_feature(coords, knn_idx)
        else:
            edge_feat = gather_neighbor_features(node_in, knn_idx)
        # Padded query rows are NOT zero here: kNN only ever picks valid
        # neighbours, so a padded point's feature is [nbr - 0, 0] = [nbr, 0].
        # Those rows are ~24% of the (K, k) grid and feed the BatchNorms inside
        # `node_gen` — masking after `node_gen` (as the line below does) is too
        # late, the statistics have already absorbed them. Mask here instead.
        edge_feat = edge_feat * valid[:, None, :, None].to(edge_feat.dtype)
        node = self.node_gen(edge_feat)
        node = node * valid[:, None, :].float()

        adj = self.edge_gen(node, valid, knn_mask)
        x = node.transpose(1, 2).contiguous()
        x = self.gconv(adj, x)
        if not self.is_first:
            res = node_in.transpose(1, 2).contiguous()
            if self.skip is not None:
                res = self.skip(res)
            x = x + res
        x = x * valid[:, :, None].float()
        return x.transpose(1, 2).contiguous(), adj


class GnnRadarEncoder(nn.Module):
    """L=3 stacked graph layers producing (N_l, E_l) per layer.

    Output:
        N_list: list[(B, K, C_l)]
        E_list: list[(B, K, K)]
    """

    def __init__(
        self,
        channels: Tuple[int, int, int] = (64, 128, 512),
        k_neighbors: int = 20,
    ) -> None:
        super().__init__()
        self.channels = channels
        self.k = k_neighbors
        layers = []
        prev = channels[0]
        for i, c in enumerate(channels):
            layers.append(_GraphRadarLayer(in_ch=prev, out_ch=c,
                                           k_neighbors=k_neighbors,
                                           is_first=(i == 0)))
            prev = c
        self.layers = nn.ModuleList(layers)

    def forward(
        self,
        radar_points: torch.Tensor,   # (B, K, 6) hybrid (front, left, up, x_pix, y_pix, depth)
        radar_mask: torch.Tensor,     # (B, K) bool
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        # Use the ego-frame 3D channels (front, left, up) for kNN distance
        # and node feature construction so that pairwise distances live in
        # consistent meter units (image-pixel + depth would mix units).
        coords = radar_points[:, :, :3].transpose(1, 2).contiguous()               # (B,3,K)
        coords = coords * radar_mask[:, None, :].float()
        knn_idx = masked_knn(coords, radar_mask, self.k)
        # (B, K, K) boolean adjacency of the geometric graph, reused by every
        # layer's `_EdgeGenerator` to restrict its softmax candidates.
        knn_mask = torch.zeros(radar_points.shape[0], radar_points.shape[1],
                               radar_points.shape[1], dtype=torch.bool,
                               device=radar_points.device)
        knn_mask.scatter_(2, knn_idx, True)
        N_list, E_list = [], []
        node = None
        for layer in self.layers:
            node, adj = layer(coords, node, knn_idx, radar_mask, knn_mask)
            N_list.append(node.transpose(1, 2).contiguous())                       # (B,K,C_l)
            E_list.append(adj)
        return N_list, E_list
