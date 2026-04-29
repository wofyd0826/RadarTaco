"""Depth visualization helpers (returns HxWx3 uint8 numpy arrays)."""
from typing import Optional

import numpy as np


def colorize_depth(
    depth: np.ndarray,
    valid_mask: Optional[np.ndarray] = None,
    vmin: float = 0.0,
    vmax: float = 80.0,
    cmap: str = "magma",
) -> np.ndarray:
    """Map a [H, W] depth map to a [H, W, 3] uint8 RGB image via matplotlib cmap."""
    import matplotlib.cm as cm
    d = np.clip((depth - vmin) / max(vmax - vmin, 1e-8), 0.0, 1.0)
    rgb = (cm.get_cmap(cmap)(d)[..., :3] * 255).astype(np.uint8)
    if valid_mask is not None:
        rgb[~valid_mask] = 0
    return rgb


def colorize_error(
    pred: np.ndarray,
    gt: np.ndarray,
    valid_mask: np.ndarray,
    vmax: float = 10.0,
    cmap: str = "inferno",
) -> np.ndarray:
    """|pred − gt| → uint8 RGB error map clipped at vmax meters."""
    err = np.abs(pred - gt)
    err = np.where(valid_mask, err, 0.0)
    return colorize_depth(err, valid_mask, vmin=0.0, vmax=vmax, cmap=cmap)


def overlay_radar_points(
    rgb_uint8: np.ndarray,
    radar_points: np.ndarray,        # [K, 3]
    radar_mask: np.ndarray,          # [K] bool
    vmin: float = 0.0,
    vmax: float = 80.0,
    radius: int = 4,
) -> np.ndarray:
    """Draw radar points on an RGB image, color-coded by depth."""
    import matplotlib.cm as cm
    import cv2  # type: ignore

    out = rgb_uint8.copy()
    H, W, _ = out.shape
    cmap = cm.get_cmap("magma")
    for i in range(radar_points.shape[0]):
        if not bool(radar_mask[i]):
            continue
        x, y, d = float(radar_points[i, 0]), float(radar_points[i, 1]), float(radar_points[i, 2])
        if not (0 <= x < W and 0 <= y < H):
            continue
        c = np.clip((d - vmin) / max(vmax - vmin, 1e-8), 0.0, 1.0)
        color = tuple(int(v * 255) for v in cmap(c)[:3])
        cv2.circle(out, (int(x), int(y)), radius, color, -1, cv2.LINE_AA)
    return out


def rgb_to_uint8(rgb_norm: np.ndarray) -> np.ndarray:
    """[3, H, W] in [-1, 1] → [H, W, 3] uint8."""
    if rgb_norm.ndim == 3 and rgb_norm.shape[0] == 3:
        rgb = rgb_norm.transpose(1, 2, 0)
    else:
        rgb = rgb_norm
    rgb = ((rgb + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
    return rgb


def build_grid(images, ncols: Optional[int] = None) -> np.ndarray:
    """Concatenate a list of HxWx3 uint8 images horizontally (or in a grid)."""
    if ncols is None:
        ncols = len(images)
    rows = []
    for i in range(0, len(images), ncols):
        rows.append(np.concatenate(images[i:i + ncols], axis=1))
    return np.concatenate(rows, axis=0)
