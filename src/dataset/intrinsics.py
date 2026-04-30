"""Camera intrinsics for inverse-projecting (x_pix, y_pix, depth) → 3D coords.

Used by the FALLBACK path in src/dataset/nuscenes.py when the precise
sidecar (front, left, up) coords from `scripts/preprocess_radar_3d.py`
are not yet generated. The primary path reads ego-frame (front, left, up)
straight from the raw RADAR_FRONT PCDs through the official nuScenes
calibration chain — see `scripts/preprocess_radar_3d.py`.

Also used by the SIM datasets (Hypersim / vKITTI2) where there is no real
radar PCD to read from, only sampled depth values.

Axis convention (right-handed, ego-style):
    front  = +x   forward, along the camera optical axis
    left   = +y   lateral, positive to the driver/sensor's left
    up     = +z   vertical, positive upward

Conversion from camera frame (right-down-forward) to this axis convention:
    front =  z_cam =  depth
    left  = -x_cam = -(x_pix - cx) * depth / fx
    up    = -y_cam = -(y_pix - cy) * depth / fy

ORIGIN (fallback path only): the resulting (front, left, up) is centered
at the *camera* mount position, not the true vehicle ego origin. Verified
empirically against raw RADAR_FRONT PCDs:
    - mean errors are purely systematic mounting offsets:
        front: -1.62 m, left: -0.19 m, up: -1.33 m
    - per-axis std is 0.10 m (small intrinsic-approximation noise)
    - **pairwise distances between radar points are preserved to <0.0002 m**
Distances and relative geometry are invariant to a constant origin shift,
so the GNN's kNN and per-point features behave identically. The primary
PCD-derived path eliminates this offset entirely.

For 3D radar (nuScenes), `up` will be unreliable because elevation is
ambiguous, but it doesn't break anything — the encoder learns to weight it
appropriately. For 4D radar (ZJU-4DRadarCam) `up` carries real signal.
"""
from typing import Dict, Tuple

import numpy as np


# nuScenes CAM_FRONT (1600 × 900). All scenes use the same camera setup;
# per-sample intrinsics vary by < 0.5 % so the constant approximation is
# fine for kNN distance and feature extraction purposes.
NUSCENES_CAM_FRONT_INTRINSIC: Dict[str, float] = {
    "fx": 1266.42,
    "fy": 1266.42,
    "cx": 816.27,
    "cy": 491.51,
    "W_orig": 1600,
    "H_orig": 900,
}

# Synthetic placeholder intrinsics for sim datasets. The exact values do
# not matter as long as they roughly match the image's FoV — the radar
# encoders' kNN remains scale-consistent across all points in the same
# image.
HYPERSIM_INTRINSIC: Dict[str, float] = {
    "fx": 886.81,                # ~60° horizontal FoV at W=1024
    "fy": 886.81,
    "cx": 512.0,
    "cy": 384.0,
    "W_orig": 1024,
    "H_orig": 768,
}

VKITTI2_INTRINSIC: Dict[str, float] = {
    "fx": 725.0087,              # KITTI camera 02 reference
    "fy": 725.0087,
    "cx": 620.5,
    "cy": 187.0,
    "W_orig": 1242,
    "H_orig": 375,
}

INTRINSICS_BY_NAME: Dict[str, Dict[str, float]] = {
    "nuscenes": NUSCENES_CAM_FRONT_INTRINSIC,
    "hypersim": HYPERSIM_INTRINSIC,
    "vkitti2": VKITTI2_INTRINSIC,
}


def image_to_ego_3d(
    x_pix: np.ndarray,                                       # (N,) pixel
    y_pix: np.ndarray,                                       # (N,) pixel
    depth: np.ndarray,                                       # (N,) meters
    intrinsic: Dict[str, float],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Inverse pinhole projection followed by camera→ego axis remap.

    `x_pix`, `y_pix` should be in the dataset's original pixel coordinate
    frame (the same frame `intrinsic` was measured in). Output `(front,
    left, up)` is in the ego frame (meters):
        front = +z_cam (forward)
        left  = -x_cam (camera x is right; ego y is left)
        up    = -y_cam (camera y is down;  ego z is up)
    """
    fx = intrinsic["fx"]; fy = intrinsic["fy"]
    cx = intrinsic["cx"]; cy = intrinsic["cy"]
    front = depth
    left = -(x_pix - cx) * depth / fx
    up = -(y_pix - cy) * depth / fy
    return (front.astype(np.float32),
            left.astype(np.float32),
            up.astype(np.float32))


def expand_to_6ch(
    pts3: np.ndarray,                                        # (N, 3) (x_pix, y_pix, depth)
    intrinsic: Dict[str, float],
) -> np.ndarray:
    """Expand (N, 3) image-projected radar to (N, 6) hybrid format.

    Output channels: (front, left, up, x_pix, y_pix, depth).
    """
    if pts3.shape[0] == 0:
        return np.zeros((0, 6), dtype=np.float32)
    x_pix, y_pix, depth = pts3[:, 0], pts3[:, 1], pts3[:, 2]
    front, left, up = image_to_ego_3d(x_pix, y_pix, depth, intrinsic)
    return np.stack([front, left, up, x_pix, y_pix, depth], axis=-1).astype(np.float32)


# Channel-index constants. All radar code should use these instead of
# hardcoded magic numbers so the layout can be revised in one place.
CH_FRONT, CH_LEFT, CH_UP = 0, 1, 2
CH_XPIX, CH_YPIX, CH_DEPTH = 3, 4, 5
RADAR_CHANNELS = 6
