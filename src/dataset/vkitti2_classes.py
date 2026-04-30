"""vKITTI2 classSegmentation color → class → radar reflectivity mapping.

vKITTI2 ships per-pixel semantic labels as RGB-encoded PNGs. Colors are
fixed across all scenes/variants (verified empirically across Scene01..20).

The 14 classes and their canonical RGB codes are taken from the vKITTI2
release notes (Cabon et al. 2020). For radar simulation we approximate each
class' radar reflectivity (specular return strength) so that
semantic-aware sampling concentrates on the surfaces real automotive
mmWave radar actually sees most often.

Reflectivity weights — rough proxy, NOT a calibrated RCS estimate:
  vehicles (metal body)            → strongest    (5.0)
  guard rail (metal)               → very strong  (4.0)
  poles / traffic signs / lights   → strong       (3.0)
  building corners (concrete)      → moderate     (1.5)
  road / terrain / vegetation      → ignored      (0.0)
  sky / misc                       → ignored      (0.0)

These weights are tunable via SimRadarDepthDataset constructor / config.
"""
from typing import Dict, Tuple

import numpy as np


# Color (R, G, B) → (class_name, default_reflectivity_weight)
VKITTI2_CLASSES: Dict[Tuple[int, int, int], Tuple[str, float]] = {
    (210,   0, 200): ("Terrain",      0.0),
    ( 90, 200, 255): ("Sky",          0.0),
    (  0, 199,   0): ("Tree",         0.0),
    ( 90, 240,   0): ("Vegetation",   0.0),
    (140, 140, 140): ("Building",     1.5),
    (100,  60, 100): ("Road",         0.0),
    (250, 100, 255): ("GuardRail",    4.0),
    (255, 255,   0): ("TrafficSign",  3.0),
    (200, 200,   0): ("TrafficLight", 3.0),
    (255, 130,   0): ("Pole",         3.0),
    ( 80,  80,  80): ("Misc",         0.0),
    (160,  60,  60): ("Truck",        5.0),
    (255, 127,  80): ("Car",          5.0),
    (  0, 139, 139): ("Van",          5.0),
}


def reflectivity_map_from_classgt(
    classgt_rgb: np.ndarray,                                 # (H, W, 3) uint8
    class_weights: Dict[str, float] = None,
) -> np.ndarray:
    """Convert a vKITTI2 classSegmentation PNG → per-pixel reflectivity weight.

    Args:
        classgt_rgb:    (H, W, 3) uint8 array (output of np.array(Image.open(...)))
        class_weights:  optional override of the default per-class weights
                        keyed by class name (e.g. {"Car": 6.0, "Building": 0.5}).
                        Unmentioned classes keep their VKITTI2_CLASSES default.
    Returns:
        weights:  (H, W) float32, ≥ 0. Pixels not matching any known class
                  receive 0.0 (treated as unsampled).
    """
    if classgt_rgb.ndim != 3 or classgt_rgb.shape[-1] != 3:
        raise ValueError(f"expected (H, W, 3) RGB, got {classgt_rgb.shape}")
    H, W, _ = classgt_rgb.shape
    out = np.zeros((H, W), dtype=np.float32)
    # Build effective weights dict
    eff = {}
    for color, (name, default_w) in VKITTI2_CLASSES.items():
        w = float(class_weights.get(name, default_w)) if class_weights else default_w
        eff[color] = w

    # Vectorized per-color comparison (≤ 14 classes so cheap).
    for color, w in eff.items():
        if w == 0.0:
            continue
        mask = ((classgt_rgb[..., 0] == color[0])
                & (classgt_rgb[..., 1] == color[1])
                & (classgt_rgb[..., 2] == color[2]))
        if mask.any():
            out[mask] = w
    return out


def derive_classgt_path(rgb_path: str) -> str:
    """Map a vKITTI2 rgb file path to its matching classSegmentation file.

    Layout (vKITTI2 official):
        .../rgb/Scene01/clone/frames/rgb/Camera_0/rgb_00000.jpg
        .../classSegmentation/Scene01/clone/frames/classSegmentation/Camera_0/classgt_00000.png
    """
    p = (rgb_path
         .replace("/rgb/", "/classSegmentation/", 1)
         .replace("/frames/rgb/", "/frames/classSegmentation/")
         .replace("/rgb_", "/classgt_"))
    if p.endswith(".jpg"):
        p = p[:-4] + ".png"
    return p
