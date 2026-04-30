#!/usr/bin/env python3
"""Sanity-check that our inverse-projected (front, left, up) matches the
ground-truth ego-frame radar coordinates obtained by loading the raw
nuScenes RADAR_FRONT PCD and applying calibration.

This is NOT a regular pipeline script — it's a one-off verification.
"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import RadarPointCloud
from pyquaternion import Quaternion

from src.dataset.intrinsics import expand_to_6ch, NUSCENES_CAM_FRONT_INTRINSIC


def load_first_sample_with_radar(nusc: NuScenes):
    """Pick the first sample with both CAM_FRONT and RADAR_FRONT entries."""
    for s in nusc.sample:
        sd_cam = nusc.get("sample_data", s["data"]["CAM_FRONT"])
        sd_radar = nusc.get("sample_data", s["data"]["RADAR_FRONT"])
        return s, sd_cam, sd_radar


def radar_pcd_to_ego(nusc: NuScenes, sd_radar) -> np.ndarray:
    """Load PCD, return (N, 3) points in ego frame at sample timestamp."""
    pcd_path = os.path.join(nusc.dataroot, sd_radar["filename"])
    pc = RadarPointCloud.from_file(pcd_path)
    # Apply radar→ego calibration (rotation + translation)
    cal = nusc.get("calibrated_sensor", sd_radar["calibrated_sensor_token"])
    pc.rotate(Quaternion(cal["rotation"]).rotation_matrix)
    pc.translate(np.array(cal["translation"]))
    return pc.points[:3, :].T                                      # (N, 3) in ego at radar ts


def cam_intrinsic(nusc: NuScenes, sd_cam) -> dict:
    cal = nusc.get("calibrated_sensor", sd_cam["calibrated_sensor_token"])
    K = np.array(cal["camera_intrinsic"])
    return {"fx": float(K[0, 0]), "fy": float(K[1, 1]),
            "cx": float(K[0, 2]), "cy": float(K[1, 2])}


def main():
    nusc = NuScenes(version="v1.0-trainval",
                    dataroot="/data/public/nuScenes/trainval",
                    verbose=False)

    sample, sd_cam, sd_radar = load_first_sample_with_radar(nusc)
    print(f"sample token: {sample['token']}")

    # 1) Ground truth: load raw PCD → ego frame.
    pts_ego_gt = radar_pcd_to_ego(nusc, sd_radar)                  # (N, 3)
    print(f"raw PCD points in ego: {pts_ego_gt.shape[0]}")

    # 2) Reproduction: take what we'd compute from our preprocessed data.
    #    Since our preprocessed npy stores (x_pix, y_pix, depth), we build
    #    that from the GT ego points via the official camera projection
    #    (radar ego → camera) and then inverse-project back to ego using
    #    our hardcoded CAM_FRONT intrinsic.
    cal_cam = nusc.get("calibrated_sensor", sd_cam["calibrated_sensor_token"])
    K = np.array(cal_cam["camera_intrinsic"])
    R = Quaternion(cal_cam["rotation"]).rotation_matrix              # ego→camera frame rot
    t = np.array(cal_cam["translation"])                              # camera position in ego

    # ego → camera (right-down-forward)
    pts_cam = (pts_ego_gt - t[None, :]) @ R                           # (N, 3)
    # Keep only points in front of camera
    in_front = pts_cam[:, 2] > 0.1
    pts_cam = pts_cam[in_front]
    pts_ego_kept = pts_ego_gt[in_front]
    # Project to image plane
    proj = pts_cam @ K.T
    x_pix = proj[:, 0] / proj[:, 2]
    y_pix = proj[:, 1] / proj[:, 2]
    depth = pts_cam[:, 2]

    pts3 = np.stack([x_pix, y_pix, depth], axis=-1).astype(np.float32)
    pts6 = expand_to_6ch(pts3, NUSCENES_CAM_FRONT_INTRINSIC)          # uses our hardcoded K
    front_pred = pts6[:, 0]
    left_pred = pts6[:, 1]
    up_pred = pts6[:, 2]

    # nuScenes ego frame: x = front, y = left, z = up — matches our convention.
    front_gt = pts_ego_kept[:, 0]
    left_gt = pts_ego_kept[:, 1]
    up_gt = pts_ego_kept[:, 2]

    # 3) Compare. For an ideal pinhole + identity calibration, these would
    #    match exactly. Differences come from (a) using the per-sample real
    #    intrinsic vs our hardcoded one, and (b) the radar→camera mounting
    #    offset (small lever arm).
    n = len(front_gt)
    print(f"\ncompared {n} points (those projecting in front of CAM_FRONT)")
    for name, pred, gt in [("front", front_pred, front_gt),
                           ("left",  left_pred,  left_gt),
                           ("up",    up_pred,    up_gt)]:
        err = pred - gt
        # The systematic mean is the camera-vs-ego origin offset; the std
        # of (err - mean) is the actual coordinate noise.
        de_mean = err.mean()
        de_std  = err.std()
        print(f"  {name:5s}  mean_err={de_mean:+8.4f} m  std={de_std:.4f}  "
              f"|err| mean={np.abs(err).mean():.4f}  max={np.abs(err).max():.4f}")
    # Pairwise distance preservation: even with a constant origin shift,
    # kNN distances should be identical. Verify.
    pred_xyz = np.stack([front_pred, left_pred, up_pred], axis=-1)
    gt_xyz   = np.stack([front_gt,   left_gt,   up_gt],   axis=-1)
    pd_pred = np.linalg.norm(pred_xyz[:, None] - pred_xyz[None], axis=-1)
    pd_gt   = np.linalg.norm(gt_xyz[:, None]   - gt_xyz[None],   axis=-1)
    print(f"\npairwise distance preservation:")
    print(f"  max |dist_pred - dist_gt| = {np.abs(pd_pred - pd_gt).max():.6f} m")
    print(f"  mean |dist_pred - dist_gt| = {np.abs(pd_pred - pd_gt).mean():.6f} m")

    print("\nNote: 'up' will differ a lot whenever the GT 'up' is far from "
          "0 — that's the radar elevation ambiguity (3D radar barely "
          "measures up). 'front' should match within a few cm; 'left' "
          "within tens of cm depending on radar mounting offset.")


if __name__ == "__main__":
    main()
