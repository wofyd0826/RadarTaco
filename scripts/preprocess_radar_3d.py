#!/usr/bin/env python3
"""Generate (front, left, up) ego-frame radar coordinates from raw nuScenes
RADAR_FRONT PCDs as a sidecar to the existing 7-channel radar npy.

Output: <data_root>/radar_3d/scene_XXXX/<cam_filename>.npy
        shape (N_kept, 3), dtype float32, channels = (front, left, up)

`N_kept` matches the existing 7-channel `radar/` npy line-by-line — both
are produced by the same RADAR_FRONT → CAM_FRONT projection pipeline and
filtered by the same in-front-of-camera mask.

The dataset loader (src/dataset/nuscenes.py) reads both files at sample
access time and concatenates them into the (N, 6) hybrid layout
(front, left, up, x_pix, y_pix, depth). No inverse-projection / camera
intrinsic approximation is involved — these coords come straight from the
PCDs through the same calibration chain nuScenes ships.

Usage:
    python scripts/preprocess_radar_3d.py
    python scripts/preprocess_radar_3d.py --workers 8
    python scripts/preprocess_radar_3d.py --splits train val
"""
import argparse
import copy
import logging
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import RadarPointCloud
from nuscenes.utils.geometry_utils import view_points
from PIL import Image
from pyquaternion import Quaternion
from tqdm import tqdm

logging.basicConfig(format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                    level=logging.INFO)
logger = logging.getLogger("preprocess_radar_3d")


def transform_pc_to_camera_keep_ego(nusc, pc, sensor_token, camera_token):
    """Same 5-step transform as RadarMarigold's preprocess_nuscenes.py, but
    we snapshot the ego-frame coordinates AFTER step 3 (camera timestamp
    ego) and BEFORE step 4 (ego → camera). Returns ego (front/left/up),
    image-plane (x_pix/y_pix) and the in-front-of-camera mask.
    """
    camera = nusc.get("sample_data", camera_token)
    sensor = nusc.get("sample_data", sensor_token)
    image_path = os.path.join(nusc.dataroot, camera["filename"])
    H, W = np.array(Image.open(image_path)).shape[:2]

    # Step 1: sensor → ego (radar timestamp)
    cal_sensor = nusc.get("calibrated_sensor", sensor["calibrated_sensor_token"])
    pc.rotate(Quaternion(cal_sensor["rotation"]).rotation_matrix)
    pc.translate(np.array(cal_sensor["translation"]))
    # Step 2: ego → global (radar timestamp)
    ego_pose = nusc.get("ego_pose", sensor["ego_pose_token"])
    pc.rotate(Quaternion(ego_pose["rotation"]).rotation_matrix)
    pc.translate(np.array(ego_pose["translation"]))
    # Step 3: global → ego (camera timestamp)
    cam_ego_pose = nusc.get("ego_pose", camera["ego_pose_token"])
    pc.translate(-np.array(cam_ego_pose["translation"]))
    pc.rotate(Quaternion(cam_ego_pose["rotation"]).rotation_matrix.T)

    # Snapshot ego coords here. nuScenes ego frame: x = front, y = left, z = up.
    ego_xyz = pc.points[:3, :].copy().T                        # (N, 3)

    # Step 4: ego → camera
    cal_cam = nusc.get("calibrated_sensor", camera["calibrated_sensor_token"])
    pc.translate(-np.array(cal_cam["translation"]))
    pc.rotate(Quaternion(cal_cam["rotation"]).rotation_matrix.T)
    # Step 5: camera → image
    K = np.array(cal_cam["camera_intrinsic"])
    depths = pc.points[2, :]
    points_2d = view_points(pc.points[:3, :], K, normalize=True)
    mask = ((depths > 1.0)
            & (points_2d[0] > 1) & (points_2d[0] < W - 1)
            & (points_2d[1] > 1) & (points_2d[1] < H - 1))
    return ego_xyz, points_2d, depths, mask, (H, W)


def process_sample(nusc, sample_token):
    """Returns the (N, 3) (front, left, up) array for this sample, ordered
    identically to the existing radar npy (same de-dup pixel rule).
    """
    sample = nusc.get("sample", sample_token)
    radar_token = sample["data"]["RADAR_FRONT"]
    camera_token = sample["data"]["CAM_FRONT"]

    RadarPointCloud.disable_filters()
    radar_path = nusc.get_sample_data(radar_token)[0]
    pc = RadarPointCloud.from_file(radar_path)
    pc_copy = copy.deepcopy(pc)

    ego_xyz, points_2d, depths, mask, hw = transform_pc_to_camera_keep_ego(
        nusc, pc_copy, radar_token, camera_token,
    )

    pts_x = np.round(points_2d[0, mask]).astype(int)
    pts_y = np.round(points_2d[1, mask]).astype(int)
    pts_d = depths[mask]
    pts_e = ego_xyz[mask]                                       # (M, 3)

    H, W = hw
    # Same de-dup as RadarMarigold's process_radar: keep the closest point
    # per pixel; multiple radar returns can hit the same pixel due to
    # rounding. Track which original index won so we can emit ego coords
    # in the same per-pixel order as the existing 7-channel radar npy.
    radar_image = np.full((H, W), np.inf, dtype=np.float32)
    winner_idx = np.full((H, W), -1, dtype=np.int32)
    for i in range(len(pts_x)):
        x, y = pts_x[i], pts_y[i]
        if pts_d[i] < radar_image[y, x]:
            radar_image[y, x] = pts_d[i]
            winner_idx[y, x] = i

    ys, xs = np.where(np.isfinite(radar_image))
    if len(ys) == 0:
        return np.zeros((0, 3), dtype=np.float32), camera_token

    # Sort the (xs, ys) iteration order to match np.where(validity) in the
    # original RadarMarigold script (it walks rows in row-major order too).
    order = np.lexsort((xs, ys))
    ys = ys[order]; xs = xs[order]
    win = winner_idx[ys, xs]
    front_left_up = pts_e[win]                                  # (N_kept, 3)
    return front_left_up.astype(np.float32), camera_token


def out_relpath_for_sample(nusc, sample_token):
    """Return the same scene_XXXX/cam_filename path used by the existing
    radar/ npy directory."""
    sample = nusc.get("sample", sample_token)
    cam_data = nusc.get("sample_data", sample["data"]["CAM_FRONT"])
    cam_fname = os.path.basename(cam_data["filename"]).rsplit(".", 1)[0]
    # The split files use scene_NNNN local numbering; recover it by reading
    # the existing radar/ tree.
    return cam_fname  # caller resolves scene_XXXX from existing radar/ index


def build_camfname_to_relpath_index(radar_root: str) -> dict:
    """Walk derived/radar/ to map cam_filename → scene_XXXX/cam_filename
    so we can mirror the same path layout for the new radar_3d/ dir."""
    out = {}
    for scene_dir in sorted(os.listdir(radar_root)):
        sd = os.path.join(radar_root, scene_dir)
        if not os.path.isdir(sd):
            continue
        for fn in os.listdir(sd):
            if fn.endswith(".npy"):
                stem = fn[:-4]
                out[stem] = os.path.join(scene_dir, stem)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="/data/public/nuScenes",
                        help="root containing trainval/ test/ derived/")
    parser.add_argument("--out-subdir", default="radar_3d",
                        help="subdir under <data-root>/derived/ to write to")
    parser.add_argument("--versions", nargs="+", default=["v1.0-trainval", "v1.0-test"])
    parser.add_argument("--workers", type=int, default=1,
                        help="not parallelized — nuscenes-devkit is not "
                             "process-safe; flag reserved for future use")
    args = parser.parse_args()

    derived = os.path.join(args.data_root, "derived")
    radar_root = os.path.join(derived, "radar")
    out_root = os.path.join(derived, args.out_subdir)
    os.makedirs(out_root, exist_ok=True)

    # Map cam filename → scene_XXXX/cam_filename so the new dir mirrors
    # the existing layout exactly.
    fname_to_relpath = build_camfname_to_relpath_index(radar_root)
    logger.info(f"existing radar npy entries: {len(fname_to_relpath)}")

    n_done = n_skip = n_miss = 0
    for version in args.versions:
        # nuScenes test data lives at trainval root / test root respectively
        if version == "v1.0-trainval":
            ds_root = os.path.join(args.data_root, "trainval")
        elif version == "v1.0-test":
            ds_root = os.path.join(args.data_root, "test")
        else:
            ds_root = os.path.join(args.data_root, version)
        logger.info(f"loading {version} from {ds_root}…")
        nusc = NuScenes(version=version, dataroot=ds_root, verbose=False)

        for sample in tqdm(nusc.sample, desc=version):
            try:
                cam_data = nusc.get("sample_data", sample["data"]["CAM_FRONT"])
                cam_stem = os.path.basename(cam_data["filename"]).rsplit(".", 1)[0]
                rel = fname_to_relpath.get(cam_stem)
                if rel is None:
                    n_miss += 1
                    continue
                out_path = os.path.join(out_root, rel + ".npy")
                if os.path.exists(out_path):
                    n_skip += 1
                    continue

                ego_xyz, _ = process_sample(nusc, sample["token"])
                os.makedirs(os.path.dirname(out_path), exist_ok=True)
                np.save(out_path, ego_xyz)
                n_done += 1
            except Exception as e:                               # pragma: no cover
                logger.warning(f"sample {sample['token']} failed: {e}")
        del nusc

    logger.info(f"DONE  wrote={n_done}  skipped(existing)={n_skip}  not-in-splits={n_miss}")


if __name__ == "__main__":
    main()
