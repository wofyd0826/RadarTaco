#!/usr/bin/env python3
"""Convert points_compare.json into per-sample CSVs.

Each row = one GT lidar point (x, y), with the colocated prediction
(same pixel) and the nearest radar point (in image-pixel space) attached
for comparison.

Output:
    output/baseline/eval_test/points_compare/{label}.csv
    output/baseline/eval_test/points_compare/all.csv     # 4 samples concatenated
"""
import csv
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IN_PATH = os.path.join(ROOT, "output/baseline/eval_test/points_compare.json")
OUT_DIR = os.path.join(ROOT, "output/baseline/eval_test/points_compare")

FIELDS = [
    "sample_id", "label",
    "x", "y",
    "gt_depth", "pred_depth", "pred_abs_err",
    "nearest_radar_x", "nearest_radar_y", "nearest_radar_depth",
    "radar_pix_dist", "radar_depth_abs_err",
]


def rows_for_sample(sample):
    gt = sample["gt_points"]
    pred = sample["pred_points"]
    radar = sample["radar_points"]
    assert len(gt) == len(pred)

    if len(radar) == 0:
        rad_xy = np.zeros((0, 2), dtype=np.float32)
        rad_d = np.zeros((0,), dtype=np.float32)
    else:
        rad_xy = np.array([[r["x"], r["y"]] for r in radar], dtype=np.float32)
        rad_d = np.array([r["depth"] for r in radar], dtype=np.float32)

    rows = []
    for g, p in zip(gt, pred):
        x, y = g["x"], g["y"]
        row = {
            "sample_id": sample["sample_id"],
            "label": sample["label"],
            "x": x, "y": y,
            "gt_depth": g["depth"],
            "pred_depth": p["depth"],
            "pred_abs_err": p["abs_err"],
        }
        if len(radar) == 0:
            row.update({
                "nearest_radar_x": "", "nearest_radar_y": "",
                "nearest_radar_depth": "", "radar_pix_dist": "",
                "radar_depth_abs_err": "",
            })
        else:
            d = np.hypot(rad_xy[:, 0] - x, rad_xy[:, 1] - y)
            j = int(np.argmin(d))
            row.update({
                "nearest_radar_x": float(rad_xy[j, 0]),
                "nearest_radar_y": float(rad_xy[j, 1]),
                "nearest_radar_depth": float(rad_d[j]),
                "radar_pix_dist": float(d[j]),
                "radar_depth_abs_err": float(abs(rad_d[j] - g["depth"])),
            })
        rows.append(row)
    return rows


def main():
    with open(IN_PATH) as f:
        data = json.load(f)
    os.makedirs(OUT_DIR, exist_ok=True)

    all_rows = []
    for s in data["samples"]:
        rows = rows_for_sample(s)
        all_rows.extend(rows)
        path = os.path.join(OUT_DIR, f"{s['label']}.csv")
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            w.writeheader()
            w.writerows(rows)
        print(f"  {s['label']:8s}  {len(rows):5d} rows  →  {path}")

    path_all = os.path.join(OUT_DIR, "all.csv")
    with open(path_all, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(all_rows)
    print(f"  ALL       {len(all_rows):5d} rows  →  {path_all}")


if __name__ == "__main__":
    sys.exit(main())
