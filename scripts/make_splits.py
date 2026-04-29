#!/usr/bin/env python3
"""Generate day/night sub-splits for nuScenes from `scene.json` descriptions.

Produces (next to existing `splits/{train,val,test}.txt`):
    splits/night_ids.txt   — set of "scene_XXXX/cam_filename" sample_ids
                             belonging to night scenes (used by the dataset
                             loader to tag per-sample `is_night`)
    splits/val_day.txt     — val.txt rows whose sample_id is NOT in night_ids
    splits/val_night.txt   — val.txt rows whose sample_id IS in night_ids
    splits/train_day.txt
    splits/train_night.txt

Detects night via case-insensitive substring "night" in `scene.description`.

Usage:
    python scripts/make_splits.py
    python scripts/make_splits.py --splits-dir /data/public/nuScenes/derived/splits
"""
import argparse
import json
import logging
import os
import sys

logger = logging.getLogger("make_splits")
logging.basicConfig(format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                    level=logging.INFO)


def collect_night_scene_names(meta_dirs):
    """Return set of nuScenes scene names (e.g. 'scene-0001') marked as night."""
    night = set()
    for meta_dir in meta_dirs:
        path = os.path.join(meta_dir, "scene.json")
        if not os.path.exists(path):
            logger.warning("missing %s", path)
            continue
        with open(path) as f:
            scenes = json.load(f)
        for s in scenes:
            desc = (s.get("description") or "").lower()
            if "night" in desc:
                night.add(s["name"])
        logger.info("%s → %d total / %d night cumulative", path, len(scenes), len(night))
    return night


def night_scene_to_sample_id_prefix(name: str) -> str:
    """nuScenes scene name 'scene-0001' → split sample_id prefix 'scene_0001/'."""
    return name.replace("-", "_") + "/"


def split_by_night(split_path: str, night_prefixes: set):
    if not os.path.exists(split_path):
        logger.warning("missing %s — skipping", split_path)
        return None, None
    day, night = [], []
    with open(split_path) as f:
        for line in f:
            line_s = line.rstrip("\n")
            if not line_s.strip():
                continue
            sample_id = line_s.split("\t", 1)[0]
            if any(sample_id.startswith(p) for p in night_prefixes):
                night.append(line_s)
            else:
                day.append(line_s)
    return day, night


def write_lines(path: str, lines):
    with open(path, "w") as f:
        f.write("\n".join(lines) + ("\n" if lines else ""))
    logger.info("wrote %d lines → %s", len(lines), path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits-dir", default="/data/public/nuScenes/derived/splits")
    parser.add_argument("--meta-dirs", nargs="+", default=[
        "/data/public/nuScenes/trainval/v1.0-trainval",
        "/data/public/nuScenes/test/v1.0-test",
    ])
    args = parser.parse_args()

    if not os.path.isdir(args.splits_dir):
        logger.error("splits dir does not exist: %s", args.splits_dir)
        sys.exit(2)

    night_names = collect_night_scene_names(args.meta_dirs)
    night_prefixes = {night_scene_to_sample_id_prefix(n) for n in night_names}
    logger.info("night scene prefixes: %d", len(night_prefixes))

    # Build sample_id-level night index by scanning all splits.
    night_ids = []
    for split in ("train", "val", "test"):
        sp = os.path.join(args.splits_dir, f"{split}.txt")
        if not os.path.exists(sp):
            continue
        with open(sp) as f:
            for line in f:
                sample_id = line.split("\t", 1)[0].strip()
                if any(sample_id.startswith(p) for p in night_prefixes):
                    night_ids.append(sample_id)
    write_lines(os.path.join(args.splits_dir, "night_ids.txt"), night_ids)

    for split in ("train", "val"):
        sp = os.path.join(args.splits_dir, f"{split}.txt")
        day, night = split_by_night(sp, night_prefixes)
        if day is None:
            continue
        write_lines(os.path.join(args.splits_dir, f"{split}_day.txt"), day)
        write_lines(os.path.join(args.splits_dir, f"{split}_night.txt"), night)

    logger.info("done.")


if __name__ == "__main__":
    main()
