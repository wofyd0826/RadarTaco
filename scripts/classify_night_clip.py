#!/usr/bin/env python3
"""CLIP zero-shot day/night classification for nuScenes.

Why this exists: nuScenes test set has its scene descriptions deliberately
masked ("## No descriptions available for the test set. ##"), so the
keyword-based `make_splits.py` cannot tag night samples there. This script
runs CLIP-ViT-B/32 zero-shot on one representative frame per scene and
outputs a complete `night_ids.txt` covering train + val + test.

Approach:
  1. Group all sample_ids in {train,val,test}.txt by scene_XXXX prefix.
  2. Pick the first frame of each scene (~1000 scenes total).
  3. CLIP zero-shot vs prompts:
        positive: "a photo taken at night"
        negative: "a photo taken during the day"
  4. (Optional) validate accuracy against trainval ground truth from
     scene.json (99 known night scenes / 850 trainval scenes).
  5. Write `night_ids.txt` (one sample_id per line) covering ALL frames
     of every classified-night scene.

Usage:
    python scripts/classify_night_clip.py
    python scripts/classify_night_clip.py --validate-only       # just check accuracy on trainval
    python scripts/classify_night_clip.py --batch-size 64
"""
import argparse
import json
import logging
import os
import sys
from collections import defaultdict
from typing import Dict, List, Set, Tuple

import torch
from PIL import Image
from tqdm import tqdm

logging.basicConfig(format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                    level=logging.INFO)
logger = logging.getLogger("classify_night_clip")

PROMPT_NIGHT = "a photo taken at night"
PROMPT_DAY = "a photo taken during the day"


def load_split(path: str) -> List[Tuple[str, str]]:
    """Return list of (sample_id, image_path) from a split TSV."""
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            out.append((parts[0], parts[1]))
    return out


def gt_night_scenes(meta_dirs: List[str]) -> Set[str]:
    """Load ground-truth night scene names (e.g. 'scene-0992') from scene.json."""
    night = set()
    for d in meta_dirs:
        p = os.path.join(d, "scene.json")
        if not os.path.exists(p):
            continue
        with open(p) as f:
            for s in json.load(f):
                if "night" in (s.get("description") or "").lower():
                    night.add(s["name"])
    return night


def scene_prefix_from_sample_id(sample_id: str) -> str:
    """'scene_0001/n008-...' → 'scene_0001'"""
    return sample_id.split("/", 1)[0]


def cam_fname_from_sample_id(sample_id: str) -> str:
    """'scene_0001/n008-..._CAM_FRONT__123' → 'n008-..._CAM_FRONT__123'"""
    return sample_id.split("/", 1)[1] if "/" in sample_id else sample_id


def build_camfname_to_scene_map(meta_dirs: List[str]) -> Dict[str, str]:
    """Map cam filename (basename without extension) → official nuScenes scene name.

    The split files in `derived/splits/` use a local scene numbering
    ('scene_0000' through 'scene_10149') that does NOT match the official
    nuScenes scene names ('scene-0001' through 'scene-1153'). The only
    reliable bridge is the cam_data filename, which is unique across the
    whole dataset and present in `sample_data.json`.
    """
    out: Dict[str, str] = {}
    for d in meta_dirs:
        sample_p = os.path.join(d, "sample.json")
        sd_p = os.path.join(d, "sample_data.json")
        scene_p = os.path.join(d, "scene.json")
        if not (os.path.exists(sample_p) and os.path.exists(sd_p) and os.path.exists(scene_p)):
            continue
        with open(sample_p) as f:
            sample_to_scene = {s["token"]: s["scene_token"] for s in json.load(f)}
        with open(scene_p) as f:
            scene_token_to_name = {s["token"]: s["name"] for s in json.load(f)}
        with open(sd_p) as f:
            for sd in json.load(f):
                fn = os.path.basename(sd["filename"]).rsplit(".", 1)[0]
                scene_tok = sample_to_scene.get(sd["sample_token"])
                if scene_tok:
                    out[fn] = scene_token_to_name[scene_tok]
    return out


@torch.inference_mode()
def classify_scenes(
    image_paths: List[str],
    device: str = "cuda",
    batch_size: int = 64,
    model_name: str = "openai/clip-vit-base-patch32",
) -> List[float]:
    """Return P(night) ∈ [0, 1] for each image using CLIP zero-shot."""
    from transformers import CLIPModel, CLIPProcessor
    logger.info(f"loading CLIP model: {model_name}")
    model = CLIPModel.from_pretrained(model_name).to(device).eval()
    processor = CLIPProcessor.from_pretrained(model_name)

    text_inputs = processor(text=[PROMPT_NIGHT, PROMPT_DAY],
                            return_tensors="pt", padding=True)
    text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
    text_out = model.get_text_features(**text_inputs)
    # transformers 5.5 returns BaseModelOutputWithPooling; older returns Tensor
    text_feats = text_out.pooler_output if hasattr(text_out, "pooler_output") else text_out
    text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)

    p_night = []
    for i in tqdm(range(0, len(image_paths), batch_size), desc="CLIP"):
        batch_paths = image_paths[i:i + batch_size]
        images = [Image.open(p).convert("RGB") for p in batch_paths]
        img_inputs = processor(images=images, return_tensors="pt")
        img_inputs = {k: v.to(device) for k, v in img_inputs.items()}
        img_out = model.get_image_features(**img_inputs)
        img_feats = img_out.pooler_output if hasattr(img_out, "pooler_output") else img_out
        img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)
        # cosine similarity, scaled by CLIP's logit_scale, softmax over [night, day]
        logits = (model.logit_scale.exp() * img_feats @ text_feats.t())
        probs = logits.softmax(dim=-1)
        p_night.extend(probs[:, 0].cpu().tolist())
    return p_night


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits-dir", default="/data/public/nuScenes/derived/splits")
    parser.add_argument("--meta-dirs", nargs="+", default=[
        "/data/public/nuScenes/trainval/v1.0-trainval",
        "/data/public/nuScenes/test/v1.0-test",
    ])
    parser.add_argument("--out", default=None,
                        help="output night_ids.txt (default: <splits-dir>/night_ids.txt)")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="P(night) > threshold → night (default 0.5)")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--validate-only", action="store_true",
                        help="run on trainval only, report accuracy, do NOT write output")
    parser.add_argument("--predictions-csv", default=None,
                        help="optional CSV dump: scene_prefix, p_night, gt_night, predicted_night")
    args = parser.parse_args()

    out_path = args.out or os.path.join(args.splits_dir, "night_ids.txt")

    # 1) Collect (sample_id, image_path) per split.
    all_samples: List[Tuple[str, str, str]] = []  # (split, sample_id, image_path)
    for split in ("train", "val", "test"):
        sp = os.path.join(args.splits_dir, f"{split}.txt")
        if not os.path.exists(sp):
            logger.warning(f"missing {sp}")
            continue
        rows = load_split(sp)
        all_samples.extend((split, sid, ip) for sid, ip in rows)
        logger.info(f"{split}: {len(rows)} frames")

    # 2) Group by scene; pick first sample per scene as the representative frame.
    scene_to_samples: Dict[str, List[Tuple[str, str, str]]] = defaultdict(list)
    for split, sid, ip in all_samples:
        scene_to_samples[scene_prefix_from_sample_id(sid)].append((split, sid, ip))
    scene_prefixes = sorted(scene_to_samples.keys())
    rep_paths = [scene_to_samples[s][0][2] for s in scene_prefixes]
    logger.info(f"unique scenes: {len(scene_prefixes)}")

    # 3) CLIP inference.
    p_night = classify_scenes(rep_paths, device=args.device, batch_size=args.batch_size)
    pred_night_scenes = {s for s, p in zip(scene_prefixes, p_night) if p > args.threshold}
    logger.info(f"predicted night scenes: {len(pred_night_scenes)}")

    # 4) Validate against trainval ground truth.
    # Build cam_fname → official_scene_name map and look up split scenes.
    logger.info("building cam_filename → nuScenes scene_name map "
                "(needed because split-local scene numbering ≠ official scene names)…")
    fname_to_scene = build_camfname_to_scene_map(args.meta_dirs)
    logger.info(f"  {len(fname_to_scene)} cam filenames in nuScenes metadata")

    # For each split scene_prefix, take its representative frame's filename
    # and find the official scene_name via the metadata mapping.
    split_to_official = {}
    for s in scene_prefixes:
        rep_fname = cam_fname_from_sample_id(scene_to_samples[s][0][1])
        official = fname_to_scene.get(rep_fname)
        if official is not None:
            split_to_official[s] = official

    # Determine which split scenes belong to trainval (have descriptions).
    trainval_official = set()
    for d in args.meta_dirs:
        if "trainval" not in d:
            continue
        p = os.path.join(d, "scene.json")
        if os.path.exists(p):
            with open(p) as f:
                trainval_official.update(s["name"] for s in json.load(f))

    gt_night_official = gt_night_scenes(args.meta_dirs)  # in trainval only

    eval_scenes = [s for s in scene_prefixes
                   if split_to_official.get(s) in trainval_official]
    p_by_scene = dict(zip(scene_prefixes, p_night))
    tp = fp = fn = tn = 0
    for s in eval_scenes:
        official = split_to_official[s]
        is_night_gt = official in gt_night_official
        is_night_pred = p_by_scene[s] > args.threshold
        if is_night_pred and is_night_gt:    tp += 1
        elif is_night_pred and not is_night_gt: fp += 1
        elif not is_night_pred and is_night_gt: fn += 1
        else:                                tn += 1
    n = max(tp + fp + fn + tn, 1)
    acc = (tp + tn) / n
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-8)
    logger.info(f"trainval validation:  acc={acc:.4f}  prec={prec:.4f}  rec={rec:.4f}  f1={f1:.4f}  "
                f"(TP={tp}  FP={fp}  FN={fn}  TN={tn},  n={n})")
    logger.info(f"  GT night scenes (trainval): {sum(1 for s in eval_scenes if split_to_official[s] in gt_night_official)}")
    logger.info(f"  predicted night (trainval only): "
                f"{sum(1 for s in eval_scenes if p_by_scene[s] > args.threshold)}")

    # Optional CSV dump for inspection.
    if args.predictions_csv:
        import csv
        with open(args.predictions_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["scene_prefix", "official_scene", "p_night", "gt_night", "predicted_night"])
            for s, p in zip(scene_prefixes, p_night):
                official = split_to_official.get(s, "")
                in_gt = int(official in gt_night_official) if official in trainval_official else -1
                w.writerow([s, official, f"{p:.6f}", in_gt, int(p > args.threshold)])
        logger.info(f"wrote predictions CSV → {args.predictions_csv}")

    if args.validate_only:
        logger.info("--validate-only: not writing night_ids.txt")
        return

    # 5) Expand predicted night scenes back to all sample_ids and write.
    night_sample_ids: List[str] = []
    for split, sid, _ in all_samples:
        if scene_prefix_from_sample_id(sid) in pred_night_scenes:
            night_sample_ids.append(sid)
    with open(out_path, "w") as f:
        f.write("\n".join(night_sample_ids) + ("\n" if night_sample_ids else ""))
    logger.info(f"wrote {len(night_sample_ids)} night sample_ids → {out_path}")
    # Per-split count for sanity.
    by_split = defaultdict(int)
    for split, sid, _ in all_samples:
        if scene_prefix_from_sample_id(sid) in pred_night_scenes:
            by_split[split] += 1
    for split, n_split in by_split.items():
        logger.info(f"  {split}: {n_split} night frames")


if __name__ == "__main__":
    main()
