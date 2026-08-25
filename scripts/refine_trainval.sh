#!/usr/bin/env bash
# Refine nuScenes train + val splits with DA-v2 + MoGe-2 ROE.
#
# Anchor source: depth_lidar (single-frame sparse LiDAR). This matches the
# anchor used on the test split, so the refined depth is computed under a
# consistent procedure across train/val/test. (depth_acc is denser on
# train/val but not available on test — using it would make the splits
# inconsistent.)
#
# Output: <data_root>/depth_refined/<sample_id>.png  (uint16 ×256 m)
# Resumes via --skip_existing (default ON).
#
# Usage:
#   CUDA_VISIBLE_DEVICES=3 bash scripts/refine_trainval.sh
#
# Env overrides (optional):
#   DATA_ROOT, OUT_DIR, ENCODER, GT_DIR, EXTRA_ARGS, LOG_DIR
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"

DATA_ROOT="${DATA_ROOT:-/data/public/nuScenes/derived}"
OUT_DIR="${OUT_DIR:-${DATA_ROOT}/depth_refined}"
ENCODER="${ENCODER:-vitl}"
GT_DIR="${GT_DIR:-depth_lidar}"
LOG_DIR="${LOG_DIR:-${ROOT}/output/da_v2_refined_logs}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

mkdir -p "$OUT_DIR" "$LOG_DIR"

echo "[refine_trainval] data_root=$DATA_ROOT"
echo "[refine_trainval] out_dir=$OUT_DIR"
echo "[refine_trainval] gt_dir=$GT_DIR  encoder=$ENCODER"
echo "[refine_trainval] log_dir=$LOG_DIR"

for SPLIT in val train; do
    SPLIT_FILE="${DATA_ROOT}/splits/${SPLIT}.txt"
    LOG_FILE="${LOG_DIR}/${SPLIT}.log"
    if [[ ! -f "$SPLIT_FILE" ]]; then
        echo "[skip] missing split file: $SPLIT_FILE"
        continue
    fi
    N=$(wc -l < "$SPLIT_FILE")
    echo "[run] split=$SPLIT  N=$N  →  $LOG_FILE"
    python3 -u "${ROOT}/scripts/infer_da_v2_refined.py" \
        --split "$SPLIT_FILE" \
        --data_root "$DATA_ROOT" \
        --out_root "$OUT_DIR" \
        --gt_dir "$GT_DIR" \
        --encoder "$ENCODER" \
        $EXTRA_ARGS \
        2>&1 | tee "$LOG_FILE"
    echo "[done] split=$SPLIT"
done

# Sanity report
echo
echo "[summary] refined files per split:"
for SPLIT in train val test; do
    SPLIT_FILE="${DATA_ROOT}/splits/${SPLIT}.txt"
    [[ -f "$SPLIT_FILE" ]] || continue
    # Count refined files whose sample_id appears in this split.
    M=$(awk -F'\t' '{print $1}' "$SPLIT_FILE" \
        | while read sid; do
            [[ -f "${OUT_DIR}/${sid}.png" ]] && echo 1
          done | wc -l)
    N=$(wc -l < "$SPLIT_FILE")
    printf "  %-5s  %d / %d\n" "$SPLIT" "$M" "$N"
done
