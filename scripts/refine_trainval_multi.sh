#!/usr/bin/env bash
# Run DA-v2 + MoGe ROE refinement (3 modes: global / grid / random) on the
# nuScenes train + val splits. Anchor source: depth_lidar (matches test
# split's procedure for consistency).
#
# Outputs (uint16 ×256 m PNG under each dir, partitioned by scene_id):
#   <data_root>/depth_refined/<sample_id>.png         (global)
#   <data_root>/depth_refined_grid/<sample_id>.png    (grid K=8)
#   <data_root>/depth_refined_random/<sample_id>.png  (MoGe-style multi-scale)
#
# Resumes via --skip_existing (default ON). Test set already populated for
# all three modes by infer_da_v2_multi_refine.py — running this wrapper
# does NOT touch test files.
#
# Usage:
#   CUDA_VISIBLE_DEVICES=3 HF_HUB_OFFLINE=1 bash scripts/refine_trainval_multi.sh
#
# Env overrides (optional):
#   DATA_ROOT  — defaults to /data/public/nuScenes/derived
#   GT_DIR     — defaults to depth_lidar
#   ENCODER    — defaults to vitl
#   LOG_DIR    — defaults to <repo>/output/da_v2_multi_refined_logs
#   EXTRA_ARGS — passed through (e.g. "--limit 100", "--grid_K 16")
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"

DATA_ROOT="${DATA_ROOT:-/data/public/nuScenes/derived}"
GT_DIR="${GT_DIR:-depth_lidar}"
ENCODER="${ENCODER:-vitl}"
LOG_DIR="${LOG_DIR:-${ROOT}/output/da_v2_multi_refined_logs}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

OUT_GLOBAL="${DATA_ROOT}/depth_refined"
OUT_GRID="${DATA_ROOT}/depth_refined_grid"
OUT_RANDOM="${DATA_ROOT}/depth_refined_random"

mkdir -p "$OUT_GLOBAL" "$OUT_GRID" "$OUT_RANDOM" "$LOG_DIR"

echo "[refine_trainval_multi] data_root=$DATA_ROOT  gt_dir=$GT_DIR  encoder=$ENCODER"
echo "  → global = $OUT_GLOBAL"
echo "  → grid   = $OUT_GRID"
echo "  → random = $OUT_RANDOM"
echo "  log_dir  = $LOG_DIR"

for SPLIT in val train; do
    SPLIT_FILE="${DATA_ROOT}/splits/${SPLIT}.txt"
    LOG_FILE="${LOG_DIR}/${SPLIT}.log"
    if [[ ! -f "$SPLIT_FILE" ]]; then
        echo "[skip] missing split file: $SPLIT_FILE"
        continue
    fi
    N=$(wc -l < "$SPLIT_FILE")
    echo "[run] split=$SPLIT  N=$N  →  $LOG_FILE"
    python3 -u "${ROOT}/scripts/infer_da_v2_multi_refine.py" \
        --split     "$SPLIT_FILE" \
        --data_root "$DATA_ROOT" \
        --out_global "$OUT_GLOBAL" \
        --out_grid   "$OUT_GRID" \
        --out_random "$OUT_RANDOM" \
        --gt_dir    "$GT_DIR" \
        --encoder   "$ENCODER" \
        $EXTRA_ARGS \
        2>&1 | tee "$LOG_FILE"
    echo "[done] split=$SPLIT"
done

echo
echo "[summary] PNGs per mode × split (matched by sample_id):"
printf "  %-10s  %-7s  %-7s  %-7s\n" "split" "global" "grid" "random"
for SPLIT in train val test; do
    SPLIT_FILE="${DATA_ROOT}/splits/${SPLIT}.txt"
    [[ -f "$SPLIT_FILE" ]] || continue
    N=$(wc -l < "$SPLIT_FILE")
    counts=""
    for OUT in "$OUT_GLOBAL" "$OUT_GRID" "$OUT_RANDOM"; do
        C=$(awk -F'\t' '{print $1}' "$SPLIT_FILE" \
            | while read sid; do [[ -f "${OUT}/${sid}.png" ]] && echo 1; done \
            | wc -l)
        counts+=" ${C}/${N}"
    done
    printf "  %-10s  %s\n" "$SPLIT" "$counts"
done
