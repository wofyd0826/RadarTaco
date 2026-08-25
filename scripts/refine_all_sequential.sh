#!/usr/bin/env bash
# Sequentially refine all splits (val → test → train) with DA-v2 + ROE,
# single worker, single GPU. Computes grid and random; global is
# auto-skipped where PNGs already exist (default --skip_existing).
#
# Order: val (small) → test (medium) → train (large) so smaller splits
# finish first, giving fast confidence the pipeline works end-to-end.
#
# Usage:
#   CUDA_VISIBLE_DEVICES=3 HF_HUB_OFFLINE=1 \
#     bash scripts/refine_all_sequential.sh
#
# Env (with defaults):
#   DATA_ROOT=/data/public/nuScenes/derived
#   GT_DIR=depth_lidar
#   ENCODER=vitl
#   LOG_DIR=<repo>/output/da_v2_multi_refined_logs
#   EXTRA_ARGS=                          (e.g. "--grid_K 16")
#   SPLITS="val test train"              order to process
#
# Resumes via --skip_existing (default ON in the python script). To force
# recompute, set EXTRA_ARGS="--no_skip_existing".
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"

DATA_ROOT="${DATA_ROOT:-/data/public/nuScenes/derived}"
GT_DIR="${GT_DIR:-depth_lidar}"
ENCODER="${ENCODER:-vitl}"
LOG_DIR="${LOG_DIR:-${ROOT}/output/da_v2_multi_refined_logs}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
SPLITS="${SPLITS:-val test train}"

OUT_GLOBAL="${DATA_ROOT}/depth_refined"
OUT_GRID="${DATA_ROOT}/depth_refined_grid"
OUT_RANDOM="${DATA_ROOT}/depth_refined_random"

mkdir -p "$OUT_GLOBAL" "$OUT_GRID" "$OUT_RANDOM" "$LOG_DIR"

echo "[refine_all_sequential] data_root=$DATA_ROOT  gt_dir=$GT_DIR  encoder=$ENCODER"
echo "  splits in order: $SPLITS"
echo "  → global = $OUT_GLOBAL  (auto-skipped if PNG already exists)"
echo "  → grid   = $OUT_GRID"
echo "  → random = $OUT_RANDOM"
echo "  log_dir  = $LOG_DIR"
echo

for SPLIT in $SPLITS; do
    SPLIT_FILE="${DATA_ROOT}/splits/${SPLIT}.txt"
    LOG_FILE="${LOG_DIR}/${SPLIT}.log"
    if [[ ! -f "$SPLIT_FILE" ]]; then
        echo "[skip] missing split file: $SPLIT_FILE"
        continue
    fi
    N=$(wc -l < "$SPLIT_FILE")
    echo "============================================================"
    echo "[run] split=$SPLIT  N=$N  log=$LOG_FILE"
    echo "============================================================"
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
    echo
done

echo
echo "[summary] PNGs per mode × split:"
printf "  %-6s  %-13s  %-13s  %-13s\n" "split" "global" "grid" "random"
for SPLIT in train val test; do
    SPLIT_FILE="${DATA_ROOT}/splits/${SPLIT}.txt"
    [[ -f "$SPLIT_FILE" ]] || continue
    N=$(wc -l < "$SPLIT_FILE")
    counts=""
    for OUT in "$OUT_GLOBAL" "$OUT_GRID" "$OUT_RANDOM"; do
        C=$(awk -F'\t' '{print $1}' "$SPLIT_FILE" \
            | while read sid; do [[ -f "${OUT}/${sid}.png" ]] && echo 1; done \
            | wc -l)
        counts+="  ${C}/${N}"
    done
    printf "  %-6s%s\n" "$SPLIT" "$counts"
done
