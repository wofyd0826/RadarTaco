#!/usr/bin/env bash
# Launch N parallel workers on the SAME GPU to refine the nuScenes train
# (or val) split in parallel. Each worker handles a shard of the split
# (row % N == i) using the --shard flag of infer_da_v2_multi_refine.py.
#
# Multiple DA-v2 ViT-L copies fit comfortably in 97GB VRAM:
#   weights ~1.3GB + activations at 900×1600 ~6-8GB → ~8GB per worker.
#   N=4 fits with margin; N=6-8 if you want to push it.
#
# Usage:
#   GPU=3 N=4 SPLIT=train bash scripts/refine_train_parallel.sh
#   GPU=3 N=4 SPLIT=val   bash scripts/refine_train_parallel.sh
#
# Env (with defaults):
#   GPU=3
#   N=4
#   SPLIT=train                          ("train" / "val" / "test" or a path ending in .txt)
#   DATA_ROOT=/data/public/nuScenes/derived
#   GT_DIR=depth_lidar
#   ENCODER=vitl
#   LOG_DIR=<repo>/output/da_v2_multi_refined_logs
#   EXTRA_ARGS=                           (e.g. "--grid_K 16")
#
# After launching, all workers run in the background. Monitor via:
#   tail -f <LOG_DIR>/<split>_shard*.log
#   jobs -l   (in this shell)
#   ps -fC python3 | grep infer_da_v2_multi
#
# To stop all workers in this GPU:
#   pkill -f "infer_da_v2_multi_refine.*--shard"
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"

GPU="${GPU:-3}"
N="${N:-4}"
SPLIT="${SPLIT:-train}"
DATA_ROOT="${DATA_ROOT:-/data/public/nuScenes/derived}"
GT_DIR="${GT_DIR:-depth_lidar}"
ENCODER="${ENCODER:-vitl}"
LOG_DIR="${LOG_DIR:-${ROOT}/output/da_v2_multi_refined_logs}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

OUT_GLOBAL="${DATA_ROOT}/depth_refined"
OUT_GRID="${DATA_ROOT}/depth_refined_grid"
OUT_RANDOM="${DATA_ROOT}/depth_refined_random"

# Resolve split file
if [[ "$SPLIT" == *.txt ]]; then
    SPLIT_FILE="$SPLIT"
    SPLIT_NAME="$(basename "$SPLIT" .txt)"
else
    SPLIT_FILE="${DATA_ROOT}/splits/${SPLIT}.txt"
    SPLIT_NAME="$SPLIT"
fi
[[ -f "$SPLIT_FILE" ]] || { echo "missing split file: $SPLIT_FILE" >&2; exit 1; }

mkdir -p "$OUT_GLOBAL" "$OUT_GRID" "$OUT_RANDOM" "$LOG_DIR"

N_TOTAL=$(wc -l < "$SPLIT_FILE")
echo "[refine_parallel] GPU=$GPU  N_workers=$N  split=$SPLIT_NAME  total=$N_TOTAL"
echo "  → global = $OUT_GLOBAL"
echo "  → grid   = $OUT_GRID"
echo "  → random = $OUT_RANDOM"
echo "  log_dir  = $LOG_DIR"

PIDS=()
for ((i=0; i<N; i++)); do
    LOG_FILE="${LOG_DIR}/${SPLIT_NAME}_shard${i}of${N}.log"
    CUDA_VISIBLE_DEVICES="$GPU" HF_HUB_OFFLINE=1 \
        nohup python3 -u "${ROOT}/scripts/infer_da_v2_multi_refine.py" \
            --split     "$SPLIT_FILE" \
            --data_root "$DATA_ROOT" \
            --out_global "$OUT_GLOBAL" \
            --out_grid   "$OUT_GRID" \
            --out_random "$OUT_RANDOM" \
            --gt_dir    "$GT_DIR" \
            --encoder   "$ENCODER" \
            --shard     "${i}/${N}" \
            $EXTRA_ARGS \
            > "$LOG_FILE" 2>&1 &
    PID=$!
    PIDS+=($PID)
    echo "[launched] shard ${i}/${N}  pid=$PID  log=$LOG_FILE"
done

echo
echo "[summary] PIDs: ${PIDS[*]}"
echo "  monitor:  tail -f ${LOG_DIR}/${SPLIT_NAME}_shard*.log"
echo "  wait for all:  wait ${PIDS[*]}"
