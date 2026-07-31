#!/usr/bin/env bash
# Train + eval the four per-tag specialist models derived from the
# shape_lidar_grad_shape_edge_res recipe. Each trains a fresh model on a
# single (day/night × clear/rain) subset, then evaluates on that subset's
# test split.
#
# Usage:
#   # All 4 specialists sequentially on GPU 3 (default)
#   bash scripts/run_specialists.sh
#
#   # A single specialist
#   TAGS=day_rain bash scripts/run_specialists.sh
#
#   # Custom GPU / skip eval
#   GPU=0 SKIP_EVAL=1 bash scripts/run_specialists.sh
#
# Env overrides (with defaults):
#   GPU=3                  # CUDA_VISIBLE_DEVICES
#   TAGS="day_clear day_rain night_clear night_rain"   # space-separated
#   OUT_ROOT=./output      # per-run dir: $OUT_ROOT/specialist_<tag>
#   SKIP_TRAIN=0
#   SKIP_EVAL=0
#   EXTRA_TRAIN_ARGS=      # extra Hydra overrides for train.py
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
cd "$ROOT"

GPU="${GPU:-3}"
TAGS="${TAGS:-day_clear day_rain night_clear night_rain}"
OUT_ROOT="${OUT_ROOT:-$ROOT/output}"
SKIP_TRAIN="${SKIP_TRAIN:-0}"
SKIP_EVAL="${SKIP_EVAL:-0}"
EXTRA_TRAIN_ARGS="${EXTRA_TRAIN_ARGS:-}"

banner() {
    echo
    echo "================================================================"
    echo "$1"
    echo "================================================================"
}

banner "Plan"
echo "  GPU              = $GPU"
echo "  TAGS             = $TAGS"
echo "  OUT_ROOT         = $OUT_ROOT"
echo "  SKIP_TRAIN/EVAL  = $SKIP_TRAIN/$SKIP_EVAL"
echo "  EXTRA_TRAIN_ARGS = ${EXTRA_TRAIN_ARGS:-(none)}"

T_START=$(date +%s)

for TAG in $TAGS; do
    RUN_NAME="specialist_${TAG}"
    OUT_DIR="$OUT_ROOT/$RUN_NAME"

    banner "[$TAG] training"
    if [[ "$SKIP_TRAIN" != "1" ]]; then
        TRAIN_CMD=(python3 -u scripts/train.py
                   "+experiment=specialist_${TAG}"
                   "wandb.name=$RUN_NAME")
        if [[ -n "$EXTRA_TRAIN_ARGS" ]]; then
            # shellcheck disable=SC2206
            TRAIN_CMD+=($EXTRA_TRAIN_ARGS)
        fi
        echo "[cmd] CUDA_VISIBLE_DEVICES=$GPU ${TRAIN_CMD[*]}"
        CUDA_VISIBLE_DEVICES="$GPU" "${TRAIN_CMD[@]}"

        # train.py auto-suffixes _2, _3, ... if dir exists — pick newest.
        LATEST="$(ls -dt "$OUT_ROOT"/"$RUN_NAME"* 2>/dev/null | head -1 || true)"
        if [[ -n "$LATEST" && -f "$LATEST/best.pt" ]]; then
            OUT_DIR="$LATEST"
        fi
        if [[ ! -f "$OUT_DIR/best.pt" ]]; then
            echo "[error] no best.pt produced at $OUT_DIR" >&2
            exit 1
        fi
        echo "[ok] training done → $OUT_DIR"
    else
        echo "[skip] training"
    fi

    banner "[$TAG] eval on test_${TAG}"
    if [[ "$SKIP_EVAL" != "1" ]]; then
        EVAL_LOG="$OUT_DIR/eval_test_${TAG}.log"
        CUDA_VISIBLE_DEVICES="$GPU" python3 -u scripts/eval.py \
            "+checkpoint=$OUT_DIR/best.pt" \
            "+eval_split=test_${TAG}" \
            2>&1 | tee "$EVAL_LOG"
        echo "[ok] eval done — log: $EVAL_LOG"
    else
        echo "[skip] eval"
    fi
done

T_END=$(date +%s)
DURATION=$((T_END - T_START))
HOURS=$((DURATION / 3600))
MINUTES=$(( (DURATION % 3600) / 60 ))
banner "[done] all requested tags"
echo "  total runtime = ${HOURS}h ${MINUTES}m"
