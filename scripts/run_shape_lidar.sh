#!/usr/bin/env bash
# End-to-end pipeline: train → eval → diagnose for a shape_lidar variant.
#
# Defaults match the "reduce LiDAR + add edge smoothness" recipe for
# attenuating LiDAR-ring stripes:
#     loss.w_lidar=0.5, loss.w_smooth=0.1
#
# All three phases run sequentially on a single GPU. Each phase can be
# skipped via env vars (useful for re-running just eval/diagnose on an
# existing checkpoint).
#
# Usage:
#   # Default: train+eval+diagnose on GPU 0 with the recipe above
#   bash scripts/run_shape_lidar.sh
#
#   # Different hyperparams, different GPU, custom run name
#   GPU=1 W_LIDAR=0.3 W_SMOOTH=0.05 bash scripts/run_shape_lidar.sh
#   RUN_NAME=my_experiment bash scripts/run_shape_lidar.sh
#
#   # Warm-start from an earlier ckpt
#   LOAD_WEIGHTS=output/shape_lidar/best.pt bash scripts/run_shape_lidar.sh
#
#   # Skip training (already trained, re-run only eval+diagnose)
#   SKIP_TRAIN=1 RUN_NAME=shape_lidar_w0.5_smooth0.1 \
#     bash scripts/run_shape_lidar.sh
#
#   # Diagnose only, all 4 splits
#   SKIP_TRAIN=1 SKIP_EVAL=1 RUN_NAME=shape_lidar_w0.5_smooth0.1 \
#     DIAG_SPLITS="val val_day val_night test" \
#     bash scripts/run_shape_lidar.sh
#
# Env overrides (with defaults):
#   GPU=3                  # CUDA_VISIBLE_DEVICES
#   W_LIDAR=0.5            # loss.w_lidar override
#   W_SMOOTH=0.1           # loss.w_smooth override
#   RUN_NAME=shape_lidar_w${W_LIDAR}_smooth${W_SMOOTH}
#   EXPERIMENT=shape_lidar # +experiment=<name>
#   LOAD_WEIGHTS=          # training.load_weights_from=<path>
#   EVAL_SPLIT=test        # eval_split=<name>
#   DIAG_SPLITS="test"     # space-separated list of splits to diagnose
#   EXTRA_TRAIN_ARGS=      # extra Hydra overrides for train.py
#   SKIP_TRAIN=0           # 1 to skip
#   SKIP_EVAL=0
#   SKIP_DIAG=0
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
cd "$ROOT"

GPU="${GPU:-3}"
W_LIDAR="${W_LIDAR:-0.5}"
W_SMOOTH="${W_SMOOTH:-0.1}"
RUN_NAME="${RUN_NAME:-shape_lidar_w${W_LIDAR}_smooth${W_SMOOTH}}"
EXPERIMENT="${EXPERIMENT:-shape_lidar}"
LOAD_WEIGHTS="${LOAD_WEIGHTS:-}"
EVAL_SPLIT="${EVAL_SPLIT:-test}"
DIAG_SPLITS="${DIAG_SPLITS:-test}"
EXTRA_TRAIN_ARGS="${EXTRA_TRAIN_ARGS:-}"
SKIP_TRAIN="${SKIP_TRAIN:-0}"
SKIP_EVAL="${SKIP_EVAL:-0}"
SKIP_DIAG="${SKIP_DIAG:-0}"

OUT_DIR="$ROOT/output/$RUN_NAME"

banner() {
    echo
    echo "================================================================"
    echo "$1"
    echo "================================================================"
}

banner "Plan"
echo "  GPU             = $GPU"
echo "  RUN_NAME        = $RUN_NAME"
echo "  EXPERIMENT      = $EXPERIMENT"
echo "  W_LIDAR         = $W_LIDAR"
echo "  W_SMOOTH        = $W_SMOOTH"
echo "  LOAD_WEIGHTS    = ${LOAD_WEIGHTS:-(none)}"
echo "  EVAL_SPLIT      = $EVAL_SPLIT"
echo "  DIAG_SPLITS     = $DIAG_SPLITS"
echo "  EXTRA_TRAIN_ARGS= ${EXTRA_TRAIN_ARGS:-(none)}"
echo "  SKIP_TRAIN/EVAL/DIAG = $SKIP_TRAIN/$SKIP_EVAL/$SKIP_DIAG"
echo "  expected OUT_DIR= $OUT_DIR"

T_START=$(date +%s)

# -------- 1. Training -----------------------------------------------------
if [[ "$SKIP_TRAIN" != "1" ]]; then
    banner "[1/3] Training: $RUN_NAME"
    TRAIN_CMD=(python3 -u scripts/train.py
               "+experiment=$EXPERIMENT"
               "loss.w_lidar=$W_LIDAR"
               "loss.w_smooth=$W_SMOOTH"
               "wandb.name=$RUN_NAME")
    if [[ -n "$LOAD_WEIGHTS" ]]; then
        TRAIN_CMD+=("training.load_weights_from=$LOAD_WEIGHTS")
    fi
    if [[ -n "$EXTRA_TRAIN_ARGS" ]]; then
        # shellcheck disable=SC2206
        TRAIN_CMD+=($EXTRA_TRAIN_ARGS)
    fi
    echo "[cmd] CUDA_VISIBLE_DEVICES=$GPU ${TRAIN_CMD[*]}"
    CUDA_VISIBLE_DEVICES="$GPU" "${TRAIN_CMD[@]}"

    # train.py auto-suffixes the dir (_2, _3, ...) if RUN_NAME already
    # exists. Pick the most recently modified directory matching the
    # prefix as the actual output.
    LATEST="$(ls -dt "$ROOT"/output/"$RUN_NAME"* 2>/dev/null | head -1 || true)"
    if [[ -n "$LATEST" && -f "$LATEST/best.pt" ]]; then
        OUT_DIR="$LATEST"
    fi
    if [[ ! -f "$OUT_DIR/best.pt" ]]; then
        echo "[error] no best.pt produced at $OUT_DIR" >&2
        exit 1
    fi
    echo "[ok] training done — actual output dir: $OUT_DIR"
else
    if [[ ! -f "$OUT_DIR/best.pt" ]]; then
        echo "[error] SKIP_TRAIN=1 but $OUT_DIR/best.pt does not exist." >&2
        echo "        Set RUN_NAME to the name of an existing run."     >&2
        exit 1
    fi
    echo "[skip] training — using existing ckpt at $OUT_DIR/best.pt"
fi

# -------- 2. Evaluation ---------------------------------------------------
if [[ "$SKIP_EVAL" != "1" ]]; then
    banner "[2/3] Eval on $EVAL_SPLIT"
    EVAL_LOG="$OUT_DIR/eval_${EVAL_SPLIT}.log"
    # Note: eval.py's root config doesn't define `checkpoint` or
    # `eval_split` — they must be added via `+` (Hydra's append syntax).
    CUDA_VISIBLE_DEVICES="$GPU" python3 -u scripts/eval.py \
        "+checkpoint=$OUT_DIR/best.pt" \
        "+eval_split=$EVAL_SPLIT" \
        2>&1 | tee "$EVAL_LOG"
    echo "[ok] eval done — $OUT_DIR/eval_independent/ (log: $EVAL_LOG)"
else
    echo "[skip] eval"
fi

# -------- 3. Diagnose ------------------------------------------------------
if [[ "$SKIP_DIAG" != "1" ]]; then
    banner "[3/3] Diagnose shape losses (splits: $DIAG_SPLITS)"
    for SPLIT in $DIAG_SPLITS; do
        echo
        echo "  --- $SPLIT ---"
        CUDA_VISIBLE_DEVICES="$GPU" python3 -u scripts/diagnose_shape_night.py \
            --checkpoint "$OUT_DIR/best.pt" \
            --split "$SPLIT"
    done
    echo "[ok] diagnose done — $OUT_DIR/diagnose_shape/"
else
    echo "[skip] diagnose"
fi

T_END=$(date +%s)
DURATION=$((T_END - T_START))
HOURS=$((DURATION / 3600))
MINUTES=$(( (DURATION % 3600) / 60 ))

banner "[done] all requested phases for $RUN_NAME"
echo "  total runtime  = ${HOURS}h ${MINUTES}m"
echo "  ckpt           = $OUT_DIR/best.pt"
echo "  config         = $OUT_DIR/config.yaml"
echo "  eval results   = $OUT_DIR/eval_independent/"
echo "  diag results   = $OUT_DIR/diagnose_shape/"
