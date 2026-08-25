#!/usr/bin/env bash
# End-to-end pipeline for the two stripe-countermeasure experiments:
#   1) shape_lidar_anti_stripe — baseline shape_lidar + AntiStripeLoss
#   2) shape_lidar_grad_shape  — log-depth gradient shape, AffineInvariantL1
#                                 fully replaced
#
# Each experiment goes through train → eval (test split) → diagnose. The two
# experiments run sequentially on a single GPU. All weights are picked up
# from the experiment's yaml — set EXTRA_TRAIN_ARGS to override.
#
# Usage:
#   # Default: both experiments, train+eval+diagnose, GPU 3
#   bash scripts/run_stripe_experiments.sh
#
#   # Single experiment
#   EXPERIMENTS=shape_lidar_anti_stripe \
#     bash scripts/run_stripe_experiments.sh
#
#   # Different GPU
#   GPU=2 bash scripts/run_stripe_experiments.sh
#
#   # Warm-start each run from baseline shape_lidar
#   LOAD_WEIGHTS=output/shape_lidar/best.pt \
#     bash scripts/run_stripe_experiments.sh
#
#   # Skip training (already trained, re-run only eval+diagnose).
#   # RUN_NAMES must match existing output/ directories in the same
#   # order as EXPERIMENTS.
#   SKIP_TRAIN=1 \
#     EXPERIMENTS="shape_lidar_anti_stripe shape_lidar_grad_shape" \
#     RUN_NAMES="shape_lidar_anti_stripe shape_lidar_grad_shape" \
#     bash scripts/run_stripe_experiments.sh
#
#   # Tweak anti-stripe weight
#   EXTRA_TRAIN_ARGS="loss.w_anti_stripe=0.3" \
#     EXPERIMENTS=shape_lidar_anti_stripe \
#     bash scripts/run_stripe_experiments.sh
#
# Env overrides (with defaults):
#   GPU=3                                    # CUDA_VISIBLE_DEVICES
#   EXPERIMENTS="shape_lidar_anti_stripe shape_lidar_grad_shape"
#   RUN_NAMES=                               # space-separated, same length;
#                                              empty = use EXPERIMENTS as names
#   LOAD_WEIGHTS=                            # training.load_weights_from
#   EVAL_SPLIT=test                          # eval split
#   DIAG_SPLITS="test"                       # space-separated splits to diagnose
#   EXTRA_TRAIN_ARGS=                        # extra Hydra overrides for train.py
#   SKIP_TRAIN=0
#   SKIP_EVAL=0
#   SKIP_DIAG=0
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
cd "$ROOT"

GPU="${GPU:-3}"
EXPERIMENTS="${EXPERIMENTS:-shape_lidar_anti_stripe shape_lidar_grad_shape}"
RUN_NAMES="${RUN_NAMES:-}"
LOAD_WEIGHTS="${LOAD_WEIGHTS:-}"
EVAL_SPLIT="${EVAL_SPLIT:-test}"
DIAG_SPLITS="${DIAG_SPLITS:-test}"
EXTRA_TRAIN_ARGS="${EXTRA_TRAIN_ARGS:-}"
SKIP_TRAIN="${SKIP_TRAIN:-0}"
SKIP_EVAL="${SKIP_EVAL:-0}"
SKIP_DIAG="${SKIP_DIAG:-0}"

# Build aligned (EXPERIMENT, RUN_NAME) arrays.
read -ra EXPS_ARR  <<< "$EXPERIMENTS"
read -ra NAMES_ARR <<< "$RUN_NAMES"
if [[ ${#NAMES_ARR[@]} -eq 0 ]]; then
    NAMES_ARR=("${EXPS_ARR[@]}")
elif [[ ${#NAMES_ARR[@]} -ne ${#EXPS_ARR[@]} ]]; then
    echo "[error] RUN_NAMES length (${#NAMES_ARR[@]}) must match EXPERIMENTS (${#EXPS_ARR[@]})" >&2
    exit 1
fi

banner() {
    echo
    echo "================================================================"
    echo "$1"
    echo "================================================================"
}

banner "Plan"
echo "  GPU             = $GPU"
echo "  EXPERIMENTS     = ${EXPS_ARR[*]}"
echo "  RUN_NAMES       = ${NAMES_ARR[*]}"
echo "  LOAD_WEIGHTS    = ${LOAD_WEIGHTS:-(none)}"
echo "  EVAL_SPLIT      = $EVAL_SPLIT"
echo "  DIAG_SPLITS     = $DIAG_SPLITS"
echo "  EXTRA_TRAIN_ARGS= ${EXTRA_TRAIN_ARGS:-(none)}"
echo "  SKIP_TRAIN/EVAL/DIAG = $SKIP_TRAIN/$SKIP_EVAL/$SKIP_DIAG"

T_START=$(date +%s)
SUMMARY=()

# ============================== per-experiment loop ==========================
for i in "${!EXPS_ARR[@]}"; do
    EXPERIMENT="${EXPS_ARR[$i]}"
    RUN_NAME="${NAMES_ARR[$i]}"
    OUT_DIR="$ROOT/output/$RUN_NAME"

    EXP_T_START=$(date +%s)
    banner "[${i}/${#EXPS_ARR[@]}] ▸ EXPERIMENT=$EXPERIMENT  RUN_NAME=$RUN_NAME"

    # -------- 1. Training --------------------------------------------------
    if [[ "$SKIP_TRAIN" != "1" ]]; then
        banner "  [1/3] Training: $RUN_NAME"
        TRAIN_CMD=(python3 -u scripts/train.py
                   "+experiment=$EXPERIMENT"
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

        # train.py auto-suffixes the dir (_2, _3, ...) if RUN_NAME exists.
        # Pick the most recently modified directory matching the prefix.
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
            exit 1
        fi
        echo "[skip] training — using existing ckpt at $OUT_DIR/best.pt"
    fi

    # -------- 2. Evaluation -----------------------------------------------
    if [[ "$SKIP_EVAL" != "1" ]]; then
        banner "  [2/3] Eval on $EVAL_SPLIT"
        EVAL_LOG="$OUT_DIR/eval_${EVAL_SPLIT}.log"
        CUDA_VISIBLE_DEVICES="$GPU" python3 -u scripts/eval.py \
            "+checkpoint=$OUT_DIR/best.pt" \
            "+eval_split=$EVAL_SPLIT" \
            2>&1 | tee "$EVAL_LOG"
        echo "[ok] eval done — $OUT_DIR/eval_independent/ (log: $EVAL_LOG)"
    else
        echo "[skip] eval"
    fi

    # -------- 3. Diagnose --------------------------------------------------
    if [[ "$SKIP_DIAG" != "1" ]]; then
        banner "  [3/3] Diagnose shape (splits: $DIAG_SPLITS)"
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

    EXP_T_END=$(date +%s)
    EXP_DUR=$((EXP_T_END - EXP_T_START))
    SUMMARY+=("$RUN_NAME : $((EXP_DUR/3600))h $(((EXP_DUR%3600)/60))m  @ $OUT_DIR")
done

T_END=$(date +%s)
DURATION=$((T_END - T_START))
HOURS=$((DURATION / 3600))
MINUTES=$(( (DURATION % 3600) / 60 ))

banner "[done] all experiments finished"
echo "  total runtime  = ${HOURS}h ${MINUTES}m"
echo
echo "  per-experiment summary:"
for line in "${SUMMARY[@]}"; do
    echo "    $line"
done
echo
echo "  Each output dir contains:"
echo "    best.pt                  - selected checkpoint"
echo "    config.yaml              - resolved Hydra config"
echo "    eval_${EVAL_SPLIT}.log   - eval stdout"
echo "    eval_independent/        - metrics.json / summary.txt / per_sample.csv"
echo "    diagnose_shape/          - diagnostic plots / stripe visuals"
