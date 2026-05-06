#!/usr/bin/env bash
# Sequential train + eval for vKITTI2-only sim ablations.
#
# Modes
#   sim50_vkitti2_only   real 50% + vKITTI2 50% (semantic_aware radar)
#   sim100_vkitti2_only  vKITTI2 100% (no real data in train; val/test still nuScenes)
#
# Usage
#   bash scripts/sweep_vkitti2_only.sh                     # both, paper hparams
#   EPOCHS=15 bash scripts/sweep_vkitti2_only.sh           # cheaper epochs
#   BATCH_SIZE=8 bash scripts/sweep_vkitti2_only.sh        # smaller GPU
#   ONLY=sim50_vkitti2_only bash scripts/sweep_vkitti2_only.sh   # subset
#   SKIP_TRAIN=1 bash scripts/sweep_vkitti2_only.sh        # eval-only (existing ckpts)
#   WANDB=false bash scripts/sweep_vkitti2_only.sh
#   OUTPUT_ROOT=output bash scripts/sweep_vkitti2_only.sh

set -uo pipefail
trap 'echo "[interrupted] cleaning up..."; exit 130' INT

ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"

EPOCHS="${EPOCHS:-}"
BATCH_SIZE="${BATCH_SIZE:-}"
SKIP_TRAIN="${SKIP_TRAIN:-0}"
WANDB="${WANDB:-true}"
SWEEP_TAG="${SWEEP_TAG:-$(date +%Y-%m-%dT%H-%M-%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-output}"
mkdir -p "$OUTPUT_ROOT"
LOG="$OUTPUT_ROOT/_sweep_log_vkitti2_only_${SWEEP_TAG}.log"

# ────────────────────────────────────────────────────── per-mode definitions ──
# format: NAME|hydra-args
declare -a EXPERIMENTS=(
  "sim50_vkitti2_only|+experiment=mixed_semantic_radar dataset.sim.hypersim.enabled=false dataset.sim.ratio_real=0.5 dataset.sim.ratio_sim=0.5"
  "sim100_vkitti2_only|+experiment=mixed_semantic_radar dataset.sim.hypersim.enabled=false dataset.sim.ratio_real=0.0 dataset.sim.ratio_sim=1.0"
)

# optional subset filter
declare -a ONLY_LIST=()
if [ -n "${ONLY:-}" ]; then
  IFS=',' read -ra ONLY_LIST <<< "$ONLY"
fi
in_subset() {
  local name="$1"
  if [ ${#ONLY_LIST[@]} -eq 0 ]; then return 0; fi
  for o in "${ONLY_LIST[@]}"; do [ "$o" = "$name" ] && return 0; done
  return 1
}

# common overrides applied to every run
EXTRA_ARGS=""
[ -n "$EPOCHS" ]     && EXTRA_ARGS+=" training.epochs=$EPOCHS"
[ -n "$BATCH_SIZE" ] && EXTRA_ARGS+=" training.batch_size=$BATCH_SIZE"
[ "$WANDB" = "false" ] && EXTRA_ARGS+=" wandb.enabled=false"

declare -a SUMMARY=()

# ────────────────────────────────────────────────────────── header ──
{
  echo "════════════════════════════════════════════════════════════════"
  echo "  RadarTaco vKITTI2-only sweep — $SWEEP_TAG"
  echo "  output_root: $OUTPUT_ROOT"
  echo "  log:         $LOG"
  echo "  epochs:      ${EPOCHS:-<config default>}"
  echo "  batch_size:  ${BATCH_SIZE:-<config default>}"
  echo "  skip_train:  $SKIP_TRAIN"
  echo "  wandb:       $WANDB"
  if [ ${#ONLY_LIST[@]} -gt 0 ]; then
    echo "  subset:      ${ONLY_LIST[*]}"
  fi
  echo "  extra_args:  $EXTRA_ARGS"
  echo "  start:       $(date)"
  echo "════════════════════════════════════════════════════════════════"
} | tee "$LOG"

# ───────────────────────────────────────────────────── per-experiment loop ──
for entry in "${EXPERIMENTS[@]}"; do
  NAME="${entry%%|*}"
  ARGS="${entry#*|}"

  if ! in_subset "$NAME"; then
    echo ""; echo "[$NAME] skipped (not in ONLY set)" | tee -a "$LOG"
    continue
  fi

  EXP_DIR="$OUTPUT_ROOT/$NAME"

  {
    echo ""
    echo "────────────────────────────────────────────────────────────────"
    echo "  [$NAME]  $(date '+%H:%M:%S')"
    echo "  args: $ARGS"
    echo "  dir:  $EXP_DIR"
    echo "────────────────────────────────────────────────────────────────"
  } | tee -a "$LOG"

  # ───── train ─────
  if [ "$SKIP_TRAIN" -ne 1 ]; then
    echo "[$NAME] train…" | tee -a "$LOG"
    if python scripts/train.py $ARGS \
        training.output_dir="$OUTPUT_ROOT" \
        wandb.name="$NAME" \
        wandb.group="$SWEEP_TAG" \
        $EXTRA_ARGS 2>&1 | tee -a "$LOG"; then
      echo "[$NAME] train: OK" | tee -a "$LOG"
    else
      rc=$?
      echo "[$NAME] train: FAILED (exit=$rc)" | tee -a "$LOG"
      SUMMARY+=("$NAME | TRAIN FAILED (exit=$rc)")
      continue
    fi
  else
    echo "[$NAME] train skipped (SKIP_TRAIN=1)" | tee -a "$LOG"
  fi

  # ───── locate best.pt ─────
  CKPT=$(ls -dt "$EXP_DIR"/best.pt "$EXP_DIR"[0-9]*/best.pt 2>/dev/null | head -1)
  if [ -z "$CKPT" ]; then
    echo "[$NAME] no best.pt under ${EXP_DIR}* — skipping eval" | tee -a "$LOG"
    SUMMARY+=("$NAME | NO CHECKPOINT")
    continue
  fi

  # ───── eval on test ─────
  EVAL_DIR="$(dirname "$CKPT")/eval_test"
  echo "[$NAME] eval test split (ckpt: $CKPT)…" | tee -a "$LOG"
  if python scripts/eval.py $ARGS \
      +checkpoint="$CKPT" \
      +eval_split=test \
      +eval_out_dir="$EVAL_DIR" \
      wandb.enabled=false 2>&1 | tee -a "$LOG"; then
    echo "[$NAME] eval: OK → $EVAL_DIR" | tee -a "$LOG"
  else
    rc=$?
    echo "[$NAME] eval: FAILED (exit=$rc)" | tee -a "$LOG"
    SUMMARY+=("$NAME | EVAL FAILED (exit=$rc)")
    continue
  fi

  # ───── extract headline metrics ─────
  if [ -f "$EVAL_DIR/metrics.json" ]; then
    LINE=$(python - <<PY
import json
m = json.load(open("$EVAL_DIR/metrics.json"))
ovr  = m.get("overall", {}).get("0-80m", {})
o100 = m.get("overall", {}).get("0-100m", {})
far  = m.get("far", {}).get("50-80m", {})
f100 = m.get("far", {}).get("80-100m", {})
night = m.get("night", {}).get("0-80m", {})
day   = m.get("day", {}).get("0-80m", {})
def fmt(d, k):
    v = d.get(k); return f"{v:.1f}" if v is not None else "n/a"
print(
    f"0-80 MAE={fmt(ovr,'mae')}/RMSE={fmt(ovr,'rmse')}  "
    f"0-100 MAE={fmt(o100,'mae')}  "
    f"far50-80 MAE={fmt(far,'mae')}  far80-100 MAE={fmt(f100,'mae')}  "
    f"day MAE={fmt(day,'mae')}  night MAE={fmt(night,'mae')}"
)
PY
)
    SUMMARY+=("$NAME | $LINE")
    echo "[$NAME] headline: $LINE" | tee -a "$LOG"
  fi
done

# ───────────────────────────────────────────────────────── footer ──
{
  echo ""
  echo "════════════════════════════════════════════════════════════════"
  echo "                   vKITTI2-only SWEEP SUMMARY"
  echo "════════════════════════════════════════════════════════════════"
  for line in "${SUMMARY[@]}"; do
    printf "  %s\n" "$line"
  done
  echo ""
  echo "  finished:    $(date)"
  echo "  full log:    $LOG"
  echo "  outputs:     $OUTPUT_ROOT/{sim50_vkitti2_only,sim100_vkitti2_only}/"
  echo "════════════════════════════════════════════════════════════════"
} | tee -a "$LOG"
