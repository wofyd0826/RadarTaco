#!/usr/bin/env bash
# Two-stage sim-pretrain → nuScenes-finetune pipeline (Solution B1).
#
# Stage 1: train on vKITTI2 + Hypersim only (sim ratio 1.0)
# Stage 2: load stage-1 ckpt, fine-tune on nuScenes only with smaller LR
# Stage 3: eval the fine-tuned model on the nuScenes test split
#
# Output layout
#   $OUTPUT_ROOT/sim_pretrain/{best.pt, last.pt, config.yaml, ...}
#   $OUTPUT_ROOT/pretrain_finetune/{best.pt, eval_test/, ...}
#
# Usage
#   bash scripts/sweep_pretrain_finetune.sh                       # full pipeline
#   PRETRAIN_EPOCHS=10 FINETUNE_EPOCHS=30 bash scripts/sweep_pretrain_finetune.sh
#   BATCH_SIZE=8 bash scripts/sweep_pretrain_finetune.sh
#   SKIP_PRETRAIN=1 bash scripts/sweep_pretrain_finetune.sh       # use existing pretrain ckpt
#   SKIP_FINETUNE=1 bash scripts/sweep_pretrain_finetune.sh       # pretrain only (no FT, no eval)
#   WANDB=false bash scripts/sweep_pretrain_finetune.sh
#   OUTPUT_ROOT=output bash scripts/sweep_pretrain_finetune.sh

set -uo pipefail
trap 'echo "[interrupted] cleaning up..."; exit 130' INT

ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"

PRETRAIN_EPOCHS="${PRETRAIN_EPOCHS:-}"
FINETUNE_EPOCHS="${FINETUNE_EPOCHS:-}"
BATCH_SIZE="${BATCH_SIZE:-}"
SKIP_PRETRAIN="${SKIP_PRETRAIN:-0}"
SKIP_FINETUNE="${SKIP_FINETUNE:-0}"
WANDB="${WANDB:-true}"
SWEEP_TAG="${SWEEP_TAG:-$(date +%Y-%m-%dT%H-%M-%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-output}"
PRETRAIN_NAME="sim_pretrain"
FINETUNE_NAME="pretrain_finetune"
mkdir -p "$OUTPUT_ROOT"
LOG="$OUTPUT_ROOT/_sweep_log_${FINETUNE_NAME}_${SWEEP_TAG}.log"

EXTRA=""
[ "$WANDB" = "false" ] && EXTRA+=" wandb.enabled=false"
[ -n "$BATCH_SIZE" ] && EXTRA+=" training.batch_size=$BATCH_SIZE"

{
  echo "════════════════════════════════════════════════════════════════"
  echo "  RadarTaco sim-pretrain → finetune sweep — $SWEEP_TAG"
  echo "  output_root:      $OUTPUT_ROOT"
  echo "  pretrain epochs:  ${PRETRAIN_EPOCHS:-<config default 30>}"
  echo "  finetune epochs:  ${FINETUNE_EPOCHS:-<config default 50>}"
  echo "  batch_size:       ${BATCH_SIZE:-<config default>}"
  echo "  skip_pretrain:    $SKIP_PRETRAIN"
  echo "  skip_finetune:    $SKIP_FINETUNE"
  echo "  wandb:            $WANDB"
  echo "  start:            $(date)"
  echo "════════════════════════════════════════════════════════════════"
} | tee "$LOG"

# ───────────────────────────────────────────────────── STAGE 1: sim pretrain ──
PRETRAIN_DIR="$OUTPUT_ROOT/$PRETRAIN_NAME"
PRETRAIN_CKPT="$PRETRAIN_DIR/best.pt"

if [ "$SKIP_PRETRAIN" -ne 1 ]; then
  PRE_ARGS="+experiment=sim_pretrain_finetune training=sim_pretrain dataset=mixed dataset.sim.ratio_real=0.0 dataset.sim.ratio_sim=1.0"
  [ -n "$PRETRAIN_EPOCHS" ] && PRE_ARGS+=" training.epochs=$PRETRAIN_EPOCHS"
  {
    echo ""
    echo "──── STAGE 1 · sim pretrain · $(date '+%H:%M:%S') ────"
    echo "  args: $PRE_ARGS"
  } | tee -a "$LOG"
  if python scripts/train.py $PRE_ARGS \
      training.output_dir="$OUTPUT_ROOT" \
      wandb.name="$PRETRAIN_NAME" \
      wandb.group="$SWEEP_TAG" \
      $EXTRA 2>&1 | tee -a "$LOG"; then
    echo "[$PRETRAIN_NAME] OK" | tee -a "$LOG"
  else
    rc=$?
    echo "[$PRETRAIN_NAME] FAILED (exit=$rc)" | tee -a "$LOG"
    exit $rc
  fi
else
  echo "[$PRETRAIN_NAME] skipped (SKIP_PRETRAIN=1)" | tee -a "$LOG"
fi

# locate stage-1 best.pt (could be under a re-run suffix like sim_pretrain1/)
PRETRAIN_CKPT=$(ls -dt "$PRETRAIN_DIR"*/best.pt 2>/dev/null | head -1)
if [ -z "$PRETRAIN_CKPT" ]; then
  echo "[$PRETRAIN_NAME] no best.pt under ${PRETRAIN_DIR}* — abort" | tee -a "$LOG"
  exit 2
fi
echo "[$PRETRAIN_NAME] using ckpt: $PRETRAIN_CKPT" | tee -a "$LOG"

if [ "$SKIP_FINETUNE" -eq 1 ]; then
  echo "[$FINETUNE_NAME] skipped (SKIP_FINETUNE=1) — pretrain only" | tee -a "$LOG"
  exit 0
fi

# ───────────────────────────────────────────────────── STAGE 2: finetune ──
FT_ARGS="+experiment=sim_pretrain_finetune training=nuscenes_finetune"
[ -n "$FINETUNE_EPOCHS" ] && FT_ARGS+=" training.epochs=$FINETUNE_EPOCHS"
{
  echo ""
  echo "──── STAGE 2 · nuScenes finetune · $(date '+%H:%M:%S') ────"
  echo "  args: $FT_ARGS"
  echo "  init from: $PRETRAIN_CKPT"
} | tee -a "$LOG"
if python scripts/train.py $FT_ARGS \
    training.output_dir="$OUTPUT_ROOT" \
    training.load_weights_from="$PRETRAIN_CKPT" \
    wandb.name="$FINETUNE_NAME" \
    wandb.group="$SWEEP_TAG" \
    $EXTRA 2>&1 | tee -a "$LOG"; then
  echo "[$FINETUNE_NAME] OK" | tee -a "$LOG"
else
  rc=$?
  echo "[$FINETUNE_NAME] FAILED (exit=$rc)" | tee -a "$LOG"
  exit $rc
fi

# ───────────────────────────────────────────────────── STAGE 3: eval test ──
FT_DIR="$OUTPUT_ROOT/$FINETUNE_NAME"
FT_CKPT=$(ls -dt "$FT_DIR"*/best.pt 2>/dev/null | head -1)
if [ -z "$FT_CKPT" ]; then
  echo "[$FINETUNE_NAME] no best.pt under ${FT_DIR}* — skipping eval" | tee -a "$LOG"
  exit 0
fi
EVAL_DIR="$(dirname "$FT_CKPT")/eval_test"
{
  echo ""
  echo "──── STAGE 3 · eval on nuScenes test · $(date '+%H:%M:%S') ────"
  echo "  ckpt: $FT_CKPT"
  echo "  out:  $EVAL_DIR"
} | tee -a "$LOG"
if python scripts/eval.py +experiment=sim_pretrain_finetune training=nuscenes_finetune \
    +checkpoint="$FT_CKPT" \
    +eval_split=test \
    +eval_out_dir="$EVAL_DIR" \
    wandb.enabled=false 2>&1 | tee -a "$LOG"; then
  echo "[$FINETUNE_NAME] eval: OK" | tee -a "$LOG"
else
  rc=$?
  echo "[$FINETUNE_NAME] eval: FAILED (exit=$rc)" | tee -a "$LOG"
  exit $rc
fi

# ───────────────────────────────────────────────────── headline metrics ──
if [ -f "$EVAL_DIR/metrics.json" ]; then
  python - <<PY 2>&1 | tee -a "$LOG"
import json
m = json.load(open("$EVAL_DIR/metrics.json"))
def fmt(d, k):
    v = d.get(k); return f"{v:.1f}" if v is not None else "n/a"
ovr  = m.get("overall", {}).get("0-80m", {})
o100 = m.get("overall", {}).get("0-100m", {})
far  = m.get("far", {}).get("50-80m", {})
f100 = m.get("far", {}).get("80-100m", {})
night = m.get("night", {}).get("0-80m", {})
day   = m.get("day", {}).get("0-80m", {})
print("")
print("════════════════════════════════════════════════════════════════")
print("  PRETRAIN→FINETUNE  TEST METRICS")
print("════════════════════════════════════════════════════════════════")
print(f"  0-80   MAE={fmt(ovr,'mae')}  RMSE={fmt(ovr,'rmse')}")
print(f"  0-100  MAE={fmt(o100,'mae')} RMSE={fmt(o100,'rmse')}")
print(f"  far 50-80   MAE={fmt(far,'mae')}")
print(f"  far 80-100  MAE={fmt(f100,'mae')}")
print(f"  day  MAE={fmt(day,'mae')}    night MAE={fmt(night,'mae')}")
PY
fi

{
  echo ""
  echo "  finished:  $(date)"
  echo "  full log:  $LOG"
  echo "  outputs:   $OUTPUT_ROOT/{${PRETRAIN_NAME},${FINETUNE_NAME}}/"
  echo "════════════════════════════════════════════════════════════════"
} | tee -a "$LOG"
