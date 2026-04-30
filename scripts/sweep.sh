#!/usr/bin/env bash
# Sequential train + eval sweep across 6 ablation modes.
#
# Modes
#   baseline             paper-faithful (metric, no aug)
#   sim50_mixed          mixed-batch (Hypersim + vKITTI2 augmented), 50% sim
#   sim50_vkitti2_only   vKITTI2 only, semantic_aware radar, 50% sim
#   inv_depth            inverse-depth output (Option B)
#   multi_scale          multi-scale aux heads (Option A)
#   dark_aug             photometric night-like augmentation
#
# Usage
#   bash scripts/sweep.sh                       # default: all 6, paper hparams
#   EPOCHS=15 bash scripts/sweep.sh             # cheaper epochs (ablation)
#   BATCH_SIZE=8 bash scripts/sweep.sh          # smaller GPU
#   ONLY=baseline,dark_aug bash scripts/sweep.sh   # subset
#   SKIP_TRAIN=1 bash scripts/sweep.sh          # eval-only (use existing ckpts)
#   WANDB=false bash scripts/sweep.sh           # disable wandb
#   SWEEP_DIR=/path/to/dir bash scripts/sweep.sh   # custom output root
#
# Layout
#   $SWEEP_DIR/
#     sweep.log
#     <mode>/<mode>/                       (train.py auto-suffix per ckpt)
#       config.yaml  last.pt  best.pt  epoch_N.pt
#       eval_test/
#         metrics.json  summary.txt  viz/

set -uo pipefail
trap 'echo "[interrupted] cleaning up..."; exit 130' INT

ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"

EPOCHS="${EPOCHS:-}"
BATCH_SIZE="${BATCH_SIZE:-}"
SKIP_TRAIN="${SKIP_TRAIN:-0}"
WANDB="${WANDB:-true}"
SWEEP_TAG="${SWEEP_TAG:-$(date +%Y%m%d_%H%M%S)}"
SWEEP_DIR="${SWEEP_DIR:-output/sweep_$SWEEP_TAG}"
mkdir -p "$SWEEP_DIR"
LOG="$SWEEP_DIR/sweep.log"

# ────────────────────────────────────────────────────── per-mode definitions ──
# format: NAME|hydra-args
declare -a EXPERIMENTS=(
  "baseline|+experiment=baseline_gnn"
  "sim50_mixed|+experiment=mixed_with_aux dataset.sim.ratio_real=0.5 dataset.sim.ratio_sim=0.5"
  "sim50_vkitti2_only|+experiment=mixed_semantic_radar dataset.sim.hypersim.enabled=false dataset.sim.ratio_real=0.5 dataset.sim.ratio_sim=0.5"
  "inv_depth|+experiment=baseline_gnn_inverse"
  "multi_scale|+experiment=baseline_gnn_multiscale"
  "dark_aug|+experiment=baseline_gnn_dark_aug"
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
  echo "  RadarTaco sweep — $SWEEP_TAG"
  echo "  output:      $SWEEP_DIR"
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

  EXP_BASE="$SWEEP_DIR/$NAME"

  {
    echo ""
    echo "────────────────────────────────────────────────────────────────"
    echo "  [$NAME]  $(date '+%H:%M:%S')"
    echo "  args: $ARGS"
    echo "────────────────────────────────────────────────────────────────"
  } | tee -a "$LOG"

  # ───── train ─────
  if [ "$SKIP_TRAIN" -ne 1 ]; then
    echo "[$NAME] train…" | tee -a "$LOG"
    # `wandb.name` doubles as the run-dir leaf (train.py auto-resolves
    # output_dir = $training.output_dir/$wandb.name).
    if python scripts/train.py $ARGS \
        training.output_dir="$EXP_BASE" \
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

  # ───── locate best.pt (might have a numeric suffix if dir existed) ─────
  CKPT=$(find "$EXP_BASE" -maxdepth 3 -name "best.pt" 2>/dev/null \
         | sort | tail -1)
  if [ -z "$CKPT" ]; then
    echo "[$NAME] no best.pt under $EXP_BASE — skipping eval" | tee -a "$LOG"
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
ovr = m.get("overall", {}).get("0-80m", {})
ovr100 = m.get("overall", {}).get("0-100m", {})
far  = m.get("far", {}).get("50-80m", {})
far100 = m.get("far", {}).get("80-100m", {})
night = m.get("night", {}).get("0-80m", {})
day   = m.get("day", {}).get("0-80m", {})
def fmt(d, k, p="{:.1f}"):
    v = d.get(k)
    return p.format(v) if v is not None else "n/a"
print(
    f"0-80 MAE={fmt(ovr,'mae')}/RMSE={fmt(ovr,'rmse')}  "
    f"0-100 MAE={fmt(ovr100,'mae')}  "
    f"far50-80 MAE={fmt(far,'mae')}  far80-100 MAE={fmt(far100,'mae')}  "
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
  echo "                       SWEEP SUMMARY"
  echo "════════════════════════════════════════════════════════════════"
  for line in "${SUMMARY[@]}"; do
    printf "  %s\n" "$line"
  done
  echo ""
  echo "  finished:    $(date)"
  echo "  full log:    $LOG"
  echo "  outputs:     $SWEEP_DIR"
  echo "════════════════════════════════════════════════════════════════"
} | tee -a "$LOG"
