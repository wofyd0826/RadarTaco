#!/usr/bin/env bash
# Re-run the two reference experiments against the fixed radar path.
#
#   1. shape_lidar_grad_shape_edge_res_fix   (single stage, 50 ep)
#   2. radartaco_moe_stage1_v3_fix           (teacher forcing, 30 ep)
#   3. radartaco_moe_stage2_v3_fix           (self-routing,   20 ep,
#                                             warm start from step 2)
#
# Steps 2 and 3 are chained: stage 2 loads stage 1's best.pt, and
# `load_checkpoint(weights_only=True)` uses strict=False, so a stale path
# would silently drop the parameters the fixes introduced. The script checks
# the file exists before starting stage 2.
#
# Usage:
#   bash scripts/run_radar_path_fix.sh              # all three, GPU 0
#   GPU=3 bash scripts/run_radar_path_fix.sh
#   ONLY=shape bash scripts/run_radar_path_fix.sh   # shape | moe
#   DRY=1 bash scripts/run_radar_path_fix.sh        # print commands only
set -euo pipefail

GPU="${GPU:-0}"
ONLY="${ONLY:-all}"
DRY="${DRY:-0}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

run() {
  echo "+ $*"
  [ "$DRY" = "1" ] || CUDA_VISIBLE_DEVICES="$GPU" "$@"
}

if [ "$ONLY" = "all" ] || [ "$ONLY" = "shape" ]; then
  echo "=== 1/3  shape_lidar_grad_shape_edge_res_fix (50 ep) ==="
  run python scripts/train.py +experiment=shape_lidar_grad_shape_edge_res_fix
fi

if [ "$ONLY" = "all" ] || [ "$ONLY" = "moe" ]; then
  echo "=== 2/3  radartaco_moe_stage1_v3_fix (30 ep, teacher forcing) ==="
  run python scripts/train.py +experiment=radartaco_moe_stage1_v3_fix

  STAGE1_CKPT="output/radartaco_moe_stage1_v3_fix/best.pt"
  if [ "$DRY" != "1" ] && [ ! -f "$STAGE1_CKPT" ]; then
    echo "ERROR: $STAGE1_CKPT missing — stage 1 may have written to a" >&2
    echo "       suffixed directory (the trainer appends _2, _3 when the" >&2
    echo "       name already exists). Point stage 2 at the right path:" >&2
    echo "         python scripts/train.py +experiment=radartaco_moe_stage2_v3_fix \\" >&2
    echo "           training.load_weights_from=<stage1_dir>/best.pt" >&2
    exit 1
  fi

  echo "=== 3/3  radartaco_moe_stage2_v3_fix (20 ep, self-routing) ==="
  run python scripts/train.py +experiment=radartaco_moe_stage2_v3_fix
fi

echo "done. baselines to compare against:"
echo "  output/shape_lidar_grad_shape_edge_res"
echo "  output/radartaco_moe_stage1_v3 , output/radartaco_moe_stage2_v3"
