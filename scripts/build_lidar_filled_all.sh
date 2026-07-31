#!/usr/bin/env bash
# Build depth_lidar_filled for train + val + test splits.
#
# Each LiDAR point's residual (lidar - dense_gt) is propagated to nearby
# pixels (within MAX_FILL_DIST px) via nearest-neighbor (per-LiDAR Voronoi).
# Pixels outside that band are 0 (invalid → not used in supervision).
#
# Usage:
#   bash scripts/build_lidar_filled_all.sh
#
# Env overrides:
#   DATA_ROOT       — nuScenes derived root (default /data/public/nuScenes/derived)
#   DENSE_DIR       — dense gt dir under DATA_ROOT (default depth_refined_grid)
#   LIDAR_DIR       — sparse lidar gt dir (default depth_lidar)
#   OUT_DIR         — output dir under DATA_ROOT (default depth_lidar_filled)
#   MAX_FILL_DIST   — fill radius in pixels (default 20)
#   RESIDUAL_CLIP   — residual clip in meters (default 3.0)
#   METHOD          — nearest | linear (default nearest)
#   SPLITS          — space-separated splits to run (default "train val test")
#   SKIP_EXISTING   — 1=skip already-saved PNGs (default 1)
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
cd "$ROOT"

DATA_ROOT="${DATA_ROOT:-/data/public/nuScenes/derived}"
DENSE_DIR="${DENSE_DIR:-depth_refined_grid}"
LIDAR_DIR="${LIDAR_DIR:-depth_lidar}"
OUT_DIR="${OUT_DIR:-depth_lidar_filled}"
MAX_FILL_DIST="${MAX_FILL_DIST:-20}"
RESIDUAL_CLIP="${RESIDUAL_CLIP:-3.0}"
METHOD="${METHOD:-nearest}"
SPLITS="${SPLITS:-train val test}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"

SKIP_FLAG=""
if [[ "$SKIP_EXISTING" != "1" ]]; then
    SKIP_FLAG="--no_skip_existing"
fi

banner() {
    echo
    echo "================================================================"
    echo "$1"
    echo "================================================================"
}

banner "Plan"
echo "  DATA_ROOT     = $DATA_ROOT"
echo "  DENSE_DIR     = $DENSE_DIR"
echo "  LIDAR_DIR     = $LIDAR_DIR"
echo "  OUT_DIR       = $DATA_ROOT/$OUT_DIR"
echo "  MAX_FILL_DIST = $MAX_FILL_DIST px"
echo "  RESIDUAL_CLIP = ${RESIDUAL_CLIP} m"
echo "  METHOD        = $METHOD"
echo "  SPLITS        = $SPLITS"
echo "  SKIP_EXISTING = $SKIP_EXISTING"

T0=$(date +%s)
for SPLIT in $SPLITS; do
    SPLIT_FILE="$DATA_ROOT/splits/${SPLIT}.txt"
    if [[ ! -f "$SPLIT_FILE" ]]; then
        echo "[warn] split file not found: $SPLIT_FILE — skip" >&2
        continue
    fi
    N=$(wc -l < "$SPLIT_FILE")
    banner "[$SPLIT] $N samples"
    python3 scripts/build_lidar_filled_gt.py \
        --split "$SPLIT_FILE" \
        --data_root "$DATA_ROOT" \
        --dense_dir "$DENSE_DIR" \
        --lidar_dir "$LIDAR_DIR" \
        --out_dir   "$DATA_ROOT/$OUT_DIR" \
        --method "$METHOD" \
        --max_fill_dist "$MAX_FILL_DIST" \
        --residual_clip "$RESIDUAL_CLIP" \
        $SKIP_FLAG
done
T1=$(date +%s)
DUR=$((T1 - T0))
H=$((DUR / 3600))
M=$(((DUR % 3600) / 60))
banner "[done] all splits"
echo "  total time = ${H}h ${M}m"
echo "  out dir    = $DATA_ROOT/$OUT_DIR"
