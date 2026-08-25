#!/usr/bin/env bash
# Depth-specialist ablation — NEAR bin [0, 20) m.
# Base: shape_lidar_grad_shape_edge_res. Only loss mask differs.
set -euo pipefail
cd "$(dirname "$0")/.."
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-3} \
    python3 -u scripts/train.py \
        +experiment=shape_edge_res_bin_near "$@"
