#!/usr/bin/env bash
# Prepare all raw splits, fixed worlds/paths and the three active protocols.
# Repeating this command reuses an unchanged, verified preparation receipt.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
export CUDA_VISIBLE_DEVICES=""
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"

.venv/bin/python -m cot_navigation.cli data prepare \
  --config configs/experiments/main_context_direct.yaml \
  --set data.audit_fraction=1.0 "$@"
.venv/bin/python -m cot_navigation.cli data prepare \
  --config configs/experiments/main_stage_b_path.yaml \
  --set data.audit_fraction=1.0 "$@"
