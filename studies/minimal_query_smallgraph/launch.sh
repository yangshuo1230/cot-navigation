#!/usr/bin/env bash
# Foreground by default. This script is never invoked automatically on upload.
set -euo pipefail
REPO="/mnt/ningxuefei/yangshuo/cot-navigation"
cd "$REPO"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export CUDA_VISIBLE_DEVICES="${GPU:-0}"
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export PYTHONUNBUFFERED=1
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
PY="$REPO/.venv/bin/python"

# CPU-only preflight; does not initialize a model, W&B or checkpoints.
if [[ "${1:-}" == "--check-only" ]]; then
  shift
  exec "$PY" "$SCRIPT_DIR/experiment.py" --check-only "$@"
fi

# Do not accidentally run two launchers on the same physical GPU.
exec 9>"$SCRIPT_DIR/.gpu${CUDA_VISIBLE_DEVICES}.lock"
flock -n 9 || { echo "A launcher already owns GPU $CUDA_VISIBLE_DEVICES" >&2; exit 1; }
USED="$(nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-gpu=memory.used --format=csv,noheader,nounits)"
if (( USED > 1000 )); then
  echo "GPU $CUDA_VISIBLE_DEVICES has ${USED} MiB in use; refusing to start." >&2
  exit 1
fi

"$PY" "$SCRIPT_DIR/experiment.py" --check-only "$@"
# Single GPU, sequential jobs. Reinvoke with identical arguments to resume.
# Optional: JOBS="bfs_control bfs_small_mix dfs_control dfs_small_mix" bash launch.sh
read -r -a JOB_LIST <<< "${JOBS:-direct_single direct_multi bfs_control bfs_small_mix}"
for job in "${JOB_LIST[@]}"; do
  "$PY" "$SCRIPT_DIR/experiment.py" --job "$job" "$@"
done
