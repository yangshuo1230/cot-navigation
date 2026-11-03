#!/usr/bin/env bash
# Manual entrypoint: exactly GPU0, six runs sequentially, automatic resume.
set -euo pipefail
cd /mnt/ningxuefei/yangshuo/cot-navigation
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export PYTHONUNBUFFERED=1
PY="$PWD/.venv/bin/python"

if [[ "${1:-}" == "--check-only" ]]; then
  shift
  exec "$PY" "$SCRIPT_DIR/suite.py" --check-only "$@"
fi

# Shared with the earlier minimal-study launcher, plus GPU process/memory check.
exec 9>"/tmp/cot-navigation-gpu0.lock"
flock -n 9 || { echo 'Another Direct launcher owns GPU0' >&2; exit 1; }
exec 8>"$PWD/studies/minimal_query_smallgraph/.gpu0.lock"
flock -n 8 || { echo 'The previous minimal-study launcher owns GPU0' >&2; exit 1; }
assert_gpu_free() {
  local used processes
  used="$(nvidia-smi -i 0 --query-gpu=memory.used --format=csv,noheader,nounits)"
  processes="$(nvidia-smi -i 0 --query-compute-apps=pid --format=csv,noheader,nounits)"
  if (( used > 1000 )) || [[ -n "$processes" ]]; then
    echo "GPU0 is busy (${used} MiB); refusing to start. No process was stopped." >&2
    exit 1
  fi
}
assert_gpu_free
"$PY" "$SCRIPT_DIR/suite.py" --check-only "$@"
read -r -a JOB_LIST <<< "${JOBS:-ordinary24 paired24 paired24_interleaved curriculum paired24_low_lr paired24_long}"
for job in "${JOB_LIST[@]}"; do
  assert_gpu_free
  "$PY" "$SCRIPT_DIR/suite.py" --job "$job" "$@"
done
