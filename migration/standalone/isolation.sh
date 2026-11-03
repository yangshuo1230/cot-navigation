#!/bin/bash
set -euo pipefail
ROOT=/mnt/ningxuefei/yangshuo/cot-navigation-unified
cd "$ROOT"
mkdir -p migration/standalone/empty migration/standalone/tmp
mount --make-rprivate /
for blocked in /mnt/ningxuefei/yangshuo/cot-navigation "$ROOT/research" "$ROOT/.code_backups"; do
    mount --bind "$ROOT/migration/standalone/empty" "$blocked"
    mount -o remount,bind,ro "$blocked"
done
export PYTHONPATH="$ROOT/src:$ROOT"
export PYTHONPYCACHEPREFIX="$ROOT/migration/standalone/pycache"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export TMPDIR="$ROOT/migration/standalone/tmp"
if [ "$#" -eq 0 ]; then
    export CUDA_VISIBLE_DEVICES=''
    exec .venv/bin/python -m pytest tests -o addopts='' -q
else
    exec "$@"
fi
