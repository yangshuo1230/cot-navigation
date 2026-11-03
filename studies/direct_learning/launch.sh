#!/usr/bin/env bash
set -euo pipefail
ROOT=/mnt/ningxuefei/yangshuo/cot-navigation
STUDY="$ROOT/studies/direct_learning"
cd "$ROOT"
mkdir -p "$STUDY/runs/direct_learning_v1"
nohup "$ROOT/.venv/bin/python" -u "$STUDY/launch.py" \
  > "$STUDY/runs/direct_learning_v1/launcher.log" 2>&1 < /dev/null &
echo "Launcher PID: $!"
echo "Log: $STUDY/runs/direct_learning_v1/launcher.log"
