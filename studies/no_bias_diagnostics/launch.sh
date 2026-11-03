#!/usr/bin/env bash
set -euo pipefail
cd /mnt/ningxuefei/yangshuo/cot-navigation
nohup .venv/bin/python -u studies/no_bias_diagnostics/launch.py > studies/no_bias_diagnostics/launcher_current.log 2>&1 < /dev/null &
echo "Frozen diagnostics launcher PID: $!"
