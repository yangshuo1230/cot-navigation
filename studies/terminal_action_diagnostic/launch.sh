#!/usr/bin/env bash
set -euo pipefail
cd /mnt/ningxuefei/yangshuo/cot-navigation
nohup .venv/bin/python -u studies/terminal_action_diagnostic/launch.py > studies/terminal_action_diagnostic/launcher_current.log 2>&1 < /dev/null &
echo "Frozen diagnostics launcher PID: $!"
