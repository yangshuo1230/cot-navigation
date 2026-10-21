#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
exec .venv/bin/python scripts/task_only_n32.py "$@"
