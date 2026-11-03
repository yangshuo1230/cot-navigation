#!/usr/bin/env python3
"""Backward-compatible entry point; implementation lives in the package."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from cot_navigation.experiments.scheduler import *
from cot_navigation.experiments.first_round import parse_args, build_jobs, main

if __name__ == '__main__':
    def interrupt(_signum, _frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, interrupt)
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print('Interrupted; resume with the same run name.', file=sys.stderr)
        raise SystemExit(130)
    except (RuntimeError, ValueError, subprocess.CalledProcessError) as exc:
        print(f'Error: {exc}', file=sys.stderr)
        raise SystemExit(1)
