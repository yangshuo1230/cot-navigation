"""Detach the scheduler with its own output directory created before redirection."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--repo-root', type=Path, required=True)
p.add_argument('--mode', choices=['pilot', 'formal'], default='pilot')
p.add_argument('--gpus', type=int, nargs='+', default=list(range(8)))
p.add_argument('--execute', action='store_true')
args = p.parse_args()
command = [sys.executable, str(HERE / 'launch.py'), '--repo-root', str(args.repo_root),
           '--mode', args.mode, '--gpus', *map(str, args.gpus)]
if not args.execute:
    subprocess.run(command, check=True)
else:
    root = args.repo_root / 'outputs/fixed_navigation_v5' / args.mode
    root.mkdir(parents=True, exist_ok=True)
    log = root / 'launch.log'
    with log.open('a') as stream:
        proc = subprocess.Popen(command + ['--execute'], cwd=args.repo_root,
                                stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT,
                                start_new_session=True)
    print(json.dumps({'scheduler_pid': proc.pid, 'log': str(log), 'mode': args.mode}))
