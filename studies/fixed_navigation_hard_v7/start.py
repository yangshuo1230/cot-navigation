"""Detached launcher; create the output directory before opening its log."""
import argparse
from pathlib import Path
import json
import subprocess
import sys
from prepare import HERE

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--repo-root', type=Path, required=True)
p.add_argument('--mode', choices=['pilot'], default='pilot')
p.add_argument('--protocols', nargs='+', choices=['direct', 'path', 'dfs_state', 'dfs_nodes'])
p.add_argument('--gpus', type=int, nargs='+', default=list(range(8)))
p.add_argument('--execute', action='store_true')
args = p.parse_args()
repo = args.repo_root.resolve()
command = [sys.executable, str(HERE / 'launch.py'), '--repo-root', str(repo),
           '--mode', args.mode, '--gpus', *map(str, args.gpus)]
if args.protocols:
    command += ['--protocols', *args.protocols]
if not args.execute:
    subprocess.run(command, check=True)
else:
    root = repo / 'outputs/fixed_navigation_hard_v7' / args.mode
    root.mkdir(parents=True, exist_ok=True)
    with (root / 'launch.log').open('a') as stream:
        process = subprocess.Popen(command + ['--execute'], stdin=subprocess.DEVNULL,
            stdout=stream, stderr=subprocess.STDOUT, cwd=repo, start_new_session=True)
    print(json.dumps({'scheduler_pid': process.pid, 'log': str(root / 'launch.log')}))
