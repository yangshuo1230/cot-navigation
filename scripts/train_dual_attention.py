#!/usr/bin/env python3
"""Launch DFS/BFS x parallel/serial x causal/bidirectional on eight GPUs.

Reuses audited hard_dag_v1 data and trajectories without regenerating anything.
The common 140k curriculum is 40k/30k/30k/40k at N16/36/64/100.
Use --background to detach, --resume to resume the exact saved launch plan.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import signal
import sys

for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'TORCHINDUCTOR_COMPILE_THREADS'):
    os.environ.setdefault(key, '2')

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))
import train_first_round as engine
from cot_navigation.config import load_config, to_plain


def arguments(argv):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-name', default='dual_attention_v1_s0_' + datetime.now().strftime('%Y%m%d_%H%M'))
    p.add_argument('--output-root', default='outputs/dual_attention_v1/launches')
    p.add_argument('--gpus', type=int, nargs='+', default=list(range(8)))
    p.add_argument('--tracking', choices=('online', 'offline', 'disabled'), default='online')
    p.add_argument('--background', action='store_true')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--wait', action=argparse.BooleanOptionalAction, default=True)
    args = p.parse_args(argv)
    if args.run_name in ('', '.', '..') or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-' for c in args.run_name):
        p.error('run-name must be a plain name')
    if len(args.gpus) != len(set(args.gpus)):
        p.error('GPU indices must be unique')
    return args


def build_jobs(args):
    output = ROOT / args.output_root / args.run_name
    configs = ROOT / 'configs/generated/dual_attention_v1' / args.run_name
    configs.mkdir(parents=True, exist_ok=True)
    jobs = []
    for protocol in ('dfs_state', 'bfs_state'):
        reference = ROOT / 'outputs/hard_dag_v1/launches/hard_dag_v1_s0/context/curriculum' / protocol / 'resolved_config.yaml'
        for order in ('parallel', 'serial'):
            for graph in ('causal', 'bidirectional'):
                name = f'{protocol}/{order}/{graph}'
                cfg = load_config([reference, ROOT / 'configs/train/throughput.yaml'], [
                    f'output_dir={output / name}', 'model.architecture=dual', 'model.heads=2',
                    f'model.attention_order={order}', f'model.graph_attention={graph}',
                    'model.graph_max_tokens=868', 'model.compile=true',
                    'train.max_steps=140000', 'train.scale_stage_ends=[40000,70000,100000,140000]',
                    'evaluation.selection_min_step=100000',
                    'evaluation.save_trace_tokens=true',
                    # Keep full best/last and all scale/shape tests; only rollout throughput changes.
                    f'tracking.mode={args.tracking}', f'tracking.group={args.run_name}',
                    f'tracking.name={args.run_name}/{name}', f'tracking.variant=dual-{order}-{graph}',
                    f'tracking.tags=[dual_attention_v1,hard_dag_v1,{protocol},{order},{graph}]',
                    'tracking.notes=Two 64-dimensional branches; fixed graph+QUERY memory; causal CoT history; matched parameters across eight cells; 140k shared curriculum.'])
                path = configs / (name.replace('/', '__') + '.yaml')
                raw = yaml.safe_dump(to_plain(cfg), sort_keys=False)
                if path.exists() and path.read_text() != raw and (output / 'launch_plan.json').exists():
                    raise ValueError(f'existing configuration differs: {path}')
                path.write_text(raw)
                jobs.append(engine.Job(name, path, [], output / name, cfg.train.max_steps))
    return output, jobs


def verify_data(jobs, output, env):
    """Read existing preparation inventories once; never silently regenerate."""
    from cot_navigation.data.prepared import _inventory
    from cot_navigation.data.shards import is_complete
    reports = []
    cfg = load_config([jobs[0].config])
    roots = list(cfg.train.scale_roots) + [x.split('=', 1)[1] for x in cfg.evaluation.extra_sets]
    for root in dict.fromkeys(roots):
        path = ROOT / root
        prepared = json.loads((path / 'prepared_data.json').read_text())
        actual = _inventory(path)
        # Archival copies changed ctime without changing bytes. Recheck content
        # against completion checksums instead of accepting stale stat receipts.
        for name, old in prepared['inventory'].items():
            if name not in actual or actual[name][0] != old[0]:
                raise RuntimeError(f'prepared file missing or size changed: {path / name}')
        complete = list(path.rglob('*.complete'))
        if not complete or any(not is_complete(Path(str(marker)[:-9])) for marker in complete):
            raise RuntimeError(f'dataset completion checksum mismatch: {path}')
        reports.append({'root': root, 'manifest_sha256': hashlib.sha256((path / 'manifest.json').read_bytes()).hexdigest(),
                        'files': len(actual), 'completion_checksums_verified': len(complete)})
    engine.write_json(output / 'data_reuse.json', reports)
    print(f'Validated {len(reports)} existing dataset inventories; no regeneration.', flush=True)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    args = arguments(argv)
    os.chdir(ROOT)
    output, jobs = build_jobs(args)
    return engine.execute(args, output, jobs, argv, script_path=Path(__file__).resolve(), data_preparer=verify_data)


if __name__ == '__main__':
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(130))
    raise SystemExit(main())
