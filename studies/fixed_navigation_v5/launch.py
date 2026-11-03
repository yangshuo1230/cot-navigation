"""PATH-only scheduler. Default is a dry plan; --execute starts formal jobs.

Claims only idle GPUs, never kills jobs, stops dispatch on failure. Each job
starts from random initialization and jointly trains short/long queries + ADJ.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from data import HERE, atomic_json, digest


def idle_gpus():
    text = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.used,utilization.gpu',
                                    '--format=csv,noheader,nounits'], text=True)
    return {int(p[0]) for line in text.splitlines() if (p := line.split(','))
            and int(p[1]) < 256 and int(p[2]) < 5}


def jobs_for(mode, cfg):
    jobs = []
    worlds = [(80, 0, s) for s in cfg['model_seeds']] if mode == 'pilot' else [
        (160, g, s) for g in cfg['graph_seeds'] for s in cfg['model_seeds']]
    for n, graph, seed in worlds:
        prefix = f'n{n}/g{graph}_s{seed}'
        jobs.append({'id': f'{prefix}/path', 'protocol': 'path',
                     'world': f'n{n}_g{graph}.json', 'seed': seed})
    return jobs


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo-root', type=Path, required=True)
    p.add_argument('--mode', choices=['pilot', 'formal'], default='pilot')
    p.add_argument('--gpus', type=int, nargs='+', default=list(range(8)))
    p.add_argument('--execute', action='store_true')
    p.add_argument('--eager', action='store_true')
    p.add_argument('--tracking', choices=['online', 'disabled'], default='online')
    args = p.parse_args()
    cfg = json.loads((HERE / 'experiment.json').read_text())
    jobs = jobs_for(args.mode, cfg)
    root = args.repo_root / 'outputs/fixed_navigation_v5' / args.mode
    plan = {'mode': args.mode, 'jobs': jobs, 'output': str(root), 'gpus': args.gpus,
            'training_started': args.execute}
    if not args.execute:
        print(json.dumps(plan, indent=2))
        return
    if len(args.gpus) != len(set(args.gpus)):
        raise ValueError('Duplicate GPU identifiers')
    # Preparation and checks must be reviewable before any GPU worker starts.
    manifest = json.loads((HERE / 'manifest.json').read_text())
    verification = json.loads((HERE / 'verification.json').read_text())
    if verification['status'] != 'PASS' or not verification.get('supervision_checked') or verification['manifest_sha256'] != digest(manifest):
        raise ValueError('Dataset check missing or stale')
    from train import source_hash
    if verification.get('source_sha256') != source_hash(args.repo_root):
        raise ValueError('Runtime changed since verification; rerun check.py')
    root.mkdir(parents=True, exist_ok=True)
    import fcntl
    lock = (root / '.scheduler.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    atomic_json(root / 'plan.json', plan)
    active, completed, failed = {}, set(), []
    for job in jobs:
        if (root / job['id'] / 'COMPLETE.json').is_file():
            completed.add(job['id'])
    streams = []
    try:
        while len(completed) < len(jobs):
            for gpu, (job, proc) in list(active.items()):
                result = proc.poll()
                if result is None:
                    continue
                del active[gpu]
                if result == 0 and (root / job['id'] / 'COMPLETE.json').exists():
                    completed.add(job['id'])
                else:
                    failed.append({'job': job['id'], 'returncode': result})
            atomic_json(root / 'status.json', {'completed': sorted(completed), 'failed': failed,
                'active': {str(g): {'job': j['id'], 'pid': p.pid} for g, (j, p) in active.items()}})
            if failed:
                raise RuntimeError('Worker failed; dispatch stopped; other workers left running')
            busy = {j['id'] for j, _ in active.values()}
            for gpu in sorted(idle_gpus() & set(args.gpus) - set(active)):
                job = next((j for j in jobs if j['id'] not in completed | busy), None)
                if job is None:
                    continue
                out = root / job['id']
                command = [sys.executable, str(HERE / 'train.py'), '--repo-root', str(args.repo_root),
                           '--world', job['world'], '--protocol', job['protocol'], '--seed', str(job['seed']),
                           '--gpu', str(gpu), '--output', str(out), '--tracking', args.tracking, '--execute']
                if (out / 'latest.pt').exists():
                    command.append('--resume')
                elif out.exists():
                    raise FileExistsError(f'Partial job without checkpoint; inspect {out}')
                if args.eager:
                    command.append('--eager')
                stream = (root / (job['id'].replace('/', '__') + '.log')).open('a')
                streams.append(stream)
                env = {**os.environ, 'OMP_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2',
                       'TORCHINDUCTOR_COMPILE_THREADS': '2'}
                proc = subprocess.Popen(command, cwd=args.repo_root, stdout=stream,
                                        stderr=subprocess.STDOUT, env=env)
                active[gpu] = job, proc
                busy.add(job['id'])
                print(json.dumps({'start': job['id'], 'gpu': gpu, 'pid': proc.pid}), flush=True)
            if len(completed) < len(jobs):
                time.sleep(10)
    finally:
        for stream in streams:
            stream.close()
        lock.close()


if __name__ == '__main__':
    main()
