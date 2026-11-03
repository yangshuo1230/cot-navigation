"""Finite independent protocol jobs; dry plan unless --execute is supplied."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from prepare import HERE, digest, atomic_json
from common import source_hash


def jobs_for(mode, cfg, protocols):
    worlds = [(80, 0)] if mode == 'pilot' else [(160, s) for s in cfg['graph_seeds']]
    return [{'id': f'n{n}/g{graph}_s{seed}/{p}', 'world': f'n{n}_g{graph}', 'seed': seed, 'protocol': p}
            for n, graph in worlds for seed in cfg['model_seeds'] for p in protocols]


def idle_gpus():
    rows = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.used,utilization.gpu',
                                    '--format=csv,noheader,nounits'], text=True)
    return {int(p[0]) for line in rows.splitlines() if (p := line.split(',')) and int(p[1]) < 256 and int(p[2]) < 5}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo-root', type=Path, required=True)
    p.add_argument('--mode', choices=['pilot', 'formal'], default='pilot')
    p.add_argument('--protocols', nargs='+', choices=['direct', 'dfs_state', 'dfs_nodes'])
    p.add_argument('--gpus', nargs='+', type=int, default=list(range(8)))
    p.add_argument('--execute', action='store_true')
    args = p.parse_args()
    repo = args.repo_root.resolve()
    cfg = json.loads((HERE / 'experiment.json').read_text())
    protocols = args.protocols or cfg['protocols']
    if len(set(protocols)) != len(protocols) or len(set(args.gpus)) != len(args.gpus):
        raise ValueError('Duplicate protocols or GPU IDs')
    jobs = jobs_for(args.mode, cfg, protocols)
    root = repo / 'outputs/fixed_navigation_protocols_v6' / args.mode
    if not args.execute:
        print(json.dumps({'jobs': jobs, 'tasks': len(jobs), 'output': str(root), 'training_started': False}, indent=2))
        return
    manifest = json.loads((HERE / 'manifest.json').read_text())
    verified = json.loads((HERE / 'verification.json').read_text())
    assert verified['status'] == 'PASS' and verified['manifest_sha256'] == digest(manifest)
    assert verified['source_sha256'] == source_hash(repo), 'Recheck modified source'
    assert {j['world'] for j in jobs} <= {w['name'] for w in manifest['worlds']}
    root.mkdir(parents=True, exist_ok=True)
    import fcntl
    lock = (root / '.scheduler.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    atomic_json(root / 'plan.json', {'jobs': jobs, 'gpus': args.gpus, 'config_sha256': digest(cfg)})
    done = {j['id'] for j in jobs if (root / j['id'] / 'COMPLETE.json').exists()}
    active, failed, streams = {}, [], []
    try:
        while len(done) < len(jobs):
            for gpu, (job, proc) in list(active.items()):
                status = proc.poll()
                if status is None:
                    continue
                del active[gpu]
                if status == 0 and (root / job['id'] / 'COMPLETE.json').exists():
                    done.add(job['id'])
                else:
                    failed.append({'job': job['id'], 'returncode': status})
            atomic_json(root / 'status.json', {'completed': sorted(done), 'failed': failed,
                'active': {str(g): {'job': j['id'], 'pid': p.pid} for g, (j, p) in active.items()}})
            if failed:
                raise RuntimeError('Worker failed; dispatch stopped; other processes left running')
            busy = {j['id'] for j, _ in active.values()}
            for gpu in sorted(idle_gpus() & (set(args.gpus) - set(active))):
                job = next((j for j in jobs if j['id'] not in done | busy), None)
                if job is None:
                    continue
                out = root / job['id']
                command = [sys.executable, str(HERE / 'train_protocol.py'), '--repo-root', str(repo),
                    '--world', job['world'], '--protocol', job['protocol'], '--seed', str(job['seed']),
                    '--gpu', str(gpu), '--output', str(out), '--execute']
                if (out / 'latest.pt').exists():
                    command.append('--resume')
                elif out.exists():
                    raise FileExistsError(f'Partial run without checkpoint: {out}')
                stream = (root / (job['id'].replace('/', '__') + '.log')).open('a')
                streams.append(stream)
                env = {**os.environ, 'OMP_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2', 'TORCHINDUCTOR_COMPILE_THREADS': '2'}
                proc = subprocess.Popen(command, cwd=repo, stdout=stream, stderr=subprocess.STDOUT, env=env)
                active[gpu] = job, proc
                busy.add(job['id'])
                print(json.dumps({'start': job['id'], 'gpu': gpu, 'pid': proc.pid}), flush=True)
            if len(done) < len(jobs):
                time.sleep(10)
    finally:
        for stream in streams:
            stream.close()
        lock.close()


if __name__ == '__main__':
    main()
