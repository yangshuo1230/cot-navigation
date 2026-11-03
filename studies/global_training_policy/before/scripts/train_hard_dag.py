#!/usr/bin/env python3
"""Prepare immutable hard DAG data, wait for prior launchers, fill eight GPUs.

Uses independent single-GPU trials, not eight-way DDP for a tiny model.
Resume with the same arguments plus --resume; raw data/caches are reused.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
import gzip
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

for _key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'TORCHINDUCTOR_COMPILE_THREADS'):
    os.environ.setdefault(_key, '2')

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))
from cot_navigation.experiments import scheduler as engine
from cot_navigation.config import load_config, to_plain
from cot_navigation.data.shards import atomic_json

FAMILY = 'multiscale_dag_v1'
PROTOCOLS = ('direct', 'path', 'dfs_state', 'bfs_state', 'dfs_nodes', 'bfs_nodes')
MIX = ((4, 4), (4, 6), (4, 8), (6, 6), (8, 8), (10, 10))
OOD = ((20, 5), (5, 20), (12, 12))


def arguments(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-name', default='hard_dag_v1_s0')
    p.add_argument('--output-root', default='outputs/hard_dag_v1/launches')
    p.add_argument('--gpus', type=int, nargs='+', default=list(range(8)))
    p.add_argument('--context-train', type=int, default=50000)
    p.add_argument('--holdout', type=int, default=512)
    p.add_argument('--variants', type=int, default=4)
    p.add_argument('--prepare-jobs', type=int, default=3)
    p.add_argument('--data-workers', type=int, default=4)
    p.add_argument('--tracking', choices=('online', 'offline', 'disabled'), default='online')
    p.add_argument('--compile', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--background', action='store_true')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--prepare-only', action='store_true')
    p.add_argument('--wait', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--after-pids', type=int, nargs='*', default=[])
    args = p.parse_args(argv)
    if any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-' for c in args.run_name) or args.run_name in ('', '.', '..'):
        p.error('run-name must be a plain name')
    if min(args.context_train, args.holdout, args.variants, args.prepare_jobs, args.data_workers) < 1 or args.context_train % 2 or args.holdout % 2:
        p.error('positive sizes; dataset counts must be even')
    return args


def root_for(kind, l, w, args):
    suffix = f'_train{args.context_train}' if kind == 'context' and (l, w) in MIX else ''
    return f'data/hard_dag_v1/{kind}_r{l}b{w}_n{l*w}_m{2*l*w}{suffix}_s0'


def data_flags(kind, l, w, args):
    is_train = (l, w) in MIX
    n_train = args.context_train if is_train else 2
    return [f'data.root={root_for(kind,l,w,args)}', f'data.n_nodes={l*w}',
            f'data.n_edges={2*l*w}', f'data.graph_family={FAMILY}',
            f'data.shapes={{main: {{layers: {l}, width: {w}}}}}',
            'data.graph_shape=main', 'data.graph_seed=0', 'data.graph_seeds=[0]',
            'data.data_seed=0', f'data.workers={args.data_workers}', 'data.max_attempts=2000',
            f'data.splits.train={n_train if kind == "context" else 0}',
            f'data.splits.val={args.holdout if kind == "context" else 0}',
            f'data.splits.test={args.holdout if kind == "context" else 0}',
            f'data.context_holdout_first={str(kind == "context").lower()}',
            f'data.audit_fraction={0.05 if kind == "context" else 1.0}',
            f'data.include_one_hop_queries={str(kind == "fixed").lower()}',
            'data.query_ratios=[0.7,0.15,0.15]']


def shared(args):
    return ['model.max_seq_len=32768', f'model.compile={str(args.compile).lower()}',
            'model.swa_records=0', 'train.trace_loss_lambda=1.0', 'protocol.cot_margin=8',
            f'protocol.trajectory_variants={args.variants}', 'runtime.device=cuda',
            'runtime.cuda_device=0', 'runtime.smoke=false', 'train.persist_outputs=true',
            'train.model_seed=0', 'train.eval_interval=5000', 'train.save_interval=5000',
            'train.eval_samples=128', 'train.free_gen_samples=64', 'train.keep_last=1',
            'train.max_tokens_per_microbatch=65536', 'evaluation.batch_size=16',
            'evaluation.test_samples=256', 'evaluation.trace_diagnostics=false',
            'evaluation.final_checkpoints=[best,last]', f'tracking.mode={args.tracking}',
            f'tracking.group={args.run_name}', 'tracking.tags=[hard_dag_v1,nonlayered]',
            'tracking.notes=Hidden-topological multiscale DAG; distance is rank-band gap, not shortest hops. Full reference traces; no truncation.']


def weights(main, count=6):
    if main == 0:
        return [1.] + [0.] * (count - 1)
    return [.3 / main] * main + [.7] + [0.] * (count - main - 1)


def build_jobs(args):
    output = ROOT / args.output_root / args.run_name
    configs = ROOT / 'configs/generated/hard_dag_v1' / args.run_name
    configs.mkdir(parents=True, exist_ok=True)
    jobs = []

    def add(name, preset, steps, flags, parent=None):
        out = output / name
        cfg = load_config([ROOT / f'configs/experiments/{preset}.yaml'], shared(args) + flags + [
            f'output_dir={out}', f'train.max_steps={steps}',
            f'tracking.name={args.run_name}/{name}', f'tracking.variant={name.replace("/", "-")}'])
        path = configs / (name.replace('/', '__') + '.yaml')
        raw = yaml.safe_dump(to_plain(cfg), sort_keys=False, allow_unicode=True)
        if path.exists() and path.read_text() != raw and (output / 'launch_plan.json').exists():
            raise ValueError(f'configuration already differs: {path}')
        path.write_text(raw)
        jobs.append(engine.Job(name, path, [], out, steps, parent))

    roots = [root_for('context', l, w, args) for l, w in MIX]
    stage_weights = [x for main in (0, 3, 4, 5) for x in weights(main)]
    extras = [f'n{l*w}={root_for("context",l,w,args)}' for l, w in MIX[:-1]] + [
        f'{name}={root_for("context",l,w,args)}' for name, (l, w) in zip(('deep','wide','larger'), OOD)]
    for protocol in PROTOCOLS[1:]:
        add(f'context/curriculum/{protocol}', 'retrain_context', 200000,
            data_flags('context', 10, 10, args) + [f'protocol.name={protocol}',
            f'protocol.negative_random_path={str(protocol == "path").lower()}',
            f'protocol.random_search_order={str(protocol not in ("direct", "path")).lower()}',
            'train.shuffle_context_edges=true', f'train.scale_roots={json.dumps(roots)}',
            'train.scale_stage_ends=[50000,100000,150000,200000]',
            f'train.scale_stage_weights={json.dumps(stage_weights)}',
            'evaluation.selection_min_step=150000', f'evaluation.extra_sets={json.dumps(extras)}'])
    for l, w, steps in ((6, 6, 75000), (10, 10, 125000)):
        root = root_for('fixed', l, w, args)
        common = data_flags('fixed', l, w, args) + [f'train.scale_roots=[{root}]', 'train.scale_weights=[1.0]']
        parent = f'fixed/n{l*w}/stage_a'
        add(parent, 'retrain_stage_a', 1000, common + ['evaluation.final_checkpoints=[]'])
        for protocol in PROTOCOLS:
            add(f'fixed/n{l*w}/stage_b/{protocol}', 'retrain_stage_b', steps, common + [
                f'protocol.name={protocol}', 'train.fixed_query_pool=full',
                'train.balance_query_labels=true', 'train.one_hop_fraction=0.25',
                f'protocol.negative_random_path={str(protocol == "path").lower()}',
                f'protocol.random_search_order={str(protocol not in ("direct", "path")).lower()}',
                'evaluation.selection_metric=balanced_accuracy', 'evaluation.fixed_query_view=full'], parent)
    return output, jobs


def distribution(values):
    a = np.asarray(values)
    return {'n': len(a), 'mean': float(a.mean()), 'p95': float(np.quantile(a,.95)),
            'max': int(a.max()), 'histogram': {str(v):int(n) for v,n in zip(*np.unique(a, return_counts=True))}}


def audit_source(source, cfg, cache_reports):
    """Compact structural and query statistics; actual hops never inferred from bands."""
    from cot_navigation.data.dataset import ContextStore
    from cot_navigation.data.hard_graph import graph_statistics, shortest_distances
    is_context = isinstance(source, ContextStore)
    arrays = source.arrays if is_context else source.queries
    structural, shortest = [], []
    for i in range(len(arrays['label'])):
        adj = source.adj(i) if is_context else source.adj()
        if is_context or i == 0:
            structural.append(graph_statistics(adj, next(iter(cfg.data.shapes.values())).width))
        s, t = (int(arrays[k][i]) for k in ('s_topo', 't_topo'))
        hops = shortest_distances(adj, s)[t]
        if (hops >= 0) != bool(arrays['label'][i]):
            raise AssertionError('independent shortest-path/label mismatch')
        shortest.append(hops)
    report = dict(graph_family=FAMILY, distance_semantics='hidden_topological_rank_band_gap_NOT_hops',
                  actual_shortest_hops=distribution([x for x in shortest if x >= 0]),
                  structure={k:distribution([x[k] for x in structural]) for k in structural[0]},
                  caches=cache_reports, references_truncated=0,
                  raw_validation=cfg.data.audit_fraction,
                  query_labels=distribution(arrays['label']))
    if not is_context:
        report['split_counts'] = {name:{'total':len(source.indices(name)),
            'positive':int(arrays['label'][source.indices(name)].sum())} for name in ('train','val','test')}
        report['all_prepared_paths'] = source.info['prepared_paths']
    atomic_json(Path(cfg.data.root) / 'hard_dag_audit.json', report)
    return report


def prepare_one(payload):
    kind, l, w, params = payload
    args = argparse.Namespace(**params)
    from cot_navigation.data.dataset import ContextStore, ParametricWorld
    from cot_navigation.data.prepared import prepare_dataset, _preparation_lock, _inventory
    from cot_navigation.data.scale_prepared import prepare_variants
    from cot_navigation.data.shards import load_npz
    from cot_navigation.protocols.tokenizer import Tokenizer
    preset = 'retrain_context' if kind == 'context' else 'retrain_stage_b'
    # A one-member scale mixture activates the cached training path for fixed.
    cfg = load_config([ROOT / f'configs/experiments/{preset}.yaml'], shared(args) +
                      data_flags(kind,l,w,args) + ['protocol.name=direct'])
    report = prepare_dataset(cfg, ['direct'])
    source = ContextStore(cfg.data.root) if kind == 'context' else ParametricWorld(Path(cfg.data.root)/'main-seed0')
    tok = Tokenizer.from_config(cfg)
    cache_reports = {}
    with _preparation_lock(Path(cfg.data.root)):
        for protocol in (PROTOCOLS[1:] if kind == 'context' else PROTOCOLS):
            print(f'CACHE {kind} n{l*w} {protocol}', flush=True)
            path = prepare_variants(source, cfg, tok, protocol)
            data = load_npz(path)
            for metadata in data['metadata']:
                if json.loads(str(metadata)).get('truncated'):
                    raise AssertionError('truncated reference')
            if kind == 'fixed' and protocol == 'path':
                actual = np.diff(data['query_offsets'])
                positive = source.queries['label'] == 1
                if not np.array_equal(actual[positive], np.diff(source.paths['query_offsets'])[positive]):
                    raise AssertionError('fixed PATH cache did not include all paths')
            cache_reports[protocol] = dict(rows=len(data['fields']),
                max_tokens=int(np.diff(data['offsets']).max()), bytes=path.stat().st_size)
        audit = audit_source(source, cfg, cache_reports)
        report['inventory'] = _inventory(Path(cfg.data.root))
        atomic_json(Path(cfg.data.root) / 'prepared_data.json', {k:v for k,v in report.items() if k != 'reused'})
    return {'root':cfg.data.root, 'kind':kind, 'shape':[l,w], 'audit':audit}


def process_identity(pid):
    try:
        value = Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()
        return None if value[0] == 'Z' else value[19]
    except (FileNotFoundError, ProcessLookupError):
        return None


def capture_barrier(args, output):
    path = output / 'queue_barrier.json'
    if path.exists():
        return json.loads(path.read_text())
    barriers = {str(pid):process_identity(pid) for pid in args.after_pids}
    if any(value is None for value in barriers.values()):
        print('Some prior launchers have already exited.', flush=True)
    atomic_json(path, barriers)
    return barriers


def data_preparer(args, output):
    barriers = capture_barrier(args, output)
    specs = [('fixed',6,6), ('fixed',10,10)] + [('context',l,w) for l,w in (*MIX,*OOD)]
    state_path = output / 'preparation_status.json'
    ready = []
    atomic_json(state_path, dict(phase='preparing_data', total=len(specs), completed=[]))
    import multiprocessing as mp
    with ProcessPoolExecutor(max_workers=args.prepare_jobs, mp_context=mp.get_context('spawn')) as pool:
        futures = {pool.submit(prepare_one,(kind,l,w,vars(args))):(kind,l,w) for kind,l,w in specs}
        for future in as_completed(futures):
            item = future.result()
            ready.append(item)
            atomic_json(state_path, dict(phase='preparing_data', total=len(specs), completed=ready))
            print(f'DATA READY {item["root"]} ({len(ready)}/{len(specs)})', flush=True)
    atomic_json(state_path, dict(phase='waiting_for_prior_training', total=len(specs), completed=ready))
    while True:
        live = [pid for pid,start in barriers.items() if start is not None and process_identity(pid) == start]
        atomic_json(output/'queue_status.json',dict(prior_launchers_alive=live,
            data_ready=True, phase='waiting_for_prior_training' if live else 'ready_for_gpu_scheduler'))
        if not live:
            break
        print(f'All data ready; waiting for prior launcher PIDs {live}', flush=True)
        time.sleep(30)
    atomic_json(state_path, dict(phase='data_ready_training_scheduler', total=len(specs), completed=ready))


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    args = arguments(argv)
    os.chdir(ROOT)
    output, jobs = build_jobs(args)
    if args.prepare_only:
        output.mkdir(parents=True, exist_ok=True)
        data_preparer(args, output)
        return 0
    return engine.execute(args, output, jobs, argv, script_path=Path(__file__).resolve(),
        data_preparer=lambda _jobs,_output,_env:data_preparer(args,_output))


if __name__ == '__main__':
    signal.signal(signal.SIGTERM, lambda *_:sys.exit(130))
    raise SystemExit(main())
