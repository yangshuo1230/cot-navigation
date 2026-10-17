#!/usr/bin/env python3
"""Controlled exploration branches; reuses the existing GPU scheduler and W&B.

Default: standard graph family, 50k independent context graphs, 50k task
updates, 1k shared knowledge updates per fixed world. No Filler. --dry-run is
read-only; --prepare-only does CPU preparation without creating a training run.
"""
from __future__ import annotations
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
sys.path.insert(0, str(ROOT/'scripts'))
import train_first_round as engine
from cot_navigation.config import load_config
from retrain import prepare_only

# Each key is an isolated experimental change, not a Cartesian product of
# all changes. Shared baselines keep comparisons interpretable and affordable.
EXPERIMENTS = {
    'baseline': ('direct', 'path', 'dfs_state'),
    'direct_onehop': ('direct',),
    'direct_curriculum': ('direct',),
    'walk_balanced': ('direct', 'path', 'dfs_state'),
    'walk_none': ('direct', 'path', 'dfs_state'),
    'path_negative': ('path',),
    'dfs_random': ('dfs_state',),
    'bfs_state': ('bfs_state',),
    'dfs_nodes': ('dfs_nodes',),
    'bfs_nodes': ('bfs_nodes',),
}
FIXED_ONLY = {'walk_balanced', 'walk_none', 'dfs_random'}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--setting', choices=('all','context','fixed'), default='all')
    p.add_argument('--experiments', nargs='+', choices=('all',*EXPERIMENTS), default=['all'])
    p.add_argument('--shapes', nargs='+', choices=('standard','wide','deep'), default=['standard'])
    p.add_argument('--context-sizes', nargs='+', type=int, default=[50000])
    p.add_argument('--context-val-size', type=int, default=10000)
    p.add_argument('--context-test-size', type=int, default=10000)
    p.add_argument('--graph-seeds', nargs='+', type=int, default=[0,1,2])
    p.add_argument('--model-seed', type=int, default=0)
    p.add_argument('--data-seed', type=int, default=0)
    p.add_argument('--context-steps', type=int, default=50000)
    p.add_argument('--task-steps', type=int, default=50000)
    p.add_argument('--knowledge-steps', type=int, default=1000)
    p.add_argument('--gpus', nargs='+', type=int)
    p.add_argument('--run-name')
    p.add_argument('--output-root', default='outputs/explore_v1/launches')
    p.add_argument('--wait', action='store_true')
    p.add_argument('--background', action='store_true')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--prepare-only', action='store_true')
    p.add_argument('--compile', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--tracking', choices=('online','offline','disabled'), default='online')
    p.add_argument('--set', action='append', default=[], metavar='KEY=VALUE')
    args = p.parse_args(argv)
    if args.resume and not args.run_name:
        p.error('--resume requires the original --run-name')
    if args.prepare_only and (args.background or args.resume):
        p.error('--prepare-only cannot be combined with background/resume')
    for key in ('context_sizes','context_val_size','context_test_size'):
        values = getattr(args,key) if isinstance(getattr(args,key),list) else [getattr(args,key)]
        if any(n<2 or n%2 for n in values):
            p.error(f'{key} must contain positive even counts')
    if any(getattr(args,k)<1 for k in ('context_steps','task_steps','knowledge_steps')):
        p.error('training steps must be positive')
    if args.gpus is not None and (min(args.gpus)<0 or len(set(args.gpus))!=len(args.gpus)):
        p.error('--gpus must contain distinct nonnegative indices')
    if any(seed not in (0,1,2) for seed in args.graph_seeds):
        p.error('prepared fixed worlds support graph seeds 0, 1, 2')
    for key in ('experiments','shapes','context_sizes','graph_seeds'):
        setattr(args,key,list(dict.fromkeys(getattr(args,key))))
    if 'all' in args.experiments:
        if args.experiments != ['all']:
            p.error('use all alone, or list individual experiments')
        args.experiments = list(EXPERIMENTS)
    owned = {'output_dir','protocol.name','protocol.negative_random_path','protocol.random_search_order',
             'train.stage','train.max_steps','train.model_seed','train.query_hops','train.curriculum_stage_steps',
             'train.walk_loss_mode','train.walk_batch_size','train.fixed_query_pool','train.balance_query_labels',
             'train.one_hop_fraction','train.shuffle_context_edges','train.persist_outputs','runtime.smoke',
             'data.root','data.n_nodes','data.n_edges','data.shapes','data.graph_shape','data.graph_seed',
             'data.graph_seeds','data.data_seed','data.splits.train','data.splits.val','data.splits.test',
             'data.include_one_hop_queries','data.context_holdout_first',
             'tracking.mode','tracking.group','tracking.name','tracking.variant'}
    for item in args.set:
        if '=' not in item or item.split('=',1)[0] in owned or item.split('=',1)[0].startswith('data.shapes.'):
            p.error(f'runner owns this option or invalid override: {item}')
    args.run_name = args.run_name or datetime.now(ZoneInfo('Asia/Shanghai')).strftime('explore_%Y%m%d_%H%M%S')
    if not re.fullmatch(r'[A-Za-z0-9_.-]+',args.run_name) or args.run_name in {'.','..'}:
        p.error('--run-name must be a plain name')
    return args


def stage_budgets(total, hops):
    """Equal stages up to the unavoidable one-update integer remainder."""
    if total < hops:
        raise ValueError('task steps must cover every curriculum stage')
    return [total//hops + int(i < total%hops) for i in range(hops)]


def geometry(setting, shape):
    if shape == 'standard':
        return (16,24,4,4) if setting == 'context' else (32,48,8,4)
    return (32,48,4,8) if shape == 'wide' else (32,48,8,4)


def build_jobs(args, root=ROOT):
    output = Path(args.output_root)
    output = (output if output.is_absolute() else root/output)/args.run_name
    shared = args.set + [f'model.compile={str(args.compile).lower()}', f'tracking.mode={args.tracking}',
        f'train.model_seed={args.model_seed}', f'data.data_seed={args.data_seed}',
        'runtime.device=cuda','runtime.smoke=false','train.persist_outputs=true',
        f'tracking.group={json.dumps(args.run_name)}']
    jobs = []
    def add(name,preset,steps,extra,parent=None):
        out = output/name
        overrides = shared + extra + [f'train.max_steps={steps}',f'output_dir={json.dumps(str(out))}',
                     f'tracking.name={json.dumps(args.run_name+"/"+name)}']
        path = root/'configs/experiments'/f'{preset}.yaml'
        cfg = load_config([path],overrides)
        if cfg.train.warmup_steps >= cfg.train.max_steps:
            raise ValueError(f'{name}: warmup must be shorter than the stage budget')
        jobs.append(engine.Job(name,path,overrides,out,steps,parent))

    for setting in ('context','fixed'):
        if args.setting not in ('all',setting):
            continue
        experiments = [e for e in args.experiments if setting=='fixed' or e not in FIXED_ONLY]
        if not experiments:
            continue
        for shape in args.shapes:
            n,m,layers,width = geometry(setting,shape)
            common = [f'data.n_nodes={n}',f'data.n_edges={m}',
                      f'data.shapes={{balanced: {{layers: {layers}, width: {width}}}}}',
                      'data.graph_shape=balanced']
            for size_or_seed in (args.context_sizes if setting=='context' else args.graph_seeds):
                if setting=='context':
                    data_root = f'data/explore_v1/context_{shape}_n{n}_m{m}_train{size_or_seed}_val{args.context_val_size}_test{args.context_test_size}_s{args.data_seed}'
                    extra = common + [f'data.root={data_root}', f'data.splits.train={size_or_seed}',
                        f'data.splits.val={args.context_val_size}',f'data.splits.test={args.context_test_size}',
                        'train.shuffle_context_edges=true']
                    prefix = f'context/{shape}/train{size_or_seed}'
                    steps, preset, parent = args.context_steps, 'retrain_context', None
                else:
                    data_root = f'data/explore_v1/fixed_{shape}_n{n}_m{m}_onehop_s{args.data_seed}'
                    extra = common + [f'data.root={data_root}',f'data.graph_seed={size_or_seed}']
                    prefix = f'fixed/{shape}/seed{size_or_seed}'
                    parent = prefix+'/stage_a'
                    add(parent,'retrain_stage_a',args.knowledge_steps,extra+['tracking.variant=knowledge'])
                    steps, preset = args.task_steps, 'retrain_stage_b'
                for experiment in experiments:
                    changes = [f'tracking.variant={experiment}']
                    if experiment in ('direct_onehop','direct_curriculum'):
                        changes += [f'train.query_hops={"one_hop" if experiment=="direct_onehop" else "curriculum"}',
                                    'train.one_hop_fraction=0.0']
                        if experiment=='direct_curriculum':
                            changes += [f'train.curriculum_stage_steps={json.dumps(stage_budgets(steps,layers-1))}']
                    if experiment in ('walk_balanced','walk_none'):
                        changes += [f'train.walk_loss_mode={"balanced" if experiment=="walk_balanced" else "none"}']
                        if experiment=='walk_none':
                            changes += ['train.walk_batch_size=0']
                    if experiment=='path_negative':
                        changes += ['protocol.negative_random_path=true']
                    if experiment=='dfs_random':
                        changes += ['protocol.random_search_order=true']
                    for protocol in EXPERIMENTS[experiment]:
                        add(prefix+f'/{experiment}/{protocol}',preset,steps,
                            extra+changes+[f'protocol.name={protocol}'],parent)
    if not jobs:
        raise ValueError('no experiments match this setting')
    return output,jobs


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    args = parse_args(argv)
    os.chdir(ROOT)
    output,jobs = build_jobs(args)
    if args.prepare_only and not args.dry_run:
        return prepare_only(jobs)
    return engine.execute(args,output,jobs,argv,script_path=Path(__file__).resolve())


if __name__ == '__main__':
    def interrupt(_signum,_frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM,interrupt)
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
    except (RuntimeError,ValueError,subprocess.CalledProcessError) as exc:
        print(f'Error: {exc}',file=sys.stderr)
        raise SystemExit(1)
