"""Depth/rate extension of the frozen DFS correction reproduction protocol.

Reuse its worker, length-dependent loss, immutable data and public scheduler.
Preparing performs CPU checks only. --launch --background detaches the queue.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BASE = HERE.parent / 'correction_reproduction'
sys.path[:0] = [str(ROOT / 'src'), str(BASE)]
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
            'TORCHINDUCTOR_COMPILE_THREADS'):
    os.environ.setdefault(key, '2')

import yaml
from cot_navigation.config import load_config
from cot_navigation.experiments import scheduler
from cot_navigation.data.shards import file_sha256
from suite import Job
from trace_weight import build_policy

SOURCE = ROOT / 'outputs/correction_reproduction_v1/launches/dfs_correction_repro_20261028/dfs_state/correction/s0/resolved_config.yaml'


def hashes():
    paths = [Path(__file__), SOURCE, BASE / 'worker.py', BASE / 'trace_weight.py',
             BASE / 'trace_weight_policy.json', BASE / 'data_audit.json',
             HERE.parent / 'accuracy_source/run.py']
    paths += sorted((ROOT / 'src/cot_navigation').rglob('*.py'))
    return {str(p.relative_to(ROOT)): file_sha256(p) for p in paths}


def prepare(args):
    template = yaml.safe_load(SOURCE.read_text())
    output = ROOT / 'outputs/correction_depth_rate/launches' / args.run_name
    if (output / 'DEPRECATED.json').exists():
        raise ValueError('Cohort withdrawn by user; training/resume disabled')
    generated = ROOT / 'configs/generated/correction_depth_rate' / args.run_name
    generated.mkdir(parents=True, exist_ok=True)
    jobs = []
    # Queue the larger models first; each GPU owns one independent trial.
    for depth in sorted(args.layers, reverse=True):
        for percent in sorted(args.error_percent, reverse=True):
            for seed in args.seeds:
                cfg = copy.deepcopy(template)
                name = f'dfs_state/l{depth}_p{percent}/s{seed}'
                cfg['output_dir'] = str(output / name)
                cfg['model']['layers'] = depth
                cfg['protocol'].update(mistake_rate=percent / 100, mistake_seed=seed)
                cfg['train'].update(model_seed=seed, exposure_seed=seed)
                cfg['tracking'].update(mode=args.tracking, group=args.run_name,
                    name=f'{args.run_name}/{name}', variant='correction',
                    tags=['correction_depth_rate', 'dfs_state', f'layers{depth}',
                          f'error{percent}', f'seed{seed}'],
                    notes='From scratch; frozen observed-error pool; full loss; '
                          'lambda=clip(actual CoT tokens/128,1,16). '
                          'Only depth and error fraction differ from prior correction protocol.')
                path = generated / f'l{depth}_p{percent}_s{seed}.yaml'
                text = yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True)
                if path.exists() and path.read_text() != text:
                    raise ValueError(f'Existing configuration differs: {path}; use another run name')
                if not path.exists():
                    path.write_text(text)
                load_config([path])
                # Reuse the exact original worker, including loss and semantic evaluation.
                jobs.append(Job(name, path, [], output / name, 200000, arm='correction'))
    return output, generated, jobs


def audit(jobs):
    """Validate reuse against the full prior semantic audit; never regenerate data."""
    import numpy as np
    import torch
    from cot_navigation.data.dataset import ContextStore
    from cot_navigation.data.natural_mistakes import NaturalCorrectionPool, pool_path
    from cot_navigation.data.scale_prepared import ScalePreparedRows
    from cot_navigation.models.transformer import build_model
    from cot_navigation.protocols.tokenizer import Tokenizer
    from cot_navigation.tracking.logger import _has_wandb_credentials
    from worker import weight_hash
    import wandb

    torch.set_num_threads(2)
    reference = load_config([SOURCE])
    template = yaml.safe_load(SOURCE.read_text())
    prior = json.loads((BASE / 'data_audit.json').read_text())
    assert json.loads((BASE / 'trace_weight_policy.json').read_text()) == build_policy()
    allowed = {'output_dir', 'model.layers', 'protocol.mistake_rate',
               'protocol.mistake_seed', 'train.model_seed', 'train.exposure_seed',
               'tracking.mode', 'tracking.online_failure', 'tracking.group', 'tracking.name', 'tracking.variant',
               'tracking.tags', 'tracking.notes'}

    def flattened(obj, prefix=''):
        result = {}
        for key, value in obj.items():
            path = f'{prefix}.{key}' if prefix else key
            if isinstance(value, dict):
                result.update(flattened(value, path))
            else:
                result[path] = value
        return result

    original = flattened(template)
    for job in jobs:
        actual = flattened(yaml.safe_load(job.config.read_text()))
        differences = {k for k in original.keys() | actual.keys() if original.get(k) != actual.get(k)}
        assert differences <= allowed, differences - allowed

    tok = Tokenizer.from_config(reference)
    scales = []
    for item in prior['scales']:
        path = ROOT / item['pool']
        assert file_sha256(path) == item['sha256'], f'Frozen error pool changed: {path}'
        source = ContextStore(item['root'])
        prepared = ScalePreparedRows(source, reference, tok)
        pool = NaturalCorrectionPool(source, reference, tok)
        assert pool_path(source, reference, tok) == Path(item['pool'])
        assert all(len(pool.by_label[y]) for y in (0, 1))
        for index in np.flatnonzero(pool.arrays['split'] == 0)[:8]:
            row = pool.row(int(index), 0)
            clean = prepared.row(int(row.meta['index']), int(row.meta['natural_variant']))
            assert row.label == clean.label and row.tokens[:row.prompt_len] == clean.tokens[:clean.prompt_len]
            assert len(row.tokens) <= reference.model.max_seq_len
        scales.append(item)
        del pool, prepared, source

    # Verify every held-out reference cache, including the special hard-DAG cache.
    for name, item in prior['eval_sets'].items():
        cfg = reference if name != 'hard' else load_config([
            ROOT / 'outputs/hard_dag_v1/launches/hard_dag_v1_s0/context/curriculum/dfs_state/resolved_config.yaml'])
        prepared = ScalePreparedRows(ContextStore(item['root']), cfg, Tokenizer.from_config(cfg))
        del prepared

    initial = {}
    for job in jobs:
        cfg = load_config([job.config])
        key = f'l{cfg.model.layers}_s{cfg.train.model_seed}'
        torch.manual_seed(cfg.train.model_seed)
        model = build_model(cfg.model, len(tok), pad_id=tok.pad)
        digest = weight_hash(model)
        if key in initial:
            assert initial[key]['sha256'] == digest
        initial[key] = dict(sha256=digest, parameters=sum(p.numel() for p in model.parameters()))
        if cfg.model.layers == 6:
            assert digest == prior['initial_weights'][str(cfg.train.model_seed)]
        del model
    if any(load_config([j.config]).tracking.mode == 'online' for j in jobs):
        assert _has_wandb_credentials(wandb), 'W&B online credentials are unavailable'
    return dict(passed=True, jobs=len(jobs), cpu_only=True, navigation_training_started=False,
                inherited_settings_verified=True, scales=scales, initial_models=initial,
                trace_weight_policy=build_policy(), prior_semantic_audit=str(BASE / 'data_audit.json'))


def verify_plan(jobs, output, env):
    manifest = json.loads((output.parent / f'{output.name}.prepared.json').read_text())
    assert manifest['source_hashes'] == hashes(), 'Source changed after preparation'
    assert manifest['jobs'] == scheduler.plan_record(jobs), 'Job plan changed after preparation'
    assert manifest['audit']['passed']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-name', default='dfs_depth_rate_20261031')
    parser.add_argument('--layers', type=int, nargs='+', default=[6, 9])
    parser.add_argument('--error-percent', type=int, nargs='+', default=[30, 60, 90])
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1])
    parser.add_argument('--gpus', type=int, nargs='+', default=list(range(8)))
    parser.add_argument('--tracking', choices=['online', 'offline', 'disabled'], default='disabled')
    for flag in ('launch', 'background', 'resume', 'wait'):
        parser.add_argument('--' + flag, action='store_true')
    args = parser.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', args.run_name):
        parser.error('Invalid run name')
    for name in ('layers', 'error_percent', 'seeds', 'gpus'):
        values = getattr(args, name)
        if len(values) != len(set(values)) or min(values) < (1 if name == 'layers' else 0):
            parser.error(f'Invalid {name}')
    if max(args.error_percent) > 100:
        parser.error('Error percentage must be <=100')
    if args.background and not args.launch:
        parser.error('--background requires --launch')
    os.chdir(ROOT)
    output, generated, jobs = prepare(args)
    manifest = output.parent / f'{args.run_name}.prepared.json'
    if not args.launch:
        receipt = dict(run_name=args.run_name, source_hashes=hashes(),
                       jobs=scheduler.plan_record(jobs), audit=audit(jobs))
        manifest.parent.mkdir(parents=True, exist_ok=True)
        scheduler.write_json(manifest, receipt)
        print(f'PREPARATION PASSED: {manifest}', flush=True)
    else:
        verify_plan(jobs, output, os.environ)
    args.dry_run = not args.launch
    return scheduler.execute(args, output, jobs, sys.argv[1:], script_path=Path(__file__),
                             data_preparer=verify_plan)


if __name__ == '__main__':
    raise SystemExit(main())
