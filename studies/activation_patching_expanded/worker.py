"""Prepare large legal paired datasets, then stream frozen activation swaps.

One worker owns one GPU; only compact recipes and key results are retained.
Existing completed results are verified and reused, never silently overwritten.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import gzip
import hashlib
import json
import os
from pathlib import Path
import time


def read_gzip(path):
    with gzip.open(path, 'rt', encoding='utf-8') as f:
        return json.load(f)


def save_gzip(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with gzip.open(tmp, 'wt', encoding='utf-8', compresslevel=6) as f:
        json.dump(value, f, ensure_ascii=False, allow_nan=False, separators=(',', ':'))
    tmp.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models', nargs='+', required=True)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--mode', choices=['prepare', 'run', 'both'], default='both')
    parser.add_argument('--cases', type=int, default=128, help='Per cohort, per matched field')
    parser.add_argument('--queries', type=int, default=1800)
    parser.add_argument('--variants', type=int, default=3)
    parser.add_argument('--max-prefix', type=int, default=3000)
    parser.add_argument('--max-new', type=int, default=384)
    parser.add_argument('--batch', type=int, default=96, help='Includes the unpatched twins')
    parser.add_argument('--chunk-cases', type=int, default=24)
    parser.add_argument('--seed', type=int, default=20261021)
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    import numpy as np
    import torch
    from cot_navigation.analysis.common import write_json, sha256
    from cot_navigation.analysis.scale_error_awareness import ScaleModel
    from data import prepare, restore_pairs
    from engine import execute, FIELDS, GROUPS
    torch.set_num_threads(4)
    directory = Path(__file__).resolve().parent
    selection = json.loads((directory / 'selection.json').read_text())['models']
    entries = {e['id']: e for e in selection}
    # Experiment identity excludes device/batching but includes the data recipe
    # version and source hash. Numerical execution settings are saved per run.
    recipe_settings = {k: getattr(args, k) for k in ('cases', 'queries', 'variants', 'max_prefix', 'seed')}
    recipe_settings['generator_sha256'] = sha256(directory / 'data.py')
    for model_id in args.models:
        started = time.time()
        entry = entries[model_id]
        am = ScaleModel(entry, Path.cwd(), device='cpu' if args.mode == 'prepare' else 'cuda:0')
        assert am.checkpoint['sha256'] == entry['sha256']
        dataset_path = directory / 'datasets' / f'{model_id}.json.gz'
        if dataset_path.exists():
            dataset = read_gzip(dataset_path)
            assert dataset['settings'] == recipe_settings, 'Use a new study directory for a changed dataset'
            assert dataset['checkpoint_sha256'] == entry['sha256']
        else:
            assert args.mode != 'run', 'Prepare the reproducible dataset first'
            records, audit, coverage = prepare(am, count=args.cases, queries=args.queries, variants=args.variants,
                seed=args.seed, max_prefix=min(args.max_prefix, am.model.max_seq_len - args.max_new - 1))
            dataset = dict(model=model_id, checkpoint_sha256=entry['sha256'], settings=recipe_settings,
                           audit=audit, coverage=coverage, records=records)
            save_gzip(dataset_path, dataset)
            print(json.dumps(dict(model=model_id, stage='prepared', pairs=len(records), coverage=coverage)), flush=True)
        if args.mode == 'prepare':
            del am
            continue
        dataset_hash = sha256(dataset_path)
        execution_settings = {k: getattr(args, k) for k in ('max_new', 'batch', 'chunk_cases')}
        execution_settings['engine_sha256'] = sha256(directory / 'engine.py')
        for readout in ('next_record', 'current_record'):
            destination = directory / 'results' / f'{model_id}__{readout}.json.gz'
            if destination.exists():
                prior = read_gzip(destination)
                assert prior['dataset_sha256'] == dataset_hash and prior['execution_settings'] == execution_settings
                print(json.dumps(dict(model=model_id, readout=readout, status='verified_existing')), flush=True)
                continue
            observations, examples = [], []
            # Mix depths and field/cohort groups so sparse controls cover all.
            order = np.random.default_rng(args.seed + 41).permutation(len(dataset['records'])).tolist()
            checks = dict(self=0, disconnected_final_layer=0)
            for start in range(0, len(order), args.chunk_cases):
                ids = order[start:start + args.chunk_cases]
                recipes = [dataset['records'][i] for i in ids]
                pairs = restore_pairs(am, recipes, readout)
                with torch.inference_mode():
                    rows, details = execute(am, pairs, [1, 3], list(GROUPS), args.batch, args.max_new)
                for row in rows:
                    local = row['case']
                    row['case'] = ids[local]
                    row['cohort'] = recipes[local]['cohort']
                    row['depth_bin'] = recipes[local]['depth_bin']
                    row['unit'] = recipes[local]['unit']
                    if row['source'] == 'self':
                        assert row['probability_max_abs_delta'] == 0 and not row['continuation_changed']
                        checks['self'] += 1
                    if row['layer'] == am.model.n_layers and row['source'] in ('no_history', 'cross_example') and row['group'] in (
                        ('nodes', 'bar', 'operator', 'nodes_bar') if readout == 'next_record' else ('nodes',)):
                        assert row['probability_max_abs_delta'] == 0 and not row['continuation_changed']
                        checks['disconnected_final_layer'] += 1
                    observations.append(row)
                for detail in details.values():
                    detail['case'] = ids[detail['case']]
                    examples.append(detail)
                examples.sort(key=lambda x: x['interesting_rank'], reverse=True)
                examples = examples[:8]
                write_json(directory / 'progress' / f'{model_id}.json', dict(model=model_id, readout=readout,
                    done=min(start + len(ids), len(order)), total=len(order), seconds=time.time() - started))
                print(json.dumps(dict(model=model_id, readout=readout, cases=min(start + len(ids), len(order)), total=len(order))), flush=True)
            assert sha256(Path.cwd() / entry['run'] / 'checkpoints/best.pt') == entry['sha256']
            save_gzip(destination, dict(model=entry, checkpoint=am.checkpoint, dataset_sha256=dataset_hash,
                recipe_settings=recipe_settings, execution_settings=execution_settings, readout=readout,
                physical_gpu=args.gpu, checks=checks, observations=observations, examples=examples,
                wall_seconds=time.time() - started))
            print(json.dumps(dict(model=model_id, readout=readout, stage='complete', rows=len(observations))), flush=True)
        del am
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
