#!/usr/bin/env python3
"""Eight paired simple-graph curriculum runs: protocol x rate x error loss.

Graph data and clean trajectory caches are reused without modification.
Prepare compact audited edits once per protocol/source, then start one trial
per idle GPU. The training loop never parses traces or searches a graph.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import multiprocessing as mp
import os
from pathlib import Path
import re
import signal
import sys

for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','TORCHINDUCTOR_COMPILE_THREADS'):
    os.environ.setdefault(key,'2')

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'),str(ROOT/'scripts')]
from cot_navigation.experiments import scheduler as engine
from cot_navigation.config import load_config, apply_training_profile
from cot_navigation.data.shards import atomic_json

OLD = ROOT/'outputs/scale_v1/launches/scale_v1_s0/context/curriculum'


def prepare_one(task):
    config, root = task
    from cot_navigation.data.dataset import ContextStore
    from cot_navigation.data.scale_prepared import ScalePreparedRows
    from cot_navigation.data.mistakes import prepare_catalog
    from cot_navigation.protocols.tokenizer import Tokenizer
    cfg = load_config([config])
    source = ContextStore(root)
    tok = Tokenizer.from_config(cfg)
    prepared = ScalePreparedRows(source,cfg,tok)
    path = prepare_catalog(source,prepared,cfg,tok)
    meta = json.loads(path.with_suffix('.json').read_text())
    return dict(protocol=cfg.protocol.name,source=root,path=str(path),**meta)


def prepare_data(jobs, output, env):
    tasks = {}
    for job in jobs:
        cfg = load_config([job.config])
        for root in cfg.train.scale_roots:
            tasks.setdefault((cfg.protocol.name,root),(str(job.config),root))
    prepared = []
    workers = min(6,len(tasks))
    with ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context('spawn')) as pool:
        futures = [pool.submit(prepare_one,t) for t in tasks.values()]
        for future in as_completed(futures):
            item = future.result()
            prepared.append(item)
            atomic_json(output/'data_preparation.json',dict(total=len(tasks),ready=len(prepared),datasets=prepared))
            print(f"DATA READY {item['protocol']} {item['source']}: {item['plans']} audited edits ({len(prepared)}/{len(tasks)})",flush=True)
    # Verify held-out clean caches too, without creating new data.
    from cot_navigation.data.dataset import ContextStore
    from cot_navigation.data.scale_prepared import ScalePreparedRows
    from cot_navigation.protocols.tokenizer import Tokenizer
    for protocol in ('dfs_state','bfs_state'):
        job = next(j for j in jobs if load_config([j.config]).protocol.name == protocol)
        cfg = load_config([job.config]);tok = Tokenizer.from_config(cfg)
        for spec in cfg.evaluation.extra_sets:
            source = ContextStore(spec.split('=',1)[1])
            cache = ScalePreparedRows(source,cfg,tok)
            cache.row(int(source.indices('test')[0]),0)
    print('All correction plans and existing held-out caches verified.',flush=True)


def build_jobs(args):
    output = ROOT/args.output_root/args.run_name
    configs = ROOT/'configs/generated/self_correction'/args.run_name
    configs.mkdir(parents=True,exist_ok=True)
    jobs, manifest = [], []
    for protocol in ('dfs_state','bfs_state'):
        original = OLD/protocol/'resolved_config.yaml'
        for percent in (15,30):
            for mask in (True,False):
                arm = f'p{percent:02d}_'+('masked' if mask else 'full_loss')
                name = f'context/{protocol}/{arm}'
                cfg = apply_training_profile(yaml.safe_load(original.read_text()))
                cfg['output_dir'] = str(output/name)
                cfg['protocol'].update(enable_mistake=True,mistake_rate=percent/100,
                    mistake_mask_errors=mask,mistake_seed=0,mistake_cache_root='data/mistake_v1')
                cfg['evaluation']['mistake_samples'] = 16
                cfg['tracking'].update(mode=args.tracking,group=args.run_name,
                    name=f'{args.run_name}/{name}',variant=arm,
                    tags=['self_correction','simple_layered','context',protocol,arm],
                    notes='Original simple-graph 4K curriculum; one erroneous candidate/visited record, MISTAKE, then canonical repair. No retry or oracle at inference.')
                path = configs/(name.replace('/','__')+'.yaml')
                if args.resume and path.exists():
                    cfg = yaml.safe_load(path.read_text())
                raw = yaml.safe_dump(cfg,sort_keys=False,allow_unicode=True)
                if path.exists() and path.read_text()!=raw:
                    raise ValueError(f'Immutable configuration changed: {path}')
                path.write_text(raw)
                load_config([path])
                jobs.append(engine.Job(name,path,[],output/name,cfg['train']['max_steps']))
                manifest.append(dict(job=name,original=str(original),config=str(path),rate=percent/100,mask_errors=mask))
    atomic_json(configs/'manifest.json',dict(jobs=manifest,from_scratch=True,
        events_per_corrupted_example=1,error_fields={'cands':0.5,'visited':0.5},
        correction='wrong state | MISTAKE ; followed by the complete original state/action record',
        clean_validation_for_selection=True,canonical_data_unchanged=True))
    return output,jobs


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-name',default='self_correction_s0')
    p.add_argument('--output-root',default='outputs/self_correction_v1/launches')
    p.add_argument('--gpus',type=int,nargs='+',default=list(range(8)))
    p.add_argument('--tracking',choices=['online','offline','disabled'],default='online')
    p.add_argument('--background',action='store_true')
    p.add_argument('--resume',action='store_true')
    p.add_argument('--dry-run',action='store_true')
    p.add_argument('--wait',action='store_true')
    args = p.parse_args(argv)
    if not re.fullmatch(r'[A-Za-z0-9_-][A-Za-z0-9_.-]*',args.run_name):p.error('invalid run name')
    if len(set(args.gpus))!=len(args.gpus) or not args.gpus or min(args.gpus)<0:p.error('invalid GPU list')
    os.chdir(ROOT)
    output,jobs = build_jobs(args)
    return engine.execute(args,output,jobs,argv,script_path=Path(__file__),data_preparer=prepare_data)


if __name__=='__main__':
    signal.signal(signal.SIGTERM,lambda *_:sys.exit(130))
    raise SystemExit(main())
