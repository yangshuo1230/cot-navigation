#!/usr/bin/env python3
"""Prepare/run the paired five-condition in-context DFS-state study.

Clones the original resolved curriculum configurations, including data roots,
scale weights, optimizer, validation selection and all final evaluation sets.
Does not generate data or start any fixed-graph/BFS experiment.
"""
from pathlib import Path
import argparse
import json
import re
import sys
import yaml

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from cot_navigation.experiments import scheduler as engine
from cot_navigation.config import load_config

ARMS={'full_l1':(1,0),'full_l4':(4,0),'full_l16':(16,0),'swa1_l1':(1,1),'swa2_l1':(1,2)}
OLD=ROOT/'outputs/scale_v1/launches/scale_v1_s0'

def prepare_data(jobs,output,env):
    """Read-only inventory: fail early instead of rebuilding shared datasets."""
    from cot_navigation.data.dataset import ContextStore
    from cot_navigation.data.scale_prepared import ScalePreparedRows
    from cot_navigation.protocols.tokenizer import Tokenizer
    checked=set()
    for job in jobs:
        cfg=load_config([job.config])
        roots=list(cfg.train.scale_roots)+[x.split('=',1)[1] for x in cfg.evaluation.extra_sets]
        for root in dict.fromkeys(roots):
            path=Path(root)
            key=(str(path),cfg.protocol.name)
            if key in checked:continue
            source=ContextStore(path)
            tok=Tokenizer.from_config(cfg)
            cache=ScalePreparedRows(source,cfg,tok)
            assert len(source.indices('test'))>0,(str(path),'empty test set')
            # Verify a real prepared record; no generation and no data writes.
            cache.row(int(source.indices('test')[0]),0)
            checked.add(key)
    print(f'Existing prepared datasets verified: {len(checked)} protocol/source pairs',flush=True)

def build_jobs(args):
    out=ROOT/args.output_root/args.run_name
    generated=ROOT/'configs/generated/curriculum_loss_swa'/args.run_name
    generated.mkdir(parents=True,exist_ok=True)
    jobs=[];manifest=[]
    for setting in ['context']:
        for protocol in args.protocols:
            rel=Path(setting)/'curriculum'
            original=OLD/rel/protocol/'resolved_config.yaml'
            for arm in args.arms:
                lam,window=ARMS[arm]
                cfg=yaml.safe_load(original.read_text())
                name=f'{setting}/{protocol}/{arm}'
                cfg['output_dir']=str(out/name)
                cfg['train']['trace_loss_lambda']=float(lam)
                cfg['model']['swa_records']=window
                cfg['model']['swa_backend']=args.swa_backend
                if window:cfg['model']['compile']=False
                cfg['tracking'].update(mode=args.tracking,group=args.run_name,
                    name=f'{args.run_name}/{name}',variant=arm,
                    tags=['curriculum_loss_swa',setting,protocol,arm],
                    notes='Normalized task loss; record SWA preserves input and absolute RoPE.')
                path=generated/(name.replace('/','__')+'.yaml')
                text=yaml.safe_dump(cfg,sort_keys=False)
                if path.exists() and path.read_text()!=text:
                    raise ValueError(f'Immutable generated configuration differs: {path}; use a new run name')
                path.write_text(text)
                load_config([path])
                jobs.append(engine.Job(name,path,[],out/name,cfg['train']['max_steps']))
                manifest.append({'name':name,'original_config':str(original),
                    'config':str(path),'lambda':lam,'swa_records':window})
    (generated/'manifest.json').write_text(json.dumps({'formula':'2*(answer+lambda*mean_trace)/(1+lambda)',
        'prompt_visibility':'always causal-visible','window_unit':'semicolon-completed state record',
        'current_record_counts_toward_window':False,'jobs':manifest},indent=2)+'\n')
    return out,jobs

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--setting',choices=['context'],default='context')
    parser.add_argument('--protocols',nargs='+',choices=['dfs_state'],default=['dfs_state'])
    parser.add_argument('--arms',nargs='+',choices=list(ARMS),default=list(ARMS))
    parser.add_argument('--gpus',nargs='+',type=int,default=list(range(8)))
    parser.add_argument('--tracking',choices=['online','offline','disabled'],default='online')
    parser.add_argument('--swa-backend',choices=['flex','sdpa'],default='flex')
    parser.add_argument('--run-name',default='curriculum_loss_swa_s0')
    parser.add_argument('--output-root',default='outputs/curriculum_loss_swa_v1/launches')
    parser.add_argument('--dry-run',action='store_true')
    parser.add_argument('--prepare-only',action='store_true')
    parser.add_argument('--background',action='store_true')
    parser.add_argument('--wait',action='store_true')
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args(argv)
    if not re.fullmatch(r'[A-Za-z0-9_-][A-Za-z0-9_.-]*',args.run_name):parser.error('invalid run name')
    if not args.gpus or min(args.gpus)<0 or len(set(args.gpus))!=len(args.gpus):parser.error('GPUs must be distinct nonnegative indices')
    if args.prepare_only and any([args.background,args.resume,args.dry_run]):parser.error('prepare-only cannot combine with background/resume/dry-run')
    output,jobs=build_jobs(args)
    if args.prepare_only:
        prepare_data(jobs,output,{})
        print('Configurations ready; no training started.')
        return 0
    return engine.execute(args,output,jobs,list(argv if argv is not None else sys.argv[1:]),
                          script_path=Path(__file__),data_preparer=prepare_data)

if __name__=='__main__':raise SystemExit(main())
