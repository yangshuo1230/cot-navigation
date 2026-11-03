#!/usr/bin/env python3
"""Single-world N32 WALK ablations, including joint training from random init.

Reuses the scale N32 4x8 world, its query split, prepared trajectory variants,
node names and 286-token vocabulary. GPUs 1-7; GPU0 is never selected.
"""
import argparse
import json
from pathlib import Path
import re
import sys
import os
import subprocess
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
from cot_navigation.experiments import scheduler as engine
from cot_navigation.config import load_config
from cot_navigation.data.dataset import ParametricWorld
from cot_navigation.data.scale_prepared import ScalePreparedRows
from cot_navigation.data.prepared import PreparedRows
from cot_navigation.protocols.tokenizer import Tokenizer

PROTOCOLS=('direct','path','dfs_state','bfs_state','dfs_nodes','bfs_nodes')

def build_jobs(args):
    output=ROOT/'outputs/task_only_n32_v1/launches'/args.run_name
    jobs=[]
    def add(name,config,steps,parent=None):
        dest=output/name
        overrides=[f'output_dir={dest}',f'tracking.name={args.run_name}/{name}',
                   f'tracking.group={args.run_name}',f'tracking.mode={args.tracking}',
                   f'train.max_steps={steps}']
        jobs.append(engine.Job(name,ROOT/f'configs/experiments/task_only_n32_v1/{config}.yaml',overrides,dest,steps,parent))
    if getattr(args,'arm','paired') == 'joint_from_scratch':
        for p in args.protocols:
            add('joint_from_scratch/'+p,'joint_'+p,args.steps)
    else:
        for p in args.protocols:
            add('task_only/'+p,p,args.steps)
        if args.with_walk_control:
            add('walk_control/stage_a','knowledge',1000)
            for p in args.protocols:
                add('walk_control/'+p,'control_'+p,args.steps,'walk_control/stage_a')
    return output,jobs

def verify_cached_data(jobs,output,env):
    """Read existing query/trajectory caches and the WALK cache when needed."""
    evidence=[]
    for job in jobs:
        cfg=load_config([job.config],job.overrides)
        if job.name.startswith('task_only/'):
            assert cfg.train.task_only and cfg.train.stage=='B' and cfg.train.walk_loss_mode=='none'
        else:
            assert not cfg.train.task_only and cfg.train.walk_loss_mode=='fixed'
        if job.name.startswith('joint_from_scratch/'):
            assert cfg.train.skip_knowledge_stage and cfg.train.stage=='B' and job.depends_on is None
        else:
            assert not cfg.train.skip_knowledge_stage
        assert len(cfg.train.scale_roots)==1 and cfg.data.n_nodes==32
        world=ParametricWorld(Path(cfg.data.root)/'main-seed0')
        tok=Tokenizer.from_config(cfg)
        cache=ScalePreparedRows(world,cfg,tok)
        if not cfg.train.task_only:
            PreparedRows(world,'walk',tok)
        train=set(map(int,world.indices('train','multi_hop')))|set(map(int,world.indices('train','one_hop')))
        val=set(map(int,world.indices('val','full')));test=set(map(int,world.indices('test','full')))
        assert not (train&val or train&test or val&test)
        assert len(train)==288 and len(val)==37 and len(test)==59
        assert len(tok)==286
        evidence.append(dict(job=job.name,protocol=cfg.protocol.name,train_queries=len(train),val_queries=len(val),test_queries=len(test),
                             data_fingerprint=world.fingerprint,initialization='random' if cfg.train.task_only or cfg.train.skip_knowledge_stage or cfg.train.stage=='A' else 'shared_stage_a',
                             stage_a=not (cfg.train.task_only or cfg.train.skip_knowledge_stage),walk_sampling=not cfg.train.task_only,
                             cached_first_query_variants=cache.variants(min(train)),vocab_hash=tok.hash()))
    engine.write_json(output/'data_contract.json',evidence)

def audit_final_trajectories(jobs,output,gpus):
    """After training, replay every non-Direct final-test trajectory."""
    selected=[dict(name=job.name,folder=str(job.output),cases=[['main',None,256]])
              for job in jobs if job.name!='walk_control/stage_a' and not job.name.endswith('/direct')]
    jobfile=output/'semantic_jobs.json'
    engine.write_json(jobfile,selected)
    while True:
        available=[row['index'] for row in engine.idle_gpus() if row['idle'] and row['index'] in gpus]
        if available:
            break
        time.sleep(10)
    argv=[sys.executable,str(ROOT/'studies/scale_trace_audit/audit.py'),
          '--job-file',str(jobfile),'--output-root',str(output/'semantic_audit')]
    with (output/'semantic_audit.log').open('a') as log:
        subprocess.run(argv,cwd=ROOT,env={**os.environ,'CUDA_VISIBLE_DEVICES':str(available[0]),'WANDB_MODE':'disabled'},
                       stdout=log,stderr=subprocess.STDOUT,check=True)

def main(argv=None):
    argv=sys.argv[1:] if argv is None else argv
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm',choices=['paired','joint_from_scratch'],default='paired',
                        help='paired: existing task-only/WALK comparison; joint: skip Stage A, train QUERY+WALK')
    parser.add_argument('--run-name')
    parser.add_argument('--gpus',nargs='+',type=int,default=list(range(1,8)))
    parser.add_argument('--protocols',nargs='+',choices=PROTOCOLS,default=list(PROTOCOLS))
    parser.add_argument('--steps',type=int,default=50000)
    parser.add_argument('--tracking',choices=['online','offline','disabled'],default='online')
    parser.add_argument('--with-walk-control',action=argparse.BooleanOptionalAction,default=True,
                        help='matched 1k Stage A + QUERY/WALK control for each protocol')
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--background',action='store_true')
    parser.add_argument('--wait',action='store_true')
    parser.add_argument('--dry-run',action='store_true')
    args=parser.parse_args(argv)
    args.run_name=args.run_name or ('joint_walk_n32_s0' if args.arm=='joint_from_scratch' else 'task_only_n32_s0')
    if not re.fullmatch(r'[A-Za-z0-9_.-]+',args.run_name) or args.run_name in {'.','..'}:
        parser.error('run-name must be a plain directory name')
    if args.steps<1 or not args.gpus or not set(args.gpus)<=set(range(1,8)) or len(set(args.gpus))!=len(args.gpus):
        parser.error('positive steps and distinct GPUs 1-7 required; GPU0 remains free')
    args.protocols=list(dict.fromkeys(args.protocols))
    output,jobs=build_jobs(args)
    code=engine.execute(args,output,jobs,argv,script_path=Path(__file__),data_preparer=verify_cached_data)
    if not code and not args.dry_run and not args.background:
        audit_final_trajectories(jobs,output,args.gpus)
    return code

if __name__=='__main__':
    raise SystemExit(main())
