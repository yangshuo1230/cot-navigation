#!/usr/bin/env python3
"""Mine observed errors on eight GPUs, audit, then launch eight fresh trials."""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import multiprocessing as mp
import os
from pathlib import Path
import re
import signal
import sys

for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','TORCHINDUCTOR_COMPILE_THREADS'):
    os.environ.setdefault(key,'2')
import yaml

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from cot_navigation.experiments import scheduler as engine
from cot_navigation.config import load_config
from cot_navigation.data.shards import atomic_json

OLD=ROOT/'outputs/scale_v1/launches/scale_v1_s0/context/curriculum'


def worker_init(queue):
    os.environ['CUDA_VISIBLE_DEVICES']=str(queue.get())


def mine_task(task):
    from cot_navigation.data.mine_natural_candidates import mine_one
    return mine_one(**task)


def prepare_data(jobs,output,env,args):
    tasks={}
    for job in jobs:
        cfg=load_config([job.config])
        for root in cfg.train.scale_roots:
            tasks.setdefault((cfg.protocol.name,root),dict(config=str(job.config),root=root,
                train_queries=args.train_queries,heldout_queries=512,states_per_query=4,batch_size=32))
    ready=[];ctx=mp.get_context('spawn');queue=ctx.Queue()
    for gpu in args.gpus:queue.put(gpu)
    atomic_json(output/'data_preparation.json',dict(phase='mining_observed_errors',total=len(tasks),ready=0))
    with ProcessPoolExecutor(max_workers=len(args.gpus),mp_context=ctx,
                             initializer=worker_init,initargs=(queue,)) as pool:
        futures=[pool.submit(mine_task,t) for t in tasks.values()]
        for future in as_completed(futures):
            item=future.result();ready.append(item)
            atomic_json(output/'data_preparation.json',dict(phase='mining_observed_errors',
                total=len(tasks),ready=len(ready),datasets=ready))
            print(f"OBSERVED DATA READY {item['source']} {item['identity']['canonical']['protocol']} ({len(ready)}/{len(tasks)}): {item['statistics']['episodes']}",flush=True)
    # All observed corpora must be valid before any optimizer is started.
    from cot_navigation.data.dataset import ContextStore
    from cot_navigation.data.natural_mistakes import NaturalCorrectionPool
    from cot_navigation.protocols.tokenizer import Tokenizer
    checked=set()
    for job in jobs:
        cfg=load_config([job.config]);tok=Tokenizer.from_config(cfg)
        for root in cfg.train.scale_roots:
            if (cfg.protocol.name,root) in checked:continue
            source=ContextStore(root);corpus=NaturalCorrectionPool(source,cfg,tok)
            corpus.cases('val',64,cfg.evaluation.seed);corpus.cases('test',64,cfg.evaluation.seed)
            checked.add((cfg.protocol.name,root))
    atomic_json(output/'data_preparation.json',dict(phase='audited_ready_for_training',
        total=len(tasks),ready=len(ready),datasets=ready))
    print('All observed corpora audited; GPU miners exited; launching eight training jobs.',flush=True)


def build_jobs(args):
    output=ROOT/args.output_root/args.run_name
    generated=ROOT/'configs/generated/natural_correction'/args.run_name
    generated.mkdir(parents=True,exist_ok=True)
    jobs=[];manifest=[]
    for protocol in ('dfs_state','bfs_state'):
        original=OLD/protocol/'resolved_config.yaml'
        for percent in (30,50):
            for weight in (1,16):
                name=f'context/{protocol}/p{percent}_l{weight}'
                cfg=yaml.safe_load(original.read_text())
                cfg['output_dir']=str(output/name)
                cfg['protocol'].update(enable_mistake=True,mistake_source='natural',
                    mistake_rate=percent/100,mistake_mask_errors=False,mistake_seed=0,
                    natural_complex_share=0.5,mistake_cache_root=f'data/natural_correction/{args.run_name}')
                cfg['train']['trace_loss_lambda']=float(weight)
                cfg['train']['free_gen_samples']=64
                cfg['evaluation']['mistake_samples']=64
                cfg['tracking'].update(mode=args.tracking,group=args.run_name,
                    name=f'{args.run_name}/{name}',variant=f'p{percent}_l{weight}',
                    tags=['natural_correction','simple_layered',protocol,f'p{percent}',f'lambda{weight}'],
                    notes='Observed greedy candidate fields from frozen clean donors on disjoint train/val/test graphs. Full loss, up to four observed errors per episode, original 4K curriculum.')
                path=generated/(name.replace('/','__')+'.yaml')
                text=yaml.safe_dump(cfg,sort_keys=False,allow_unicode=True)
                if path.exists() and path.read_text()!=text:raise ValueError(f'immutable config changed: {path}')
                path.write_text(text);load_config([path])
                jobs.append(engine.Job(name,path,[],output/name,cfg['train']['max_steps']))
                manifest.append(dict(job=name,original=str(original),rate=percent/100,
                    trace_loss_lambda=weight,full_loss=True,config=str(path)))
    atomic_json(generated/'manifest.json',dict(jobs=manifest,from_scratch=True,
        source='actual completed greedy candidate fields; no hand-coded node edits',
        train_queries_per_scale=args.train_queries,states_per_query=4,
        donor_models='historical clean DFS/BFS best and last, not prior synthetic-error models',
        data_isolation='train episodes only in optimizer; separate val/test error pools',
        validation_for_checkpoint_selection='clean free generation, 64 queries',
        loss='2*(answer+lambda*mean_trace)/(1+lambda)'))
    return output,jobs


def main(argv=None):
    argv=list(sys.argv[1:] if argv is None else argv)
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-name',default='natural_correction_s0')
    p.add_argument('--output-root',default='outputs/natural_correction_v1/launches')
    p.add_argument('--gpus',nargs='+',type=int,default=list(range(8)))
    p.add_argument('--train-queries',type=int,default=8192)
    p.add_argument('--tracking',choices=['online','offline','disabled'],default='online')
    for flag in ('background','resume','dry-run','wait'):p.add_argument('--'+flag,action='store_true')
    args=p.parse_args(argv)
    if not re.fullmatch(r'[A-Za-z0-9_-][A-Za-z0-9_.-]*',args.run_name):p.error('invalid run name')
    if not args.gpus or min(args.gpus)<0 or len(set(args.gpus))!=len(args.gpus):p.error('invalid GPU list')
    if args.train_queries<32:p.error('train-queries must be at least 32')
    os.chdir(ROOT);output,jobs=build_jobs(args)
    return engine.execute(args,output,jobs,argv,script_path=Path(__file__),
        data_preparer=lambda j,o,e:prepare_data(j,o,e,args))


if __name__=='__main__':
    signal.signal(signal.SIGTERM,lambda *_:sys.exit(130))
    raise SystemExit(main())
