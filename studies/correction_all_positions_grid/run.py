"""Fresh DFS depth/rate grid on the all-position v2 pool; shared trainer/scheduler."""
import argparse,copy,importlib.util,json,os,sys
from pathlib import Path
P=Path(__file__).resolve().parent;ROOT=P.parents[1];BASE=P.parent/'correction_reproduction'
sys.path[:0]=[str(ROOT/'src'),str(BASE)]
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','TORCHINDUCTOR_COMPILE_THREADS']:os.environ.setdefault(k,'2')
import yaml
from cot_navigation.config import load_config
from cot_navigation.experiments import scheduler
from cot_navigation.data.shards import file_sha256
from suite import Job
from trace_weight import build_policy
SOURCE=ROOT/'outputs/correction_reproduction_v1/launches/dfs_correction_repro_20261028/dfs_state/correction/s0/resolved_config.yaml'
POOL='data/natural_correction/all_positions_v2_20261102'
BUDGET=ROOT/'outputs/correction_all_positions_v2/token_matched_plan_20261102.json'


def training_schedule(percent,uniform_steps=None):
    if uniform_steps is None:
        return json.loads(BUDGET.read_text())['schedule'][str(percent)]
    if uniform_steps<4 or uniform_steps%4:
        raise ValueError('--uniform-steps must be positive and divisible by four')
    return dict(max_steps=uniform_steps,stage_ends=[uniform_steps*i//4 for i in range(1,5)])


def sources():
    paths=[Path(__file__),SOURCE,BUDGET,BASE/'worker.py',BASE/'trace_weight.py',BASE/'trace_weight_policy.json',P.parent/'accuracy_source/run.py']
    paths+=sorted((ROOT/'src/cot_navigation').rglob('*.py'))
    return {str(p.relative_to(ROOT)):file_sha256(p) for p in paths}


def build(args):
    template=yaml.safe_load(SOURCE.read_text());out=ROOT/'outputs/correction_all_positions_v2/launches'/args.run_name
    if (out/'STOPPED_BY_USER.json').exists():
        raise ValueError('This equal-step cohort was stopped by user; do not resume. Prepare a new token-matched run.')
    generated=ROOT/'configs/generated/correction_all_positions_v2'/args.run_name;generated.mkdir(parents=True,exist_ok=True)
    jobs=[]
    for depth in [9,6]:
        for percent in [60,30,0]:
            for seed in [0,1]:
                cfg=copy.deepcopy(template);name=f'dfs_state/l{depth}_p{percent}/s{seed}'
                cfg['output_dir']=str(out/name);cfg['model']['layers']=depth
                cfg['protocol'].update(mistake_cache_root=POOL,mistake_rate=percent/100,mistake_seed=0)
                schedule=training_schedule(percent,args.uniform_steps)
                cfg['train'].update(model_seed=seed,exposure_seed=0,max_steps=schedule['max_steps'],scale_stage_ends=schedule['stage_ends'])
                notes=('Fresh initialization; all-position v2 error pool. Initialization seeds0/1; exposure/error RNG fixed0. '
                       f'Uniform {args.uniform_steps} updates, four equal curriculum stages; same length loss. p0 uses clean rows, identical vocabulary.'
                       if args.uniform_steps is not None else
                       'Fresh initialization; all-position v2 error pool. Initialization seeds0/1; exposure/error RNG fixed0. Token-matched per curriculum stage to p60 200k; p0 262370, p30 226929 steps. Same length loss. p0 uses clean rows, identical vocabulary.')
                cfg['tracking'].update(mode='disabled',group=args.run_name,name=f'{args.run_name}/{name}',variant='correction',
                    tags=['all_positions_v2',f'layers{depth}',f'error{percent}',f'init{seed}'],
                    notes=notes)
                path=generated/f'l{depth}_p{percent}_s{seed}.yaml';text=yaml.safe_dump(cfg,sort_keys=False)
                if path.exists():assert path.read_text()==text,f'Config changed: {path}'
                else:path.write_text(text)
                load_config([path]);jobs.append(Job(name,path,[],out/name,schedule['max_steps'],arm='correction'))
    return out,jobs


def audit(jobs,uniform_steps=None):
    import torch
    from cot_navigation.data.dataset import ContextStore
    from cot_navigation.data.scale_prepared import ScalePreparedRows
    from cot_navigation.data.natural_mistakes import NaturalCorrectionPool,pool_path
    from cot_navigation.protocols.tokenizer import Tokenizer
    from cot_navigation.models.transformer import build_model
    from worker import weight_hash
    torch.set_num_threads(2)
    cfg=load_config([jobs[0].config]);tok=Tokenizer.from_config(cfg)
    assert json.loads((BASE/'trace_weight_policy.json').read_text())==build_policy()
    original=yaml.safe_load(SOURCE.read_text())
    allowed={'output_dir','model.layers','protocol.mistake_cache_root','protocol.mistake_rate','protocol.mistake_seed',
             'train.model_seed','train.exposure_seed','train.max_steps','train.scale_stage_ends','tracking.mode','tracking.group','tracking.name','tracking.variant','tracking.tags','tracking.notes'}
    def flatten(d,prefix=''):
        out={}
        for k,v in d.items():
            key=f'{prefix}.{k}' if prefix else k
            if isinstance(v,dict):out.update(flatten(v,key))
            else:out[key]=v
        return out
    expected=flatten(original)
    for job in jobs:
        actual=flatten(yaml.safe_load(job.config.read_text()))
        differences={k for k in expected.keys()|actual.keys() if expected.get(k)!=actual.get(k)}
        assert differences<=allowed,differences-allowed
        c=load_config([job.config]);s=training_schedule(round(c.protocol.mistake_rate*100),uniform_steps)
        assert c.train.max_steps==s['max_steps'] and list(c.train.scale_stage_ends)==s['stage_ends'] and job.steps==s['max_steps']
    pools=[]
    for root in cfg.train.scale_roots:
        store=ContextStore(root);prepared=ScalePreparedRows(store,cfg,tok);pool=NaturalCorrectionPool(store,cfg,tok)
        path=pool_path(store,cfg,tok);header=json.loads(path.with_suffix('.json').read_text())
        assert header['identity']['version']==2 and header['provenance']['settings']['states_per_query']==0
        raw=Path(header['statistics']['raw_error_file']);assert file_sha256(raw)==header['statistics']['raw_error_sha256']
        events=0
        for row in pool.train_rows:
            q=int(row.meta['index']);clean=prepared.row(q,int(row.meta['natural_variant']))
            assert int(store.arrays['split_id'][q])==int(row.meta['split'])==0
            assert row.label==clean.label and row.tokens[:row.prompt_len]==clean.tokens[:clean.prompt_len]
            assert len(row.tokens)<=4096
            for e in row.meta['natural_events']:
                assert row.tokens[e['candidate_start']:e['candidate_end']]==e['predicted']
                assert row.tokens[e['repair_start']:e['repair_end']]==clean.tokens[e['original_start']:e['original_end']]
                events+=1
        pools.append(dict(root=root,path=str(path),sha256=file_sha256(path),train_episodes=len(pool.train_rows),train_events=events))
        del pool,store,prepared
    # Verify all final evaluation caches before launching long training.
    evals=[]
    for name,root in [('main',cfg.data.root)]+[v.split('=',1) for v in cfg.evaluation.extra_sets]:
        ref=cfg if name!='hard' else load_config([ROOT/'outputs/hard_dag_v1/launches/hard_dag_v1_s0/context/curriculum/dfs_state/resolved_config.yaml'])
        store=ContextStore(root);prepared=ScalePreparedRows(store,ref,Tokenizer.from_config(ref));evals.append(dict(name=name,root=root));del store,prepared
    initial={}
    for job in jobs:
        c=load_config([job.config]);key=f'l{c.model.layers}_s{c.train.model_seed}'
        torch.manual_seed(c.train.model_seed);model=build_model(c.model,len(tok),pad_id=tok.pad)
        digest=weight_hash(model)
        if key in initial:assert initial[key]['sha256']==digest
        initial[key]=dict(sha256=digest,parameters=sum(p.numel() for p in model.parameters()));del model
    assert len(initial)==4
    return dict(passed=True,cpu_only=True,pools=pools,evals=evals,initial=initial,trace_weight_policy=build_policy(),
                initialization_seeds=[0,1],exposure_seed=0,error_seed=0,training_started=False,
                budget_policy=dict(kind='uniform_steps',steps=uniform_steps,stage_ends=training_schedule(60,uniform_steps)['stage_ends']) if uniform_steps is not None else json.loads(BUDGET.read_text()))


def verify(jobs,out,env):
    p=json.loads((out.parent/f'{out.name}.prepared.json').read_text())
    assert p['source_hashes']==sources(),'Sources changed; re-prepare before launch'
    assert p['jobs']==scheduler.plan_record(jobs) and p['audit']['passed']
    for pool in p['audit']['pools']:assert file_sha256(Path(pool['path']))==pool['sha256']


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-name',default='dfs_all_positions_tokenmatched_20261102')
    p.add_argument('--gpus',type=int,nargs='+',default=list(range(8)))
    p.add_argument('--uniform-steps',type=int,help='Use identical steps and four equal curriculum stages instead of token matching')
    for k in ['launch','background','resume','wait']:p.add_argument('--'+k,action='store_true')
    a=p.parse_args()
    import re
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*',a.run_name):p.error('Invalid run name')
    if a.background and not a.launch:p.error('--background requires --launch')
    if a.uniform_steps is not None and (a.uniform_steps<4 or a.uniform_steps%4):p.error('--uniform-steps must be positive and divisible by four')
    os.chdir(ROOT);out,jobs=build(a)
    if not a.launch:
        receipt=dict(jobs=scheduler.plan_record(jobs),source_hashes=sources(),audit=audit(jobs,a.uniform_steps))
        out.parent.mkdir(parents=True,exist_ok=True);scheduler.write_json(out.parent/f'{out.name}.prepared.json',receipt)
        print('ALL SIX DFS V2 POOLS VERIFIED; 12 JOBS PREPARED',flush=True)
    else:verify(jobs,out,os.environ)
    a.dry_run=not a.launch
    return scheduler.execute(a,out,jobs,sys.argv[1:],script_path=Path(__file__),data_preparer=verify)


if __name__=='__main__':raise SystemExit(main())
