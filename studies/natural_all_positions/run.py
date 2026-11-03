"""Rebuild DFS/BFS observed-error pools at every reference candidate field.

Data generation only. Reuses frozen donors and immutable graph/query splits.
Shares GPU0 under the user's existing authorization; no navigation training.
"""
import argparse,json,os,sys,subprocess
from pathlib import Path
P=Path(__file__).resolve().parent;ROOT=P.parents[1]
sys.path.insert(0,str(ROOT/'src'));os.environ['WANDB_MODE']='disabled'
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','TORCHINDUCTOR_COMPILE_THREADS']:os.environ.setdefault(k,'2')
from cot_navigation.experiments.scheduler import write_json,file_lock
from cot_navigation.data.shards import file_sha256
OUT=ROOT/'outputs/data_preparation/natural_all_positions_v2_20261102'
NEW=ROOT/'data/natural_correction/all_positions_v2_20261102'
OLD=ROOT/'data/natural_correction/natural_correction_s0'


def prepare():
    import yaml
    from cot_navigation.config import load_config
    jobs=[]
    generated=ROOT/'configs/generated/natural_all_positions_v2_20261102';generated.mkdir(parents=True,exist_ok=True)
    for protocol in ['dfs_state','bfs_state']:
        source=ROOT/f'outputs/natural_correction_v1/launches/natural_correction_s0/context/{protocol}/p30_l16/resolved_config.yaml'
        cfg=yaml.safe_load(source.read_text())
        cfg['protocol'].update(mistake_cache_root=str(NEW),mistake_seed=0,mistake_source='natural')
        cfg['tracking']['mode']='disabled'
        config=generated/f'{protocol}.yaml';text=yaml.safe_dump(cfg,sort_keys=False)
        if config.exists():assert config.read_text()==text
        else:config.write_text(text)
        loaded=load_config([config])
        for root in loaded.train.scale_roots:
            jobs.append(dict(config=str(config),root=root,train_queries=8192,heldout_queries=512,
                             states_per_query=0,batch_size=32))
    receipt=dict(schema=2,new_pool=str(NEW),old_pool=str(OLD),jobs=jobs,
        selection='all candidate fields in every selected prepared reference trajectory; no 4-position cap',
        semantics='independent completions from correct prefixes, not free-running erroneous trajectories',
        raw_retention='all wrong closed fields plus malformed/unclosed observations; packing exclusions counted',
        donor_policy='unchanged historical clean best/latest; one donor per query, one completion per position',
        source_sha256={str(p.relative_to(ROOT)):file_sha256(p) for p in [Path(__file__),ROOT/'src/cot_navigation/data/mine_natural_candidates.py',ROOT/'src/cot_navigation/data/natural_mistakes.py']})
    OUT.mkdir(parents=True,exist_ok=True);write_json(OUT/'plan.json',receipt)
    write_json(OLD/'DEPRECATED.json',dict(retired=True,replacement=str(NEW),reason='4-position capped error mining retired by user; historical files retained, new loading prohibited'))
    NEW.mkdir(parents=True,exist_ok=True);write_json(NEW/'POLICY.json',receipt)
    return jobs


def execute(jobs):
    import torch
    from cot_navigation.config import load_config
    from cot_navigation.data.mine_natural_candidates import mine_one
    from cot_navigation.data.dataset import ContextStore
    from cot_navigation.data.natural_mistakes import NaturalCorrectionPool
    from cot_navigation.protocols.tokenizer import Tokenizer
    torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.2)
    results=[]
    with file_lock(OUT/'.lock',blocking=False):
        for i,job in enumerate(jobs):
            write_json(OUT/'status.json',dict(complete=False,phase='mining',ready=i,total=len(jobs),current=job['root'],protocol=load_config([job['config']]).protocol.name))
            item=mine_one(**job)
            cfg=load_config([job['config']]);tok=Tokenizer.from_config(cfg);store=ContextStore(job['root']);pool=NaturalCorrectionPool(store,cfg,tok)
            assert item['provenance']['settings']['states_per_query']==0
            assert item['statistics']['raw_closed_error_events']>=item['statistics']['packed_events']
            raw=Path(item['statistics']['raw_error_file']);assert file_sha256(raw)==item['statistics']['raw_error_sha256']
            for row in pool.train_rows:
                assert int(row.meta['split'])==0 and len(row.tokens)<=cfg.model.max_seq_len
            results.append(item);write_json(OUT/'results.json',results)
            print('POOL READY',i+1,len(jobs),item['statistics']['episodes'],flush=True)
            del pool,store;torch.cuda.empty_cache()
        write_json(OUT/'status.json',dict(complete=True,phase='all_pools_ready_no_training_launched',ready=len(jobs),total=len(jobs)))


def main():
    p=argparse.ArgumentParser();p.add_argument('--background',action='store_true');p.add_argument('--execute',action='store_true');a=p.parse_args();os.chdir(ROOT)
    jobs=prepare()
    if a.background:
        with (OUT/'run.log').open('a') as log:
            child=subprocess.Popen([sys.executable,str(__file__),'--execute'],cwd=ROOT,env={**os.environ,'CUDA_VISIBLE_DEVICES':'0','PYTHONUNBUFFERED':'1'},stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        write_json(OUT/'process.json',dict(pid=child.pid));print('MINING PID',child.pid);return
    if a.execute:
        try:execute(jobs)
        except Exception as exc:write_json(OUT/'status.json',dict(complete=False,phase='failed',error=str(exc)));raise
    else:print('PREPARED',len(jobs),'data jobs; use --background to mine; no training')


if __name__=='__main__':main()
