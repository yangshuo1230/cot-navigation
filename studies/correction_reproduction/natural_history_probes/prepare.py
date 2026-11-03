"""Unedited, naturally generated CoT prefixes on graph-disjoint official splits.

Two frozen seed-0 donors supply probe train/validation examples; all eight
matched/correction checkpoints supply test examples. Stop only after two
records following the first error, without changing any generated token.
"""
import argparse
import gzip
import json
import os
from pathlib import Path
import runpy
import time

os.environ.setdefault('CUDA_VISIBLE_DEVICES','0')
os.environ['WANDB_MODE']='disabled'
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','TORCHINDUCTOR_COMPILE_THREADS']:
    os.environ.setdefault(key,'2')
P=Path(__file__).resolve().parent;ROOT=P.parents[2]

import torch
from cot_navigation.evaluation.rollout import generate
from cot_navigation.models.compiled_inference import CompiledInference
from cot_navigation.protocols.correction_replay import replay
from cot_navigation.experiments.artifacts import atomic_json,sha_file

OLD=runpy.run_path(str(ROOT/'studies/correction_reproduction/history_probes/run.py'))
Model=OLD['Model']


def read_cache(path):
    with gzip.open(path,'rt',encoding='utf-8') as f:return json.load(f)


def write_cache(path,data):
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp')
    with gzip.open(tmp,'wt',encoding='utf-8',compresslevel=3) as f:
        json.dump(data,f,separators=(',',':'),ensure_ascii=False)
    tmp.replace(path)


def graph_tokens(am,query):
    nodes=am.store.node_vocab_ids(query,am.tok);adj=am.store.adj(query)
    return ({nodes[u]:[nodes[v] for v in vs] for u,vs in enumerate(adj)},
            nodes[int(am.store.arrays['s_topo'][query])],
            nodes[int(am.store.arrays['t_topo'][query])])


def prepare(am,arm,seed,split,count,batch,model):
    path=P/'native'/f'{arm}_s{seed}_{split}.json.gz'
    queries=am.sample_ids(split,count)
    identity=dict(checkpoint=am.checkpoint,source_fingerprint=am.store.fingerprint,
                  queries=queries,split=split,protocol='dfs_state',precision='bf16',batch=batch,
                  stop='first illegal token record + two subsequent complete records',
                  schema='natural_unedited_v1')
    data=read_cache(path) if path.exists() else dict(identity=identity,arm=arm,model_seed=seed,complete=False,rows=[])
    assert data['identity']==identity, f'Cache identity mismatch: {path}'
    if data['complete']:return
    done=[r['query'] for r in data['rows']];assert done==queries[:len(done)]
    started=time.time();initial=len(done)
    for start in range(initial,len(queries),batch):
        qs=queries[start:start+batch]
        prompts=[am.example(q,split).prompt for q in qs]
        graphs=[graph_tokens(am,q) for q in qs];record_counts=[0]*len(qs);first_records=[None]*len(qs)
        def record_stop(i,trace):
            record_counts[i]+=1
            if first_records[i] is None:
                adj,src,dst=graphs[i]
                first=replay(trace,None,adj,src,dst,am.tok)['first']
                if first is not None:first_records[i]=first['record']
            return first_records[i] is not None and record_counts[i]>=first_records[i]+2
        outputs=generate(model,prompts,'dfs_state',am.tok,'bf16',cot_margin=am.cfg.protocol.cot_margin,
                         record_stop=record_stop)
        for q,g,graph in zip(qs,outputs,graphs):
            trace=g['trace_tokens'];stop=am.tok.stoi.get('<STOP>')
            if stop in trace:trace=trace[:trace.index(stop)]
            adj,src,dst=graph;audit=replay(trace,None,adj,src,dst,am.tok)
            marker=am.tok.stoi.get('<MISTAKE>',-1)
            data['rows'].append(dict(query=q,trace=trace,first=audit['first'],
                graph_hash=OLD['graph_id'](am,q),marker_at=trace.index(marker) if marker in trace else None,
                analysis_stopped=g['analysis_stopped'],generation_limit=g['generation_limit_reached'],
                forced_stop=g['forced_stop'],source=f'{arm}_s{seed}',split=split))
        data['elapsed_current_process']=time.time()-started
        write_cache(path,data)
        print('NATURAL',arm,seed,split,len(data['rows']),'/',len(queries),
              'errors',sum(r['first'] is not None for r in data['rows']),
              'seconds',round(time.time()-started),flush=True)
    data.update(complete=True,compile_fallbacks=sorted(model.compile_fallbacks))
    write_cache(path,data)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train-queries',type=int,default=1536)
    parser.add_argument('--val-queries',type=int,default=256)
    parser.add_argument('--test-queries',type=int,default=256)
    parser.add_argument('--batch',type=int,default=32)
    args=parser.parse_args();os.chdir(ROOT);torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(.32)
    P.mkdir(exist_ok=True,parents=True)
    atomic_json(P/'preparation_plan.json',dict(vars(args),gpu=0,
        population='positive N100 official context splits; truly unedited model generations',
        training_donors=['matched_clean_s0','correction_s0'],test_donors='all 8 checkpoints',
        source_sha256={str(p.relative_to(ROOT)):sha_file(p) for p in [Path(__file__),ROOT/'src/cot_navigation/evaluation/rollout.py']}))
    for seed in range(4):
        for arm in ['matched_clean','correction']:
            am=Model(arm,seed);model=CompiledInference(am.model).eval()
            splits=[('train',args.train_queries),('val',args.val_queries),('test',args.test_queries)] if seed==0 else [('test',args.test_queries)]
            for split,n in splits:prepare(am,arm,seed,split,n,args.batch,model)
            assert sha_file(Path(am.checkpoint['path']))==am.checkpoint['sha256']
            del model,am;torch.cuda.empty_cache()
    files=list((P/'native').glob('*.json.gz'))
    assert len(files)==12 and all(read_cache(p)['complete'] for p in files)
    atomic_json(P/'prepared.json',dict(complete=True,files=[str(p.relative_to(P)) for p in files]))


if __name__=='__main__':main()
