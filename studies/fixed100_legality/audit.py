"""Full N100 Fixed test-pool legality; frozen best checkpoints.

Semantic-first full pools and action-only diagnostic panels have distinct
denominators. Early-stopped rows have unknown native final answers, not NO.
No reference trajectory equality is required.
"""
from pathlib import Path
from collections import Counter
from types import SimpleNamespace
from io import BytesIO
import argparse
import gzip
import hashlib
import importlib.util
import json
import os
import sys
import time

os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('MKL_NUM_THREADS','2')
import numpy as np
import torch

ROOT=Path.cwd();OUT=ROOT/'studies/fixed100_legality'
sys.path[:0]=[str(ROOT/'src'),str(OUT)]
from replay import StateReplay,ActionReplay,SequenceReplay
from stream import generate_checked
from cache_prefix import enable
from cot_navigation.config import load_config
from cot_navigation.models.transformer import build_model
from cot_navigation.protocols.tokenizer import Tokenizer
from cot_navigation.protocols.render import parametric_prompt
from cot_navigation.data.dataset import ParametricWorld
from cot_navigation.data.hard_graph import shortest_distances

spec=importlib.util.spec_from_file_location('previous_audit',ROOT/'studies/curriculum_trace_errors/audit.py')
previous=importlib.util.module_from_spec(spec);spec.loader.exec_module(previous)

RUNS={'simple':ROOT/'outputs/fixed_short3_v1/launches/short3_n100_s0/all_hops',
      'hard':ROOT/'outputs/hard_dag_v1/launches/hard_dag_v1_s0/fixed/n100/stage_b'}
PROTOCOLS=('path','dfs_nodes','bfs_nodes','dfs_state','bfs_state')


def write(path,value):
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(value,ensure_ascii=False,indent=2));temp.replace(path)


def counts(rows):
    n=len(rows)
    return dict(n=n,semantic_valid=sum(r['audit']['semantic_valid'] for r in rows),
        protocol_valid=sum(r['audit']['protocol_valid'] for r in rows),
        semantic_valid_rate=sum(r['audit']['semantic_valid'] for r in rows)/n if n else None,
        protocol_valid_rate=sum(r['audit']['protocol_valid'] for r in rows)/n if n else None,
        first_errors=dict(Counter(r['audit']['first_error']['kind'] for r in rows if r['audit']['first_error'])),
        interrupted=sum(r['interrupted'] for r in rows),early_censored=sum(r['early_censored'] for r in rows),
        completed_native_answers=sum(r['pred'] is not None for r in rows))


def panel_ids(ids,q):
    rng=np.random.default_rng(20261022);out=[]
    for gap,label in sorted({(int(q['distance'][i]),int(q['label'][i])) for i in ids}):
        pool=[i for i in ids if int(q['distance'][i])==gap and int(q['label'][i])==label]
        out.extend(rng.choice(pool,min(4,len(pool)),replace=False).tolist())
    return sorted(out)


def run(setting,protocol,batch=32):
    name=f'{setting}_{protocol}';folder=RUNS[setting]/protocol
    cfg=load_config([folder/'resolved_config.yaml']);tok=Tokenizer.from_config(cfg)
    world=ParametricWorld(Path(cfg.data.root)/f'{cfg.data.graph_shape}-seed{cfg.data.graph_seed}')
    q=world.queries;rawadj=world.adj();nodes=world.node_vocab_ids(tok)
    adj={nodes[u]:set(nodes[v] for v in nbrs) for u,nbrs in enumerate(rawadj)}
    width=next(iter(cfg.data.shapes.values())).width
    ids=[int(i) for i in world.indices('test') if int(q['t_topo'][i])//width>int(q['s_topo'][i])//width]
    checkpoint=folder/'checkpoints/best.pt';data=checkpoint.read_bytes();checksum=hashlib.sha256(data).hexdigest()
    blob=torch.load(BytesIO(data),map_location='cpu',weights_only=False);step=int(blob['step'])
    model=build_model(cfg.model,len(tok),tok.pad).cuda().eval().requires_grad_(False)
    model.load_state_dict(blob['model'],strict=True);del blob,data
    enable(model)
    header=[]
    if cfg.train.world_roots:
        worlds=[Path(x).resolve() for x in cfg.train.world_roots]
        header=[tok.id(f'<WORLD_{worlds.index(world.root.resolve())}>')]
    phases=[('full_test_semantic',ids)]
    if protocol.endswith('_state'):phases.append(('action_panel',panel_ids(ids,q)))
    summaries={};examples=[]
    for mode,selected in phases:
        rows=[]
        for start in range(0,len(selected),batch):
            chosen=selected[start:start+batch];prompts=[];replays=[]
            for i in chosen:
                s,t=int(q['s_topo'][i]),int(q['t_topo'][i])
                prompts.append(header+parametric_prompt(tok,nodes,s,t))
                if protocol.endswith('_state'):
                    cls=ActionReplay if mode=='action_panel' else StateReplay
                    replays.append(cls(adj,nodes[s],nodes[t],tok,protocol=='bfs_state'))
                else:
                    replays.append(SequenceReplay(adj,nodes[s],nodes[t],int(q['label'][i]),protocol,tok,previous.sequence_audit))
            generated=generate_checked(model,prompts,tok,cfg.train.precision,replays,cfg.protocol.cot_margin)
            for i,out in zip(chosen,generated):
                s,t=int(q['s_topo'][i]),int(q['t_topo'][i]);a=out['audit']
                row=dict(query_id=i,label=int(q['label'][i]),source=s,target=t,
                    rank_gap=int(q['distance'][i]),shortest_hops=shortest_distances(rawadj,s)[t],
                    pred=out['pred'],early_censored=out['early_censored'],interrupted=out['interrupted'],
                    generated_tokens=len(out['trace']),audit={k:v for k,v in a.items() if k not in ('last_record','previous_record')})
                if row['audit']['protocol_valid']:assert row['pred']==row['label']
                rows.append(row)
                if a['first_error'] and sum(e['mode']==mode for e in examples)<8:
                    examples.append(dict(mode=mode,query_id=i,source=tok.name(nodes[s]),target=tok.name(nodes[t]),
                        label=row['label'],pred=row['pred'],first_error=a['first_error'],
                        previous_record=tok.decode(a['previous_record']),bad_record=tok.decode(a['last_record'])))
            print(name,mode,min(start+batch,len(selected)),'/',len(selected),flush=True)
        with gzip.open(OUT/f'{name}_{mode}.jsonl.gz','wt') as f:
            for row in rows:f.write(json.dumps(row)+'\n')
        summary=dict(all=counts(rows),positive=counts([r for r in rows if r['label']]),negative=counts([r for r in rows if not r['label']]))
        if mode=='action_panel':
            summary['full_state_on_same_panel']=dict(n=len(rows),semantic_valid=sum(r['audit']['full_state']['semantic_valid'] for r in rows),
                protocol_valid=sum(r['audit']['full_state']['protocol_valid'] for r in rows))
        summaries[mode]=summary
    native=None
    report=folder/'eval/free_test_best.json'
    if report.exists():
        report=json.loads(report.read_text())
        if report.get('checkpoint_step')==step:native=report.get('overall')
    write(OUT/f'{name}.json',dict(setting=setting,protocol=protocol,run=str(folder),checkpoint_step=step,
        checkpoint_sha256=checksum,batch_size=batch,summaries=summaries,examples=examples,
        previously_saved_native_test=native,
        note='Full-test first-invalid stopping gives exact whole-trace failure for that rollout; no final answer is imputed. Action-panel denominators differ.'))
    del model;torch.cuda.empty_cache()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('setting');parser.add_argument('protocol');args=parser.parse_args()
    torch.set_num_threads(2);OUT.mkdir(exist_ok=True)
    run(args.setting,args.protocol)
