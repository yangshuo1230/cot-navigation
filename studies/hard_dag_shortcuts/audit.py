"""Frozen fixed-DAG Direct/Path shortcut audit. No optimizer or W&B runs.

Evaluate the entire held-out test pool in the same order/batches for both
protocols. WALK exposure means available in the knowledge corpus; it does
not claim a particular training update consumed that walk. All-path exposure
means a witness is available inside a supervised training-query PATH.
"""
from pathlib import Path
from collections import Counter, defaultdict
import gzip
import hashlib
import json
import os
import sys
import time

os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('MKL_NUM_THREADS','2')
import numpy as np
import torch

ROOT=Path.cwd()
sys.path.insert(0,str(ROOT/'src'))
from cot_navigation.config import load_config
from cot_navigation.data.dataset import ParametricWorld
from cot_navigation.data.hard_graph import shortest_distances
from cot_navigation.data.reachability import build_reach,bfs_reaches
from cot_navigation.models.transformer import build_model
from cot_navigation.protocols.tokenizer import Tokenizer
from cot_navigation.evaluation.rollout import generate

OUT=ROOT/'studies/hard_dag_shortcuts'
RUN=ROOT/'outputs/hard_dag_v1/launches/hard_dag_v1_s0'


def write(path,obj):
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,indent=2,ensure_ascii=False))
    tmp.replace(path)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rate(rows,key='correct'):
    return dict(n=len(rows),count=sum(bool(x[key]) for x in rows),
                rate=sum(bool(x[key]) for x in rows)/len(rows) if rows else None)


def metadata(world):
    a=world.adj();reach=build_reach(a);q=world.queries
    paths={}
    for i in range(world.n_walks()):
        p=world.walk_nodes_at(i)
        for j in range(len(p)):
            for k in range(j+1,len(p)):
                paths.setdefault((p[j],p[k]),p[j:k+1])
    train=[(int(q['s_topo'][i]),int(q['t_topo'][i])) for i in world.indices('train') if q['label'][i]]
    result={}
    for i in world.indices('test').tolist():
        s,t=int(q['s_topo'][i]),int(q['t_topo'][i]);hops=shortest_distances(a,s)[t]
        assert bfs_reaches(a,s,t)==bool(q['label'][i])==(hops>0)
        witness=next(((u,v) for u,v in train if (u==s or (reach[u]>>s)&1) and
                      (v==t or (reach[t]>>v)&1)),None) if hops>0 else None
        result[i]=dict(query_id=i,source=s,target=t,label=int(q['label'][i]),
                       rank_gap=int(q['distance'][i]),shortest_hops=hops,
                       walk_witness=(s,t) in paths,walk_example=paths.get((s,t)),
                       training_path_witness_available=witness is not None,
                       enclosing_train_query=witness)
    return result


def path_audit(tokens,world,tok,s,t,label):
    inverse={node:u for u,node in enumerate(world.node_vocab_ids(tok))}
    body=list(map(int,tokens));stop=tok.id('<STOP>')
    interrupted=stop in body
    # A STOP is interruption, not a node and not a successful full witness.
    nonnodes=[(i,x) for i,x in enumerate(body) if x not in inverse]
    nodes=[inverse[x] for x in body if x in inverse]
    adj=world.adj()
    bad_edges=[(i,u,v) for i,(u,v) in enumerate(zip(nodes,nodes[1:])) if v not in adj[u]]
    wrong_start=not nodes or nodes[0]!=s
    repeats=len(nodes)!=len(set(nodes))
    basic=bool(nodes) and not nonnodes and not bad_edges and not wrong_start and not repeats and not interrupted
    reaches_goal=bool(nodes) and nodes[-1]==t
    valid_positive_witness=basic and reaches_goal
    # A negative random path proves only local path legality, not unreachability.
    return dict(nodes=nodes,nonnode_tokens=nonnodes,non_edges=bad_edges,
        wrong_start=wrong_start,repeated_nodes=repeats,interrupted=interrupted,
        locally_legal_path=basic,ends_at_target=reaches_goal,
        valid_positive_witness=valid_positive_witness,
        negative_sink_end=basic and not adj[nodes[-1]] if nodes else False)


def summarize(rows):
    groups={'all':rows,'positive':[x for x in rows if x['label']],
            'negative':[x for x in rows if not x['label']]}
    pos=groups['positive']
    groups.update(walk_seen=[x for x in pos if x['walk_witness']],
        walk_unseen=[x for x in pos if not x['walk_witness']],
        true_hops_ge5=[x for x in pos if x['shortest_hops']>=5],
        no_walk_no_training_path=[x for x in pos if not x['walk_witness'] and not x['training_path_witness_available']])
    for h in sorted({x['shortest_hops'] for x in pos}):
        groups[f'hops_{h}']=[x for x in pos if x['shortest_hops']==h]
    result={name:rate(g) for name,g in groups.items()}
    if rows and 'path' in rows[0]:
        for name,g in groups.items():
            result[name]['local_path_legality']=rate([dict(correct=x['path']['locally_legal_path']) for x in g])
            if name!='negative':
                gp=[x for x in g if x['label']]
                result[name]['positive_witness_validity']=rate([dict(correct=x['path']['valid_positive_witness']) for x in gp])
        result['correct_yes_without_valid_witness']=rate([dict(correct=not x['path']['valid_positive_witness']) for x in pos if x['correct']])
        result['path_error_counts']={k:sum(bool(x['path'][k]) for x in rows) for k in
            ('nonnode_tokens','non_edges','wrong_start','repeated_nodes','interrupted')}
    return result


def feature_baselines(world,meta):
    a=world.adj();indeg=Counter(v for nbrs in a for v in nbrs);q=world.queries
    train=world.indices('train').tolist();ids=sorted(meta)
    def key(i,kind):
        s,t=int(q['s_topo'][i]),int(q['t_topo'][i])
        return s if kind=='source_id' else t if kind=='target_id' else (
            len(a[s]),indeg[s],len(a[t]),indeg[t]) if kind=='endpoint_degrees' else 0
    all_results={}
    prior=float(q['label'][train].mean())
    for kind in ('majority','source_id','target_id','endpoint_degrees'):
        counts=defaultdict(lambda:[0,0])
        for i in train:counts[key(i,kind)][int(q['label'][i])]+=1
        rows=[]
        for i in ids:
            no,yes=counts[key(i,kind)];p=(yes+prior*2)/(no+yes+2)
            rows.append(dict(label=meta[i]['label'],correct=int(p>=.5)==meta[i]['label']))
        yp=rate([x for x in rows if x['label']])['rate'];np_=rate([x for x in rows if not x['label']])['rate']
        all_results[kind]=dict(**rate(rows),positive_accuracy=yp,negative_accuracy=np_,balanced_accuracy=(yp+np_)/2)
    return all_results


@torch.inference_mode()
def run_one(n,protocol):
    folder=RUN/f'fixed/n{n}/stage_b/{protocol}'
    cfg=load_config([folder/'resolved_config.yaml']);tok=Tokenizer.from_config(cfg)
    checkpoint=folder/'checkpoints/best.pt';blob=torch.load(checkpoint,map_location='cpu',weights_only=False)
    model=build_model(cfg.model,len(tok),tok.pad).cuda().eval().requires_grad_(False)
    model.load_state_dict(blob['model'],strict=True)
    world=ParametricWorld(Path(cfg.data.root)/'main-seed0');meta=metadata(world)
    ids=sorted(meta);rows=[];batch=16
    for offset in range(0,len(ids),batch):
        selected=ids[offset:offset+batch]
        prompts=[world.render_query(i,protocol,tok).tokens[:4] for i in selected]
        outputs=generate(model,prompts,protocol,tok,cfg.train.precision,
                         max_new_tokens=0,cot_margin=cfg.protocol.cot_margin)
        for i,g in zip(selected,outputs):
            r=dict(meta[i],pred=g['pred'],correct=g['pred']==meta[i]['label'] and not g['format_fail'],
                   format_fail=g['format_fail'],generation_limit_reached=g['generation_limit_reached'])
            if protocol=='path':r['path']=path_audit(g['trace_tokens'],world,tok,r['source'],r['target'],r['label'])
            rows.append(r)
        print(n,protocol,len(rows),'/',len(ids),flush=True)
    prefix=f'n{n}_{protocol}'
    with gzip.open(OUT/f'{prefix}_queries.jsonl.gz','wt') as f:
        for r in rows:f.write(json.dumps(r)+'\n')
    old={int(x['query_id']):x for x in map(json.loads,(folder/'eval/predictions_test_best.jsonl').read_text().splitlines())}
    paired=[x for x in rows if x['query_id'] in old]
    report=dict(n=n,protocol=protocol,checkpoint=str(checkpoint),checkpoint_step=int(blob['step']),
        checkpoint_sha256=sha(checkpoint),batch_size=batch,summary=summarize(rows),
        previous_panel=dict(n=len(paired),prediction_changes=sum(x['pred']!=old[x['query_id']]['pred'] for x in paired)),
        baselines=feature_baselines(world,meta),
        interpretation='WALK availability is not proof of exact optimizer exposure; negative path legality is not an unreachability certificate.')
    write(OUT/f'{prefix}.json',report)
    examples={}
    if protocol=='path':
        tests={'correct_answer_invalid_witness':lambda x:x['label']==1 and x['correct'] and not x['path']['valid_positive_witness'],
               'long_unseen_correct_valid':lambda x:x['shortest_hops']>=5 and not x['walk_witness'] and x['correct'] and x['path']['valid_positive_witness'],
               'non_edge':lambda x:bool(x['path']['non_edges'])}
        for name,test in tests.items():
            examples[name]=[x for x in rows if test(x)][:5]
    write(OUT/f'{prefix}_examples.json',examples)
    del model,blob;torch.cuda.empty_cache()


def self_test():
    from types import SimpleNamespace
    tok=Tokenizer(4,enable_stop=True)
    world=SimpleNamespace(node_vocab_ids=lambda _:list(range(tok.node_lo,tok.node_lo+4)),adj=lambda:[[1],[2],[],[]])
    v=world.node_vocab_ids(tok)
    assert path_audit(v[:3],world,tok,0,2,1)['valid_positive_witness']
    assert path_audit([v[0],v[2]],world,tok,0,2,1)['non_edges']
    assert not path_audit([v[0],v[2]],world,tok,0,2,1)['valid_positive_witness']
    assert not path_audit(v[:2]+[tok.id('<STOP>')],world,tok,0,2,1)['locally_legal_path']
    assert path_audit(v[:3],world,tok,0,3,0)['negative_sink_end']


if __name__=='__main__':
    OUT.mkdir(exist_ok=True)
    self_test();write(OUT/'status.json',dict(phase='running',pid=os.getpid(),started=time.time()))
    for n in (36,100):
        for protocol in ('direct','path'):run_one(n,protocol)
    write(OUT/'status.json',dict(phase='complete',pid=os.getpid(),finished=time.time()))
