"""Classify actual saved PATH failures; read-only CPU, no new generations."""
from pathlib import Path
import json,gzip,sys
import numpy as np
ROOT=Path.cwd();OUT=ROOT/'studies/path_failures'
sys.path.insert(0,str(ROOT/'src'))
from cot_navigation.config import load_config
from cot_navigation.protocols.tokenizer import Tokenizer
from cot_navigation.data.dataset import ParametricWorld

def run(setting):
    old=ROOT/'studies/fixed100_layer_probes'
    manifest=json.loads((old/f'{setting}_path.json').read_text())['manifest']
    cfg=load_config([Path(manifest['run'])/'resolved_config.yaml']);tok=Tokenizer.from_config(cfg)
    world=ParametricWorld(Path(cfg.data.root)/'main-seed0');adj=world.adj();nodes=world.node_vocab_ids(tok);inv={v:i for i,v in enumerate(nodes)}
    n=len(nodes);reach=np.eye(n,dtype=bool)
    for u in reversed(range(n)):
        for v in adj[u]:reach[u]|=reach[v]
    rows=[json.loads(v) for v in gzip.open(old/f'{setting}_path_queries.jsonl.gz','rt')]
    rows=[r for r in rows if r['split']==2];failures=[];counts={}
    for row in rows:
        s,t=row['source'],row['target'];path=[inv.get(v,-1) for v in row['trace']]
        legal=bool(path and path[0]==s and all(v>=0 for v in path));nonedges=[];first_bad_branch=None
        for step,(u,v) in enumerate(zip(path,path[1:]),1):
            if u<0 or v<0 or v not in adj[u]:legal=False;nonedges.append(step)
            if row['label'] and first_bad_branch is None and t not in path[:step] and u>=0 and v>=0 and v in adj[u] and reach[u,t] and not reach[v,t]:
                first_bad_branch=dict(step=step,current=u,chosen=v,
                    useful=[w for w in adj[u] if reach[w,t]],dead=[w for w in adj[u] if not reach[w,t]])
        reached=t in path;goal_pos=path.index(t) if reached else None
        overshot=reached and goal_pos<len(path)-1
        witness=bool(legal and path[-1]==t)
        answer_ok=row['pred']==row['label'];negative_terminal=bool(legal and not adj[path[-1]])
        trace_ok=witness if row['label'] else negative_terminal
        if trace_ok and answer_ok:continue
        if not path or path[0]!=s:kind='wrong_or_missing_start'
        elif row['label'] and overshot:kind='continued_after_goal'
        elif nonedges:kind='nonedge_or_nonnode'
        elif row['label'] and first_bad_branch is not None:kind='legal_edge_to_dead_branch'
        elif not trace_ok:kind='stopped_before_required_endpoint'
        else:kind='answer_only_wrong'
        counts[kind]=counts.get(kind,0)+1
        def name(u):return tok.name(nodes[u]) if u>=0 else 'INVALID'
        failure=dict(query=row['query_ids'],label=row['label'],pred=row['pred'],answer_correct=answer_ok,
            source=name(s),target=name(t),trace=[name(u) for u in path],kind=kind,
            all_edges_legal=legal,nonedge_steps=nonedges,reached_goal=reached,continued_after_goal=overshot,
            ends_at_sink=bool(path and path[-1]>=0 and not adj[path[-1]]),
            interrupted=row.get('interrupted'),generation_limit=row.get('limit'))
        if first_bad_branch:
            failure['first_bad_branch']={k:([name(v) for v in value] if isinstance(value,list) else name(value) if k in ('current','chosen') else value) for k,value in first_bad_branch.items()}
        if overshot:failure['goal_reached_at_index']=goal_pos
        failures.append(failure)
    result=dict(manifest=manifest,test_queries=len(rows),positive_queries=sum(r['label'] for r in rows),
        answer_errors=sum(r['pred']!=r['label'] for r in rows),failed_trace_or_answer=len(failures),
        positive_failures=sum(r['label'] for r in failures),negative_failures=sum(not r['label'] for r in failures),
        kinds=counts,failures=failures,
        method='Original saved test greedy traces, no audit-time stopping or new generation. Positive trace must be wholly legal and end at target; negative trace legal and end at sink. '
               'Kinds use continued-after-goal before nonedge as primary classification; nonedge indices remain recorded, so kinds are exclusive while properties may overlap.')
    (OUT/f'{setting}.json').write_text(json.dumps(result,indent=2));print(setting,json.dumps(result,ensure_ascii=False),flush=True)

if __name__=='__main__':
    OUT.mkdir(exist_ok=True)
    for setting in ('simple','hard'):run(setting)
