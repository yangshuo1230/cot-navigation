"""Exhaustive small-DAG reference checks and deliberate state corruption."""
from pathlib import Path
from itertools import combinations,permutations
import json
import sys

ROOT=Path.cwd();OUT=ROOT/'studies/fixed100_legality'
sys.path[:0]=[str(ROOT/'src'),str(OUT),str(ROOT/'studies/scale_trace_audit')]
from replay import StateReplay,ActionReplay
from cot_navigation.protocols.tokenizer import Tokenizer
from cot_navigation.protocols.dfs import dfs_search
from cot_navigation.protocols.bfs import bfs_search
from cot_navigation.protocols.render import render_steps
from legacy_trace_audit import validate_states,validate_execution


def replay(raw,s,t,fifo,body,pred,cls=StateReplay):
    a={nodes[u]:set(nodes[v] for v in vs) for u,vs in enumerate(raw)}
    checker=cls(a,nodes[s],nodes[t],tok,fifo);prefix=[]
    for token in body:
        prefix.append(token)
        if checker.check(prefix):break
    return checker.finish(prefix,pred)


tok=Tokenizer(4,enable_stop=True);nodes=[tok.node_id(i) for i in range(4)]
pairs=list(combinations(range(4),2));n=0
for mask in range(64):
    raw=[[] for _ in nodes]
    for i,(u,v) in enumerate(pairs):
        if mask>>i&1:raw[u].append(v)
    for s,t in permutations(range(4),2):
        for fifo,search in ((False,dfs_search),(True,bfs_search)):
            steps=search(raw,s,t);pred=int(steps[-1].action=='FOUND');body=render_steps(steps,nodes,tok)
            result=replay(raw,s,t,fifo,body,pred)
            assert result['semantic_valid'] and result['protocol_valid']
            assert replay(raw,s,t,fifo,body,pred,ActionReplay)['semantic_valid']
            assert not replay(raw,s,t,fifo,body,1-pred)['semantic_valid']
            n+=1
raw=[[1,2],[3],[],[]];steps=dfs_search(raw,0,3);body=render_steps(steps,nodes,tok)
# Missing a real eligible candidate must invalidate full-state semantics while
# the unchanged action sequence may remain executable.
broken=body.copy();bars=[i for i,x in enumerate(broken) if x==tok.id('|')]
del broken[bars[1]+2]
assert not replay(raw,0,3,False,broken,1)['semantic_valid']
assert replay(raw,0,3,False,broken,1,ActionReplay)['semantic_valid']
# A real non-edge PUSH is rejected even in action-only replay.
broken=body.copy();push=broken.index(tok.id('<PUSH>'));broken[push+1]=nodes[3]
assert not replay(raw,0,3,False,broken,1,ActionReplay)['semantic_valid']
result=dict(reference_dag_query_search_cases=n,wrong_answers_rejected=True,
            missing_candidate_detected=True,annotation_vs_action_separated=True,nonedge_push_detected=True)
(OUT/'validator_tests.json').write_text(json.dumps(result,indent=2));print(result)
