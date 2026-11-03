"""Exhaustive independent node-language membership on all 4-node DAGs."""
from itertools import combinations,permutations,product
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src'),str(Path(__file__).resolve().parent)]
from legacy_trace_audit import validate_nodes
from cot_navigation.protocols.dfs import dfs_search
from cot_navigation.protocols.bfs import bfs_search,first_expansion_order
from cot_navigation.protocols.tokenizer import Tokenizer

tok=Tokenizer(enable_stop=True,world_tokens=15)
nodes=[tok.node_id(i) for i in range(4)]
pairs=list(combinations(range(4),2));checked=0
for mask in range(64):
    raw=[[] for _ in nodes]
    for i,(u,v) in enumerate(pairs):
        if mask>>i&1:raw[u].append(v)
    orders=list(product(*(list(permutations(vs)) for vs in raw)))
    adj={nodes[u]:[nodes[v] for v in vs] for u,vs in enumerate(raw)}
    for s,t in permutations(range(4),2):
        for fifo,search in [(False,dfs_search),(True,bfs_search)]:
            legal=set()
            for ordered in orders:
                steps=search(ordered,s,t)
                legal.add((tuple(nodes[v] for v in first_expansion_order(steps)),int(steps[-1].action=='FOUND')))
            others=[n for n in nodes if n!=nodes[s]]
            for size in range(4):
                for tail in permutations(others,size):
                    trace=(nodes[s],)+tail
                    for answer in [0,1]:
                        actual=validate_nodes(list(trace),answer,adj,nodes[s],nodes[t],tok,fifo)['valid']
                        assert actual==((trace,answer) in legal),(mask,s,t,fifo,trace,answer)
                        checked+=1
print('PASS: node-only validators equal all possible reference priority orders for',checked,'complete candidate sequences')
