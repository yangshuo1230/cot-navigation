"""Cancellable records and first-token locations extend the shared replayer."""
import itertools
from cot_navigation.protocols.tokenizer import Tokenizer
from cot_navigation.protocols.correction_replay import replay,self_test
from cot_navigation.protocols.render import render_steps
from cot_navigation.protocols.dfs import dfs_search
from cot_navigation.protocols.bfs import bfs_search


def test_explicit_error_types_and_cancellation():
    tok=Tokenizer(256,enable_stop=True,enable_mistake=True)
    assert self_test(tok)>=15


def test_first_error_positions_and_valid_prefix_censoring():
    tok=Tokenizer(256,enable_stop=True,enable_mistake=True)
    a,b,c=[tok.node_id(i) for i in range(3)];bar=tok.id('|');semi=tok.id(';')
    prefix=[a,bar,a,bar]
    x=replay(prefix+[c,bar,tok.id('<PUSH>'),b,semi],1,{a:[b],b:[],c:[]},a,b,tok)
    assert x['first']['kind']=='candidate_nonedge' and x['first']['token']==5
    assert x['first']['search_step']==1 and x['risk_steps_by_bin']==[1,0,0,0,0,0,0]
    y=replay(prefix,1,{a:[b],b:[]},a,b,tok,interrupted=True)
    assert y['first'] is None and y['first_outcome']=='no_error_prefix_budget_cutoff'


def test_all_four_node_dags_and_negative_labels():
    # Existing DFS reference/rendering is the independent clean-trace oracle.
    tok=Tokenizer(256,enable_stop=True,enable_mistake=True)
    nodes=[tok.node_id(i) for i in range(4)]
    edges=list(itertools.combinations(range(4),2))
    for mask in range(64):
        raw=[[] for _ in nodes]
        for i,(u,v) in enumerate(edges):
            if mask>>i&1:raw[u].append(v)
        adj={nodes[u]:[nodes[v] for v in vs] for u,vs in enumerate(raw)}
        for s,t in itertools.permutations(range(4),2):
            for fifo,search in [(False,dfs_search),(True,bfs_search)]:
                steps=search(raw,s,t);label=int(steps[-1].action=='FOUND')
                tokens=render_steps(steps,nodes,tok)
                result=replay(tokens,label,adj,nodes[s],nodes[t],tok,fifo=fifo)
                assert result['literal_valid'] and result['correct'] and result['first'] is None
                assert not replay(tokens,1-label,adj,nodes[s],nodes[t],tok,fifo=fifo)['correct']
                a=nodes[s];bar=tok.id('|')
                cancelled=[a,bar,a,bar,a,bar,tok.id('<MISTAKE>'),tok.id(';')]
                fixed=replay(cancelled+tokens,label,adj,nodes[s],nodes[t],tok,fifo=fifo)
                assert fixed['trace_valid'] and fixed['first_detected'] and fixed['first_repaired']
