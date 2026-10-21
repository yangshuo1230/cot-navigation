"""Independent traversal-language checks and scale input boundary tests."""
from itertools import combinations, permutations, product
from types import SimpleNamespace

from cot_navigation.analysis.scale_error_awareness import node_slots, body_tokens, synthetic_events, slots, IndexedCorruptions
import numpy as np
from cot_navigation.protocols.tokenizer import Tokenizer
from cot_navigation.protocols.dfs import dfs_search
from cot_navigation.protocols.bfs import bfs_search, first_expansion_order


def test_node_prefix_oracle_matches_all_neighbor_orders():
    edges = list(combinations(range(4), 2))
    checked = 0
    for mask in range(64):
        adj = [[] for _ in range(4)]
        for i, (u,v) in enumerate(edges):
            if mask >> i & 1:
                adj[u].append(v)
        orders = list(product(*(list(permutations(vs)) for vs in adj)))
        for source, target in permutations(range(4), 2):
            ex = SimpleNamespace(adj=adj,nodes=list(range(4)),source=source,target=target)
            for protocol, search in [('dfs_nodes',dfs_search),('bfs_nodes',bfs_search)]:
                legal_next = {}
                for order in orders:
                    full = tuple(first_expansion_order(search(order,source,target)))
                    for i in range(len(full)+1):
                        legal_next.setdefault(full[:i],set())
                        if i<len(full):
                            legal_next[full[:i]].add(full[i])
                for prefix, expected in legal_next.items():
                    yielded = list(node_slots(ex,protocol,list(prefix)+[999]))
                    assert len(yielded)==len(prefix)+1
                    assert yielded[-1][1]==expected,(mask,source,target,protocol,prefix,expected,yielded[-1])
                    checked += 1
    assert checked >= 4900


def test_bfs_equal_distance_does_not_imply_legal_queue_order():
    # 0 -> (1,2); 1 -> 3; 2 -> 4. Expanding 1 before 2 forces 3 before 4.
    ex=SimpleNamespace(adj=[[1,2],[3],[4],[],[]],nodes=list(range(5)),source=0,target=4)
    result=list(node_slots(ex,'bfs_nodes',[0,1,2,4]))
    assert result[-1][1]=={3}


def test_stop_is_not_labeled_as_reasoning_error():
    tok=Tokenizer(enable_stop=True)
    ex=SimpleNamespace(row=SimpleNamespace(prompt_len=2))
    assert body_tokens(ex,tok,[tok.id('<QUERY>'),tok.id('|'),tok.node_id(0),tok.id('<STOP>'),tok.yes])==[tok.node_id(0)]


def test_synthetic_node_prefix_pairs_are_valid_and_do_not_append_future():
    tok=Tokenizer(enable_stop=True)
    nodes=[tok.node_id(i) for i in range(4)]
    prompt=[tok.id('<QUERY>'),nodes[0],nodes[3],tok.id('|')]
    body=[nodes[i] for i in [0,1,2,3]]
    ex=SimpleNamespace(index=8,adj=[[1,2],[3],[],[]],nodes=nodes,source=0,target=3,
                       prompt=prompt,row=SimpleNamespace(prompt_len=4,tokens=prompt+body+[tok.yes]))
    evs=synthetic_events([ex],'bfs_nodes',tok,15,31)
    assert len(evs)>5
    for e in evs:
        assert len(e['wrong'])==len(e['correct'])==e['position']+1
        assert e['wrong'][:-1]==e['correct'][:-1]
        for key,expected in [('wrong',False),('correct',True)]:
            trace=e[key][4:]
            oracle=list(slots(ex,'bfs_nodes',tok,trace))
            assert len(oracle)==len(trace)
            assert (trace[-1] in oracle[-1][1])==expected


def test_indexed_sampler_matches_expanded_shuffle_and_rng_state():
    for seed in range(8):
        rows=[(i,{100+i},list(range(i+1)), 'field','kind') for i in range(1,24)]
        expanded=[(at,good,bad,field,kind) for at,good,options,field,kind in rows for bad in options]
        a,b=np.random.default_rng(seed),np.random.default_rng(seed)
        a.shuffle(expanded)
        indexed=IndexedCorruptions(rows,b)
        assert a.bit_generator.state==b.bit_generator.state
        while expanded:
            assert indexed.pop()==expanded.pop()
        assert len(indexed)==0
