"""Exhaustive small-DAG agreement and temporal attribution fixtures."""
import json,gzip,itertools
from pathlib import Path
import run as audit
from cot_navigation.protocols.tokenizer import Tokenizer

tok=Tokenizer(256)
def events(nodes,answer):return [tok.node_id(v) for v in nodes]+[answer]
def check(adj,s,t,seq,answer):
    inverse={tok.node_id(v):v for v in range(len(adj))}
    return audit.analyze(adj,s,t,events(seq,answer),inverse,tok,dist=audit.distances(adj))
adj=[[1,2],[3],[],[],[]]
assert check(adj,0,4,[0,1,3,2],tok.no)['protocol_valid']
assert check(adj,0,4,[0,1,2],tok.no)['error']=='skipped_live_frontier'
assert check(adj,0,4,[0,1,3,0],tok.no)['detail']['repeat_subtype']=='printed_implicit_return_top'
assert check([[1],[2],[]],0,2,[0,2],tok.yes)['detail']['jump_subtype']=='reachable_descendant_skip'
assert check([[1],[]],0,1,[0],tok.yes)['error']=='premature_yes'
assert check([[1],[]],0,1,[0,1],tok.no)['error']=='wrong_answer_after_goal'
assert check([[1],[]],0,1,[0,1],tok.id('<EOS>'))['error']=='missing_answer_after_goal'
assert check([[1,2],[],[]],0,1,[0,1,2],tok.yes)['detail']['follows_DFS_if_goal_ignored']
inverse={tok.node_id(v):v for v in range(3)}
r=audit.analyze([[1],[2],[]],0,2,[tok.node_id(0),tok.node_id(2),tok.id('|'),tok.yes],inverse,tok)
assert r['error']=='non_edge_jump' and r['first_error_position']==1
assert audit.replay_nodes([[1],[2],[]],0,2,[tok.node_id(0),tok.node_id(2),tok.id('|')],inverse)['error']=='non_node'

n=4;edges=[(u,v) for u in range(n) for v in range(u+1,n)]
sequences=[seq for k in range(n+1) for seq in itertools.product(range(n),repeat=k)]
tested=0
for bits in range(1<<len(edges)):
    graph=[[] for _ in range(n)]
    for k,(u,v) in enumerate(edges):
        if bits>>k&1:graph[u].append(v)
    dist=audit.distances(graph);inverse={tok.node_id(v):v for v in range(n)}
    for source in range(n):
        for target in range(n):
            answer=tok.yes if target in dist[source] else tok.no
            for seq in sequences:
                body=[tok.node_id(v) for v in seq]
                a=audit.analyze(graph,source,target,body+[answer],inverse,tok,dist=dist)
                b=audit.replay_nodes(graph,source,target,body,inverse)
                assert a['semantic_valid']==b['valid'],(graph,source,target,seq,a,b)
                assert a['protocol_valid']==b['valid']
                tested+=1
out={'status':'PASS','exhaustive_4node_DAG_cases':tested,'temporal_first_error_fixtures':8,
     'implicit_backtracking_and_wrong_terminal_answers':'PASS','goal_ignored_continuation_fixture':'PASS'}
root=Path(__file__).resolve().parent
if (root/'results.json').exists():
    d=json.loads((root/'results.json').read_text())
    prior=json.loads((audit.ROOT/'studies/dfs_v6_state_probes/results.json').read_text())
    total=0
    for s in d['seeds']:
        assert s['checkpoint_sha256']==prior['models'][f"seed{s['seed']}"]['checkpoint_sha256']
        with gzip.open(root/f"seed{s['seed']}_queries.jsonl.gz",'rt') as stream:rows=[json.loads(line) for line in stream]
        assert len(rows)==2712 and len({r['query'] for r in rows})==2712
        assert s['legacy_semantic_agreements']==2712
        for panel,p in s['panels'].items():
            rr=[r for r in rows if r['panel']==panel]
            assert len(rr)==p['n']==sum(p['errors'].values())
            assert sum(r['protocol_valid'] for r in rr)==p['protocol_valid']
        total+=len(rows)
    out.update(seeds=len(d['seeds']),queries=total,legacy_semantic_agreement='100%',checkpoint_sha='PASS')
(root/'verification.json').write_text(json.dumps(out,indent=2))
print(out)
