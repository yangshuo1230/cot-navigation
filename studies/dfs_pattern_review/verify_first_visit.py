"""Check field-attempt counting, merge filtering and first-error censoring."""
import json
import candidate_first_visit as c
from cot_navigation.protocols.dfs import dfs_search
from cot_navigation.protocols.render import render_steps
tok=c.m.base.Tokenizer(8)
nodes=[tok.node_id(i) for i in range(5)]
adj=[[1,2],[3],[3],[],[]]
graph={nodes[u]:set(nodes[v] for v in vs) for u,vs in enumerate(adj)}
def replay(body,finish=True):
    r=c.VisitReplay(graph,nodes[0],nodes[4],tok);seen=[]
    for t in body:
        seen.append(t)
        if r.check(seen):break
    if finish:r.finish(seen,0,False)
    return r
clean=render_steps(dfs_search(adj,0,4),nodes,tok)
r=replay(clean)
assert r.error is None and len(r.events)==7
assert sum(e['fresh'] for e in r.events)==4
assert sum(not e['fresh'] for e in r.events)==3
assert any(e['current']==nodes[2] and e['fresh'] and e['blocked_successors']==1 and not e['nonempty'] for e in r.events)
bar=tok.id('|');semi=tok.id(';');push=tok.id('<PUSH>')
prefix=[nodes[0],bar,nodes[0],bar]
bad=replay(prefix+[nodes[3],bar,push,nodes[3],semi])
assert bad.error['kind']=='candidate_mismatch' and bad.events[0]['bad']
outside=replay(prefix+[tok.node_id(7)])
assert outside.error['kind']=='candidate_outside_graph' and outside.events[0]['bad']
invalid=replay(prefix+[push])
assert invalid.error['kind']=='candidate_invalid_token' and invalid.events[0]['bad']
unclosed=replay(prefix)
assert unclosed.error['kind']=='candidate_unclosed' and unclosed.events[0]['bad']
wrongvis=replay([nodes[0],bar,nodes[0],nodes[0],bar])
assert wrongvis.error['kind']=='visited_mismatch' and not wrongvis.events
wrongaction=replay(prefix+[nodes[1],nodes[2],bar,tok.id('<POP>'),semi])
assert wrongaction.error['kind']=='wrong_action' and len(wrongaction.events)==1 and not wrongaction.events[0]['bad']
(c.m.OUT/'candidate_first_visit_verification.json').write_text(json.dumps(dict(
    first_vs_revisit=True,merge_requires_filtering_on_first_visit=True,
    candidate_set_error=True,outside_graph=True,invalid_token=True,unclosed=True,
    wrong_visited_excluded=True,wrong_action_keeps_candidate_success=True),indent=2)+'\n')
print('Field-level counting and merge checks passed')
