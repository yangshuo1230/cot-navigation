"""Independent combinatorial checks and result integrity; no model training."""
import json
from pathlib import Path
import numpy as np
import run as study

study.gen.self_test()
# A merge can be visited before reaching the same current node via another branch.
adj=[[1,2],[3],[3],[],[]]
def states(fifo):
    search=study.ref.bfs_search if fifo else study.ref.dfs_search
    rows=[]
    for order in [[1,2],[2,1]]:
        a=[list(v) for v in adj];a[0]=order
        steps=search(a,0,4)
        seen=set()
        for step in steps:
            if step.cur in seen:continue
            seen.add(step.cur)
            rows.append(dict(query=1,current=step.cur,succ=adj[step.cur],
                             allowed=list(step.cands),frontier=list(step.stack),
                             visited=list(step.visited),nodes=list(range(5))))
    return rows
dfs=states(False);bfs=states(True)
assert study.matched_pairs(dfs,exact_stack=True)
assert study.matched_pairs(bfs)
for rows in [dfs,bfs]:
    for a,b,node in study.matched_pairs(rows):
        assert rows[a]['current']==rows[b]['current']
        assert node not in rows[a]['visited'] and node in rows[b]['visited']
        assert node in rows[a]['succ'] and node in rows[b]['succ']

root=Path(__file__).resolve().parent
if (root/'results.json').exists():
    d=json.loads((root/'results.json').read_text())
    original=root.parent/'node_search_state_generated'/'results.json'
    if not original.exists():
        original=root.parent/'node_state_audit_20261026'/'generated.json'
    baseline=json.loads(original.read_text()) if original.exists() else None
    for name,model in d['models'].items():
        if baseline:
            assert model['checkpoint']['sha256']==baseline['models'][name]['checkpoint']['sha256']
            assert model['query_ids']==baseline['models'][name]['query_ids']
        q=model['query_ids']
        assert not(set(q['train'])&set(q['val']) or set(q['test'])&set(q['train']) or set(q['test'])&set(q['val']))
        for task in ['allowed','filter']:
            for key in ['matched','matched_exact_stack']:
                p=model['tasks'][task][key]
                if p['n_pairs']:
                    assert p['directional_accuracy'][7]==.5
                    assert p['mean_logit_delta'][7]==0
        for task in model['tasks'].values():
            for result in task['scores']:
                assert result['nontrivial_rows']<=result['n_rows']
                for metric in ['pairwise_auc','topk_exact']:
                    if result.get(metric) is not None:
                        assert 0<=result[metric]<=1
    print('audited models',len(d['models']))
print('PASS: independent replay, DFS same-stack and BFS legal-history toy pairs, split and identity-control checks')
