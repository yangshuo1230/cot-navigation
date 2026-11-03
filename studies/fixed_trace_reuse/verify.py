"""Check matching definitions and final-panel accounting independently."""
from pathlib import Path
import json
import run

cfg=run.load_config([run.RUNS['hard']/'dfs_state/resolved_config.yaml'])
tok=run.Tokenizer.from_config(cfg);bank=run.Bank('dfs_state',tok)
v=[tok.id(f'<V{i}>') for i in range(6)]
bar=tok.id('|');push=tok.id('<PUSH>');end=tok.id(';')
base=[v[0],v[1],bar,v[0],v[1],v[2],bar,v[3],v[4],bar,push,v[3],end]
reordered=[v[0],v[1],bar,v[2],v[0],v[1],bar,v[4],v[3],bar,push,v[4],end]
other_frontier=[v[1],v[0],bar,v[0],v[1],v[2],bar,v[3],v[4],bar,push,v[3],end]
other_visited=[v[0],v[1],bar,v[0],v[1],v[5],bar,v[3],v[4],bar,push,v[3],end]
assert bank.state_key(base)==bank.state_key(reordered)
assert bank.state_key(base)!=bank.state_key(other_frontier)
assert bank.state_key(base)!=bank.state_key(other_visited)
trie={};bank.insert(trie,[1,2,3]);bank.insert(trie,[1,4])
assert bank.prefix(trie,[1,2,5])==2
assert bank.prefix(trie,[1,2])==2
assert bank.prefix(trie,[8,1])==0
assert run.digest([1,2])!=run.digest([2,1])
checked=[]
for setting in ('simple','hard'):
    for protocol in ('dfs_nodes','bfs_nodes','dfs_state','bfs_state'):
        p=run.OUT/f'{setting}_{protocol}.json';r=json.loads(p.read_text())
        assert r['version']==2
        assert len(r['rows'])==len(r['panel_ids'])==len(set(r['panel_ids']))
        assert r['training_bank_variants']==4*r['training_queries']
        assert r['disjoint_query_pairs']
        assert r['complete_count']==sum(v['complete'] and not v['interrupted'] for v in r['rows'])
        assert r['censored_count']==sum(v['censored'] for v in r['rows'])
        for row in r['rows']:
            assert row['novel_valid_states_after_first2']<=row['valid_states_after_first2']
            assert row['records_seen_excluding_first2']<=row['records_after_first2']
        checked.append(dict(model=p.stem,n=len(r['rows']),complete=r['complete_count']))
(run.OUT/'verification.json').write_text(json.dumps(dict(state_key_order_invariance=True,
    frontier_and_visited_discriminated=True,prefix_and_digest_checks=True,panels=checked),indent=2)+'\n')
print('All matching checks passed')
