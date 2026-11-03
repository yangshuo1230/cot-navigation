"""Independent artifact consistency checks, including all event masses."""
import json,gzip
from pathlib import Path
from collections import Counter
import numpy as np
ROOT=Path(__file__).resolve().parent
def read(name):return json.loads((ROOT/name).read_text(encoding='utf-8'))
def main():
    plan=read('plan.json');identities=[]
    for tag,parts in plan['panels'].items():
        val={s['index'] for s in parts['val']};test={s['index'] for s in parts['test']}
        assert not(val&test)
        for split,specs in parts.items():
            for s in specs:assert s['length']<=4096;identities.append((tag,split,s['index']))
    assert len(identities)==52==len(set(identities))
    counts={};queries=set()
    for name in ['events.jsonl.gz','free_events.jsonl.gz']:
        count=0
        with gzip.open(ROOT/name,'rt',encoding='utf-8') as file:
            for line in file:
                e=json.loads(line);assert e['target']==e['q']+1;assert 0<=e['q']<e['target']<e['length']
                a=np.asarray(e['group_mass']);assert a.shape==(6,19)
                assert np.allclose(a.sum(1),1,atol=3e-6);assert a.min()>=0;assert a.max()<=1+1e-6
                for layer,keys in enumerate(e['top_keys']):
                    assert all(k['position']<=e['q'] and k['weight']>=0 for k in keys)
                    assert all(keys[i]['weight']>=keys[i+1]['weight'] for i in range(len(keys)-1))
                for values in e['route_mass'].values():assert min(values)>=0 and max(values)<=1+3e-6
                if e['mode']=='reference':queries.add((e['tag'],e['split'],e['index']))
                count+=1
        counts[name]=count
    assert len(queries)==52
    rolls=read('free_rollouts.json');assert len(rolls)==12==len(set(r['index'] for r in rolls))
    assert sum(r['full_valid'] for r in rolls)==2
    swaps=read('register_swaps.json')['results'];assert len(swaps)==18
    for r in swaps:
        for c in r['conditions']:
            assert c['earlier_logit_max_difference']==0
            if c['kind']=='sham':assert not c['changed_prediction'] and c['logp_drop']==0
            if c['kind']=='swap_main':assert c['key_token']==c['donor_token']
    terminal=read('terminal_tests.json')['results'];assert len(terminal)==18
    assert all(r['full_edit_follows_alternate'] and r['local_swap_follows_alternate'] for r in terminal)
    visited=read('visited_register_swaps.json')['results'];assert len(visited)==6
    assert all(r['baseline_correct'] for r in visited)
    for r in visited:
        control=next(c for c in r['conditions'] if c['kind']=='swap_low_weight_roots')
        sham=next(c for c in r['conditions'] if c['kind']=='sham')
        assert not control['changed_prediction'] and not sham['changed_prediction']
    results=dict(passed=True,reference_queries=52,event_counts=counts,free_queries=12,
                 all_event_positions_causal=True,all_group_masses_normalized=True,shams_exact=True,
                 local_interventions_leave_earlier_logits_unchanged=True,terminal_counterfactuals=18)
    (ROOT/'validation.json').write_text(json.dumps(results,indent=2)+'\n');print(json.dumps(results))
if __name__=='__main__':main()
