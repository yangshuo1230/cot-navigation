"""Audit actual training semantic-state variation, including same-stack pairs."""
import json
from collections import defaultdict, Counter
import run as a

tok = a.Tokenizer(256)
meta = next(w for w in json.loads((a.V6/'manifest.json').read_text())['worlds'] if w['name']=='n80_g0')
bank = a.PackedBank(meta,'dfs_state',a.torch.device('cpu'))
world = bank.world
inverse = {tok.node_id(v): i for i,v in enumerate(world['node_names'])}
counts = json.loads((a.ROOT/'outputs/fixed_navigation_protocols_v6/pilot/n80/g0_s0/dfs_state/exposure_counts.json').read_text())
by_stack = defaultdict(set)
stack_candidates = defaultdict(set)
by_current = defaultdict(set)
current_candidates = defaultdict(set)
state_occurrences = 0
for row,key in enumerate(bank.keys):
    if not key.startswith('q:') or counts.get(key,0)<=0:
        continue
    q = int(bank.cpu['qids'][row])
    target = world['records'][q]['t']
    begin = int(bank.cpu['offsets'][row])+4
    end = int(bank.cpu['offsets'][row])+int(bank.cpu['answers'][row])
    body = bank.cpu['tokens'][begin:end].tolist()
    at, record = 0,0
    while at<len(body):
        fields=[]
        for _ in range(3):
            pos=body.index(tok.id('|'),at)
            fields.append(tuple(inverse[t] for t in body[at:pos]))
            at=pos+1
        at=body.index(tok.id(';'),at)+1
        stack,visited,candidates=fields
        record+=1
        if record<=2 or stack[-1]==target:
            continue
        mask=sum(1<<v for v in visited)
        cmask=sum(1<<v for v in candidates)
        assert len(visited)==len(set(visited)) and len(candidates)==len(set(candidates))
        by_stack[stack].add(mask)
        stack_candidates[stack].add(cmask)
        by_current[stack[-1]].add(mask)
        current_candidates[stack[-1]].add(cmask)
        state_occurrences+=1
doc=dict(definition='Actual exposed training variants; exclude first two records and current==target; VIS membership ignores printed order; one corpus, all four seeds saw all variants',
    state_occurrences=state_occurrences,distinct_ordered_stacks=len(by_stack),
    stacks_multiple_visited=sum(len(v)>1 for v in by_stack.values()),
    stacks_multiple_eligible_sets=sum(len(v)>1 for v in stack_candidates.values()),
    states_under_stacks_with_multiple_visited=sum(len(v) for v in by_stack.values() if len(v)>1),
    distinct_stack_visited_states=sum(len(v) for v in by_stack.values()),
    per_current=[dict(current=u,visited_sets=len(by_current[u]),eligible_sets=len(current_candidates[u]),
                      outdegree=len(world['adj'][u]),possible_subsets_upper_bound=2**len(world['adj'][u])) for u in sorted(by_current)])
(a.OUT/'history_diversity.json').write_text(json.dumps(doc,indent=2))
print(json.dumps({k:v for k,v in doc.items() if k!='per_current'}))
