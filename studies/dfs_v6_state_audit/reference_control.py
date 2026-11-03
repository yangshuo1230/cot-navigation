"""CPU-only training finite-sample and held-out random-reference controls."""
import json, math
from collections import Counter
import run as a

tok = a.Tokenizer(256)
meta = next(w for w in json.loads((a.V6/'manifest.json').read_text())['worlds'] if w['name']=='n80_g0')
bank = a.PackedBank(meta,'dfs_state',a.torch.device('cpu'))
world = bank.world
inverse = {tok.node_id(v): i for i,v in enumerate(world['node_names'])}
counts = json.loads((a.ROOT/'outputs/fixed_navigation_protocols_v6/pilot/n80/g0_s0/dfs_state/exposure_counts.json').read_text())
train, bodies, orders = a.diversity(bank,counts,tok,inverse)
degree = Counter()
for q in world['panels']['short_train'] + world['panels']['long_train']:
    k = len(world['adj'][world['records'][q]['s']])
    if k>1:
        degree[k] += 1


def expected_entropy(k, draws=8):
    return -k*sum(math.comb(draws,n)*(1/k)**n*(1-1/k)**(draws-n)
                  *(n/draws)*math.log(n/draws) for n in range(1,draws+1))/math.log(k)


doc = dict(root_degree_counts=dict(degree),
           observed_train_normalized_entropy=train['root_branch_mean_normalized_entropy'],
           expected_iid_uniform_eight_draw_entropy=sum(v*expected_entropy(k) for k,v in degree.items())/sum(degree.values()),
           panels={})
for panel in a.PANELS:
    c = Counter()
    for q in world['panels'][panel]:
        r = world['records'][q]
        for row in range(int(bank.qptr[q]),int(bank.qptr[q+1])):
            begin = int(bank.cpu['offsets'][row])+4
            end = int(bank.cpu['offsets'][row])+int(bank.cpu['answers'][row])
            body = bank.cpu['tokens'][begin:end].tolist()
            seq = tuple([r['s']]+[inverse[body[i+1]] for i,t in enumerate(body[:-1]) if t==tok.id('<PUSH>')])
            c['n']+=1
            c['body_exact']+=a.digest(body) in bodies
            c['push_order_exact']+=seq in orders
    doc['panels'][panel] = dict(c)
(a.OUT/'reference_control.json').write_text(json.dumps(doc,indent=2))
print(json.dumps(doc))
