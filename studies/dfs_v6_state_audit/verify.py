"""CPU-only independent checks of error attribution and frozen-bank replay."""
import json
import run as a

tok = a.Tokenizer(256)
a.self_test(tok)
adj = [[1,2], [3], [3], [], []]
inverse = {tok.node_id(i): i for i in range(5)}
from semantics import reference, replay_state


def encode(rows):
    result = []
    for stack, visited, candidates, action, arg in rows:
        for field in (stack, visited, candidates):
            result.extend(tok.node_id(x) for x in field)
            result.append(tok.id('|'))
        result.append(tok.id('<'+action+'>'))
        if arg is not None:
            result.append(tok.node_id(arg))
        result.append(tok.id(';'))
    return result


rows = reference(adj, 0, 4)
assert rows[5][0] == [0,2] and rows[5][2] == []
altered = [(s[:], v[:], c[:], ac, ar) for s,v,c,ac,ar in rows]
altered[5][2].append(3)
bad = a.audit(adj, 0, 4, encode(altered), tok, inverse)
assert bad['first']['phase'] == 'fresh_pushed'
assert bad['first']['flags']['extra_visited_neighbor']
assert bad['first']['record'] == 6
altered = [(s[:], v[:], c[:], ac, ar) for s,v,c,ac,ar in rows]
altered[0][2].pop()
assert a.audit(adj, 0, 4, encode(altered), tok, inverse)['first']['flags']['missing']
altered = [(s[:], v[:], c[:], ac, ar) for s,v,c,ac,ar in rows]
altered[1][1].append(4)
bad = a.audit(adj, 0, 4, encode(altered), tok, inverse)
assert bad['first']['kind'] == 'visited_mismatch' and len(bad['events']) == 1

meta = next(w for w in json.loads((a.V6/'manifest.json').read_text())['worlds'] if w['name']=='n80_g0')
bank = a.PackedBank(meta, 'dfs_state', a.torch.device('cpu'))
world = bank.world
inverse = {tok.node_id(v): i for i,v in enumerate(world['node_names'])}
ids = a.np.random.default_rng(20261027).choice(a.np.flatnonzero(bank.cpu['qids']>=0),size=200,replace=False)
for row in ids:
    q = int(bank.cpu['qids'][row])
    r = world['records'][q]
    start = int(bank.cpu['offsets'][row])+4
    end = int(bank.cpu['offsets'][row])+int(bank.cpu['answers'][row])
    body = bank.cpu['tokens'][start:end].tolist()
    new = a.audit(world['adj'],r['s'],r['t'],body,tok,inverse)
    old = replay_state(world['adj'],r['s'],r['t'],body,tok,inverse)
    assert new['valid'] and old['valid']
    assert new['terminal'] == old['terminal']
    assert len(new['events']) == old['records']
doc = dict(status='PASS',independent_frozen_reference_replays=200,
           toy_checks=['missing', 'duplicates', 'non_neighbor', 'fresh_with_visited_neighbor',
                       'visited_error_excluded_from_candidate_opportunities'],
           native_generation_checks='Every trace checked against original independent replay_state in run.py')
(a.OUT/'verification.json').write_text(json.dumps(doc,indent=2))
print(json.dumps(doc))
