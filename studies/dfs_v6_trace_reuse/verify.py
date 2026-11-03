"""Independent state-bank and substring tests. No GPU inference or fitting."""
import json,random
from pathlib import Path
import run as audit
from cot_navigation.protocols.dfs import dfs_search
import numpy as np

audit.self_test()
manifest=json.loads((audit.V6/'manifest.json').read_text())
meta=next(m for m in manifest['worlds'] if m['name']=='n80_g0')
bank=audit.PackedBank(meta,'dfs_nodes',audit.torch.device('cpu'))
world=bank.world;tok=audit.Tokenizer(256)
tokens=[tok.node_id(i) for i in world['node_names']];inv={t:i for i,t in enumerate(tokens)}
trainq=set(world['panels']['short_train'])|set(world['panels']['long_train'])
rows=[i for i,q in enumerate(bank.cpu['qids']) if int(q) in trainq]
for row in random.Random(71).sample(rows,200):
    qid=int(bank.cpu['qids'][row]);q=world['records'][qid]
    a=int(bank.cpu['offsets'][row])+4;b=int(bank.cpu['offsets'][row])+int(bank.cpu['answers'][row])
    seq=[inv[int(t)] for t in bank.cpu['tokens'][a:b]]
    # Recover a compatible neighbor order, then replay with the original project
    # DFS rather than the audit's implicit-stack implementation.
    rank={u:k for k,u in enumerate(seq)}
    adj=[sorted(vs,key=lambda u:rank.get(u,1000+u)) for vs in world['adj']]
    ref=dfs_search(adj,q['s'],q['t'])
    assert list(dict.fromkeys(s.cur for s in ref))==seq
    expected={(tuple(s.stack),sum(1<<v for v in s.visited)) for s in ref if len(s.visited)>=3}
    actual={s for s in audit.states(world['adj'],q['s'],q['t'],seq,True,True) if s[1].bit_count()>=3}
    assert actual==expected,(qid,expected-actual,actual-expected)

path=Path(__file__).resolve().parent/'results.json'
doc=json.loads(path.read_text())
assert len(doc['seeds'])==4 and doc['audit_version']==2
examples=0
prior_keys=None;corpus=None;training_states=set()
for seed in doc['seeds']:
    run=audit.ROOT/f"outputs/fixed_navigation_protocols_v6/pilot/n80/g0_s{seed['seed']}/dfs_nodes"
    counts=json.loads((run/'exposure_counts.json').read_text())
    keys={key for key,count in counts.items() if key.startswith('q:') and count>0}
    if keys!=prior_keys:
        seqs=[];training_states=set()
        for row,key in enumerate(bank.keys):
            if key not in keys:continue
            q=world['records'][int(bank.cpu['qids'][row])]
            a=int(bank.cpu['offsets'][row])+4;b=int(bank.cpu['offsets'][row])+int(bank.cpu['answers'][row])
            seq=[inv[int(t)] for t in bank.cpu['tokens'][a:b]]
            seqs.append(seq)
            training_states.update(audit.states(world['adj'],q['s'],q['t'],seq,True,True))
        corpus=audit.Corpus(seqs,world['n_nodes']);prior_keys=keys
    for ex in seed['examples']:
        q=world['records'][ex['query']]
        trace=[tok.id(v) for v in ex['trace']]
        checked=audit.replay_nodes(world['adj'],q['s'],q['t'],trace,inv)
        assert checked['valid'];assert not ex['whole_subsequence']
        assert not corpus.compare([inv[t] for t in trace])['whole_subsequence']
        stack=tuple(inv[tok.id(v)] for v in ex['novel_stack'])
        mask=sum(1<<inv[tok.id(v)] for v in ex['novel_visited'])
        assert (stack,mask) not in training_states
        examples+=1
    for p in seed['panels'].values():
        for group in ['generated','legal_generated','reference']:
            m=p[group]
            assert m['exact']<=m['whole_prefix']+1e-9
            assert m['whole_prefix']<=m['whole_contiguous']+1e-9
            assert m['whole_contiguous']<=m['whole_subsequence']+1e-9
            assert m['unique_novel_states']<=m['unique_states']
verification={'status':'PASS','training_state_replays_independently_verified':200,
              'suffix_automaton_bruteforce_tests':100,'saved_legal_counterexamples_replayed':examples,
              'training_bank_includes_implicit_pops_and_goals':True,'seeds':4}
(path.parent/'verification.json').write_text(json.dumps(verification,indent=2))
print(verification)
