import json
import analyze as a

cfg=a.load_config([a.RUN/'resolved_config.yaml']);tok=a.Tokenizer.from_config(cfg)
store=a.ContextStore(a.ROOT/'data/scale_v1'/a.SETS['n100'])
specs=json.loads((a.OUT/'field_termination_probe.json').read_text())['candidate_permutations'];results=[]
for spec in [x for x in specs if x['kind']=='all_emitted']:
    ix=spec['index'];row=store.render(ix,'dfs_state',tok);seq=row.tokens[:-1];recs=a.parse(seq,row.prompt_len,tok);r=recs[spec['record']-1]
    seq[r['b']+1],seq[r['c']-1]=seq[r['c']-1],seq[r['b']+1]
    # The protocol chooses the FIRST candidate. Validate the permuted prefix
    # with its corresponding action, without reusing the original later path.
    assert r['action']=='PUSH'
    seq[r['c']+2]=seq[r['b']+1]
    inv={t:i for i,t in enumerate(store.node_vocab_ids(ix,tok))}
    audit=a.replay_state(store.adj(ix),int(store.arrays['s_topo'][ix]),int(store.arrays['t_topo'][ix]),seq[row.prompt_len:r['end']],tok,inv)
    assert audit['error']=='missing_terminal' and audit['records']==spec['record'],audit
    results.append(dict(index=ix,record=spec['record'],prefix_legal=True,action_matches_first_candidate=True))
a.save('field_permutation_validation.json',dict(passed=True,results=results))
print(json.dumps(dict(passed=True,permutations=len(results))))
