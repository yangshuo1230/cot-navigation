"""Independently complete the current paired state with its legal candidates."""
from common import *
def main():
 cfg=a.load_config([a.RUN/'resolved_config.yaml']);tok=a.Tokenizer.from_config(cfg);store=a.ContextStore(a.ROOT/'data/scale_v1'/a.SETS['n100']);selection=json.loads((OUT/'fresh_selection.json').read_text());results=[]
 for spec in [r for r in selection if r['available']]:
  ix=spec['index'];row=store.render(ix,'dfs_state',tok);seq=row.tokens[:-1];P=row.prompt_len;recs=a.parse(seq,P,tok);ri=spec['record']-1;r=recs[ri];cur=r['stack'][-1]
  # Names are serialized token names, use tokenizer vocabulary to recover IDs.
  v=tok.id(spec['excluded']);w=tok.id(spec['alternate']);ids=store.node_vocab_ids(ix,tok);inv={int(t):i for i,t in enumerate(ids)}
  def perm(t):return w if t==v else v if t==w else t
  paired=seq[:r['b']+1]
  for p in range(0,P-4,3):
   u,d=seq[p+1:p+3]
   if u!=cur:paired[p+1],paired[p+2]=perm(u),perm(d)
  for p in range(P,len(paired)):paired[p]=perm(paired[p])
  edge=np.array(paired[:P-4]).reshape(-1,3);adj=[[] for _ in ids]
  for _,u,d in edge:adj[inv[int(u)]].append(inv[int(d)])
  neighbors=set(edge[edge[:,1]==cur,2].tolist());candidates=sorted(neighbors-set(paired[r['a']+1:r['b']]));assert v in candidates
  body=paired[P:]+candidates+[tok.id('|'),tok.id('<PUSH>'),candidates[0],tok.id(';')]
  audit=a.replay_state(adj,int(store.arrays['s_topo'][ix]),int(store.arrays['t_topo'][ix]),body,tok,inv)
  assert audit['records']==ri+1 and audit['error']=='missing_terminal',audit
  results.append(dict(index=ix,current_record=ri+1,legal_through_current_record=True))
 save('paired_validation',dict(passed=True,results=results))
if __name__=='__main__':main()
