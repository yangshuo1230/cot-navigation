"""First expansion: legal graph/visit-history pairs and explicit-field controls."""
from common import *
def main():
 cfg,tok,model,store,panel=setup();results=[];selection=[]
 for spec in panel:
  ix=spec['index'];row=store.render(ix,'dfs_state',tok);seq=row.tokens[:-1];P=row.prompt_len;recs=a.parse(seq,P,tok);edge=np.array(seq[:P-4]).reshape(-1,3);ids=store.node_vocab_ids(ix,tok);inv={int(v):i for i,v in enumerate(ids)};choices=[]
  for ri,r in enumerate(recs):
   cur=r['stack'][-1]
   if ri==0 or any(p['stack'][-1]==cur for p in recs[:ri]) or r['action']=='FOUND':continue
   neighbors=set(edge[edge[:,1]==cur,2].tolist());cross=neighbors&set(r['visited'])
   for v in sorted(cross):
    if v in [seq[P-3],seq[P-2]]:continue
    ws=[int(w) for w in ids if int(w) not in neighbors|set(r['visited'])|{seq[P-3],seq[P-2],cur} and inv[int(w)]//10==inv[v]//10]
    if ws:choices.append((ri,r,v,ws[0],neighbors))
  if not choices:selection.append(dict(index=ix,available=False));continue
  ri,r,v,w,neighbors=min(choices,key=lambda x:(abs(x[0]-len(recs)/2),x[0],x[2]));cur=r['stack'][-1];prefix=seq[:r['b']+1];gold=set(r['candidates']);selection.append(dict(index=ix,record=ri+1,available=True,cur=tok.name(cur),excluded=tok.name(v),alternate=tok.name(w)))
  # Graph node permutation v<->w, except the never-before-expanded cur's
  # outgoing edges. Earlier completed records remain a legal paired search.
  def perm(t):return w if t==v else v if t==w else t
  paired=prefix[:]
  for p in range(0,P-4,3):
   u,d=prefix[p+1:p+3]
   if u!=cur:paired[p+1],paired[p+2]=perm(u),perm(d)
  for p in range(P,len(paired)):paired[p]=perm(paired[p])
  graph=paired[:P-4];newedge=np.array(graph).reshape(-1,3);adj=[[] for _ in ids]
  for _,u,d in newedge:adj[inv[int(u)]].append(inv[int(d)])
  audit=a.replay_state(adj,int(store.arrays['s_topo'][ix]),int(store.arrays['t_topo'][ix]),paired[P:recs[ri-1]['end']],tok,inv)
  assert audit['error']=='missing_terminal' and audit['records']==ri,audit
  assert set(newedge[newedge[:,1]==cur,2].tolist())==neighbors
  newgold=neighbors-set(paired[r['a']+1:r['b']]);assert newgold==gold|{v}
  explicit=prefix[:]
  for p in range(r['a']+1,r['b']):explicit[p]=perm(explicit[p])
  paired_keep_explicit=paired[:];paired_keep_explicit[r['a']+1:r['b']]=prefix[r['a']+1:r['b']]
  graph_only=prefix[:];graph_only[:P]=paired[:P]
  # First-expansion edge control: change exactly the excluded cur->v edge
  # into cur->w, where w was unvisited. No earlier cur expansion is affected.
  edge_edit=prefix[:]
  for p in range(0,P-4,3):
   if edge_edit[p+1]==cur and edge_edit[p+2]==v:edge_edit[p+2]=w
  jobs=[('baseline',prefix,gold),('explicit_visited_only',explicit,newgold),('legal_paired_graph_and_history',paired,newgold),('paired_history_keep_original_explicit_visited',paired_keep_explicit,newgold),('paired_graph_only',graph_only,gold),('excluded_edge_to_unvisited_node',edge_edit,gold|{w})]
  # Scrub only one historical role, preserving graph/current explicit state.
  for role in ['visited','stack','argument','candidates','stack_argument','candidate_argument','selected_candidate','unselected_candidates','selected_candidate_argument','unselected_candidate_argument','all']:
   edited=prefix[:]
   for past in recs[:ri]:
    positions=list(range(past['a']+1,past['b'])) if role=='visited' else list(range(past['start'],past['a'])) if role=='stack' else list(range(past['b']+1,past['c'])) if role=='candidates' else [past['c']+2] if role=='argument' and past['action']=='PUSH' else list(range(past['start'],past['end'])) if role=='all' else []
    if role=='stack_argument':positions=list(range(past['start'],past['a']))+([past['c']+2] if past['action']=='PUSH' else [])
    if role=='candidate_argument':positions=list(range(past['b']+1,past['c']))+([past['c']+2] if past['action']=='PUSH' else [])
    selected=([past['b']+1] if past['action']=='PUSH' and past['argument']==v else [])
    unselected=[p for p in range(past['b']+1,past['c']) if p not in selected]
    if role=='selected_candidate':positions=selected
    if role=='unselected_candidates':positions=unselected
    if role=='selected_candidate_argument':positions=selected+([past['c']+2] if past['action']=='PUSH' else [])
    if role=='unselected_candidate_argument':positions=unselected+([past['c']+2] if past['action']=='PUSH' else [])
    for p in positions:
     if edited[p]==v:edited[p]=w
   jobs.append(('historical_'+role+'_only',edited,newgold))
  for condition,edited,expected in jobs:
   generated=candidate(model,tok,edited);nodes=generated[:-1] if generated[-1]==tok.id('|') else generated;well=generated[-1]==tok.id('|') and len(nodes)==len(set(nodes))
   results.append(dict(index=ix,record=ri+1,condition=condition,excluded=tok.name(v),alternate=tok.name(w),output=[tok.name(t) for t in generated],original_correct=well and set(nodes)==gold,expected_correct=well and set(nodes)==expected,excluded_returned=v in nodes,alternate_returned=w in nodes,legal_paired_previous_records=True if condition=='legal_paired_graph_and_history' else None,expected=[tok.name(t) for t in sorted(expected)]))
  save('fresh_candidates',results)
 save('fresh_selection',selection)
if __name__=='__main__':main()
