"""PUSH-state membership features and edge-identity checks on identical prefixes."""
from common import *
def main():
 store,plan=store_plan();OUT.mkdir(exist_ok=True)
 for binding in bindings():
  cfg,tok,model=load(binding);name=binding['name'];features=[];metadata=[];edge_rows=[]
  for split in ['val','test']:
   for spec in plan[split]:
    ix=spec['index'];row=store.render(ix,'dfs_state',tok);seq=row.tokens[:-1];P=row.prompt_len;recs=a.parse(seq,P,tok);ids=[int(v) for v in store.node_vocab_ids(ix,tok)];edge=np.array(seq[:P-4]).reshape(-1,3);pushes=[(i,r) for i,r in enumerate(recs) if r['action']=='PUSH'];positions=[p for i,r in pushes for p in [r['c']+1,r['c']+2]];cap={};hooks=[]
    def embed_hook(m,args,out):cap[0]=out[0,positions].clone()
    hooks.append(model.embed.register_forward_hook(embed_hook))
    for layer in range(1,10):
     def grab(m,args,out,layer=layer):cap[layer]=out[0][0,positions].clone()
     hooks.append(model.blocks[layer-1].register_forward_hook(grab))
    with torch.inference_mode():logits,ws=model(torch.tensor(seq)[None],return_attn=True)
    for h in hooks:h.remove()
    for pi,(ri,r) in enumerate(pushes):
     u=r['argument'];neighbors=set(edge[edge[:,1]==u,2].tolist())
     features.append(np.stack([np.stack([cap[l][pi*2].numpy(),cap[l][pi*2+1].numpy()]) for l in range(10)]))
     metadata.append(dict(index=ix,split=split,record=ri+1,action_position=r['c']+1,argument_position=r['c']+2,pushed=u-tok.node_lo,source=seq[P-3]-tok.node_lo,visited=[v-tok.node_lo for v in r['visited']],graph=[v-tok.node_lo for v in ids],neighbors=[v-tok.node_lo for v in neighbors],count=len(r['visited'])))
    fresh=[]
    for ri,r in enumerate(recs):
     if ri==0 or not r['candidates'] or any(p['stack'][-1]==r['stack'][-1] for p in recs[:ri]):continue
     for q in range(r['b'],r['c']-1):fresh.append((ri,r,q))
    take=np.unique(np.linspace(0,len(fresh)-1,min(24,len(fresh)),dtype=int)) if fresh else []
    for si in take:
     ri,r,q=fresh[int(si)];u=r['stack'][-1];relevant=np.flatnonzero(edge[:,1]==u).tolist();visited=set(r['visited']);eligible=[i for i in relevant if int(edge[i,2]) not in visited];excluded=[i for i in relevant if int(edge[i,2]) in visited];wrong=[i for i in range(len(edge)) if i not in relevant];controls=[];same_dest_pairs=[]
     for i in relevant:
      choices=[j for j in wrong if (int(edge[j,2]) in visited)==(int(edge[i,2]) in visited) and j not in controls]
      if choices:controls.append(min(choices,key=lambda j:abs(j-i)))
      choices=[j for j in wrong if edge[j,2]==edge[i,2]]
      if choices:same_dest_pairs.append((i,min(choices,key=lambda j:abs(j-i))))
     # destination+1 is the common carrier in both nine-layer checkpoints.
     carriers=[3*i+3 for i in range(len(edge))];rows=np.stack([w[0,0,q,carriers].numpy() for w in ws]);pred=int(logits[0,q].argmax())
     edge_rows.append(dict(model=name,index=ix,split=split,record=ri+1,q=q,outdegree=len(relevant),n_eligible=len(eligible),n_excluded=len(excluded),n_matched_wrong=len(controls),n_same_destination_pairs=len(same_dest_pairs),attention=dict(relevant=rows[:,relevant].sum(1).tolist(),eligible=rows[:,eligible].sum(1).tolist(),excluded=rows[:,excluded].sum(1).tolist(),position_matched_wrong=rows[:,controls].sum(1).tolist(),same_destination_true=[float(rows[l,[i for i,j in same_dest_pairs]].mean()) if same_dest_pairs else None for l in range(9)],same_destination_wrong=[float(rows[l,[j for i,j in same_dest_pairs]].mean()) if same_dest_pairs else None for l in range(9)]),predicted_is_legal_remaining=pred in set(seq[q+1:r['c']]),top_edges=[[dict(edge=int(i)+1,source=tok.name(int(edge[i,1])),destination=tok.name(int(edge[i,2])),is_relevant=i in relevant,is_visited=int(edge[i,2]) in visited,weight=float(rows[l,i])) for i in np.argsort(-rows[l])[:3]] for l in range(9)]))
    print(json.dumps(dict(model=name,index=ix,split=split,pushes=len(metadata),edge_events=len(edge_rows))),flush=True)
   # Save after each split so an interrupted job remains inspectable.
   np.savez_compressed(OUT/(name+'.push.npz'),features=np.array(features));save(name+'.push_metadata',metadata);save(name+'.edge_reads',edge_rows)
 save('models',bindings());save('COLLECT_COMPLETE',dict(models=2,queries_per_model=36,sites=['PUSH','argument'],layers=list(range(10))))
if __name__=='__main__':main()
