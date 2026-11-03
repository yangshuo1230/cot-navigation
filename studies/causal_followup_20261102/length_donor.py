"""Same source token, same emitted count=1, different natural planned lengths.
Full donor swap versus depth-only projection and its orthogonal complement.
"""
from common import *
def main():
 cfg,tok,model,store,panel=setup();weights=np.load(PREV/'probe_weights.npz');results=[]
 for spec in [s for s in panel if s['band']=='long']:
  row=store.render(spec['index'],'dfs_state',tok);seq=row.tokens[:-1];recs=a.parse(seq,row.prompt_len,tok)
  short=min([(i,r) for i,r in enumerate(recs) if len(r['stack'])==1],key=lambda x:abs(x[0]-len(recs)/2));long=min([(i,r) for i,r in enumerate(recs) if len(r['stack'])>=2],key=lambda x:abs(x[0]-len(recs)/2));qs=[short[1]['start'],long[1]['start']];assert seq[qs[0]]==seq[qs[1]]
  states={};handles=[]
  for layer in range(2,7):
   def capture(m,args,out,layer=layer):states[layer]=out[0][0,qs].clone()
   handles.append(model.blocks[layer-1].register_forward_hook(capture))
  with torch.inference_mode():model(torch.tensor(seq[:max(qs)+1])[None])
  for h in handles:h.remove()
  for receiver in [0,1]:
   q=qs[receiver];donor=1-receiver;ids=torch.tensor(seq[:q+1])[None]
   with torch.inference_mode():base=model(ids)
   for layer in range(2,7):
    x=states[layer][receiver].numpy();difference=(states[layer][donor]-states[layer][receiver]).numpy();w=weights['stack_L'+str(layer)];c=w[:-1,:2].T
    depth=c.T@np.linalg.solve(c@c.T,[float(c[0]@difference),0.]);both=c.T@np.linalg.solve(c@c.T,c@difference);complement=difference-both
    random=np.random.default_rng(spec['index']+q+layer).normal(size=len(x));random-=c.T@np.linalg.solve(c@c.T,c@random);random*=np.linalg.norm(difference)/np.linalg.norm(random)
    for condition,change in [('sham',np.zeros_like(difference)),('full_donor',difference),('depth_only_donor',depth),('complement_donor',complement),('matched_null',random)]:
     delta=torch.tensor(change,dtype=torch.float32)
     def patch(m,args,out):
      z=out[0].clone();z[0,q]+=delta;return z,out[1]
     h=model.blocks[layer-1].register_forward_hook(patch)
     with torch.inference_mode():log=model(ids)
     h.remove();assert torch.equal(base[:,:q],log[:,:q])
     if condition=='sham':assert torch.equal(base,log)
     p=int(log[0,q].argmax());results.append(dict(index=spec['index'],layer=layer,receiver='stop' if receiver==0 else 'continue',q=q,condition=condition,short_depth=1,long_depth=len(long[1]['stack']),baseline_correct=int(base[0,q].argmax())==seq[q+1],prediction=tok.name(p),class_switch=(tok.node_lo<=p<tok.node_lo+tok.n_node_tokens) if receiver==0 else p==tok.id('|'),probe_shift=(np.r_[x+change,1]@w-np.r_[x,1]@w).tolist(),delta_norm=float(delta.norm()),earlier_logits_unchanged=True))
  save('length_donor',results)
if __name__=='__main__':main()
