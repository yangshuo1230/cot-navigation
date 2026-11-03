"""Predeclared L2..L6 depth shifts and dose-matched null controls."""
from common import *
def main():
 cfg,tok,model,store,panel=setup();all_weights=np.load(PREV/'probe_weights.npz');results=[]
 for spec in [s for s in panel if s['band']=='long']:
  row=store.render(spec['index'],'dfs_state',tok);seq=row.tokens[:-1];recs=a.parse(seq,row.prompt_len,tok);ri,r=min([(i,r) for i,r in enumerate(recs) if len(r['stack'])>=2],key=lambda x:abs(x[0]-len(recs)/2))
  for receiver,q,sign in [('stop',r['a']-1,1),('continue',r['a']-2,-1)]:
   ids=torch.tensor(seq[:q+1])[None];cap={};hooks=[]
   for layer in range(2,7):
    def grab(m,args,out,layer=layer):cap[layer]=out[0][0,q].clone()
    hooks.append(model.blocks[layer-1].register_forward_hook(grab))
   with torch.inference_mode():base=model(ids)
   for h in hooks:h.remove()
   for layer in range(2,7):
    w=all_weights['stack_L'+str(layer)];constraints=w[:-1,:2].T;v=constraints.T@np.linalg.solve(constraints@constraints.T,[1.,0.]);assert np.allclose(constraints@v,[1,0])
    null=np.random.default_rng(20261102+spec['index']+q+layer).normal(size=len(v));null-=constraints.T@np.linalg.solve(constraints@constraints.T,constraints@null);null*=np.linalg.norm(v)/np.linalg.norm(null)
    jobs=[('sham',0,np.zeros_like(v))]+[(kind,dose,direction*sign*dose) for dose in [1,2,4] for kind,direction in [('length',v),('null',null)]]
    for condition,dose,delta in jobs:
     delta=torch.tensor(delta,dtype=torch.float32)
     def patch(m,args,out):
      x=out[0].clone();x[0,q]+=delta;return x,out[1]
     h=model.blocks[layer-1].register_forward_hook(patch)
     with torch.inference_mode():log=model(ids)
     h.remove();assert torch.equal(base[:,:q],log[:,:q])
     if condition=='sham':assert torch.equal(base,log)
     p=int(log[0,q].argmax());x=cap[layer].numpy();before=np.r_[x,1.]@w;after=np.r_[x+delta.numpy(),1.]@w
     results.append(dict(index=spec['index'],record=ri+1,layer=layer,receiver=receiver,q=q,condition=condition,dose=dose,baseline_correct=int(base[0,q].argmax())==seq[q+1],baseline=tok.name(int(base[0,q].argmax())),prediction=tok.name(p),class_switch=(tok.node_lo<=p<tok.node_lo+tok.n_node_tokens) if receiver=='stop' else p==tok.id('|'),probe_shift=(after-before).tolist(),delta_norm=float(delta.norm()),relative_norm=float(delta.norm()/cap[layer].norm()),stop_margin_delta=float((log[0,q,tok.id('|')]-log[0,q,tok.node_lo:tok.node_lo+tok.n_node_tokens].max())-(base[0,q,tok.id('|')]-base[0,q,tok.node_lo:tok.node_lo+tok.n_node_tokens].max())),earlier_logits_unchanged=True))
   save('length_layers',results)
if __name__=='__main__':main()
