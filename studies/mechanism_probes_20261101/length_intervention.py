"""Manipulate a validation-trained depth direction while holding emitted probe fixed."""
import sys,json,torch,numpy as np
from pathlib import Path
ROOT=Path('/mnt/ningxuefei/yangshuo/cot-navigation-unified');sys.path.insert(0,str(ROOT/'studies/dfs_attention_lifecycle_20261030'));import analyze as a
OUT=ROOT/'studies/mechanism_probes_20261101'
def main():
 torch.set_num_threads(4);cfg=a.load_config([a.RUN/'resolved_config.yaml']);tok=a.Tokenizer.from_config(cfg);model=a.build_model(cfg.model,len(tok),tok.pad).eval();model.requires_grad_(False);model.load_state_dict(torch.load(a.RUN/'checkpoints/best.pt',map_location='cpu',weights_only=False)['model'])
 result=json.loads((OUT/'probe_results.json').read_text());layer=result['selected_length_layer'];w=np.load(OUT/'probe_weights.npz')['stack_L'+str(layer)];constraints=w[:-1,:2].T;v=constraints.T@np.linalg.solve(constraints@constraints.T,np.array([1.,0.]));assert np.allclose(constraints@v,[1,0])
 store=a.ContextStore(a.ROOT/'data/scale_v1'/a.SETS['n100']);panel=json.loads((a.OUT/'plan.json').read_text())['panels']['n100']['test'];results=[]
 for spec in [s for s in panel if s['band']=='long']:
  row=store.render(spec['index'],'dfs_state',tok);seq=row.tokens[:-1];recs=a.parse(seq,row.prompt_len,tok);options=[(ri,r) for ri,r in enumerate(recs) if len(r['stack'])>=2];ri,r=min(options,key=lambda x:abs(x[0]-len(recs)/2))
  for label,q,direction in [('stop',r['a']-1,1),('continue',r['a']-2,-1)]:
   ids=torch.tensor(seq[:q+1])[None];cap={}
   def capture(m,args,out):cap['x']=out[0][0,q].clone()
   h=model.blocks[layer-1].register_forward_hook(capture)
   with torch.inference_mode():base=model(ids)
   h.remove();x=cap['x'].numpy();rng=np.random.default_rng(20261101+spec['index']+q);random=rng.normal(size=len(v));random-=constraints.T@np.linalg.solve(constraints@constraints.T,constraints@random);random*=np.linalg.norm(v)/np.linalg.norm(random)
   for condition,delta in [('sham',np.zeros_like(v)),('depth_shift_1',v*direction),('depth_shift_2',v*direction*2),('norm_matched_null',random*direction)]:
    delta=torch.tensor(delta,dtype=torch.float32)
    def patch(m,args,out):
     z=out[0].clone();z[0,q]+=delta;return z,out[1]
    h=model.blocks[layer-1].register_forward_hook(patch)
    with torch.inference_mode():log=model(ids)
    h.remove();assert torch.equal(base[:,:q],log[:,:q])
    if condition=='sham':assert torch.equal(base,log)
    p=int(log[0,q].argmax());before=np.r_[x,1.]@w;after=np.r_[x+delta.numpy(),1.]@w
    results.append(dict(index=spec['index'],record=ri+1,layer=layer,receiver=label,q=q,true_depth=len(r['stack']),baseline=tok.name(int(base[0,q].argmax())),baseline_correct=int(base[0,q].argmax())==seq[q+1],condition=condition,prediction=tok.name(p),expected_class_switch=(tok.node_lo<=p<tok.node_lo+tok.n_node_tokens) if label=='stop' else p==tok.id('|'),decoded_before=before.tolist(),decoded_after=after.tolist(),delta_norm=float(delta.norm()),residual_norm=float(np.linalg.norm(x)),stop_margin_change=float((log[0,q,tok.id('|')]-log[0,q,tok.node_lo:tok.node_lo+tok.n_node_tokens].max())-(base[0,q,tok.id('|')]-base[0,q,tok.node_lo:tok.node_lo+tok.n_node_tokens].max())),earlier_logits_unchanged=True))
   (OUT/'length_intervention.json').write_text(json.dumps(results))
  print(json.dumps(dict(index=spec['index'],layer=layer)),flush=True)
if __name__=='__main__':main()
