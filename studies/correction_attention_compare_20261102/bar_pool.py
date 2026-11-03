"""Prompt bar plus past stack bars: distributed source interface control."""
import sys,json,torch,numpy as np
from pathlib import Path
ROOT=Path('/mnt/ningxuefei/yangshuo/cot-navigation-unified');sys.path.insert(0,str(ROOT/'studies/dfs_attention_lifecycle_20261030'));import analyze as a
OUT=ROOT/'studies/correction_attention_compare_20261102'
def main():
 torch.set_num_threads(4);models=json.loads((OUT/'models.json').read_text());selected=json.loads((OUT/'selected_routes.json').read_text());panel=json.loads((a.OUT/'plan.json').read_text())['panels']['n100']['test'];store=a.ContextStore(a.ROOT/'data/scale_v1'/a.SETS['n100']);results=[]
 for spec_model in [m for m in models if m['name'] in ['base6','corr9_s0','corr9_s1']]:
  name=spec_model['name'];run=Path(spec_model['run']);cfg=a.load_config([run/'resolved_config.yaml']);tok=a.Tokenizer.from_config(cfg);model=a.build_model(cfg.model,len(tok),tok.pad).eval();model.requires_grad_(False);model.load_state_dict(torch.load(run/'checkpoints/best.pt',map_location='cpu',weights_only=False)['model']);layer=selected[name]['stack初始化']['layer'];module=model.blocks[layer-1].attn
  for spec in [s for s in panel if s['band']=='long']:
   row=store.render(spec['index'],'dfs_state',tok);seq=row.tokens[:-1];P=row.prompt_len;recs=a.parse(seq,P,tok);ri,r=min([(i,r) for i,r in enumerate(recs) if i and recs[i-1]['action']=='PUSH' and len(r['visited'])>=3],key=lambda x:abs(x[0]-len(recs)/2));q=r['start']-1;ids=torch.tensor(seq[:q+1])[None];keys=[P-1]+[p['a'] for p in recs[:ri]];cap={}
   def capture(m,args,out):cap['v']=out.view(1,q+1,3,1,cfg.model.d_model)[0,:,2,0].clone()
   h=module.qkv.register_forward_hook(capture)
   with torch.inference_mode():base,ws=model(ids,return_attn=True)
   h.remove();z=ws[layer-1][0,0,q].clone();del ws;delta=-torch.nn.functional.linear((cap['v'][keys]*z[keys,None]).sum(0),module.proj.weight);random=torch.tensor(np.random.default_rng(spec['index']).normal(size=cfg.model.d_model),dtype=torch.float32);random*=delta.norm()/random.norm()
   for condition,change in [('zero_bar_pool',delta),('norm_matched_random',random),('sham',torch.zeros_like(delta))]:
    def patch(m,args,out):
     x=out[0].clone();x[0,q]+=change;return x,out[1]
    h=module.register_forward_hook(patch)
    with torch.inference_mode():log,_=model(ids,return_attn=True)
    h.remove();assert torch.equal(base[:,:q],log[:,:q])
    if condition=='sham':assert torch.equal(base,log)
    results.append(dict(model=name,index=spec['index'],layer=layer,condition=condition,baseline_correct=int(base[0,q].argmax())==seq[q+1],correct=int(log[0,q].argmax())==seq[q+1],prediction=tok.name(int(log[0,q].argmax())),attention_mass=float(z[keys].sum()),delta_norm=float(change.norm())))
   (OUT/'bar_pool.json').write_text(json.dumps(results))
if __name__=='__main__':main()
