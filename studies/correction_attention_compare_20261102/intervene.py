"""Same-prefix content-path swaps; layers chosen from validation attention only."""
import sys,json,torch,numpy as np
from pathlib import Path
ROOT=Path('/mnt/ningxuefei/yangshuo/cot-navigation-unified');sys.path.insert(0,str(ROOT/'studies/dfs_attention_lifecycle_20261030'));import analyze as a
OUT=ROOT/'studies/correction_attention_compare_20261102'
def main():
 torch.set_num_threads(4);bindings=json.loads((OUT/'models.json').read_text());selected=json.loads((OUT/'selected_routes.json').read_text());panel=json.loads((a.OUT/'plan.json').read_text())['panels']['n100']['test'];store=a.ContextStore(a.ROOT/'data/scale_v1'/a.SETS['n100']);results=[]
 for binding in bindings:
  name=binding['name'];run=Path(binding['run']);cfg=a.load_config([run/'resolved_config.yaml']);tok=a.Tokenizer.from_config(cfg);model=a.build_model(cfg.model,len(tok),tok.pad).eval();model.requires_grad_(False);model.load_state_dict(torch.load(run/'checkpoints/best.pt',map_location='cpu',weights_only=False)['model'])
  for spec in [s for s in panel if s['band']=='long']:
   row=store.render(spec['index'],'dfs_state',tok);seq=row.tokens[:-1];P=row.prompt_len;recs=a.parse(seq,P,tok);ri,r=min([(i,r) for i,r in enumerate(recs) if i and recs[i-1]['action']=='PUSH' and len(r['visited'])>=3],key=lambda x:abs(x[0]-len(recs)/2));p=r['a']+1+len(r['visited'])//2;target=seq[p]
   jobs=[dict(phase='stack_init',layer=selected[name]['stack初始化']['layer'],q=r['start']-1,keys=[P-1],donor=recs[0]['a']),dict(phase='visited_init',layer=selected[name]['visited初始化']['layer'],q=r['a'],keys=[P-1],donor=recs[0]['a']),dict(phase='visited_continue',layer=selected[name]['visited续写/root']['layer'],q=p-1,keys=[past['start'] for past in recs[:ri] if past['visited'][-1]==target],donor=recs[0]['start'])]
   for job in jobs:
    q=job['q'];module=model.blocks[job['layer']-1].attn;ids=torch.tensor(seq[:q+1])[None];cap={}
    def capture(m,args,out):cap['v']=out.view(1,q+1,3,1,cfg.model.d_model)[0,:,2,0].clone()
    h=module.qkv.register_forward_hook(capture)
    with torch.inference_mode():base,weights=model(ids,return_attn=True)
    h.remove();v=cap['v'];z=weights[job['layer']-1][0,0,q].clone();del weights
    assert all(seq[k]==seq[job['donor']] for k in job['keys'])
    delta=sum((torch.nn.functional.linear((v[job['donor']]-v[k])*z[k],module.proj.weight) for k in job['keys']),torch.zeros(cfg.model.d_model));random=torch.tensor(np.random.default_rng(spec['index']+q).normal(size=cfg.model.d_model),dtype=torch.float32);random*=delta.norm()/random.norm()
    for condition,change in [('same_symbol_value_swap',delta),('norm_matched_random',random),('sham',torch.zeros_like(delta))]:
     def patch(m,args,out):
      x=out[0].clone();x[0,q]+=change;return x,out[1]
     h=module.register_forward_hook(patch)
     with torch.inference_mode():log,_=model(ids,return_attn=True)
     h.remove();assert torch.equal(base[:,:q],log[:,:q])
     if condition=='sham':assert torch.equal(base,log)
     gold=seq[q+1];results.append(dict(model=name,index=spec['index'],phase=job['phase'],layer=job['layer'],q=q,n_keys=len(job['keys']),condition=condition,baseline_correct=int(base[0,q].argmax())==gold,prediction=tok.name(int(log[0,q].argmax())),output_changed=int(log[0,q].argmax())!=int(base[0,q].argmax()),next_correct=int(log[0,q].argmax())==gold,attention_mass=float(z[job['keys']].sum()),delta_norm=float(change.norm()),gold_logp_drop=float(base[0,q].log_softmax(-1)[gold]-log[0,q].log_softmax(-1)[gold]),earlier_logits_unchanged=True))
   (OUT/'interventions.json').write_text(json.dumps(results));print(json.dumps(dict(model=name,index=spec['index'],conditions=len(results))),flush=True)
if __name__=='__main__':main()
