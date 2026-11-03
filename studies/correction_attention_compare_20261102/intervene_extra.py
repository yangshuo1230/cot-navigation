"""Exploratory route ablations after discovering model-specific carriers."""
import sys,json,torch,numpy as np
from pathlib import Path
ROOT=Path('/mnt/ningxuefei/yangshuo/cot-navigation-unified');sys.path.insert(0,str(ROOT/'studies/dfs_attention_lifecycle_20261030'));import analyze as a
OUT=ROOT/'studies/correction_attention_compare_20261102'
def main():
 torch.set_num_threads(4);bindings=json.loads((OUT/'models.json').read_text());extra=json.loads((OUT/'extra_routes.json').read_text());selected=json.loads((OUT/'selected_routes.json').read_text());panel=json.loads((a.OUT/'plan.json').read_text())['panels']['n100']['test'];store=a.ContextStore(a.ROOT/'data/scale_v1'/a.SETS['n100']);results=[]
 for binding in bindings:
  name=binding['name'];run=Path(binding['run']);cfg=a.load_config([run/'resolved_config.yaml']);tok=a.Tokenizer.from_config(cfg);model=a.build_model(cfg.model,len(tok),tok.pad).eval();model.requires_grad_(False);model.load_state_dict(torch.load(run/'checkpoints/best.pt',map_location='cpu',weights_only=False)['model'])
  for spec in [s for s in panel if s['band']=='long']:
   row=store.render(spec['index'],'dfs_state',tok);seq=row.tokens[:-1];P=row.prompt_len;recs=a.parse(seq,P,tok);ri,r=min([(i,r) for i,r in enumerate(recs) if i and recs[i-1]['action']=='PUSH' and len(r['visited'])>=3],key=lambda x:abs(x[0]-len(recs)/2));p=r['a']+1+len(r['visited'])//2;target=seq[p];jobs=[]
   jobs.append(dict(phase='init_prompt',q=r['start']-1,layer=selected[name]['stack初始化']['layer'],keys=[P-1],field=False))
   jobs.append(dict(phase='init_roots',q=r['start']-1,layer=extra[name]['init_roots']['layer'],keys=[past['start'] for past in recs[:ri]],field=False))
   jobs.append(dict(phase='visited_direct',q=p-1,layer=extra[name]['visited_direct']['layer'],keys=[k for past in recs[:ri] for k in range(past['a']+1,past['b']) if seq[k]==target],field=False))
   opts=[];repeat=[]
   for i,t in enumerate(recs):
    if not t['candidates'] or t['action']=='FOUND':continue
    old=[past for past in recs[:i] if past['stack'][-1]==t['stack'][-1] and past['action']=='PUSH']
    if i and not any(past['stack'][-1]==t['stack'][-1] for past in recs[:i]):opts.append((i,t))
    if old:repeat.append((i,t,old[-1]))
   if opts:
    i,t=min(opts,key=lambda x:abs(x[0]-len(recs)/2));offset=extra[name]['offset_scan']['offset'];edge=np.array(seq[:P-4]).reshape(-1,3);related=np.flatnonzero(edge[:,1]==t['stack'][-1]);keys=[3*int(e)+2+offset for e in related];keys=[k for k in keys if 0<=k<=t['b']]
    jobs.append(dict(phase='fresh_graph_best_offset',q=t['b'],layer=extra[name]['offset_scan']['layer'],keys=keys,field=True,gold=t['candidates'],offset=offset))
   if repeat:
    i,t,old=min(repeat,key=lambda x:abs(x[0]-len(recs)/2));jobs.append(dict(phase='revisit_tail',q=t['b'],layer=extra[name]['revisit_tail']['layer'],keys=list(range(old['b']+2,old['c'])),field=True,gold=t['candidates']))
   for job in jobs:
    q=job['q'];prefix=seq[:q+1];module=model.blocks[job['layer']-1].attn;cap={}
    def capture(m,args,out):cap['v']=out.view(1,len(prefix),3,1,cfg.model.d_model)[0,:,2,0].clone()
    h=module.qkv.register_forward_hook(capture)
    with torch.inference_mode():base,ws=model(torch.tensor(prefix)[None],return_attn=True)
    h.remove();v=cap['v'];z=ws[job['layer']-1][0,0,q].clone();del ws
    delta=-torch.nn.functional.linear((v[job['keys']]*z[job['keys'],None]).sum(0),module.proj.weight) if job['keys'] else torch.zeros(cfg.model.d_model)
    null=torch.tensor(np.random.default_rng(spec['index']+q+job['layer']).normal(size=cfg.model.d_model),dtype=torch.float32);null*=delta.norm()/null.norm()
    for condition,change in [('zero_route',delta),('norm_matched_random',null),('sham',torch.zeros_like(delta))]:
     def patch(m,args,out):
      x=out[0].clone();x[0,q]+=change;return x,out[1]
     h=module.register_forward_hook(patch)
     if job['field']:
      generated=[]
      for _ in range(12):
       with torch.inference_mode():log=model(torch.tensor(prefix+generated)[None])
       token=int(log[0,-1].argmax());generated.append(token)
       if token==tok.id('|') or not tok.node_lo<=token<tok.node_lo+tok.n_node_tokens:break
      nodes=generated[:-1] if generated[-1]==tok.id('|') else generated;correct=generated[-1]==tok.id('|') and len(nodes)==len(set(nodes)) and set(nodes)==set(job['gold']);prediction=[tok.name(x) for x in generated]
     else:
      with torch.inference_mode():log=model(torch.tensor(prefix)[None])
      correct=int(log[0,q].argmax())==seq[q+1];prediction=tok.name(int(log[0,q].argmax()))
     h.remove();results.append(dict(model=name,index=spec['index'],phase=job['phase'],layer=job['layer'],condition=condition,prediction=prediction,correct=correct,field=job['field'],n_keys=len(job['keys']),attention_mass=float(z[job['keys']].sum()),delta_norm=float(change.norm()),offset=job.get('offset')))
   (OUT/'extra_interventions.json').write_text(json.dumps(results));print(json.dumps(dict(model=name,index=spec['index'],conditions=len(results))),flush=True)
if __name__=='__main__':main()
