"""Local L5 old-PUSH value interventions during candidate generation."""
import sys,json,torch,numpy as np
from pathlib import Path
ROOT=Path('/mnt/ningxuefei/yangshuo/cot-navigation-unified');sys.path.insert(0,str(ROOT/'studies/dfs_attention_lifecycle_20261030'));import analyze as a
OUT=ROOT/'studies/mechanism_probes_20261101'
def main():
 torch.set_num_threads(4);cfg=a.load_config([a.RUN/'resolved_config.yaml']);tok=a.Tokenizer.from_config(cfg);model=a.build_model(cfg.model,len(tok),tok.pad).eval();model.requires_grad_(False);model.load_state_dict(torch.load(a.RUN/'checkpoints/best.pt',map_location='cpu',weights_only=False)['model'])
 store=a.ContextStore(a.ROOT/'data/scale_v1'/a.SETS['n100']);panel=json.loads((a.OUT/'plan.json').read_text())['panels']['n100']['test'];results=[]
 for spec in panel:
  row=store.render(spec['index'],'dfs_state',tok);seq=row.tokens[:-1];recs=a.parse(seq,row.prompt_len,tok);choices=[]
  for ri,r in enumerate(recs):
   prior=[p for p in recs[:ri] if p['stack'][-1]==r['stack'][-1] and p['action']=='PUSH']
   if prior and len(prior[-1]['candidates'])>1 and r['candidates'] and r['action']!='FOUND':choices.append((ri,r,prior[-1]))
  if not choices:continue
  ri,r,past=min(choices,key=lambda x:abs(x[0]-len(recs)/2));q=r['b'];prefix=seq[:q+1];module=model.blocks[4].attn;cap={}
  def capture(m,args,out):cap['v']=out.view(1,len(prefix),3,1,cfg.model.d_model)[0,:,2,0].clone()
  h=module.qkv.register_forward_hook(capture)
  with torch.inference_mode():base,weights=model(torch.tensor(prefix)[None],return_attn=True)
  h.remove();v=cap['v'];z=weights[4][0,0,q].clone();del weights;push,arg=past['c']+1,past['c']+2
  zero_push=-torch.nn.functional.linear(v[push]*z[push],module.proj.weight);zero_arg=-torch.nn.functional.linear(v[arg]*z[arg],module.proj.weight)
  donors=[t['c']+1 for t in recs[:ri] if t['action']=='PUSH' and t['stack'][-1]!=r['stack'][-1]];donor=donors[-1] if donors else push
  swap=torch.nn.functional.linear((v[donor]-v[push])*z[push],module.proj.weight)
  random=torch.tensor(np.random.default_rng(spec['index']).normal(size=cfg.model.d_model),dtype=torch.float32);random*=zero_push.norm()/random.norm()
  random_arg=random*(zero_arg.norm()/random.norm());random_pair=random*((zero_push+zero_arg).norm()/random.norm())
  assert seq[past['b']+1]==seq[arg]
  swap_arg=torch.nn.functional.linear((v[past['b']+1]-v[arg])*z[arg],module.proj.weight)
  for condition,delta in [('baseline',torch.zeros_like(zero_push)),('zero_old_PUSH',zero_push),('zero_old_argument',zero_arg),('zero_pair',zero_push+zero_arg),('swap_old_PUSH_same_symbol',swap),('swap_argument_same_node',swap_arg),('norm_matched_random',random),('norm_matched_random_arg',random_arg),('norm_matched_random_pair',random_pair)]:
   def patch(m,args,out):
    x=out[0].clone();x[0,q]+=delta;return x,out[1]
   h=module.register_forward_hook(patch);generated=[]
   for _ in range(12):
    with torch.inference_mode():log=model(torch.tensor(prefix+generated)[None])
    token=int(log[0,-1].argmax());generated.append(token)
    if token==tok.id('|') or not tok.node_lo<=token<tok.node_lo+tok.n_node_tokens:break
   h.remove();nodes=generated[:-1] if generated[-1]==tok.id('|') else generated;valid=generated[-1]==tok.id('|') and len(nodes)==len(set(nodes)) and set(nodes)==set(r['candidates'])
   results.append(dict(index=spec['index'],record=ri+1,condition=condition,push_attention=float(z[push]),argument_attention=float(z[arg]),delta_norm=float(delta.norm()),output=[tok.name(v) for v in generated],set_correct=valid,gold=[tok.name(v) for v in r['candidates']],donor_distinct=donor!=push))
  (OUT/'push_carrier.json').write_text(json.dumps(results));print(json.dumps(dict(index=spec['index'],record=ri+1)),flush=True)
if __name__=='__main__':main()
