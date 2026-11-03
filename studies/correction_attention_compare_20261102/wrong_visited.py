"""Artificial explicit-state errors; distinguish STOP from candidate-set failure."""
import sys,json,torch,numpy as np
from pathlib import Path
ROOT=Path('/mnt/ningxuefei/yangshuo/cot-navigation-unified');sys.path.insert(0,str(ROOT/'studies/dfs_attention_lifecycle_20261030'));import analyze as a
OUT=ROOT/'studies/correction_attention_compare_20261102'
def main():
 torch.set_num_threads(4);bindings=json.loads((OUT/'models.json').read_text());panel=json.loads((a.OUT/'plan.json').read_text())['panels']['n100']['test'];store=a.ContextStore(a.ROOT/'data/scale_v1'/a.SETS['n100']);results=[]
 for binding in bindings:
  name=binding['name'];run=Path(binding['run']);cfg=a.load_config([run/'resolved_config.yaml']);tok=a.Tokenizer.from_config(cfg);model=a.build_model(cfg.model,len(tok),tok.pad).eval();model.requires_grad_(False);model.load_state_dict(torch.load(run/'checkpoints/best.pt',map_location='cpu',weights_only=False)['model'])
  for spec in [s for s in panel if s['band']=='long']:
   row=store.render(spec['index'],'dfs_state',tok);seq=row.tokens[:-1];P=row.prompt_len;recs=a.parse(seq,P,tok);edge=np.array(seq[:P-4]).reshape(-1,3);choices={}
   for ri,r in enumerate(recs):
    if r['action']=='FOUND' or len(r['visited'])<3:continue
    old=[p for p in recs[:ri] if p['stack'][-1]==r['stack'][-1]];neighbors=set(edge[edge[:,1]==r['stack'][-1],2].tolist());cross=neighbors&set(r['visited'])
    phase='fresh_cross' if not old and cross else 'revisit' if old and r['candidates'] else None
    if phase:choices.setdefault(phase,[]).append((ri,r,neighbors))
   for phase,options in choices.items():
    ri,r,neighbors=min(options,key=lambda x:abs(x[0]-len(recs)/2));prefix=seq[:r['b']+1];edited=prefix[:]
    for p in range(r['a']+1,r['b']):edited[p]=r['visited'][0]
    for condition,tokens in [('clean',prefix),('current_visited_all_source',edited)]:
     out=[]
     for _ in range(12):
      with torch.inference_mode():log=model(torch.tensor(tokens+out)[None])
      t=int(log[0,-1].argmax());out.append(t)
      if t==tok.id('|') or not tok.node_lo<=t<tok.node_lo+tok.n_node_tokens:break
     nodes=out[:-1] if out[-1]==tok.id('|') else out;valid=out[-1]==tok.id('|') and len(nodes)==len(set(nodes));expected=neighbors-{r['visited'][0]}
     results.append(dict(model=name,index=spec['index'],record=ri+1,phase=phase,condition=condition,output=[tok.name(t) for t in out],original_set_correct=valid and set(nodes)==set(r['candidates']),edited_set_correct=valid and set(nodes)==expected,contains_STOP=any('STOP' in tok.name(t) for t in out),note='Only current explicit visited identities change; artificial inconsistent prefix, not a natural-error benchmark.'))
   (OUT/'wrong_visited.json').write_text(json.dumps(results));print(json.dumps(dict(model=name,index=spec['index'],conditions=len(results))),flush=True)
if __name__=='__main__':main()
