import json,gzip
from pathlib import Path
import numpy as np
from summarize import summary,selector
ROOT=Path(__file__).resolve().parent
SPECS=[('init_roots','stack_init_PUSH','all_stack_roots','supp'),('visited_direct','visited_continue','previous_same_field_target','main'),('fresh_argument','candidate_init_fresh','previous_push_argument','main'),('fresh_raw_dest','candidate_init_fresh','current_adjacency_raw_destinations','main'),('revisit_tail','candidate_init_revisit','old_candidate_tail','supp'),('fresh_last_edge','candidate_init_fresh','last_edge_marker','supp')]
def main():
 out={}
 for model in json.loads((ROOT/'models.json').read_text()):
  name=model['name'];layers=model['layers'];out[name]={};data={}
  for tag,suffix in [('main','.events.jsonl.gz'),('supp','.supplement.jsonl.gz')]:
   with gzip.open(ROOT/(name+suffix),'rt') as f:data[tag]=[json.loads(x) for x in f]
  for title,phase,route,tag in SPECS:
   val=selector([r for r in data[tag] if r['split']=='val'],phase,None);test=selector([r for r in data[tag] if r['split']=='test'],phase,None);vs=summary(val,lambda r:r['routes'].get(route,[0.]*layers));ts=summary(test,lambda r:r['routes'].get(route,[0.]*layers));layer=int(np.argmax(vs['mean']))
   out[name][title]=dict(layer=layer+1,phase=phase,route=route,test=ts,val=vs)
  phase='candidate_init_fresh';rows=data['supp'];val=[r for r in rows if r['split']=='val' and r['phase']==phase];test=[r for r in rows if r['split']=='test' and r['phase']==phase];best=None
  for offset in range(-6,7):
   route='adjacency_offset_'+str(offset);vs=summary(val,lambda r:r['routes'][route]);ts=summary(test,lambda r:r['routes'][route]);layer=int(np.argmax(vs['mean']));v=vs['mean'][layer]
   if best is None or v>best['val_peak']:best=dict(offset=offset,layer=layer+1,val_peak=v,test_peak=ts['mean'][layer])
  out[name]['offset_scan']=best
 (ROOT/'extra_routes.json').write_text(json.dumps(out,indent=2));print(json.dumps({n:{k:dict(layer=v['layer'],mean=v['test']['mean'][v['layer']-1]) if k!='offset_scan' else v for k,v in rr.items()} for n,rr in out.items()}))
if __name__=='__main__':main()
