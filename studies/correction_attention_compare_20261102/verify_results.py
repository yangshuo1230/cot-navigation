import gzip,json,collections
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent
def read(n):return json.loads((ROOT/(n+'.json')).read_text(encoding='utf8'))
def main():
 models=read('models');assert len(models)==5 and len({m['checkpoint_sha256'] for m in models})==5
 reference=None;counts={}
 for m in models:
  names=set();indices=collections.defaultdict(set);count=0
  with gzip.open(ROOT/(m['name']+'.events.jsonl.gz'),'rt') as f:
   for line in f:
    r=json.loads(line);g=np.array(r['groups']);assert g.shape==(m['layers'],19);assert np.all(g>=0) and np.max(abs(g.sum(1)-1))<5e-6
    assert len(r['top'])==m['layers']
    for top in r['top']:
     assert all(k['position']<=r['q'] and 0<=k['weight']<=1 for k in top)
    for route in r['routes'].values():assert len(route)==m['layers'] and all(-1e-7<=x<=1+5e-6 for x in route)
    names.add((r['index'],r['split'],r['phase'],r['q']));indices[r['split']].add(r['index']);count+=1
  assert len(indices['val'])==len(indices['test'])==18 and not indices['val']&indices['test']
  if reference is None:reference=names
  else:assert reference==names
  counts[m['name']]=count
 interventions=read('interventions');assert len(interventions)==270;assert all(r['earlier_logits_unchanged'] for r in interventions);assert all(not r['output_changed'] for r in interventions if r['condition']=='sham')
 extras=read('extra_interventions');assert len(extras)==450
 for name in [m['name'] for m in models]:
  for phase in set(r['phase'] for r in extras):
   rs=[r for r in extras if r['model']==name and r['phase']==phase];assert len(rs)==18
 pool=read('bar_pool');assert len(pool)==54 and all(r['baseline_correct'] for r in pool)
 wrong=read('wrong_visited');assert len(wrong)==120
 for key in set((r['model'],r['index'],r['phase']) for r in wrong):assert len([r for r in wrong if (r['model'],r['index'],r['phase'])==key])==2
 writer=read('graph_writer');assert len(writer)==180
 output=dict(passed=True,models=5,unique_query_ids=36,identical_model_event_positions=True,event_counts=counts,attention_normalized=True,top_keys_causal=True,content_swap_conditions=270,extra_route_conditions=450,bar_pool_conditions=54,wrong_visited_conditions=120,graph_writer_queries=180)
 (ROOT/'validation.json').write_text(json.dumps(output,indent=2));print(json.dumps(output))
if __name__=='__main__':main()
