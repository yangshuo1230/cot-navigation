"""Inspect graph reads at the actual current-node tokens, not only the
position predicting a candidate. Distinguishes direct lookup at a decision
from graph information already encoded in earlier token representations.
"""
import runpy,json,time
from pathlib import Path
import numpy as np
import torch
P=Path(__file__).resolve().parent
M=runpy.run_path(str(P/'run.py'))
M['ROLES'].extend(['frontier_current_repr','visited_current_repr'])
ROLES=M['ROLES'];FEATURES=M['FEATURES'];D=M['D'];AUDIT=M['AUDIT']

def add_representations(cases):
 for c in cases:
  additions=[]
  for p in c['probes']:
   if ROLES[p['role']] not in ['candidate_nonempty','candidate_empty']:continue
   rec=p['record'];cur=p['cur']
   for role,q in [('frontier_current_repr',p['frontier_end']-1),
                  ('visited_current_repr',p['vis_span'][0]+rec['visited'].index(cur))]:
    assert c['tokens'][q]==cur
    additions.append(dict(p,role=ROLES.index(role),q=q))
  c['probes'].extend(additions)
 return cases

def main():
 torch.set_num_threads(2)
 for case in ['n16','n32','main']:
  dest=P/('indirect_'+case+'.json')
  if dest.exists():continue
  cfg,tok,model=M['load_model']('clean')
  extra=dict(s.split('=',1) for s in cfg.evaluation.extra_sets)
  source=D['ContextStore'](cfg.data.root if case=='main' else extra[case])
  cache=D['ScalePreparedRows'](source,cfg,tok)
  folder=AUDIT/'strict' if case=='main' else AUDIT/'small'
  saved=[json.loads((folder/('dfs_'+a+'_'+case+'.json')).read_text())['rows'] for a in ['clean','error']]
  ids=[r['query_id'] for r in saved[0]]
  gains={a['query_id'] for a,b in zip(*saved) if not a['correct'] and b['correct']}
  a=source.arrays
  valids=M['stratified_ids'](source.indices('val'),a['shape_id'],a['distance'],a['label'],256,cfg.evaluation.seed)
  val=add_representations(M['build_cases'](source,cache,tok,valids))
  test=add_representations(M['build_cases'](source,cache,tok,ids))
  metrics=[]
  for arm in ['clean','error']:
   if arm=='error':
    del model;torch.cuda.empty_cache();_,_,model=M['load_model']('error')
   mu,count=M['calibrate'](model,val)
   metrics.append(M['observe'](model,test,mu,count))
  M['write'](dest,dict(roles=ROLES,features=FEATURES,attention=M['paired_summary'](*metrics,ids,gains)))
  for arm,values in zip(['clean','error'],metrics):
   for role in ['frontier_current_repr','visited_current_repr']:
    ri=ROLES.index(role);fi=FEATURES.index('relevant_dest_mass')
    print(case,arm,role,np.nanmean(values[:,:,ri,fi],axis=0).tolist(),flush=True)
  del model;torch.cuda.empty_cache()
 M['write'](P/'indirect_finished.json',dict(finished=True))

if __name__=='__main__':main()
