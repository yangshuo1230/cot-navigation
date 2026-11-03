"""CPU-only pairing of existing predictions; never select on test accuracy."""
import runpy, json, math
from pathlib import Path
from collections import Counter
D=runpy.run_path(str(Path(__file__).with_name('run.py')))
FOLDERS,OUT=D['FOLDERS'],D['OUT']

def compare(a,b):
 aa={r['query_id']:r for r in a};bb={r['query_id']:r for r in b}
 assert aa.keys()==bb.keys()
 cells=Counter();by_distance={};by_label={};changes=[]
 for q,x in aa.items():
  y=bb[q]
  assert x['label']==y['label'] and x['distance']==y['distance']
  group='both_correct' if x['correct'] and y['correct'] else 'gain' if y['correct'] else 'loss' if x['correct'] else 'both_wrong'
  cells[group]+=1
  for key,groups in [('label',by_label),('distance',by_distance)]:
   sub=groups.setdefault(str(x[key]),Counter())
   sub[group]+=1
  if group in ['gain','loss']:changes.append(dict(query_id=q,type=group,label=x['label'],distance=x['distance'],old_pred=x['pred'],new_pred=y['pred']))
 n=cells['gain']+cells['loss']
 k=min(cells['gain'],cells['loss'])
 p=min(1,2*sum(math.comb(n,i) for i in range(k+1))/2**n) if n else 1
 return dict(n=len(aa),cells=dict(cells),by_label=by_label,by_distance=by_distance,changes=changes,
             mcnemar_exact_p=p,note='Within-test paired p only; not a multi-seed training significance test; unadjusted for multiple comparisons.')

def main():
 result={}
 for protocol in ['dfs','bfs']:
  clean,error=FOLDERS[protocol+'_clean'],FOLDERS[protocol+'_error']
  for checkpoint in ['best','last']:
   for case in ['n16','n24','n32','n36','n64','main','deep','wide','larger']:
    suffix=checkpoint+('' if case=='main' else '-'+case)
    a,b=[list(map(json.loads,(folder/'eval'/('predictions_test_'+suffix+'.jsonl')).read_text().splitlines())) for folder in [clean,error]]
    result[protocol+'_'+suffix]=compare(a,b)
 D['write'](OUT/'paired_saved.json',result)
 for k,v in result.items():print(k,v['cells'],v['mcnemar_exact_p'])

if __name__=='__main__':main()
