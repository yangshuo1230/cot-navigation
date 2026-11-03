"""Linear ridge probes. Tune on query-disjoint validation partitions; test once."""
import json,numpy as np
from pathlib import Path
ROOT=Path(__file__).resolve().parent
ALPHAS=[1e-5,1e-4,1e-3,.01,.1]
def fit(x,y,alpha,w=None):
 if w is None:w=np.ones(len(x))/len(x)
 w=w/w.sum();mean=(x*w[:,None]).sum(0);scale=np.sqrt(((x-mean)**2*w[:,None]).sum(0));scale[scale<1e-5]=1
 z=np.c_[(x-mean)/scale,np.ones(len(x))];penalty=np.eye(z.shape[1])*alpha;penalty[-1,-1]=0
 b=np.linalg.solve(z.T@(z*w[:,None])+penalty,z.T@(y*w[:,None]));coef=b[:-1]/scale[:,None];intercept=b[-1]-mean@coef
 return np.vstack([coef,intercept])
def pred(x,b):return np.c_[x,np.ones(len(x))]@b
def weights(meta):
 ids=[r['index'] for r in meta];return np.array([1/ids.count(ix) for ix in ids])
def score(y,p,meta,classification=False):
 w=weights(meta);w=w/w.sum()
 if classification:
  a=y.argmax(1);b=p.argmax(1);return dict(accuracy=float(np.sum(w*(a==b))),balanced_accuracy=float(np.mean([np.mean(b[a==c]==c) for c in np.unique(a)])))
 mse=((y-p)**2*w[:,None]).sum(0);var=((y-(y*w[:,None]).sum(0))**2*w[:,None]).sum(0)
 return dict(r2=(1-mse/np.maximum(var,1e-9)).tolist(),rmse=np.sqrt(mse).tolist(),rounded_accuracy=((np.rint(p)==y)*w[:,None]).sum(0).tolist())
def main():
 meta=json.loads((ROOT/'probe_metadata.json').read_text());arrays=np.load(ROOT/'probe_activations.npz');out={};saved={};selection=[]
 for kind in ['stack','bars']:
  rows=meta[kind];x=arrays[kind].astype(float);ids=sorted(set(r['index'] for r in rows if r['split']=='val'));dev=set(ids[::3]);train=np.array([r['split']=='val' and r['index'] not in dev for r in rows]);valid=np.array([r['index'] in dev for r in rows]);allval=np.array([r['split']=='val' for r in rows]);test=~allval
  ys=np.array([[r['depth'],r['emitted'],r['remaining']] for r in rows]) if kind=='stack' else np.eye(3)[[r['field'] for r in rows]]
  if kind=='stack':control=np.array([[r['q']/4096,r['record_start']/4096,r['record']/100,r['emitted']] for r in rows])
  else:
   last={};counts={};control=[]
   for r in rows:
    ix=r['index'];control.append([r['q']/4096,(r['q']-last.get(ix,453))/100,counts.get(ix,0)//3/100]);last[ix]=r['q'];counts[ix]=counts.get(ix,0)+1
   control=np.array(control)
  trmeta=[r for r,m in zip(rows,train) if m];vmeta=[r for r,m in zip(rows,valid) if m];temeta=[r for r,m in zip(rows,test) if m];vameta=[r for r,m in zip(rows,allval) if m]
  results=[]
  for name,values in [('position_control',control)]+[(f'L{l}',x[:,l]) for l in range(7)]+[(f'L{l}+position',np.c_[x[:,l],control]) for l in range(7)]:
   candidates=[]
   for alpha in ALPHAS:
    b=fit(values[train],ys[train],alpha,weights(trmeta));p=pred(values[valid],b);s=score(ys[valid],p,vmeta,kind=='bars');utility=s['balanced_accuracy'] if kind=='bars' else s['r2'][0];candidates.append((utility,alpha,s))
   utility,alpha,devscores=max(candidates,key=lambda t:t[0]);b=fit(values[allval],ys[allval],alpha,weights(vameta));p=pred(values[test],b);s=score(ys[test],p,temeta,kind=='bars');results.append(dict(features=name,alpha=alpha,development=devscores,test=s));saved[kind+'_'+name]=b
   if kind=='stack' and name in [f'L{i}' for i in range(1,6)]:selection.append((utility,int(name[1:])))
  out[kind]=dict(targets=['depth','emitted','remaining'] if kind=='stack' else ['stack_end_bar','visited_end_bar','candidate_end_bar'],training_queries=sorted(set(r['index'] for r in trmeta)),development_queries=sorted(dev),refit_queries=ids,test_queries=sorted(set(r['index'] for r in temeta)),results=results)
 chosen=max(selection)[1];out['selected_length_layer']=chosen;out['selection_rule']='Max validation-development R2 of depth from hidden-only L1..L5, before test inspection.'
 np.savez(ROOT/'probe_weights.npz',**saved);(ROOT/'probe_results.json').write_text(json.dumps(out,indent=2));print(json.dumps(dict(selected_length_layer=chosen,stack=out['stack']['results'],bars=out['bars']['results'])))
if __name__=='__main__':main()
