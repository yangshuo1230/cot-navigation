"""Query-split visited membership probes at PUSH/action and argument sites."""
import sys,json,numpy as np
from pathlib import Path
ROOT=Path(__file__).resolve().parent
from ridge import fit,pred,weights,ALPHAS
def evaluations(scores,meta,threshold=None):
 groups={};neighbor={};ba={};identity=[]
 for i,r in enumerate(meta):
  excluded={r['source'],r['pushed']};graph=[v for v in r['graph'] if v not in excluded];visited=set(r['visited']);positive=[v for v in graph if v in visited];negative=[v for v in graph if v not in visited]
  def auc(pos,neg):
   a=scores[i,pos][:,None];b=scores[i,neg][None];return float(((a>b)+.5*(a==b)).mean())
  if positive and negative:
   groups.setdefault(r['index'],[]).append(auc(positive,negative))
   if threshold is not None:ba.setdefault(r['index'],[]).append(.5*(np.mean(scores[i,positive]>=threshold)+np.mean(scores[i,negative]<threshold)))
  pos=[v for v in r['neighbors'] if v in visited and v not in excluded];neg=[v for v in r['neighbors'] if v not in visited and v not in excluded]
  if pos and neg:neighbor.setdefault(r['index'],[]).append(auc(pos,neg))
 out=dict(auc=float(np.mean([np.mean(v) for v in groups.values()])),n_query=len(groups),n_states=sum(map(len,groups.values())),neighbor_auc=None if not neighbor else float(np.mean([np.mean(v) for v in neighbor.values()])),neighbor_queries=len(neighbor),neighbor_states=sum(map(len,neighbor.values())))
 if threshold is not None:out['balanced_accuracy']=float(np.mean([np.mean(v) for v in ba.values()]))
 return out
def main():
 result={};selection={}
 for name in ['corr9_s0','corr9_s1']:
  meta=json.loads((ROOT/(name+'.push_metadata.json')).read_text());features=np.load(ROOT/(name+'.push.npz'))['features'].astype(float);y=np.zeros((len(meta),256));identity=np.zeros_like(y)
  for i,r in enumerate(meta):y[i,r['visited']]=1;identity[i,r['pushed']]=1
  count=np.array([[r['count']] for r in meta]);full_y=np.c_[y,count,identity]
  ids=sorted(set(r['index'] for r in meta if r['split']=='val'));dev=set(ids[::3]);tr=np.array([r['split']=='val' and r['index'] not in dev for r in meta]);dv=np.array([r['index'] in dev for r in meta]);va=np.array([r['split']=='val' for r in meta]);te=~va
  mm=lambda mask:[r for r,m in zip(meta,mask) if m];trm,dvm,vam,tem=map(mm,[tr,dv,va,te]);allweights={};curves=[];candidates=[]
  for site,site_name in enumerate(['action','argument']):
   for layer in range(10):
    x=features[:,layer,site];best=None
    for alpha in ALPHAS:
     b=fit(x[tr],y[tr],alpha,weights(trm));p=pred(x[dv],b);m=evaluations(p,dvm)
     if best is None or m['auc']>best[0]:best=(m['auc'],alpha,p)
    utility,alpha,devpred=best;threshold=max(np.linspace(.01,.5,26),key=lambda t:evaluations(devpred,dvm,t)['balanced_accuracy'])
    b=fit(x[va],full_y[va],alpha,weights(vam));allweights[site_name+'_L'+str(layer)]=b;p=pred(x[te],b);metrics=evaluations(p[:,:256],tem,threshold);metrics['identity_accuracy']=float(np.mean(p[:,257:].argmax(1)==np.array([r['pushed'] for r in tem])))
    curves.append(dict(site=site_name,layer=layer,alpha=alpha,threshold=float(threshold),development_auc=utility,test=metrics))
    if 1<=layer<=8:candidates.append((utility,site_name,layer,alpha,float(threshold)))
  utility,site_name,layer,alpha,threshold=max(candidates);selection[name]=dict(site=site_name,layer=layer,alpha=alpha,threshold=threshold,development_auc=utility,rule='Max development AUC over both sites and L1..L8. L9 excluded: a past final-block residual cannot influence later positions.')
  # Selected-site baseline with explicit size/progress plus pushed identity.
  site=0 if site_name=='action' else 1;x=np.c_[features[:,0,1],count,np.array([[r['argument_position']/4096] for r in meta])];b=fit(x[va],y[va],alpha,weights(vam));control=evaluations(pred(x[te],b),tem)
  shuffled=y.copy();rng=np.random.default_rng(20261102)
  for i,r in enumerate(meta):
   shuffled[i]=0;pool=[v for v in r['graph'] if v!=r['pushed']];shuffled[i,rng.choice(pool,size=r['count'],replace=False)]=1
  x=features[:,layer,site];b=fit(x[va],shuffled[va],alpha,weights(vam));null=evaluations(pred(x[te],b),tem)
  result[name]=dict(curves=curves,selected=selection[name],control_size_position_identity=control,shuffled_membership_labels=null,train_queries=sorted(set(r['index'] for r in trm)),dev_queries=sorted(dev),refit_queries=ids,test_queries=sorted(set(r['index'] for r in tem)),n_states=len(meta),definition='Target is visited BEFORE this PUSH. Evaluate within each state among graph nodes, excluding source and pushed node; then average states within query and queries. Neighbor AUC uses only states with both visited and unvisited successors.')
  np.savez_compressed(ROOT/(name+'.probe_weights.npz'),**allweights);(ROOT/'probe_results.json').write_text(json.dumps(result,indent=2));(ROOT/'selected_probe.json').write_text(json.dumps(selection,indent=2));print(json.dumps(dict(model=name,selected=selection[name],selected_test=next(r['test'] for r in curves if r['site']==site_name and r['layer']==layer),control=control,null=null)),flush=True)
if __name__=='__main__':main()
