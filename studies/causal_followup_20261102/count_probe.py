"""Earlier-bar count readout with held-out queries and position controls."""
import sys,json,numpy as np
from pathlib import Path
ROOT=Path(__file__).resolve().parent;PREV=ROOT.parent/'mechanism_probes_20261101';sys.path.insert(0,str(PREV))
from probe import fit,pred,weights,score,ALPHAS
def main():
 rows=json.loads((PREV/'probe_metadata.json').read_text())['bars'];x=np.load(PREV/'probe_activations.npz')['bars'].astype(float);counts={};last={};y=[];control=[]
 for r in rows:
  ix=r['index'];n=counts.get(ix,0);y.append([n+1]);control.append([r['q']/4096,(r['q']-last.get(ix,453))/100,np.log1p(r['q']-453)]);counts[ix]=n+1;last[ix]=r['q']
 y=np.array(y);control=np.array(control);ids=sorted(set(r['index'] for r in rows if r['split']=='val'));dev=set(ids[::3]);tr=np.array([r['split']=='val' and r['index'] not in dev for r in rows]);dv=np.array([r['index'] in dev for r in rows]);va=np.array([r['split']=='val' for r in rows]);te=~va
 trm=[r for r,m in zip(rows,tr) if m];dvm=[r for r,m in zip(rows,dv) if m];vam=[r for r,m in zip(rows,va) if m];tem=[r for r,m in zip(rows,te) if m];out=[]
 for name,z in [('position',control)]+[(f'L{i}',x[:,i]) for i in range(7)]:
  scores=[]
  for alpha in ALPHAS:
   b=fit(z[tr],y[tr],alpha,weights(trm));s=score(y[dv],pred(z[dv],b),dvm);scores.append((s['r2'][0],alpha))
  _,alpha=max(scores);b=fit(z[va],y[va],alpha,weights(vam));p=pred(z[te],b);s=score(y[te],p,tem)
  truephase=((y[te,0]-1)%3).astype(int);predphase=((np.rint(p[:,0])-1)%3).astype(int)
  w=weights(tem);w/=w.sum();s['rounded_count_mod3_accuracy']=float(((truephase==predphase)*w).sum());out.append(dict(features=name,alpha=alpha,test=s))
 (ROOT/'count_probe.json').write_text(json.dumps(out,indent=2));print(json.dumps(out))
if __name__=='__main__':main()
