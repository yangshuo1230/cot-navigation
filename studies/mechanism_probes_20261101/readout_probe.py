"""Calibrated linear readout to distinguish decodability from actual L6 use."""
import json,numpy as np
from pathlib import Path
from probe import fit,pred,weights,ALPHAS
ROOT=Path(__file__).resolve().parent
def main():
 data=np.load(ROOT/'readout_activations.npz');x=data['features'].astype(float);y=data['logits'].astype(float);meta=json.loads((ROOT/'readout_metadata.json').read_text());ids=sorted(set(r['index'] for r in meta if r['split']=='val'));dev=set(ids[::3]);tr=np.array([r['split']=='val' and r['index'] not in dev for r in meta]);dv=np.array([r['index'] in dev for r in meta]);va=np.array([r['split']=='val' for r in meta]);te=~va
 trm=[r for r,m in zip(meta,tr) if m];vam=[r for r,m in zip(meta,va) if m];out={};predictions={}
 for li,layer in enumerate([5,6]):
  z=x[:,li];z=(z-z.mean(1,keepdims=True))/np.sqrt(z.var(1,keepdims=True)+1e-5);candidates=[]
  for alpha in ALPHAS:
   b=fit(z[tr],y[tr],alpha,weights(trm));p=pred(z[dv],b);candidates.append((float(np.mean(p.argmax(1)==y[dv].argmax(1))),alpha))
  utility,alpha=max(candidates);b=fit(z[va],y[va],alpha,weights(vam));prediction=pred(z[te],b).argmax(1);predictions['L'+str(layer)]=prediction
  rows=[r for r,m in zip(meta,te) if m];phases={}
  for phase in sorted(set(r['phase'] for r in rows)):
   positions=[i for i,r in enumerate(rows) if r['phase']==phase];groups={}
   for i in positions:groups.setdefault(rows[i]['index'],[]).append(i)
   phases[phase]=dict(n_queries=len(groups),n_events=len(positions),agrees_final=float(np.mean([np.mean([prediction[i]==rows[i]['final'] for i in ps]) for ps in groups.values()])))
  out['L'+str(layer)]=dict(alpha=alpha,development_agreement=utility,test_phases=phases)
 (ROOT/'readout_probe_results.json').write_text(json.dumps(out,indent=2));print(json.dumps(out))
if __name__=='__main__':main()
