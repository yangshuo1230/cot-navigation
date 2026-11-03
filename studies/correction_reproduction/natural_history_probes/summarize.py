"""Primary natural-test results, own-generation tests and final type controls."""
from collections import defaultdict
import json
from pathlib import Path
import numpy as np
from cot_navigation.experiments.artifacts import atomic_json

P=Path(__file__).resolve().parent
METRICS=['auroc','paired_rank','positive_p','negative_p','bce','brier','tpr','fpr']


def main():
    blobs=[json.loads(p.read_text()) for p in sorted((P/'results').glob('*_s*.json'))]
    assert len(blobs)==8 and all(d['complete'] and d['checkpoint_unchanged'] for d in blobs)
    grouped=defaultdict(list);curves=defaultdict(list)
    for d in blobs:
        for run in d['runs']:
            groups={**run['tests'],**{'natural_type/'+k:v for k,v in run['by_first_type'].items()},
                **{'field/'+k:v for k,v in run['by_field'].items()},
                **{'artificial/'+k:v for k,v in run['artificial_controls'].items()}}
            for ds,rr in groups.items():
                per_cell=defaultdict(list)
                for r in rr:
                    per_cell[(r['record'],r['head'])].append(r)
                    if 'token_offset' in r and ds in ['test','own_test']:
                        curves[(d['arm'],run['architecture'],ds,r['record'],r['head'],r['token_offset'])].append(r)
                for (phase,head),values in per_cell.items():
                    mean={m:float(np.mean([v[m] for v in values if v.get(m) is not None])) for m in METRICS}
                    grouped[(d['arm'],run['architecture'],ds,phase,head)].append(
                        dict(model_seed=d['model_seed'],probe_seed=run['probe_seed'],cells=len(values),**mean))
        for r in d['shuffle_control']:
            grouped[(d['arm'],'mlp64','shuffle_control',r['record'],r['head'])].append(
                dict(model_seed=d['model_seed'],probe_seed=0,**r))
    aggregate=[]
    for key,values in sorted(grouped.items()):
        by_seed=[]
        for seed in sorted({v['model_seed'] for v in values}):
            rr=[v for v in values if v['model_seed']==seed]
            by_seed.append(dict(model_seed=seed,**{m:float(np.mean([r[m] for r in rr])) for m in METRICS}))
        aggregate.append(dict(zip(['arm','architecture','dataset','record','head'],key)))
        aggregate[-1].update({m:float(np.mean([v[m] for v in by_seed])) for m in METRICS})
        aggregate[-1].update(per_model=by_seed,models=len(by_seed),
            auroc_seed_std=float(np.std([v['auroc'] for v in by_seed],ddof=1)) if len(by_seed)>1 else None)
    cc=[]
    for key,rr in sorted(curves.items()):
        row=dict(zip(['arm','architecture','dataset','record','head','token_offset'],key))
        row.update({m:float(np.mean([r[m] for r in rr if r.get(m) is not None])) for m in METRICS})
        row['min_queries']=min(r['queries'] for r in rr);cc.append(row)
    audits={}
    for name in ['verification','readout_verification','native_behavior']:
        path=P/f'{name}.json'
        if path.exists():audits[name]=json.loads(path.read_text())
    assert audits.get('verification',{}).get('passed') and audits.get('readout_verification',{}).get('passed')
    atomic_json(P/'summary.json',dict(complete=True,models=8,aggregates=aggregate,curves=cc,audits=audits,
        shared_sampling=blobs[0]['natural_sampling'],
        own_sampling={f"{d['arm']}_s{d['model_seed']}":d['own_sampling'] for d in blobs},
        artificial_counts=blobs[0]['artificial_control_counts'],
        checkpoints_unchanged=True,total_probe_seconds=sum(d['seconds'] for d in blobs)))
    print('NATURAL SUMMARY COMPLETE',len(aggregate),flush=True)


if __name__=='__main__':main()
