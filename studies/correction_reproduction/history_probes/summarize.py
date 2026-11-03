"""Compact all-token probe summary, retaining model/probe seed distinctions."""
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
from cot_navigation.experiments.artifacts import atomic_json,sha_file

P=Path(__file__).resolve().parent


def main():
    blobs=[json.loads(p.read_text()) for p in sorted((P/'results_paired_v2').glob('*_s*.json'))]
    assert len(blobs)==8 and all(d['finished'] for d in blobs)
    rows=[];checks=[]
    for d in blobs:
        assert d['evaluation_schema'].startswith('paired_after_error_v2')
        groups=[set(d['sampling'][s]['graph_hashes']) for s in ['train','val','test']]
        assert not(groups[0]&groups[1] or groups[0]&groups[2] or groups[1]&groups[2])
        checks.append(dict(arm=d['arm'],model_seed=d['seed'],seconds=d['seconds'],
            checkpoint_unchanged=sha_file(Path(d['checkpoint']['path']))==d['checkpoint']['sha256'],
            pre_error_max_difference=max(d['pre_error_max_difference'].values())))
        for run in d['runs']:
            for name,scores in run['tests'].items():
                for r in scores:
                    assert r['positive']*2==r['n']
                    rows.append(dict(arm=d['arm'],model_seed=d['seed'],probe_seed=run['probe_seed'],
                                     architecture=run['architecture'],dataset=name,**r))
        for name,scores in [('graph_trained',d['graph_trained']['scores']),('shuffle_control',d['shuffle_control'])]:
            for r in scores:
                assert r['positive']*2==r['n']
                rows.append(dict(arm=d['arm'],model_seed=d['seed'],probe_seed=0,
                                 architecture='mlp64_independent',dataset=name,**r))
    aggregate=[]
    grouped=defaultdict(list)
    for r in rows:grouped[(r['arm'],r['architecture'],r['dataset'],r['record'],r['head'])].append(r)
    keys=sorted(grouped)
    metrics=['auroc','paired_rank','positive_p','negative_p','bce','brier','tpr','fpr']
    for arm,arch,ds,phase,head in keys:
        rr=grouped[(arm,arch,ds,phase,head)]
        # Position macro-average, then probe-seed average, then model-seed average.
        per_model=[]
        for seed in range(4):
            sr=[r for r in rr if r['model_seed']==seed]
            item=dict(model_seed=seed,cells_per_probe={str(ps):len([r for r in sr if r['probe_seed']==ps]) for ps in sorted(set(r['probe_seed'] for r in sr))})
            for metric in metrics:
                vals=[np.mean([r[metric] for r in sr if r['probe_seed']==ps and r.get(metric) is not None])
                      for ps in sorted(set(r['probe_seed'] for r in sr))]
                item[metric]=float(np.mean(vals))
            per_model.append(item)
        aggregate.append(dict(arm=arm,architecture=arch,dataset=ds,record=phase,head=head,
            **{m:float(np.mean([r[m] for r in per_model])) for m in metrics},
            auroc_seed_std=float(np.std([r['auroc'] for r in per_model],ddof=1)),per_model=per_model))
    curves=[]
    cgroups=defaultdict(list)
    for r in rows:
        if r['architecture']=='mlp64_independent':
            cgroups[(r['arm'],r['dataset'],r['record'],r['head'],r['token_offset'])].append(r)
    ckeys=sorted(cgroups)
    for key in ckeys:
        rr=cgroups[key]
        curves.append(dict(zip(['arm','dataset','record','head','token_offset'],key)))
        curves[-1].update(auroc=float(np.mean([r['auroc'] for r in rr])),
            paired_rank=float(np.mean([r['paired_rank'] for r in rr])),
            positive_p=float(np.mean([r['positive_p'] for r in rr])),
            negative_p=float(np.mean([r['negative_p'] for r in rr])),
            min_queries=min(r['queries'] for r in rr))
    assert all(c['checkpoint_unchanged'] for c in checks)
    atomic_json(P/'SUMMARY_PAIRED_V2.json',dict(finished=True,models=8,total_seconds=sum(c['seconds'] for c in checks),
        data_pairs=dict(train=768,val=192,test=217,natural_donor_test=40),
        caveats=['positive-only ordinary N100; non-PUSH candidate edits with shared legal continuations',
                 'natural donor errors are not the probed model own free-generation errors',
                 'AUROC values are position macro-averages, not answer accuracy or universal calibrated error probabilities',
                 'minimum test support 8 queries per cell; report curves expose support counts'],
        checks=checks,aggregates=aggregate,curves=curves))
    print('Finished: 8/8; unchanged checkpoints; graph-disjoint splits; balanced score cells.')


if __name__=='__main__':main()
