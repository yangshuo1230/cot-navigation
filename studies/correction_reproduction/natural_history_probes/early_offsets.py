"""Incremental natural probes at first-error token + offsets 0..5, GPU0 only.

The original record-aligned results stay unchanged. Frozen adapters, natural
matching, feature extraction and independent per-cell probe fitting are reused.
No navigation training, edited primary text, activation caches or probe weights.
"""
import argparse
from collections import defaultdict
import gc
import json
import os
from pathlib import Path
import runpy
import time

os.environ['CUDA_VISIBLE_DEVICES']='0'
os.environ['WANDB_MODE']='disabled'
P=Path(__file__).resolve().parent
R=runpy.run_path(str(P/'run.py'))
import numpy as np
import torch
from cot_navigation.analysis.natural_history import match_readouts
from cot_navigation.analysis.history_probe import fit_positions,position_scores
from cot_navigation.experiments.artifacts import atomic_json,sha_file

METRICS=['auroc','paired_rank','positive_p','negative_p','bce','brier','tpr','fpr']


def coverage(plans):
    return {split:[dict(phase=phase,offset=offset,
        positive_queries=len({r['query'] for r in rows if r['phase']==phase and r['offset']==offset and r['y']==1}),
        negative_queries=len({r['query'] for r in rows if r['phase']==phase and r['offset']==offset and r['y']==0}))
        for phase in range(3) for offset in range(6)] for split,rows in plans.items()}


def check_rows(cases,rows,tok):
    pairs=defaultdict(list)
    for r in rows:
        c=cases[r['sid']];p=r['pos']
        assert 0<=p<len(c['trace'])
        assert r['y']==int(c['tau'] is not None and p>=c['tau'])
        if r['y']:assert p-c['tau']==r['offset']
        assert c['fields'][p]==r['field'] and c['source']==r['source']
        pairs[r['event'],r['cell']].append(r)
    for pair in pairs.values():
        assert len(pair)==2 and {r['y'] for r in pair}=={0,1}
        assert len({r['source'] for r in pair})==1
        assert len({r['query'] for r in pair})==2
        actual_offsets=[]
        for r in pair:
            c=cases[r['sid']]
            begin=next(a for a,b in c['spans'] if a<=r['pos']<b)
            actual_offsets.append(r['pos']-begin)
        assert len(set(actual_offsets))==1
        if pair[0]['exact_token']:
            assert len({cases[r['sid']]['trace'][r['pos']] for r in pair})==1


def summarize(blobs,record_coverage,aligned_coverage,identity):
    grouped=defaultdict(list)
    for d in blobs:
        for run in d['runs']:
            for ds,rr in run['tests'].items():
                for r in rr:
                    grouped[d['arm'],run['architecture'],ds,r['head'],r['token_offset']].append(
                        dict(model_seed=d['model_seed'],probe_seed=run['probe_seed'],**r))
    curves=[]
    for key,values in sorted(grouped.items()):
        by_seed=[]
        for seed in sorted({r['model_seed'] for r in values}):
            rr=[r for r in values if r['model_seed']==seed]
            by_seed.append(dict(model_seed=seed,**{m:float(np.mean([v[m] for v in rr])) for m in METRICS}))
        row=dict(zip(['arm','architecture','dataset','head','token_offset'],key))
        row.update({m:float(np.mean([r[m] for r in by_seed])) for m in METRICS})
        row.update(per_model=by_seed,models=len(by_seed),
            auroc_seed_std=float(np.std([r['auroc'] for r in by_seed],ddof=1)) if len(by_seed)>1 else None,
            min_queries=min(r['queries'] for r in values),
            min_positive_queries=min(r['positive'] for r in values))
        curves.append(row)
    assert all(any(r['dataset']=='test' and r['head']==h and r['token_offset']==o
                   and r['arm']==arm and r['architecture']==arch for r in curves)
               for h in range(8) for o in range(6) for arm in ['matched_clean','correction']
               for arch in ['linear','mlp64'])
    result=dict(complete=True,identity=identity,alignment='first_error_token',offsets=list(range(6)),
        target='Past-or-current semantic error after consuming first-error token + offset; before MISTAKE',
        record_coverage=record_coverage,aligned_coverage=aligned_coverage,curves=curves,
        audits=dict(readouts_checked=True,official_graph_query_splits_disjoint=True,
            checkpoints_unchanged=all(d['checkpoint_unchanged'] for d in blobs),
            unedited_natural_only=True),seconds=sum(d['seconds'] for d in blobs))
    atomic_json(P/'early_offsets_summary.json',result)
    print('EARLY SUMMARY COMPLETE',len(curves),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--steps',type=int,default=5000)
    args=parser.parse_args();os.chdir(R['ROOT']);torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.cuda.set_per_process_memory_fraction(.32)
    identity=dict(steps=args.steps,probe_seeds=[0,1],offsets=list(range(6)),
        source_sha256={str(p.relative_to(R['ROOT'])):sha_file(p) for p in [Path(__file__),
            R['ROOT']/'src/cot_navigation/analysis/natural_history.py',
            R['ROOT']/'src/cot_navigation/analysis/history_probe.py',P/'run.py']},
        corpus_sha256={p.name:sha_file(p) for p in sorted((P/'native').glob('*.json.gz'))})
    atomic_json(P/'early_offsets_plan.json',dict(identity,gpu=0,alignment='first_error_token'))
    dest=P/'early_offsets_results';dest.mkdir(exist_ok=True)
    cases=None;blobs=[]
    for seed in range(4):
        for arm in ['matched_clean','correction']:
            started=time.time();path=dest/f'{arm}_s{seed}.json'
            existing=json.loads(path.read_text()) if path.exists() else None
            am=None
            if cases is None:
                am=R['Model'](arm,seed)
                cases,record_plans,info=R['load_data'](am.tok)
                plans={};sampling={}
                for j,split in enumerate(['train','val','test']):
                    plans[split],sampling[split]=match_readouts(cases[split],am.tok,20261030+j,error_offsets=range(6))
                    check_rows(cases[split],plans[split],am.tok)
                record_coverage=coverage(record_plans);aligned_coverage=coverage(plans)
                print('EARLY COVERAGE',aligned_coverage,flush=True)
            if existing and existing.get('complete'):
                assert existing['identity']==identity,'Changed study identity; use a separate output folder'
                assert sha_file(Path(existing['checkpoint']['path']))==existing['checkpoint']['sha256']
                blobs.append(existing);del am;gc.collect();torch.cuda.empty_cache();continue
            if am is None:am=R['Model'](arm,seed)
            own,own_sampling=match_readouts(cases['test'],am.tok,20261102,
                positive_source=f'{arm}_s{seed}',error_offsets=range(6))
            check_rows(cases['test'],own,am.tok)
            data={s:R['extract'](am,cases[s],plans[s]) for s in ['train','val','test']}
            data['own_test']=R['extract'](am,cases['test'],own)
            result=dict(complete=False,identity=identity,arm=arm,model_seed=seed,
                checkpoint=am.checkpoint,sampling=sampling,own_sampling=own_sampling,runs=[])
            del am;gc.collect();torch.cuda.empty_cache()
            for hidden,arch in [(0,'linear'),(64,'mlp64')]:
                for ps in [0,1]:
                    fitted=fit_positions(data['train'],data['val'],args.steps,ps,hidden=hidden)
                    assert fitted is not None and set(fitted['mapping'])==set(range(6))
                    tests={s:position_scores(fitted,data[s]) for s in ['test','own_test']}
                    exact=R['subset'](data['test'],data['test']['exact_token'].astype(bool))
                    tests['exact_token_test']=position_scores(fitted,exact)
                    result['runs'].append(dict(architecture=arch,probe_seed=ps,tests=tests,
                        selected_steps=fitted['selected_steps'],val_bce=fitted['validation_bce']))
                    del fitted;gc.collect();torch.cuda.empty_cache()
                    print('EARLY FIT',arm,seed,arch,ps,round(time.time()-started),flush=True)
            shuffled={}
            for j,split in enumerate(['train','val','test']):
                d=data[split];rng=np.random.default_rng(4477+j)
                flips={int(e):int(rng.integers(2)) for e in np.unique(d['event'])}
                shuffled[split]={**d,'y':np.array([int(y)^flips[int(e)] for y,e in zip(d['y'],d['event'])],np.float32)}
            fitted=fit_positions(shuffled['train'],shuffled['val'],args.steps,0,hidden=64)
            result['runs'].append(dict(architecture='mlp64',probe_seed=0,
                tests={'shuffle_control':position_scores(fitted,shuffled['test'])}))
            result.update(complete=True,seconds=time.time()-started,
                checkpoint_unchanged=sha_file(Path(result['checkpoint']['path']))==result['checkpoint']['sha256'])
            assert result['checkpoint_unchanged'];atomic_json(path,result);blobs.append(result)
            del data,shuffled,fitted;gc.collect();torch.cuda.empty_cache()
    summarize(blobs,record_coverage,aligned_coverage,identity)


if __name__=='__main__':main()
