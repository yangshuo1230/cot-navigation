"""Calibrate ONLY on early, fully legal natural records; keep held-out tests.

Early means the first four records, defined without a future length fraction.
Only train/val correctness selects calibration rows. All original test readouts
are evaluated, including errors and later positions, without correctness filters.
"""
import gc
import json
from pathlib import Path
import runpy
import time

P=Path(__file__).resolve().parent
R=runpy.run_path(str(P/'run.py'))
np=R['np'];torch=R['torch'];ROOT=R['ROOT']
from cot_navigation.analysis.common import records
from cot_navigation.analysis.within_token import fit_lens
from cot_navigation.experiments.artifacts import atomic_json,sha_file


def early_legal_rows(am,cases,selected):
    """Select whole legal records before first illegal token."""
    limits={}
    for sid,c in enumerate(cases):
        # Semantic replay first.token is one-based, trace-relative.
        tau=c['first']['token']-1 if c['first'] is not None else len(c['trace'])
        limits[sid]={ri for ri,r in enumerate(records(c['trace'],0,am.tok)) if ri<4 and r['end']<tau}
    rows=[r for r in selected if r['record'] in limits[r['sid']]]
    # Repeated anchors (bar/state) must not receive extra calibration weight.
    unique={}
    for r in rows:unique.setdefault((r['sid'],r['pos']),r)
    rows=list(unique.values())
    assert rows
    return rows


def test_partitions(am,cases,rows):
    """Post-hoc test strata, never used to fit/select calibration parameters.

    Preserve original anchor multiplicities in every stratum. Correct means
    the whole record precedes the first error, not just a locally legal token.
    """
    rr={sid:records(c['trace'],0,am.tok) for sid,c in enumerate(cases)}
    correct=np.array([rr[r['sid']][r['record']]['end'] < (
        cases[r['sid']]['first']['token']-1 if cases[r['sid']]['first'] is not None
        else len(cases[r['sid']]['trace'])) for r in rows])
    early=np.array([r['record']<4 for r in rows])
    return dict(early_correct=early&correct,early_all=early,
                correct_record=correct,later_correct=(~early)&correct)


def fit_calibration(am,cases,selected,steps=1500):
    """Reproduce the frozen early/correct calibration; test data are untouched."""
    part={};support={};rng=np.random.default_rng(20261102)
    for split,cap in [('train',4096),('val',2048)]:
        pool=early_legal_rows(am,cases[split],selected[split])
        take=rng.permutation(len(pool))[:cap];rows=[pool[i] for i in take]
        feature=R['extract'](am,cases[split],rows)
        h=feature['x'][:,:am.model.n_layers,:am.model.d_model].copy()
        with torch.no_grad():
            z=am.model.lm_head(am.model.final_norm(torch.as_tensor(h[:,-1],device=am.device))).cpu().numpy()
        keep=z.argmax(-1)==feature['next'];assert keep.any()
        part[split]=(h[keep],z[keep]);fitted_rows=[r for r,k in zip(rows,keep) if k]
        support[split]=dict(eligible_unique_positions=len(pool),sampled_positions=len(take),
            fitted_positions=int(keep.sum()),receiver_correct_fraction=float(keep.mean()),
            distinct_queries=len({cases[split][r['sid']]['query'] for r in fitted_rows}),
            record_counts={str(ri):sum(r['record']==ri for r in fitted_rows) for ri in range(4)})
    lens=fit_lens(am.model,*part['train'],*part['val'],steps=steps,seed=20261102)
    return lens,support


def main():
    import os
    os.chdir(ROOT);torch.set_num_threads(2)
    parent=ROOT/'outputs/analysis/state_readouts/legacy6_s0123_supplement_20261102'
    out=parent/'early_clean_lens';out.mkdir(parents=True,exist_ok=True)
    plan=json.loads((parent/'plan.json').read_text())
    manifest=json.loads((ROOT/plan['args']['manifest']).read_text())
    config=dict(schema='early_clean_lens_v1',parent_plan_sha256=sha_file(parent/'plan.json'),
        fit='first four records AND wholly before first semantic error AND receiver predicts the legal natural next token; unique positions',
        validation='same early/legal filter, validation graphs only',
        test='unchanged original full selected natural windows, no correctness filtering',
        steps=plan['args']['steps'],maximum_train_positions=4096,maximum_val_positions=2048)
    if (out/'plan.json').exists():assert json.loads((out/'plan.json').read_text())==config
    atomic_json(out/'plan.json',config)
    for model in manifest['models']:
        dest=out/(model['id']+'.json')
        previous=json.loads(dest.read_text()) if dest.exists() else None
        if previous is not None and 'test_subsets' in previous:continue
        begin=time.time();print('MODEL',model['id'],flush=True)
        old=json.loads((parent/(model['id']+'.json')).read_text())
        am=R['R']['Model'](model['training_arm'],model['seed'])
        assert am.checkpoint==old['checkpoint']
        cases,sources=R['load_cases'](model['training_arm'],model['seed'],plan['args']['train_queries'],plan['args']['val_queries'])
        assert sources==old['sources']
        selected=R['select'](am,cases)
        lens,support=fit_calibration(am,cases,selected,config['steps'])
        test=R['extract'](am,cases['test'],selected['test'])
        result=R['lens_scores'](am,lens,test,selected['test'])
        if previous is not None:
            # Regenerate the discarded affine maps with the exact frozen plan.
            # Verify the original full-panel result is reproduced before adding
            # any conditional test results.
            for a,b in zip(result,previous['logit_lens']):
                assert (a['task'],a['kind'],a['layer'],a['n'])==(b['task'],b['kind'],b['layer'],b['n'])
                assert abs(a['final_agreement']-b['final_agreement'])<1e-7
        subsets={}
        for name,mask in test_partitions(am,cases['test'],selected['test']).items():
            ix=np.flatnonzero(mask);subrows=[selected['test'][i] for i in ix]
            subfeatures={k:v[ix] for k,v in test.items()}
            subsets[name]=dict(positions=len(ix),queries=len({cases['test'][r['sid']]['query'] for r in subrows}),
                logit_lens=R['lens_scores'](am,lens,subfeatures,subrows))
        # Raw lens is unchanged by calibration. Check the identical test panel.
        for a,b in zip(result,old['logit_lens']):
            assert (a['task'],a['kind'],a['layer'],a['n'])==(b['task'],b['kind'],b['layer'],b['n'])
            if a['kind']=='raw':assert abs(a['final_agreement']-b['final_agreement'])<1e-8
        atomic_json(dest,dict(model=model,checkpoint=am.checkpoint,sources=sources,support=support,
            logit_lens=result,test_subsets=subsets,lens_validation={k:lens[k] for k in ['steps','validation_kl','identity_kl']},
            complete=True,elapsed=time.time()-begin))
        print('DONE',model['id'],round(time.time()-begin),support,flush=True)
        report(parent,out,manifest)
        del am,lens,test;gc.collect();torch.cuda.empty_cache()
    atomic_json(out/'status.json',dict(complete=True,models=len(manifest['models'])))


def report(parent,out,manifest):
    complete=[m for m in manifest['models'] if (out/(m['id']+'.json')).exists()]
    lines=['# Early-correct calibration vs mixed-window calibration',f'Completed {len(complete)}/8 models.',
        'Same graph/query-disjoint split and same unfiltered natural test windows. Calibration: first 4 fully legal records AND correct receiver next-token prediction.',
        'Values are final-prediction agreement, not semantic correctness. Mean ± seed standard deviation.',
        '| group | seeds | test position | calibration | L1 | L2 | L3 | L4 | L5 | L6 |',
        '|---|---|---|---|---|---|---|---|---|---|']
    summary=[]
    for group in ['l6_p0','l6_p30']:
        mm=[m for m in complete if m['group']==group]
        if not mm:continue
        for task in ['field_node','field_bar','state']:
            for variant in ['raw','mixed','early_clean']:
                arrays=[]
                for m in mm:
                    d=json.loads(((out if variant=='early_clean' else parent)/(m['id']+'.json')).read_text())
                    kind='raw' if variant=='raw' else 'calibrated'
                    arrays.append([r['final_agreement'] for r in d['logit_lens'] if r['task']==task and r['kind']==kind])
                a=np.array(arrays);mean=a.mean(0);std=a.std(0)
                summary.append(dict(group=group,seeds=[m['seed'] for m in mm],task=task,variant=variant,mean=mean.tolist(),std=std.tolist()))
                lines.append('| '+' | '.join([group,str(len(mm)),task,variant]+[f'{100*m:.1f}±{100*s:.1f}' for m,s in zip(mean,std)])+' |')
    conditional=[]
    lines+=['\n## Held-out test strata, early-correct calibrated lens',
        'Correct = entire record before first semantic error. Early = first four records. No filtering on lens predictions.',
        'These are conditional accuracies on selected natural next tokens, not full-navigation answer accuracy.',
        '| group | seeds | population | task | mean positions/model | mean queries/model | L1 | L2 | L3 | L4 | L5 | L6 |',
        '|---|---|---|---|---|---|---|---|---|---|---|---|']
    for group in ['l6_p0','l6_p30']:
        dd=[json.loads((out/(m['id']+'.json')).read_text()) for m in complete if m['group']==group]
        dd=[d for d in dd if 'test_subsets' in d]
        if not dd:continue
        for subset in ['all','early_all','correct_record','early_correct','later_correct']:
            for task in ['field_node','field_bar','state']:
                for kind in ['raw','calibrated']:
                    arrays=[];counts=[];queries=[]
                    for d in dd:
                        panel=d if subset=='all' else d['test_subsets'][subset]
                        rr=[r for r in panel['logit_lens'] if r['task']==task and r['kind']==kind]
                        if len(rr)!=6:continue
                        arrays.append([r['accuracy'] for r in rr]);counts.append(rr[0]['n'])
                        queries.append(256 if subset=='all' else panel['queries'])
                    if not arrays:continue
                    a=np.array(arrays);mean=a.mean(0);std=a.std(0)
                    conditional.append(dict(group=group,models=len(arrays),population=subset,task=task,kind=kind,
                        positions_per_model=counts,queries_per_model=queries,mean=mean.tolist(),std=std.tolist()))
                    if kind=='calibrated' and task=='field_node':
                        lines.append('| '+' | '.join([group,str(len(arrays)),subset,task,f'{np.mean(counts):.0f}',f'{np.mean(queries):.0f}']+
                            [f'{100*m:.1f}±{100*s:.1f}' for m,s in zip(mean,std)])+' |')
    done=sum('test_subsets' in json.loads((out/(m['id']+'.json')).read_text()) for m in complete)
    atomic_json(out/'summary.json',dict(complete=len(complete)==8,models=len(complete),results=summary,
        test_subsets_complete=done==8,test_subsets_models=done,test_subsets=conditional))
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':main()
