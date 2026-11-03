"""Aggregate positive-only first errors, answer accounting and causal ablation."""
from collections import Counter
from pathlib import Path
import argparse,json
import numpy as np
from cot_navigation.protocols.correction_replay import STEP_BINS,bucket
from cot_navigation.runtime.workspace import ROOT
from cot_navigation.experiments.artifacts import atomic_json
P=Path(__file__).resolve().parent
ARMS=['original_clean','matched_clean','correction']
TASKS=['n32','n64','main','deep','hard']
OUTCOMES=['correct_clean_valid','correct_repaired_valid',
          'correct_execution_fields_invalid','correct_incomplete_or_illegal','wrong_answer']


def quantiles(xs):
    return dict(n=len(xs),q25=float(np.quantile(xs,.25)),median=float(np.median(xs)),q75=float(np.quantile(xs,.75))) if xs else dict(n=0,q25=None,median=None,q75=None)


def aggregate(rows):
    n=len(rows);first=[r['first'] for r in rows if r['first']]
    errors=len(first);kinds=Counter(r['first_outcome'] for r in rows)
    fields=Counter(e['field'] for e in first)
    positions={k:quantiles([e[k] for e in first]) for k in ['token','search_step','record','stack_depth','visited_size','context_fraction']}
    for k in ['token','search_step']:
        positions[k+'_before_goal']=quantiles([e[k] for e in first if not e['at_goal']])
    risks=np.asarray([r['risk_steps_by_bin'] for r in rows]).sum(0)
    hazard=np.zeros(len(STEP_BINS),dtype=int)
    for e in first:hazard[bucket(e['search_step'])]+=1
    candidate=[e['candidate_features'] for e in first if e['field']=='candidate' and 'candidate_features' in e]
    flags=Counter()
    for f in candidate:
        if f['closed']:
            flags['missing_any']+=int(f['missing']>0)
            flags['missing_useful']+=int(f['missing_useful']>0)
            flags['missing_only_deadends']+=int(f['missing']>0 and not f['missing_useful'])
            flags['all_useful_missing']+=int(f['useful_eligible']>0 and f['missing_useful']==f['useful_eligible'])
            emitted=f['eligible']-f['missing']+f['extra']+f['duplicate']
            flags['false_empty']+=int(f['eligible']>0 and emitted==0)
        for key in ['extra','extra_visited','extra_nonedge','duplicate']:flags[key]+=int(f[key]>0)
    by_distance={}
    for distance in sorted(set(r['distance'] for r in rows)):
        rs=[r for r in rows if r['distance']==distance];es=[r['first'] for r in rs if r['first']]
        by_distance[str(distance)]=dict(n=len(rs),correct=sum(r['correct'] for r in rs),
            first_errors=len(es),first_before_goal=sum(not e['at_goal'] for e in es),
            first_step=quantiles([e['search_step'] for e in es]),first_token=quantiles([e['token'] for e in es]))
    return dict(n=n,correct=sum(r['correct'] for r in rows),by_distance=by_distance,
        postrepair_failure_first_types=dict(Counter(
            (r['first_uncancelled'] or {}).get('kind','no_uncancelled_token_error_but_incomplete_or_wrong_answer')
            for r in rows if r['first_repaired'] and not r['trace_valid'])),
        trace_valid=sum(r['trace_valid'] for r in rows),literal_valid=sum(r['literal_valid'] for r in rows),
        outcomes={k:sum(r['outcome']==k for r in rows) for k in OUTCOMES},
        first_error_queries=errors,first_before_goal=sum(not e['at_goal'] for e in first),
        first_at_goal=sum(e['at_goal'] for e in first),first_outcome_counts=dict(kinds),
        first_error_type_rates={k:dict(count=v,all_positive_rate=v/n,
            conditional_error_rate=v/errors if not k.startswith('no_error') and errors else None) for k,v in kinds.items()},
        first_field_counts=dict(fields),candidate_first_record_features=dict(flags),positions=positions,
        hazard=dict(bins=STEP_BINS,first_errors=hazard.tolist(),risk_states=risks.tolist(),
                    rates=np.divide(hazard,risks,out=np.zeros(len(risks)),where=risks>0).tolist()),
        cumulative_first_error_by_step={str(c):sum(e['search_step']<=c for e in first)/n for c in [1,2,4,8,16,32,64,128]},
        first_detected=sum(r['first_detected'] for r in rows),first_repaired=sum(r['first_repaired'] for r in rows),
        first_repaired_correct=sum(r['first_repaired'] and r['correct'] for r in rows),
        first_repaired_full_valid=sum(r['first_repaired'] and r['trace_valid'] for r in rows),
        wrong_answer_after_valid_trace=sum(r['trace_valid'] and not r['correct'] for r in rows),
        wrong_no=sum(r['pred']==0 for r in rows),missing_answer=sum(r['pred'] is None for r in rows),
        any_raw_mistake=sum(r['raw_mistake_tokens']>0 for r in rows),
        any_true_mark=sum(r['true_marks']>0 for r in rows),any_false_mark=sum(r['false_marks']>0 for r in rows),
        any_repair=sum(r['repairs']>0 for r in rows),
        true_mark_records=sum(r['true_marks'] for r in rows),false_mark_records=sum(r['false_marks'] for r in rows),
        interrupted=sum(r['interrupted'] for r in rows),forced_stop=sum(r['forced_stop'] for r in rows),
        trace_length=quantiles([r['trace_length'] for r in rows]))


def pairs(base,new):
    assert [r['query_id'] for r in base]==[r['query_id'] for r in new]
    gains=[b for a,b in zip(base,new) if not a['correct'] and b['correct']]
    losses=[a for a,b in zip(base,new) if a['correct'] and not b['correct']]
    common_error=[(a,b) for a,b in zip(base,new) if a['first'] and b['first']]
    return dict(gains=len(gains),losses=len(losses),net=len(gains)-len(losses),
        gain_outcomes=dict(Counter(r['outcome'] for r in gains)),
        loss_baseline_outcomes=dict(Counter(r['outcome'] for r in losses)),
        gains_any_mistake=sum(r['raw_mistake_tokens']>0 for r in gains),
        gains_any_repair=sum(r['repairs']>0 for r in gains),
        gains_first_repaired=sum(r['first_repaired'] for r in gains),
        gain_first_types=dict(Counter(r['first_outcome'] for r in gains)),
        both_error_queries=len(common_error),
        paired_first_token_change=quantiles([b['first']['token']-a['first']['token'] for a,b in common_error]),
        paired_first_step_change=quantiles([b['first']['search_step']-a['first']['search_step'] for a,b in common_error]))


def main():
    p=argparse.ArgumentParser();p.add_argument('--partial',action='store_true');args=p.parse_args()
    blobs={}
    for path in (P/'results').glob('*.json'):
        b=json.loads(path.read_text());blobs[b['arm'],b['seed']]=b
    out=dict(finished=len(blobs)==12 and all(b.get('finished') for b in blobs.values()),
        positive_only=True,unique_positive_queries_per_task=256,repeats=4,
        count_unit='model-query outcomes (4 seeds x same 256 positive queries =1024), not1024 unique graphs',tasks={})
    if not args.partial:assert out['finished']
    for task in TASKS:
        groups={};native={};blocked={}
        for arm in ARMS:
            available=[(s,blobs[arm,s]['cases'][task]['native']) for s in range(4)
                if (arm,s) in blobs and blobs[arm,s].get('cases',{}).get(task,{}).get('native',{}).get('finished')]
            if len(available)!=4:continue
            allrows=[];perseed=[];native[arm]={}
            for s,case in available:
                assert len(case['rows'])==256
                rows=case['rows'];native[arm][s]=rows;allrows.extend(rows)
                historic=ROOT/f'outputs/correction_reproduction_v1/launches/dfs_correction_repro_20261028/dfs_state/{arm}/s{s}/eval/free_test_last{("" if task=="main" else "-"+task)}.json'
                old=json.loads(historic.read_text())
                historic_correct=round(old['overall']['yes_recall']*256)
                history={r['query_id']:r for r in [json.loads(line) for line in
                    historic.with_name(historic.name.replace('free_test_last','predictions_test_last')).with_suffix('.jsonl').read_text().splitlines()]}
                pred_drift=sum(r['pred']!=history[r['query_id']]['pred'] for r in rows)
                perseed.append(dict(seed=s,historical_correct=historic_correct,
                    historical_positive_mismatches=case.get('historical_positive_mismatches',0),
                    historical_positive_pred_changes=pred_drift,
                    **aggregate(rows)))
            groups[arm]=dict(pooled=aggregate(allrows),seeds=perseed)
            groups[arm]['historical_correct']=sum(s['historical_correct'] for s in perseed)
        item=dict(groups=groups,contrasts={})
        for base in ARMS[:2]:
            if base not in native or 'correction' not in native:continue
            aa=groups[base]['pooled'];bb=groups['correction']['pooled'];perseed=[]
            for s in range(4):perseed.append(dict(seed=s,**pairs(native[base][s],native['correction'][s])))
            item['contrasts'][base]=dict(
                delta_correct=bb['correct']-aa['correct'],
                delta_outcomes={k:bb['outcomes'][k]-aa['outcomes'][k] for k in OUTCOMES},
                delta_first_types={k:bb['first_outcome_counts'].get(k,0)-aa['first_outcome_counts'].get(k,0)
                    for k in sorted(set(aa['first_outcome_counts'])|set(bb['first_outcome_counts']))},
                gains=sum(r['gains'] for r in perseed),losses=sum(r['losses'] for r in perseed),
                gain_outcomes={k:sum(r['gain_outcomes'].get(k,0) for r in perseed) for k in OUTCOMES},
                gains_any_mistake=sum(r['gains_any_mistake'] for r in perseed),
                gains_any_repair=sum(r['gains_any_repair'] for r in perseed),
                seeds=perseed)
        if 'correction' in native:
            avail=[(s,blobs['correction',s]['cases'][task]['no_mistake']) for s in range(4)
                if blobs['correction',s]['cases'][task].get('no_mistake',{}).get('finished')]
            if len(avail)==4:
                stats=[];allblocked=[]
                for s,c in avail:
                    a=native['correction'][s];b=c['rows'];assert [r['query_id'] for r in a]==[r['query_id'] for r in b]
                    ab=[(x,y) for x,y in zip(a,b)]
                    helped=[x for x,y in ab if x['correct'] and not y['correct']]
                    harmed=[x for x,y in ab if not x['correct'] and y['correct']]
                    nontrigger=[(x,y) for x,y in ab if not x['raw_mistake_tokens']]
                    stats.append(dict(seed=s,native_correct=sum(x['correct'] for x in a),
                        blocked_correct=sum(x['correct'] for x in b),helped=len(helped),harmed=len(harmed),
                        nontrigger_queries=len(nontrigger),nontrigger_pred_drift=sum(x['pred']!=y['pred'] for x,y in nontrigger),
                        nontrigger_trace_drift=sum(x['trace_hash']!=y['trace_hash'] for x,y in nontrigger),
                        first_type_drift=sum(x['first_outcome']!=y['first_outcome'] for x,y in ab),
                        first_token_drift=sum((x['first'] or {}).get('token')!=(y['first'] or {}).get('token') for x,y in ab),
                        helped_true_marks=sum(x['true_marks']>0 for x in helped),
                        helped_repairs=sum(x['repairs']>0 for x in helped),
                        helped_repaired_valid=sum(x['outcome']=='correct_repaired_valid' for x in helped),
                        helped_no_native_mistake=sum(not x['raw_mistake_tokens'] for x in helped),
                        harmed_no_native_mistake=sum(not x['raw_mistake_tokens'] for x in harmed),
                        native_interrupted=sum(x['interrupted'] for x in a),blocked_interrupted=sum(x['interrupted'] for x in b)))
                    allblocked.extend(b)
                item['mistake_ablation']=dict(blocked=aggregate(allblocked),seeds=stats,
                    pooled={k:sum(s[k] for s in stats) for k in stats[0] if k!='seed'})
        out['tasks'][task]=item
    out['reproduction']=dict(models=len(blobs),native_reports=sum(
        c.get('native',{}).get('finished',False) for b in blobs.values() for c in b.get('cases',{}).values()),
        masked_reports=sum(c.get('no_mistake',{}).get('finished',False) for b in blobs.values() for c in b.get('cases',{}).values()),
        historical_native_mismatches=sum(c.get('native',{}).get('historical_mismatches',0) for b in blobs.values() for c in b.get('cases',{}).values()),
        identity_proxy_control=blobs.get(('correction',0),{}).get('identity_proxy_control',False))
    out['reproduction']['positive_replay_correct']=sum(g['pooled']['correct'] for t in out['tasks'].values() for g in t['groups'].values())
    out['reproduction']['positive_historical_correct']=sum(g['historical_correct'] for t in out['tasks'].values() for g in t['groups'].values())
    out['reproduction']['positive_mismatches']=sum(s['historical_positive_mismatches'] for t in out['tasks'].values() for g in t['groups'].values() for s in g['seeds'])
    out['reproduction']['positive_pred_changes']=sum(s['historical_positive_pred_changes'] for t in out['tasks'].values() for g in t['groups'].values() for s in g['seeds'])
    dest=P/('summary_partial.json' if args.partial else 'summary.json')
    atomic_json(dest,out)
    if not args.partial:
        examples=[]
        for task in TASKS:
            for s in range(4):
                case=blobs['correction',s]['cases'][task]
                aa=case['native']['rows'];bb=case['no_mistake']['rows']
                for effect in ['helped','harmed']:
                    candidates=[(a,b) for a,b in zip(aa,bb) if
                        (a['correct'] and not b['correct'] if effect=='helped' else not a['correct'] and b['correct'])]
                    if not candidates:continue
                    def score(pair):
                        a,b=pair
                        return (int(a['outcome']=='correct_repaired_valid')*100 if effect=='helped' else
                                int(a['interrupted'] and not b['interrupted'])*100)+a['raw_mistake_tokens']
                    a,b=max(candidates,key=score)
                    examples.append(dict(task=task,seed=s,effect=effect,query_id=a['query_id'],
                        checkpoint=blobs['correction',s]['checkpoint'],
                        native={k:a[k] for k in ['pred','outcome','first','first_repaired','raw_mistake_tokens','trace_length','interrupted','forced_stop']},
                        blocked={k:b[k] for k in ['pred','outcome','first','trace_length','interrupted','forced_stop']}))
        selected=[]
        for task in TASKS:
            for effect in ['helped','harmed']:
                candidates=[e for e in examples if e['task']==task and e['effect']==effect]
                if not candidates:continue
                if effect=='helped':
                    genuine=[e for e in candidates if e['native']['outcome']=='correct_repaired_valid']
                    selected.append((genuine or candidates)[0])
                else:selected.append(max(candidates,key=lambda e:100*int(e['native']['interrupted'] and not e['blocked']['interrupted'])+e['native']['raw_mistake_tokens']))
        atomic_json(P/'counterexamples.json',selected)
    print('FINISHED',out['finished'],out['reproduction'])
    for task,d in out['tasks'].items():
        print(task,{a:dict(correct=g['pooled']['correct'],literal=g['pooled']['literal_valid'],
            repaired_valid=g['pooled']['outcomes']['correct_repaired_valid'],first=g['pooled']['first_error_queries']) for a,g in d['groups'].items()},
            d.get('mistake_ablation',{}).get('pooled',{}))


if __name__=='__main__':main()
