"""CPU-only aggregation; reuse validated history and length estimators."""
import argparse
from collections import Counter,defaultdict
import gzip
import importlib.util
import json
from pathlib import Path
import numpy as np

P=Path(__file__).resolve().parent


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


BASE=P.parent/'correction_reproduction/error_continuation'
history=load('behavior_history_report',BASE/'report.py')
length=load('behavior_length_report',BASE/'length_analysis.py')
BINS=['0','1','2','3-4','5+']


def uncertainty(rows,bin_key):
    qs=sorted({r['query'] for r in rows});seeds=sorted({r['seed'] for r in rows})
    if not qs:return None
    qi={q:i for i,q in enumerate(qs)};si={s:i for i,s in enumerate(seeds)}
    n=np.zeros((len(qs),len(seeds)));y=n.copy()
    for r in rows:
        if r['prior_bin']==bin_key:n[qi[r['query']],si[r['seed']]]+=1;y[qi[r['query']],si[r['seed']]]+=r['bad']
    if not n.sum():return None
    rng=np.random.default_rng(731);rates=[]
    for _ in range(500):
        ix=np.ix_(rng.integers(len(qs),size=len(qs)),rng.integers(len(seeds),size=len(seeds)))
        den=n[ix].sum()
        if den:rates.append(y[ix].sum()/den)
    return np.quantile(rates,[.025,.975]).tolist() if rates else None


def structure(ts):
    dd=[t['dependencies'] for t in ts];rv=[r for d in dd for r in d['revisits']]
    extras=[r for r in rv if r['original_unused_extras']];missing=[r for r in rv if r['missing_still_needed']];clean=[r for r in rv if r['previously_clean']]
    local=[]
    for offset in [1,2]:
        rs=[r for d in dd for r in d['after_illegal'] if r['offset']==offset];ss=[r for r in rs if r['follows_executed'] is not None]
        local.append(dict(offset=offset,observed=len(rs),supported=len(ss),
            **{key:sum(r['follows_executed'][key] for r in ss) for key in ['frontier','visited','candidate','action','all']},
            reject_all=sum(r['follows_rejected']['all'] for r in rs if r['follows_rejected'] is not None),
            printed_supported=sum(r['follows_printed'] is not None for r in rs),
            printed_all=sum(r['follows_printed']['all'] for r in rs if r['follows_printed'] is not None)))
    following=[]
    for field in ['frontier','visited']:
        for cancel in [False,True]:
            rs=[r for d in dd for r in d['annotation_transitions'] if r['field']==field and r['cancel']==cancel]
            following.append(dict(field=field,cancel=cancel,n=len(rs),counts=dict(Counter(r['following'] for r in rs))))
    return dict(local=local,following=following,revisits=dict(total=len(rv),extra_anchors=sum(d['unused_extra_anchors'] for d in dd),
        extra_no_return=sum(d['unused_extra_without_observed_return'] for d in dd),extra_returns=len(extras),same_extra=sum(bool(r['repeated_unused_extras']) for r in extras),
        extra_return_bad=sum(r['candidate_bad'] for r in extras),missing_returns=len(missing),same_missing=sum(bool(r['repeated_missing']) for r in missing),
        clean_returns=len(clean),clean_then_bad=sum(r['candidate_bad'] for r in clean)))


def summarize(ts):
    rows=[];decorated=[]
    for t in ts:
        prior=Counter();events=0;records=[]
        for record in t['records']:
            r=dict(record,seed=t['seed'],query=t['query'],distance=t['distance'],prior_type_counts=dict(prior),event_bin=history.history_bin(events))
            rows.append(r);records.append(r);prior.update(r['errors']);events+=r['bad_token_events']
        decorated.append(dict(t,records=records))
    by_history={b:dict(history.counts([r for r in rows if r['prior_bin']==b]),ci95=uncertainty(rows,b)) for b in BINS}
    type_specific=[]
    for typ in sorted({typ for r in rows for typ in r['prior_type_counts']}):
        for b in BINS:type_specific.append(dict(type=typ,prior=b,**history.counts([r for r in rows if history.history_bin(r['prior_type_counts'].get(typ,0))==b])))
    hazard=[]
    for lo,hi in [(0,128),(128,256),(256,512),(512,1024),(1024,4096)]:
        exposure=events=entered=0
        for t in ts:
            first=t['first']['token']-1 if t['first'] else None;end=first+1 if first is not None else t['trace_tokens']-int(t['interrupted'])
            entered+=end>lo;exposure+=max(0,min(end,hi)-lo);events+=first is not None and lo<=first<hi
        hazard.append(dict(lo=lo,hi=hi,entered=entered,exposure=exposure,events=events,per100=100*events/exposure if exposure else None))
    profile=[dict(tokens=b,prior=k,**length.stats([r for r in rows if length.token_bin(r['token_start'])==b and min(r['prior_bad_records'],2)==k])) for b in ['<128','128-255','256-511','512-1023','1024+'] for k in [0,1,2]]
    return dict(n=len(ts),seeds=sorted({t['seed'] for t in ts}),correct=sum(t['correct'] for t in ts),valid=sum(t['trace_valid'] for t in ts),
        first_error_trajectories=sum(t['first'] is not None for t in ts),interrupted=sum(t['interrupted'] for t in ts),
        illegal_action_censored=sum(t['stopped_on_illegal_action'] for t in ts),partial=sum(t['trailing_partial'] for t in ts),
        true_marks=sum(t['true_marks'] for t in ts),false_marks=sum(t['false_marks'] for t in ts),repairs=sum(t['repairs'] for t in ts),
        all=history.counts(rows),by_history=by_history,
        by_unresolved={b:history.counts([r for r in rows if r['unresolved_bin']==b]) for b in BINS},
        by_events={b:history.counts([r for r in rows if r['event_bin']==b]) for b in BINS},
        excluding_goal={b:history.counts([r for r in rows if r['prior_bin']==b and not r['at_goal']]) for b in BINS},
        first_type_history=[dict(type=typ,prior=b,**history.counts([r for r in rows if r['prior_bin']==b and r['first_type']==typ])) for b in BINS for typ in sorted({r['first_type'] for r in rows if r['prior_bin']==b})],
        type_specific=type_specific,hazard=hazard,profile=profile,
        adjusted={k:length.adjust(rows,k) for k in ['length','length_state']},
        first_error_lags=history.transitions(decorated),structure=structure(ts))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);args=ap.parse_args();out=args.output
    if (out/'DEPRECATED.json').exists():raise ValueError('Withdrawn cohort: report regeneration is disabled')
    plan=json.loads((out/'plan.json').read_text());manifest=plan['manifest'];data=dict(protocol_version=1,cohort=manifest['cohort'],tasks={},complete_tasks=0,total_tasks=len(plan['models'])*len(manifest['tasks']),controls=manifest['controls'])
    markdown=['# 纠错模型行为分析：'+manifest['cohort'],'','固定流程v1；自然贪心生成，可达测试查询。缺失对照：'+manifest['controls'], '']
    for task in manifest['tasks']:
        grouped=defaultdict(list);entries={}
        for entry in plan['models']:
            file=out/'results'/f"{entry['id']}__{task}.json.gz"
            if not file.exists():continue
            with gzip.open(file,'rt') as f:b=json.load(f)
            if not b['complete']:continue
            data['complete_tasks']+=1;entries[entry['group']]=entry
            grouped[entry['group']].extend(dict(t,seed=entry['seed']) for t in b['rows'])
        groups={}
        for group,ts in grouped.items():
            s=summarize(ts);s['per_seed']={seed:summarize([t for t in ts if t['seed']==seed]) for seed in s['seeds']}
            groups[group]=dict(layers=entries[group]['layers'],error_rate=entries[group]['error_rate'],**s)
        data['tasks'][task]=groups
        markdown+=['## '+task,'','|模型组|seed|轨迹数|回答正确率|CoT合法率|首错轨迹数|','|---|---|---:|---:|---:|---:|']
        for group,g in groups.items():markdown.append(f"|{group}|{g['seeds']}|{g['n']}|{g['correct']/g['n']:.2%}|{g['valid']/g['n']:.2%}|{g['first_error_trajectories']}|")
        markdown+=['']
    data['complete']=data['complete_tasks']==data['total_tasks']
    markdown.insert(2,'完成状态：'+str(data['complete_tasks'])+'/'+str(data['total_tasks'])+'模型—图任务。')
    markdown+=['长度调整为共同支持上的描述性比较，非因果效应。回访事件不是独立查询，未回访不算修复。非法动作后局部一致不等于全局正确。','']
    for name,value in [('summary.json',json.dumps(data,ensure_ascii=False,separators=(',',':'))),('REPORT.md','\n'.join(markdown)),('dashboard.html',(P/'dashboard_template.html').read_text().replace('__DATA__',json.dumps(data,ensure_ascii=False).replace('</','<\\/')))]:
        temp=out/(name+'.tmp');temp.write_text(value,encoding='utf-8');temp.replace(out/name)
    print('REPORT',data['complete_tasks'],data['total_tasks'],flush=True)


if __name__=='__main__':main()
