"""Describe length-dependent risk on completed, seed-paired natural runs."""
from collections import defaultdict
import gzip
import json
from pathlib import Path
import numpy as np

P=Path(__file__).resolve().parent
TASKS=['n32','n64','main','deep','hard']
ARMS=['matched_clean','correction']


def token_bin(n):
    return '<128' if n<128 else '128-255' if n<256 else '256-511' if n<512 else '512-1023' if n<1024 else '1024+'


def size_bin(n):return '1-2' if n<=2 else '3-4' if n<=4 else '5-8' if n<=8 else '9-16' if n<=16 else '17+'


def metric(r,key):
    if key=='illegal_action':return not r['action_legal'] and not r['cancel']
    if key=='candidate_bad':return any(t.startswith('candidate_') for t in r['errors'])
    if key=='frontier_bad':return any(t.startswith('frontier_') for t in r['errors'])
    return r[key]


def stats(rr):
    return dict(n=len(rr),model_queries=len({(r['seed'],r['query']) for r in rr}),
        **{key:sum(metric(r,key) for r in rr)/len(rr) if rr else None for key in
           ['bad','action_bad','illegal_action','candidate_bad','frontier_bad']})


def adjust(rows,kind):
    cells=defaultdict(lambda:[[],[]])
    for r in rows:
        # Exactly one prior bad record; avoid changing the mixture with count.
        if r['prior_bad_records'] not in [0,1]:continue
        key=(r['seed'],token_bin(r['token_start']))
        if kind=='length_state':key+=(size_bin(r['record']),size_bin(r['visited_size']),size_bin(r['search_depth']),r['at_goal'],r['distance'])
        cells[key][int(bool(r['prior_bad_records']))].append(r)
    eligible=[]
    for clean,post in cells.values():
        if min(len(clean),len(post))<3:continue
        if min(len({r['query'] for r in clean}),len({r['query'] for r in post}))<2:continue
        eligible.append((clean,post,len(clean)*len(post)/(len(clean)+len(post))))
    den=sum(w for _,_,w in eligible)
    result=dict(cells=len(eligible),clean_records=sum(len(a) for a,b,w in eligible),post_records=sum(len(b) for a,b,w in eligible),
                total_post_records=sum(r['prior_bad_records']==1 for r in rows),metrics={})
    for key in ['bad','action_bad','illegal_action','candidate_bad','frontier_bad']:
        clean=sum(w*np.mean([metric(r,key) for r in a]) for a,b,w in eligible)/den if den else None
        post=sum(w*np.mean([metric(r,key) for r in b]) for a,b,w in eligible)/den if den else None
        result['metrics'][key]=dict(clean=clean,post=post,delta=post-clean if den else None)
    return result


def main():
    output={}
    for task in TASKS:
        blobs={}
        for arm in ARMS:
            for seed in range(4):
                f=P/'results'/f'{arm}_s{seed}_{task}.json.gz'
                if f.exists():
                    with gzip.open(f,'rt') as h:b=json.load(h)
                    if b['complete']:blobs[arm,seed]=b
        seeds=[s for s in range(4) if all((a,s) in blobs for a in ARMS)]
        output[task]=dict(paired_seeds=seeds,groups={})
        for arm in ARMS:
            trajectories=[t for s in seeds for t in blobs[arm,s]['rows']]
            rows=[dict(r,seed=s,query=t['query'],distance=t['distance']) for s in seeds for t in blobs[arm,s]['rows'] for r in t['records']]
            profile=[]
            for length in ['<128','128-255','256-511','512-1023','1024+']:
                for prior in [0,1,2]:
                    rr=[r for r in rows if token_bin(r['token_start'])==length and min(r['prior_bad_records'],2)==prior]
                    profile.append(dict(tokens=length,prior='2+' if prior==2 else str(prior),**stats(rr)))
            hazard=[]
            for lo,hi in [(0,128),(128,256),(256,512),(512,1024),(1024,4096)]:
                exposure=events=entered=0
                for t in trajectories:
                    first=t['first']['token']-1 if t['first'] else None
                    # STOP is appended only on interruption, not a CoT decision.
                    end=first+1 if first is not None else t['trace_tokens']-int(t['interrupted'])
                    if end>lo:entered+=1
                    exposure+=max(0,min(end,hi)-lo)
                    events+=first is not None and lo<=first<hi
                hazard.append(dict(token_lo=lo,token_hi=hi,entered=entered,at_risk_tokens=exposure,
                    first_errors=events,per_100_tokens=100*events/exposure if exposure else None))
            output[task]['groups'][arm]=dict(profile=profile,first_error_hazard=hazard,
                adjusted={k:adjust(rows,k) for k in ['length','length_state']})
    (P/'length_analysis.json').write_text(json.dumps(output,ensure_ascii=False,separators=(',',':')),encoding='utf-8')
    for task,t in output.items():
        print(task,'paired',t['paired_seeds'])
        for arm,g in t['groups'].items():
            for k,m in g['adjusted'].items():print(arm,k,'support',m['post_records'],'/',m['total_post_records'],'deltas',{key:round(100*x['delta'],2) if x['delta'] is not None else None for key,x in m['metrics'].items()})


if __name__=='__main__':main()
