"""Export compact natural and completed seed-paired oracle plotting data."""
import argparse
from collections import defaultdict
import gzip
import json
from pathlib import Path


def load(path):
    with gzip.open(path,'rt') as f:return json.load(f)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();p=a.output
    if (p/'DEPRECATED.json').exists():raise ValueError('Withdrawn cohort: new plots disabled')
    plan=json.loads((p/'plan.json').read_text());markers=json.loads((p/'marker_census/marker_counts.json').read_text())
    assert markers['complete']
    snap=dict(cohort=plan['manifest']['cohort'],natural={},oracle={},oracle_progress=json.loads((p/'oracle_mistake/status.json').read_text()),
              note='reachable queries only; oracle results shown only when both model seeds are complete; raw marker census hashes matched original traces')
    for task in plan['manifest']['tasks']:
        natural=defaultdict(list);oracle=defaultdict(list)
        for m in plan['models']:
            name=f"{m['id']}__{task}.json.gz";b=load(p/'results'/name);assert b['complete']
            natural[m['group']].extend(dict(t,seed=m['seed']) for t in b['rows'])
            op=p/'oracle_mistake/results'/name
            if op.exists():
                b=load(op)
                if b['complete']:oracle[m['group']].extend(dict(t,seed=m['seed']) for t in b['rows'])
        for condition,groups in [('natural',natural),('oracle',oracle)]:
            result={}
            for group,ts in groups.items():
                if len({t['seed'] for t in ts})!=2 or len(ts)!=512:continue
                row=dict(n=len(ts),correct=sum(t['correct'] for t in ts),valid=sum(t['trace_valid'] for t in ts),
                    complete_correct=sum(t['trace_valid'] and t['correct'] for t in ts),
                    first_tokens=[t['first']['token'] for t in ts if t['first']],first_steps=[t['first']['search_step'] for t in ts if t['first']],
                    error_records=sum(r['bad'] for t in ts for r in t['records']),
                    seed_accuracy=[sum(t['correct'] for t in ts if t['seed']==s)/256 for s in [0,1]],
                    seed_validity=[sum(t['trace_valid'] for t in ts if t['seed']==s)/256 for s in [0,1]])
                if condition=='natural':
                    c=markers['groups'][task][group];assert c['original_hash_match']==512
                    immediate=whole=0
                    for t in ts:
                        rr=t['records'];success=sum(r['true_cancel'] and i+1<len(rr) and not rr[i+1]['bad'] and rr[i+1]['action_legal'] and not rr[i+1]['cancel'] for i,r in enumerate(rr))
                        immediate+=success;whole+=bool(success and t['trace_valid'] and t['correct'])
                    row.update(raw_mistakes=c['raw'],true_marks=c.get('true',0),immediate_repairs=immediate,repaired_whole_trajectories=whole)
                result[group]=row
            snap[condition][task]=result
    (p/'plot_snapshot.json').write_text(json.dumps(snap,ensure_ascii=False,separators=(',',':')),encoding='utf-8')
    print('PLOT SNAPSHOT oracle paired groups',sum(len(v) for v in snap['oracle'].values()),'/30')


if __name__=='__main__':main()
