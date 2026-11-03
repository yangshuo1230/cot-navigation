"""Aggregate complete matched trajectory audits; never infer missing scores."""
from pathlib import Path
import json,math
from collections import Counter
F=Path(__file__).parent
files=list((F/'audits').rglob('*.json'));runs=[json.loads(p.read_text()) for p in files]
lookup={(x['model'],x['case']):x for x in runs};pairs=[]
for item in runs:
    if item['model'].startswith('baseline/'):continue
    key=('baseline/'+item['protocol'],item['case'])
    if key not in lookup:continue
    baseline=lookup[key];a={x['query_id']:x for x in baseline['by_query']};b={x['query_id']:x for x in item['by_query']};assert a.keys()==b.keys()
    counts=Counter(n=len(a))
    for q,x in a.items():
        y=b[q]
        for name,field in [('answer','correct'),('trace','complete_valid'),('actions','actions_valid'),('first_candidate','first_candidate_exact')]:
            counts[name+'_fixed']+=int(not x[field] and y[field]);counts[name+'_broken']+=int(x[field] and not y[field])
        if not x['correct'] and y['correct']:
            counts['answer_fixed_now_legal']+=int(y['complete_valid']);counts['answer_fixed_still_invalid']+=int(not y['complete_valid'])
        counts['answer_correct_invalid_to_legal']+=int(x['correct'] and not x['complete_valid'] and y['complete_valid'])
    n=len(a);entry={'model':item['model'],'case':item['case'],'counts':dict(counts),
       'baseline':baseline['overall'],'dual':item['overall']}
    for name,field in [('answer','correct'),('trace','complete_valid'),('actions','actions_valid')]:
        win,loss=counts[name+'_fixed'],counts[name+'_broken'];m=win+loss
        # Paired Bernoulli differences, normal approximation; also keep raw counts.
        mean=(win-loss)/n;variance=max(0,(win+loss)/n-mean**2);se=(variance/n)**.5
        exact=min(1.,2*sum(math.comb(m,k) for k in range(min(win,loss)+1))/2**m) if m else 1.
        entry[name+'_gain_pp']=100*mean;entry[name+'_gain_95ci_pp']=[100*(mean-1.96*se),100*(mean+1.96*se)];entry[name+'_paired_p_nominal']=exact
    pairs.append(entry)
compact=[{k:v for k,v in x.items() if k not in ['by_query','contract']} for x in runs]
summary={'complete':len(runs)==90,'expected_audits':90,'completed_audits':len(runs),
 'runs':compact,'paired':pairs,'baseline_complete_cases':len([x for x in runs if x['model'].startswith('baseline/')]),
 'note':'Trace legality excludes final answer correctness. Frozen baseline regenerated with the same inputs and compiled BF16/batch64/no token cap as existing dual rollouts. Training schedules remain unmatched. Nominal paired p values and approximate CIs describe query variability for these checkpoints, not training-seed variability.'}
(F/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print('AUDITS',len(runs),'/90; matched pairs',len(pairs),'/72')
for x in sorted(runs,key=lambda x:x['model']):
 if x['case']=='main':
  o=x['overall'];print(x['model'],'answer',round(o['correct']*100,2),'legal',round(o['complete_valid']*100,2),'actions',round(o['actions_valid']*100,2))
