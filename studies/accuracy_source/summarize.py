"""Summarize compact audit outputs, preserving paired cases and caveats."""
import json
from pathlib import Path
from collections import Counter
OUT=Path(__file__).resolve().parent

def load(name):return json.loads((OUT/(name+'.json')).read_text())
def aggregate(d):
 rows=d['rows'];n=len(rows)
 total=lambda k:sum(r[k] for r in rows)
 any_count=lambda k:sum(bool(r[k]) for r in rows)
 checked=total('checked_records')
 keys=['raw_bad_records','candidate_bad','candidate_bad_nonterminal','frontier_bad','visited_bad','action_bad','mistake_records','true_marks','false_marks','repairs']
 return dict(n=n,correct=total('correct'),saved_answer_mismatches=n-total('matches_saved'),
             raw_marker_queries=sum(bool(r.get('raw_mistake_tokens',r['mistake_records'])) for r in rows),
             literal_complete=total('literal_complete'),effective_complete=total('effective_complete'),
             effective_prefix_valid=total('effective_prefix_valid'),execution_complete=total('execution_complete'),
             execution_prefix_valid=total('execution_prefix_valid'),interrupted=total('interrupted'),
             forced_stop=total('forced_stop'),checked_records=checked,
             record_totals={k:total(k) for k in keys},query_counts={k:any_count(k) for k in keys},
             candidate_error_per_checked_record=total('candidate_bad')/checked,
             first_error_counts=dict(Counter(r['first_error'] for r in rows)),
             mean_tokens=total('trace_length')/n,mean_records=total('records')/n,
             answer_by_label={str(y):dict(n=sum(r['label']==y for r in rows),correct=sum(r['correct'] for r in rows if r['label']==y)) for y in [0,1]})

def paired(a,b):
 aa={r['query_id']:r for r in a['rows']};bb={r['query_id']:r for r in b['rows']}
 assert aa.keys()==bb.keys()
 groups={}
 for q,x in aa.items():
  y=bb[q];assert x['label']==y['label']
  assert not x['execution_complete'] or x['correct']
  assert not y['execution_complete'] or y['correct']
  group='both_correct' if x['correct'] and y['correct'] else 'gain' if y['correct'] else 'loss' if x['correct'] else 'both_wrong'
  groups.setdefault(group,[]).append(dict(query_id=q,label=y['label'],distance=y['distance'],
    clean_effective=x['effective_complete'],new_effective=y['effective_complete'],
    clean_execution=x['execution_complete'],new_execution=y['execution_complete'],
    clean_candidate_bad=x['candidate_bad'],new_candidate_bad=y['candidate_bad'],
    new_nonterminal_candidate_bad=y['candidate_bad_nonterminal'],
    clean_interrupted=x['interrupted'],new_interrupted=y['interrupted'],
    new_marks=y.get('raw_mistake_tokens',y['mistake_records']),new_repairs=y['repairs'],
    new_positive_witness=y.get('positive_witness',False),
    clean_first_error=x['first_error'],new_first_error=y['first_error']))
 return {g:dict(n=len(rs),new_marked=sum(bool(r['new_marks']) for r in rs),
    new_repaired=sum(bool(r['new_repairs']) for r in rs),
    new_positive_witness=sum(r['new_positive_witness'] for r in rs if r['label']==1),
    n_positive=sum(r['label']==1 for r in rs),
    new_effective=sum(r['new_effective'] for r in rs),new_execution=sum(r['new_execution'] for r in rs),
    clean_effective=sum(r['clean_effective'] for r in rs),clean_execution=sum(r['clean_execution'] for r in rs),
    new_no_candidate_error=sum(not r['new_candidate_bad'] for r in rs),
    new_no_nonterminal_candidate_error=sum(not r['new_nonterminal_candidate_bad'] for r in rs),
    examples=rs) for g,rs in groups.items()}

def ablation(normal,blocked,clean):
 a={r['query_id']:r for r in normal['rows']};b={r['query_id']:r for r in blocked['rows']}
 c={r['query_id']:r for r in clean['rows']}
 changes=[];unmarked_drift=[];gain_cases=[]
 for q,x in a.items():
  y=b[q]
  if not x['mistake_records'] and x['trace_hash']!=y['trace_hash']:
   unmarked_drift.append(q)
  if x['correct']!=y['correct']:
   changes.append(dict(query_id=q,normal_correct=x['correct'],blocked_correct=y['correct'],
                       normal_marks=x['mistake_records'],normal_repairs=x['repairs'],
                       normal_effective=x['effective_complete'],blocked_effective=y['effective_complete']))
  if x['correct'] and not c[q]['correct']:
   gain_cases.append(dict(query_id=q,normal_marks=x['mistake_records'],normal_repairs=x['repairs'],
                         blocked_correct=y['correct'],normal_effective=x['effective_complete']))
 return dict(normal_correct=sum(x['correct'] for x in a.values()),
             blocked_correct=sum(x['correct'] for x in b.values()),
             hurt=sum(x['normal_correct'] and not x['blocked_correct'] for x in changes),
             helped=sum(not x['normal_correct'] and x['blocked_correct'] for x in changes),
             changes=changes,unmarked_trace_drift=unmarked_drift,gain_cases=gain_cases,
             gains_retained=sum(x['blocked_correct'] for x in gain_cases))

def main():
 results={}
 for proto,cases in [('dfs',['main','deep','wide','larger']),('bfs',['main'])]:
  for case in cases:
   names=[proto+'_'+arm+'_'+case for arm in ['clean','error','error_blocked']]
   if not all((OUT/(n+'.json')).exists() for n in names):continue
   ds=list(map(load,names));clean,error,blocked=ds
   row=dict(aggregates={n:aggregate(d) for n,d in zip(names,ds)},
            paired=paired(clean,error),ablation=ablation(error,blocked,clean))
   pa,pb=[load('prefix_'+proto+'_'+arm+'_'+case) for arm in ['clean','error']]
   aa={(r['query_id'],r['record']):r for r in pa['rows']};bb={(r['query_id'],r['record']):r for r in pb['rows']}
   assert aa.keys()==bb.keys()
   for key in aa:
    assert aa[key]['gold']==bb[key]['gold'] and aa[key]['prefix_len']==bb[key]['prefix_len']
   row['clean_prefix']=dict(n=pa['n'],clean_correct=pa['correct'],error_correct=pb['correct'],
     gained=sum(not aa[k]['correct'] and bb[k]['correct'] for k in aa),
     lost=sum(aa[k]['correct'] and not bb[k]['correct'] for k in aa),
     nonempty_n=sum(bool(r['gold']) for r in pa['rows']),
     clean_nonempty_correct=sum(r['correct'] for r in pa['rows'] if r['gold']),
     error_nonempty_correct=sum(r['correct'] for r in pb['rows'] if r['gold']))
   results[proto+'_'+case]=row
 (OUT/'summary.json').write_text(json.dumps(results,indent=2)+'\n')
 for key,x in results.items():
  print('\n',key)
  for name,v in x['aggregates'].items():print(name,json.dumps(v))
  print('groups',json.dumps({g:{k:v for k,v in d.items() if k!='examples'} for g,d in x['paired'].items()}))
  print('ablation',json.dumps(x['ablation']))
  print('clean_prefix',json.dumps(x['clean_prefix']))

if __name__=='__main__':main()
