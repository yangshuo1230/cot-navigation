"""Query-balanced confirmation summaries; bootstrap queries, not tokens."""
import json,gzip
from pathlib import Path
from collections import defaultdict,Counter
import numpy as np
ROOT=Path(__file__).resolve().parent
RNG=np.random.default_rng(20261030)

def load(name):
    with gzip.open(ROOT/name,'rt',encoding='utf-8') as f:return [json.loads(line) for line in f]

def balanced(es,fn):
    groups=defaultdict(list)
    for e in es:
        v=fn(e)
        if v is not None:groups[e['index']].append(np.asarray(v,float))
    if not groups:return None
    values=np.stack([np.mean(v,axis=0) for v in groups.values()]);n=len(values)
    boot=values[RNG.integers(n,size=(600,n))].mean(1)
    return dict(mean=values.mean(0).tolist(),low=np.percentile(boot,2.5,axis=0).tolist(),high=np.percentile(boot,97.5,axis=0).tolist(),n_queries=n,n_events=sum(len(v) for v in groups.values()))

def summarize(es):
    out=dict(groups=balanced(es,lambda e:e['group_mass']),accuracy=balanced(es,lambda e:float(e['next_correct'])))
    names=sorted(set(k for e in es for k in e['route_mass']))
    # A zero is retained where a route has no matching positions. Never
    # silently condition on an attention weight being nonzero.
    out['routes']={k:balanced(es,lambda e:e['route_mass'].get(k,[0]*6)) for k in names}
    nodes=[e for e in es if e['two_hop']]
    if nodes:
        for key in ['total','via_symbol','via_action','via_symbol_control']:
            out['two_'+key]=balanced(nodes,lambda e:[x[key] for x in e['two_hop']])
        out['two_symbol_difference']=balanced(nodes,lambda e:[x['via_symbol']-x['via_symbol_control'] for x in e['two_hop']])
        out['three_symbol']=balanced(nodes,lambda e:[x['via_first_symbol'] for x in e['three_hop']])
    out['by_record_position']={band:balanced([e for e in es if (e['record_fraction']<1/3 if band=='early' else 1/3<=e['record_fraction']<2/3 if band=='middle' else e['record_fraction']>=2/3)],lambda e:e['group_mass']) for band in ['early','middle','late']}
    return out

def main():
    events=load('events.jsonl.gz');free=load('free_events.jsonl.gz')
    plan=json.loads((ROOT/'plan.json').read_text());groups=plan['groups'];by=defaultdict(list)
    for e in events:by[(e['tag'],e['split'],e['phase'])].append(e)
    results={}
    for (tag,split,phase),es in by.items():results.setdefault(tag,{}).setdefault(split,{})[phase]=summarize(es)
    freeby=defaultdict(list)
    for e in free:freeby[e['phase']].append(e)
    free_results={phase:summarize(es) for phase,es in freeby.items()}
    subgroups=defaultdict(list)
    for e in events:
        if e['tag']=='n100' and e['split']=='test' and e['sub']:
            subgroups[e['phase']+'::'+e['sub']].append(e)
    sub_results={key:summarize(es) for key,es in subgroups.items()}
    long_results={phase:summarize([e for e in es if e['length']>2048]) for (tag,split,phase),es in by.items()
                  if tag=='n100' and split=='test' and any(e['length']>2048 for e in es)}
    confirmed={}
    for tag in results:
        confirmed[tag]={}
        for phase,val in results[tag]['val'].items():
            if phase not in results[tag]['test']:continue
            test=results[tag]['test'][phase];features={}
            for route,disc in val['routes'].items():
                layer=int(np.argmax(disc['mean']));check=test['routes'].get(route)
                if check:features[route]=dict(layer=layer+1,discovery_mean=disc['mean'][layer],test_mean=check['mean'][layer],test_low=check['low'][layer],test_high=check['high'][layer],n_queries=check['n_queries'],n_events=check['n_events'])
            confirmed[tag][phase]=features
    examples=[]
    for phase in sorted({e['phase'] for e in events}):
        pool=[e for e in events if e['tag']=='n100' and e['split']=='test' and e['phase']==phase and e['length']>=2000]
        if not pool:continue
        # Explicit illustrative selection by relay mass. Not an average or
        # independently confirmed optimum. All events remain in raw results.
        best=max(pool,key=lambda e:max([r['path_mass'] for r in e['best_symbol_relays']] or [0]))
        examples.append(best)
    output=dict(groups=groups,panels=results,free=free_results,subphases=sub_results,long_only=long_results,confirmed_routes=confirmed,illustrative_examples=examples,
       reference_events=len(events),free_events=len(free),bootstrap='600 query resamples; per-query phase means first; percentiles 2.5/97.5',
       note='Reference next-token accuracy includes legal alternative neighbor orders as mismatches. It is not a DFS semantic accuracy score. Route columns may overlap; never sum them. Adjacent-layer attention-only products are descriptive, not causal contributions.')
    (ROOT/'statistics.json').write_text(json.dumps(output,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    for phase,stats in results['n100']['test'].items():
        matrix=np.array(stats['groups']['mean']);best=[]
        for li,row in enumerate(matrix):
            order=np.argsort(-row)[:2];best.append('L'+str(li+1)+':'+','.join(groups[k]+' '+str(round(100*row[k],1)) for k in order))
        print(phase,'nq',stats['groups']['n_queries'],'ne',stats['groups']['n_events'],' | '.join(best))
    print('EVENTS',len(events),len(free))

if __name__=='__main__':main()
