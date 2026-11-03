"""CPU-only graph-grounding and correctness decomposition on saved rollouts."""
from pathlib import Path
from collections import Counter
import json,sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from cot_navigation.config import load_config
from cot_navigation.protocols.tokenizer import Tokenizer
OUT=ROOT/'outputs/dual_attention_v1/launches/dual_attention_v1_s0_20261023_r1'
summary=json.loads(Path(__file__).with_name('summary.json').read_text())
allrows={};results=[]
for run in summary['runs']:
    name=run['job'];d=OUT/name;cfg=load_config([d/'resolved_config.yaml']);tok=Tokenizer.from_config(cfg)
    audits={r['query_id']:r for r in run['by_query']};rows=[];local=Counter();prefix=Counter()
    for line in (d/'eval/predictions_test_last.jsonl').read_text().splitlines():
        row=json.loads(line);prompt=row['prompt_tokens'];qi=prompt.index(tok.id('<QUERY>'));src=prompt[qi+1]
        adj={tok.node_id(i):set() for i in range(cfg.data.node_vocab)}
        for at in range(0,qi,3):adj[prompt[at+1]].add(prompt[at+2])
        assert src not in adj[src] # first record has no visited out-neighbor in a DAG
        body=row['trace_tokens'];start=0;records=[]
        for end,value in enumerate(body):
            if value!=tok.id(';'):continue
            fields=[[]]
            for v in body[start:end]:
                if v==tok.id('|'):fields.append([])
                else:fields[-1].append(v)
            start=end+1
            if len(fields)!=4:records.append(None);continue
            ff,vv,cc,aa=fields;records.append(fields)
            if not ff or any(x not in adj for x in ff+vv):continue
            cur=ff[0 if cfg.protocol.name=='bfs_state' else -1];neighbors=adj[cur]
            expected=neighbors-set(vv);chosen=set(cc)
            metrics={'n':1,'exact':chosen==expected and len(chosen)==len(cc),
                'nonedge':bool(chosen-neighbors),'visited_included':bool(chosen&set(vv)),
                'omission':bool(expected-chosen),'duplicate':len(chosen)!=len(cc),
                'eligible_tp':len(chosen&expected),'eligible_pred':len(cc),'eligible_gold':len(expected)}
            local.update(metrics)
            first=audits[row['query_id']]['first_error']
            # Include the first failing record itself. Before its candidate
            # field, all frontier/visited state and actions are still valid.
            index=len(records)-1
            if first is None or index<=first['record']:
                prefix.update(metrics)
        assert records and records[0] is not None
        ff,vv,cc,aa=records[0];assert ff==[src] and vv==[src]
        gold=adj[src];chosen=set(cc)
        first_exact=gold==chosen and len(chosen)==len(cc)
        rows.append({'query_id':row['query_id'],'correct':row['correct'],
            'legal':audits[row['query_id']]['complete_valid'],
            'first_candidate_exact':first_exact,'first_outdegree':len(gold),
            'first_nonedge':bool(chosen-gold),'first_omission':bool(gold-chosen),
            'first_candidate_tp':len(gold&chosen),'first_candidate_pred':len(cc)})
    n=len(rows);nonempty=[x for x in rows if x['first_outdegree']]
    def stats(counter):
        n=counter['n']
        return {'records':n,**{k:counter[k]/n for k in ['exact','nonedge','visited_included','omission','duplicate']},
                'micro_precision':counter['eligible_tp']/max(1,counter['eligible_pred']),
                'micro_recall':counter['eligible_tp']/max(1,counter['eligible_gold'])}
    result={'job':name,'n':n,'accuracy':run['tests']['last']['answer_accuracy'],
        'complete_legal_rate':run['complete_valid_rate'],
        'correct_invalid_rate':run['correct_but_invalid_rate'],
        'first_candidate_exact':sum(x['first_candidate_exact'] for x in rows)/n,
        'first_candidate_nonempty_n':len(nonempty),
        'first_candidate_nonempty_exact':sum(x['first_candidate_exact'] for x in nonempty)/len(nonempty),
        'first_candidate_nonedge_queries':sum(x['first_nonedge'] for x in rows),
        'first_candidate_omission_queries':sum(x['first_omission'] for x in rows),
        'first_candidate_precision':sum(x['first_candidate_tp'] for x in rows)/max(1,sum(x['first_candidate_pred'] for x in rows)),
        'first_candidate_recall':sum(x['first_candidate_tp'] for x in rows)/sum(x['first_outdegree'] for x in rows),
        'all_records':stats(local),'valid_prefix_including_first_error':stats(prefix),
        'by_query':rows}
    results.append(result);allrows[name]={x['query_id']:x for x in rows}
    print(json.dumps({k:v for k,v in result.items() if k not in ['by_query','all_records','valid_prefix_including_first_error']}),flush=True)
pairs=[]
for a,b in [('dfs_state/parallel/bidirectional','dfs_state/parallel/causal'),
            ('dfs_state/serial/causal','dfs_state/serial/bidirectional'),
            ('bfs_state/serial/causal','bfs_state/serial/bidirectional'),
            ('bfs_state/parallel/causal','bfs_state/serial/bidirectional')]:
    aa,bb=allrows[a],allrows[b];assert set(aa)==set(bb);counts=Counter()
    for q,x in aa.items():
        y=bb[q]
        counts['n']+=1
        counts['wrong_to_correct']+=int(not x['correct'] and y['correct'])
        counts['correct_to_wrong']+=int(x['correct'] and not y['correct'])
        counts['invalid_to_valid']+=int(not x['legal'] and y['legal'])
        counts['valid_to_invalid']+=int(x['legal'] and not y['legal'])
        counts['wrong_to_correct_and_now_legal']+=int(not x['correct'] and y['correct'] and y['legal'])
        counts['wrong_to_correct_and_still_invalid']+=int(not x['correct'] and y['correct'] and not y['legal'])
        counts['first_candidate_fixed']+=int(not x['first_candidate_exact'] and y['first_candidate_exact'])
        counts['first_candidate_broken']+=int(x['first_candidate_exact'] and not y['first_candidate_exact'])
    pair={'from':a,'to':b,**dict(counts)};pairs.append(pair);print('PAIR',json.dumps(pair),flush=True)
Path(__file__).with_name('candidate_analysis.json').write_text(json.dumps({'runs':results,'paired':pairs,
    'note':'Observational comparisons, not causal mediation. First candidate exactness is pure source out-neighbor extraction: first frontier=visited={source}, DAG has no self loops. All models share exact main-test queries. Whole-rollout record metrics include downstream errors; valid-prefix metrics stop after the first error.'},indent=2)+'\n')
