"""CPU-only analysis of candidate first-error types, position and next action."""
from pathlib import Path
from collections import Counter
import json,gzip,statistics,sys
F=Path(__file__).parent;ROOT=F.parents[1];sys.path.insert(0,str(F))
from audit import parse,TOK
MODELS=['baseline/dfs_state','dfs_state/parallel/causal','dfs_state/serial/causal','dfs_state/serial/bidirectional','dfs_state/parallel/bidirectional',
        'baseline/bfs_state','bfs_state/serial/bidirectional','bfs_state/parallel/causal','bfs_state/parallel/bidirectional','bfs_state/serial/causal']
results=[];examples=[];audits={}
for model in MODELS:
    for file in sorted((F/'audits'/model).glob('*.json')):
        audit=json.loads(file.read_text());case=audit['case'];audits[(model,case)]=audit
        firsts=[x for x in audit['by_query'] if x['first_error'] and x['first_error']['kind']=='candidate_mismatch']
        positions=[x['first_error']['record']+1 for x in firsts]
        combos=Counter('+'.join(sorted(x['first_candidate_subtypes'])) for x in firsts)
        result={'model':model,'case':case,'n':256,'candidate_first_error_n':len(firsts),
            'subtypes':dict(Counter(k for x in firsts for k in x['first_candidate_subtypes'])),
            'combinations':dict(combos),'median_first_error_record':statistics.median(positions) if positions else None,
            'first_record_errors':sum(x==1 for x in positions),'by_query':[{'query_id':x['query_id'],'record':x['first_error']['record']+1,'subtypes':x['first_candidate_subtypes']} for x in firsts]}
        if case=='main':
            source=Path(audit['source'])
            if model.startswith('baseline/'):
                rows=[]
                for q in sorted(source.glob('batch_*.json.gz')):
                    with gzip.open(q,'rt') as h:rows.extend(json.load(h)['rows'])
            else:rows=[json.loads(x) for x in source.read_text().splitlines()]
            byid={x['query_id']:x for x in rows};actions=Counter();samples=[]
            for obs in firsts:
                row=byid[obs['query_id']];prompt=row['prompt_tokens'];qi=prompt.index(TOK.id('<QUERY>'));src,dst=prompt[qi+1:qi+3];adj={TOK.node_id(i):set() for i in range(256)}
                for at in range(0,qi,3):adj[prompt[at+1]].add(prompt[at+2])
                records,_=parse(row['trace_tokens']);ff,vv,cc,aa=records[obs['first_error']['record']]
                cur=ff[0 if audit['protocol']=='bfs_state' else -1];eligible=adj[cur]-set(vv)
                expected='FOUND' if cur==dst else 'PUSH' if eligible else 'POP' if len(ff)>1 else 'EXHAUST'
                ok=bool(aa) and aa[0]==TOK.id('<'+expected+'>') and len(aa)==(2 if expected=='PUSH' else 1) and (expected!='PUSH' or aa[1] in eligible)
                if ok:kind='locally_legal_action_despite_bad_candidates'
                elif len(aa)==2 and aa[0]==TOK.id('<PUSH>') and aa[1] not in adj[cur]:kind='push_nonedge'
                elif len(aa)==2 and aa[0]==TOK.id('<PUSH>') and aa[1] in vv:kind='push_visited'
                elif aa and aa[0] in [TOK.id('<POP>'),TOK.id('<EXHAUST>')] and eligible:kind='premature_pop_or_exhaust'
                else:kind='other_illegal_action'
                actions[kind]+=1
                samples.append({'query_id':obs['query_id'],'record':obs['first_error']['record']+1,
                    'subtypes':obs['first_candidate_subtypes'],'current':TOK.name(cur),
                    'true_neighbors':[TOK.name(v) for v in sorted(adj[cur])],
                    'visited_neighbors':[TOK.name(v) for v in sorted(adj[cur]&set(vv))],
                    'expected_candidates':[TOK.name(v) for v in sorted(eligible)],
                    'shown_candidates':[TOK.name(v) for v in cc],
                    'action':[TOK.name(v) for v in aa],'action_kind':kind})
            result['first_error_action_counts']=dict(actions);result['examples']=samples
            print(model,'nerrors',len(firsts),'median',result['median_first_error_record'],'actions',dict(actions),flush=True)
        results.append(result)
pairs=[]
for old,new in [('baseline/dfs_state','dfs_state/parallel/causal'),('baseline/dfs_state','dfs_state/serial/causal'),('baseline/bfs_state','bfs_state/serial/bidirectional')]:
    for case in ['main','all9']:
        differences=[];transitions=Counter()
        cases=['n16','n24','n32','n36','n64','main','deep','wide','larger'] if case=='all9' else ['main']
        for c in cases:
            a={x['query_id']:x for x in audits[(old,c)]['by_query']};b={x['query_id']:x for x in audits[(new,c)]['by_query']}
            for q,x in a.items():
                y=b[q]
                if x['first_error'] and y['first_error']:
                    differences.append(y['first_error']['record']-x['first_error']['record'])
                    if x['first_error']['kind']==y['first_error']['kind']=='candidate_mismatch':
                        agraph='nonedge' in x['first_candidate_subtypes'];bgraph='nonedge' in y['first_candidate_subtypes']
                        transitions[('nonedge' if agraph else 'no_nonedge')+'->'+('nonedge' if bgraph else 'no_nonedge')]+=1
        pairs.append({'baseline':old,'dual':new,'case':case,'both_fail_n':len(differences),'new_fails_later':sum(v>0 for v in differences),
            'new_fails_earlier':sum(v<0 for v in differences),'same_record':sum(v==0 for v in differences),
            'median_record_shift':statistics.median(differences) if differences else None,'type_transitions':dict(transitions)})
# Save one same-query illustration per protocol, old nonedge error vs new pure visited error.
for old,new in [('baseline/dfs_state','dfs_state/parallel/causal'),('baseline/bfs_state','bfs_state/serial/bidirectional')]:
    a=next(x for x in results if x['model']==old and x['case']=='main');b=next(x for x in results if x['model']==new and x['case']=='main');aa={x['query_id']:x for x in a['examples']}
    choices=[{'baseline':aa[x['query_id']],'dual':x,'models':[old,new]} for x in b['examples'] if x['query_id'] in aa and 'nonedge' in aa[x['query_id']]['subtypes'] and x['subtypes']==['visited']]
    if choices:examples.append(min(choices,key=lambda x:x['dual']['record']))
(F/'candidate_error_modes.json').write_text(json.dumps({'results':results,'paired':pairs,'matched_examples':examples,
    'note':'First semantic error only: all prior records are valid. Subtypes overlap. First-error position is a generated-record index, not graph hops; differing paths and exposure imply competing risks. All9 pools equal-sized graph setups.'},indent=2)+'\n')
print('MATCHED EXAMPLES',json.dumps(examples),flush=True)
