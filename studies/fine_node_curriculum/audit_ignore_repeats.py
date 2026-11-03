"""Re-score existing predictions after stable first-occurrence deduplication."""
from pathlib import Path
from collections import Counter
from statistics import mean
import json
from cot_navigation.protocols.tokenizer import Tokenizer
from cot_navigation.protocols.trace_audit import validate_nodes

ROOT=Path('/mnt/ningxuefei/yangshuo/cot-navigation-unified')
OUT=ROOT/'outputs/fine_node_curriculum_20261028'
tok=Tokenizer(enable_stop=True)

def evaluate(tokens,label,adj,src,dst,fifo):
    strict=validate_nodes(tokens,label,adj,src,dst,tok,fifo)
    unique=list(dict.fromkeys(tokens))
    relaxed=validate_nodes(unique,label,adj,src,dst,tok,fifo)
    preserve_terminal=relaxed['valid'] and dst not in tokens[:-1]
    return strict,relaxed,preserve_terminal

def self_check():
    n=[tok.node_id(i) for i in range(5)]
    adj={n[0]:[n[1],n[2]],n[1]:[n[3]],n[2]:[],n[3]:[],n[4]:[]}
    cases=[
        ([0,1,1,2,3],1,3,True,True,True),
        ([0,1,1,3],1,3,True,False,False),
        ([0,1,1,3],1,3,False,True,True),
        ([0,1,1,2,3],1,3,False,False,False),
        ([0,1,2,3,3],1,3,True,True,False),
        ([0,1,3,2],1,3,True,False,False),
        ([0,1,1,2,3],0,4,True,True,True),
        ([0,1,1],0,4,True,False,False),
        ([0,1,3,2,2],0,4,False,True,True),
        ([1,1,0,3],1,3,False,False,False),
    ]
    for seq,label,target,fifo,expected,stop_expected in cases:
        _,relaxed,stop=evaluate([n[i] for i in seq],label,adj,n[0],n[target],fifo)
        assert relaxed['valid']==expected,(seq,relaxed)
        assert stop==stop_expected,(seq,stop)
    return len(cases)

runs=[]
for p in sorted(OUT.glob('s*/*/*')):
    records=[json.loads(x) for x in (p/'eval/predictions_test_last.jsonl').read_text().splitlines()]
    fifo=p.parent.name=='bfs_nodes'
    stats={name:Counter() for name in ['all','positive']}
    errors={name:Counter() for name in ['all','positive']}
    details=[]
    for r in records:
        prompt=r['prompt_tokens'];q=prompt.index(tok.id('<QUERY>'))
        assert q%3==0 and len(prompt)==q+4
        adj={tok.node_id(i):[] for i in range(tok.n_node_tokens)}
        for i in range(0,q,3):
            assert prompt[i]==tok.id('<EDGE>')
            adj[prompt[i+1]].append(prompt[i+2])
        strict,relaxed,preserve=evaluate(r['trace_tokens'],r['label'],adj,prompt[q+1],prompt[q+2],fifo)
        for key in ['all']+(['positive'] if r['label']==1 else []):
            c=stats[key];c['n']+=1;c['strict_valid']+=strict['valid'];c['dedup_valid']+=relaxed['valid']
            c['dedup_preserve_original_stop_valid']+=preserve
            c['answer_correct']+=r['correct'];c['answer_and_dedup_valid']+=r['correct'] and relaxed['valid']
            c['first_error_repeat']+=strict['error']=='repeated_node'
            c['recovered']+=relaxed['valid'] and not strict['valid']
            if not relaxed['valid']:errors[key][relaxed['error']]+=1
        assert not strict['valid'] or relaxed['valid']
        details.append(dict(query_id=r['query_id'],label=r['label'],strict=strict,
                            deduplicated=relaxed,preserve_original_target_stop=preserve))
    runs.append(dict(job=str(p.relative_to(OUT)),stats=stats,remaining_errors=errors,records=details))

groups=[]
for protocol in ['dfs_nodes','bfs_nodes']:
    for family in ['simple','hard']:
        selected=[r for r in runs if '/'+protocol+'/'+family in r['job']]
        g=dict(protocol=protocol,family=family)
        for subset in ['all','positive']:
            g[subset]={key:mean(r['stats'][subset][key]/r['stats'][subset]['n'] for r in selected)
                       for key in ['strict_valid','dedup_valid','dedup_preserve_original_stop_valid',
                                   'answer_correct','answer_and_dedup_valid','recovered']}
            errors=Counter()
            for r in selected:errors.update(r['remaining_errors'][subset])
            g[subset]['remaining_errors']=dict(errors)
        groups.append(g)
result=dict(definition='Stable first-occurrence deduplication, then unchanged DFS/BFS order, start and termination validation; denominator unchanged.',
            adversarial_checks=self_check(),runs=runs,groups=groups)
(OUT/'ignore_repeats_20261029.json').write_text(json.dumps(result,indent=2))
for g in groups:
    print(g['protocol'],g['family'])
    for subset in ['positive','all']:
        print(subset,{k:round(v*100,2) if not isinstance(v,dict) else v for k,v in g[subset].items()})
for r in runs:
    c=r['stats']['positive'];print(r['job'],{k:round(c[k]/c['n']*100,2) for k in ['strict_valid','dedup_valid','dedup_preserve_original_stop_valid','answer_and_dedup_valid']})
