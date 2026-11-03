"""Classify semantic failures from saved rollouts; no additional inference."""
from pathlib import Path
from collections import Counter
import gzip
import json
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from cot_navigation.config import load_config
from cot_navigation.data.dataset import ContextStore,ParametricWorld
from cot_navigation.protocols.tokenizer import Tokenizer

OUT=ROOT/'studies/scale_trace_audit'
details=[]
for p in (OUT/'results').rglob('*.json'):
    if p.name=='complete.json':continue
    result=json.loads(p.read_text())
    cfg=load_config([ROOT/'outputs/scale_v1/launches/scale_v1_s0'/result['job']/'resolved_config.yaml'])
    tok=Tokenizer.from_config(cfg)
    context=cfg.data.knowledge=='context_graph'
    source=ContextStore(result['source']) if context else ParametricWorld(result['source'])
    arrays=source.arrays if context else source.queries
    c=Counter(); examples=[]
    with gzip.open(p.with_suffix('.jsonl.gz'),'rt') as handle:
        for line in handle:
            row=json.loads(line);err=row['error'];idx=row['index']
            if row['complete_valid']:continue
            if cfg.protocol.name.endswith('_state'):
                c['first_record_error' if row['valid_records']==0 else 'later_record_error']+=1
            if err=='non_node_token':
                nodes=set(source.node_vocab_ids(idx,tok) if context else source.node_vocab_ids(tok))
                bad=[x for x in row['trace_tokens'] if x not in nodes and x!=tok.stoi.get('<STOP>')]
                c['outside_graph_node']+=int(any(tok.node_lo<=x<tok.node_lo+tok.n_node_tokens for x in bad))
                c['syntax_in_node_sequence']+=int(any(not tok.node_lo<=x<tok.node_lo+tok.n_node_tokens for x in bad))
            if err!='candidate_mismatch':continue
            nodes=source.node_vocab_ids(idx,tok) if context else source.node_vocab_ids(tok)
            raw=source.adj(idx) if context else source.adj()
            adj={nodes[i]:[nodes[j] for j in vs] for i,vs in enumerate(raw)}
            start=0;records=[]
            for end,x in enumerate(row['trace_tokens']):
                if x==tok.id(';'):
                    records.append(row['trace_tokens'][start:end]);start=end+1
            rec=records[row['valid_records']];fields=[[]]
            for x in rec:
                if x==tok.id('|'):fields.append([])
                else:fields[-1].append(x)
            frontier,vis,cands,action=fields
            cur=frontier[0] if cfg.protocol.name=='bfs_state' else frontier[-1]
            expected=set(adj[cur])-set(vis)
            missing=expected-set(cands);extra=set(cands)-expected
            c['candidate_errors']+=1
            c['candidate_error_at_target']+=int(cur==nodes[int(arrays['t_topo'][idx])])
            c['candidate_missing_only']+=int(bool(missing) and not extra)
            c['candidate_extra_only']+=int(bool(extra) and not missing)
            c['candidate_missing_and_extra']+=int(bool(missing) and bool(extra))
            c['candidate_duplicate']+=int(len(cands)!=len(set(cands)))
            c['extra_contains_non_successor']+=int(any(v not in adj[cur] for v in extra))
            c['extra_contains_visited_successor']+=int(any(v in adj[cur] and v in vis for v in extra))
            c['extra_contains_foreign_node']+=int(any(v not in adj for v in extra))
            if action and action[0]==tok.id('<PUSH>') and len(action)==2:
                c['push_immediately_illegal_at_candidate_error']+=int(action[1] not in expected)
            if len(examples)<3:
                examples.append(dict(index=idx,record_index=row['valid_records'],current=tok.name(cur),
                                     target=tok.name(nodes[int(arrays['t_topo'][idx])]),
                                     expected=[tok.name(x) for x in sorted(expected)],
                                     actual=[tok.name(x) for x in cands],action=[tok.name(x) for x in action],
                                     actual_successors=[tok.name(x) for x in adj[cur]],
                                     visited=[tok.name(x) for x in vis]))
    details.append(dict(job=result['job'],case=result['case'],n=result['n'],counts=dict(c),examples=examples))
(OUT/'failure_details.json').write_text(json.dumps(details,ensure_ascii=False,indent=2))
print('Failure classifications saved for',len(details),'finished conditions')
