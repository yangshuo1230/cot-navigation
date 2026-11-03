"""Find auditable natural repair examples in the four saved previews per set.

This is a qualitative prefix audit, not an estimate over all test queries.
Stop auditing a trace after a malformed or unacknowledged illegal action.
"""
from pathlib import Path
import json

from cot_navigation.config import load_config
from cot_navigation.protocols.tokenizer import Tokenizer

base=Path('outputs/natural_correction_v1/launches/natural_correction_s0')
examples=[];failures=[];summaries=[]
for out in sorted(base.glob('context/*/*')):
    cfg=load_config([out/'resolved_config.yaml']);tok=Tokenizer.from_config(cfg)
    fifo=cfg.protocol.name=='bfs_state'
    stats=dict(previews=0,trusted_mistakes=0,true_errors_marked=0,false_alarms=0,repaired=0,censored=0)
    for p in sorted((out/'traces').glob('free_test_best*.txt')):
        for preview,part in enumerate(p.read_text().strip().split('\n\n')):
            if not part.strip():continue
            ids=[tok.id(t) for t in part.split()]
            query=ids.index(tok.id('<QUERY>'));prompt_len=query+4
            src,dst=ids[query+1:query+3]
            adj={}
            for i in range(0,query,3):
                _,u,v=ids[i:i+3];adj.setdefault(u,set()).add(v)
            stack,visited=[src],[src];start=prompt_len;pending=None
            stats['previews']+=1
            for end in range(start,len(ids)):
                if ids[end]!=tok.id(';'):continue
                fields=[[]]
                for token in ids[start:end]:
                    if token==tok.id('|'):fields.append([])
                    else:fields[-1].append(token)
                record=ids[start:end+1];start=end+1
                if len(fields)!=4 or not stack:
                    stats['censored']+=1;break
                s,v,c,action=fields
                cur=stack[0] if fifo else stack[-1]
                expected_c=adj.get(cur,set())-set(visited)
                wrong=[]
                if s!=stack:wrong.append('stack')
                if len(v)!=len(set(v)) or set(v)!=set(visited):wrong.append('visited')
                if len(c)!=len(set(c)) or set(c)!=expected_c:wrong.append('cands')
                state_ok=not wrong
                if action==[tok.id('<MISTAKE>')]:
                    stats['trusted_mistakes']+=1
                    stats['true_errors_marked']+=int(not state_ok)
                    stats['false_alarms']+=int(state_ok)
                    pending=dict(wrong=wrong,record=record,expected_stack=stack.copy(),expected_visited=visited.copy(),expected_cands=sorted(expected_c))
                    continue
                if cur==dst:wanted=[tok.id('<FOUND>')]
                elif c:wanted=[tok.id('<PUSH>'),c[0]]
                elif len(stack)>1:wanted=[tok.id('<POP>')]
                else:wanted=[tok.id('<EXHAUST>')]
                if not state_ok or action!=wanted:
                    if pending and pending['wrong'] and len([e for e in failures if e['job']==str(out.relative_to(base))])<2:
                        failures.append(dict(job=str(out.relative_to(base)),file=p.name,preview=preview,
                            wrong_fields=pending['wrong'],wrong_record=tok.decode(pending['record']),
                            attempted_repair=tok.decode(record),remaining_wrong_fields=wrong,
                            action_matches_generated_state=action==wanted,
                            expected_cands=tok.decode(sorted(expected_c)),expected_visited=tok.decode(visited)))
                    stats['censored']+=1;break
                if pending and pending['wrong']:
                    stats['repaired']+=1
                    if len([e for e in examples if e['job']==str(out.relative_to(base))])<2:
                        examples.append(dict(job=str(out.relative_to(base)),file=p.name,preview=preview,
                            wrong_fields=pending['wrong'],wrong_record=tok.decode(pending['record']),
                            repair_record=tok.decode(record),
                            expected_cands=tok.decode(pending['expected_cands']),
                            expected_visited=tok.decode(pending['expected_visited'])))
                pending=None
                if action[0]==tok.id('<PUSH>'):stack.append(action[1]);visited.append(action[1])
                elif action[0]==tok.id('<POP>'):stack.pop(0 if fifo else -1)
                else:break
    summaries.append(dict(job=str(out.relative_to(base)),**stats))
study=Path('studies/natural_correction_results')
(study/'natural_preview_audit.json').write_text(json.dumps(dict(summaries=summaries,examples=examples,failures=failures),indent=2)+'\n')
for item in summaries:print(json.dumps(item),flush=True)
for item in examples[:4]:print('EXAMPLE',json.dumps(item),flush=True)
for item in failures[:2]:print('FAILED_REPAIR',json.dumps(item),flush=True)
