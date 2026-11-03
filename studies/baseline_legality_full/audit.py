"""Semantic audits of exact stored rollouts; independent of final answer tokens."""
from pathlib import Path
from types import SimpleNamespace
from collections import Counter
import sys,json,gzip,hashlib
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from cot_navigation.analysis.legal_history import state_audit
from cot_navigation.protocols.tokenizer import Tokenizer
from cot_navigation.protocols.dfs import dfs_search
from cot_navigation.protocols.bfs import bfs_search
from cot_navigation.protocols.render import render_steps
TOK=Tokenizer(256,enable_stop=True);FOLDER=Path(__file__).parent
VERSION=1

def parse(body):
    records=[];start=0
    for end,t in enumerate(body):
        if t!=TOK.id(';'):continue
        fields=[[]]
        for v in body[start:end]:
            if v==TOK.id('|'):fields.append([])
            else:fields[-1].append(v)
        records.append(fields if len(fields)==4 else None);start=end+1
    return records,start==len(body)

def executable(records,finished,adj,src,dst,fifo):
    frontier=[src];seen={src};terminal=None
    for fields in records:
        if fields is None or terminal:return False
        ff,vv,cc,aa=fields
        if ff!=frontier or len(set(vv))!=len(vv) or set(vv)!=seen:return False
        cur=frontier[0 if fifo else -1];eligible=adj[cur]-seen
        action='FOUND' if cur==dst else 'PUSH' if eligible else 'POP' if len(frontier)>1 else 'EXHAUST'
        if not aa or aa[0]!=TOK.id('<'+action+'>') or len(aa)!=(2 if action=='PUSH' else 1):return False
        if action=='PUSH':
            if aa[1] not in eligible:return False
            frontier.append(aa[1]);seen.add(aa[1])
        elif action=='POP':frontier.pop(0 if fifo else -1)
        else:terminal=action
    return finished and terminal is not None

def evaluate(row,protocol):
    prompt=row['prompt_tokens'];qi=prompt.index(TOK.id('<QUERY>'));assert qi%3==0 and len(prompt)==qi+4
    nodes=[TOK.node_id(i) for i in range(256)];adj={v:set() for v in nodes}
    for at in range(0,qi,3):
        assert prompt[at]==TOK.id('<EDGE>');adj[prompt[at+1]].add(prompt[at+2])
    src,dst=prompt[qi+1:qi+3];fifo=protocol=='bfs_state'
    adjacency=[[v-TOK.node_lo for v in sorted(adj[u])] for u in nodes]
    ex=SimpleNamespace(nodes=nodes,adj=adjacency,source=src-TOK.node_lo,target=dst-TOK.node_lo)
    gold=(bfs_search if fifo else dfs_search)(adjacency,ex.source,ex.target)
    label=int(gold[-1].action=='FOUND');assert label==row['label']
    goldbody=render_steps(gold,nodes,TOK)
    assert state_audit(ex,goldbody,label,TOK,fifo)['complete_valid']
    records,finished=parse(goldbody);assert executable(records,finished,adj,src,dst,fifo)
    # A wrong CANDS annotation must fail strict validation but preserve execution.
    bad=list(goldbody);bars=[i for i,v in enumerate(bad) if v==TOK.id('|')]
    bad.insert(bars[2],TOK.id('<DOT>'))
    assert not state_audit(ex,bad,label,TOK,fifo)['complete_valid']
    rr,ff=parse(bad);assert executable(rr,ff,adj,src,dst,fifo)
    body=list(row['trace_tokens']);stopped=TOK.id('<STOP>') in body
    if stopped:body=body[:body.index(TOK.id('<STOP>'))]
    # Use true label here: legality of the trace does NOT require the model's
    # final answer token to be correct. Report answer and joint metrics separately.
    audit=state_audit(ex,body,label,TOK,fifo)
    complete=bool(audit['complete_valid'] and not stopped)
    records,finished=parse(body);action_valid=bool(not stopped and executable(records,finished,adj,src,dst,fifo))
    first=audit['first_error'];local=Counter();valid_history=Counter();first_subtypes=[]
    for index,fields in enumerate(records):
        if fields is None:local['malformed']+=1;continue
        frontier,visited,candidates,action=fields
        if not frontier or any(x not in adj for x in frontier+visited):continue
        current=frontier[0 if fifo else -1];neighbors=adj[current];expected=neighbors-set(visited);chosen=set(candidates)
        counts={'n':1,'exact':int(chosen==expected and len(chosen)==len(candidates)),
           'nonedge':int(bool(chosen-neighbors)),'visited':int(bool(chosen&set(visited))),
           'omitted':int(bool(expected-chosen)),'duplicate':int(len(chosen)!=len(candidates)),
           'tp':len(chosen&expected),'pred':len(candidates),'gold':len(expected)}
        local.update(counts)
        before_error=(first is None or index<first['record'] or index==first['record'] and first['kind'] not in ['frontier_mismatch','visited_mismatch','malformed_record','record_after_terminal'])
        if before_error:valid_history.update(counts)
        if first and index==first['record'] and first['kind']=='candidate_mismatch':
            first_subtypes=[k for k in ['nonedge','visited','omitted','duplicate'] if counts[k]]
    first_candidates=records[0][2] if records and records[0] is not None else None
    first_exact=first_candidates is not None and set(first_candidates)==adj[src] and len(first_candidates)==len(set(first_candidates))
    return dict(query_id=row['query_id'],label=label,distance=row['distance'],correct=row['pred']==label,
        complete_valid=complete,actions_valid=action_valid,prefix_valid=audit['prefix_valid'],
        correct_invalid=row['pred']==label and not complete,joint=complete and row['pred']==label,
        valid_wrong_answer=complete and row['pred']!=label,annotation_only_invalid=action_valid and not complete,
        stopped=stopped,forced_stop=row['forced_stop'],format_fail=row['format_fail'],
        trace_records=len(records),trace_tokens=len(body),strict_prefix_records=audit['strict_valid_records'],
        first_error=first,first_candidate_subtypes=first_subtypes,local=dict(local),valid_history=dict(valid_history),
        first_candidate_exact=first_exact,first_outdegree=len(adj[src]))

def summarize(rows):
    n=len(rows);assert n
    keys=['correct','complete_valid','actions_valid','prefix_valid','correct_invalid','joint','valid_wrong_answer','annotation_only_invalid','stopped','forced_stop','format_fail','first_candidate_exact']
    out={'n':n,**{k:sum(bool(x[k]) for x in rows)/n for k in keys},
        'counts':{k:sum(bool(x[k]) for x in rows) for k in keys},
        'first_errors':dict(Counter(x['first_error']['kind'] for x in rows if x['first_error'])),
        'first_candidate_subtypes':dict(Counter(k for x in rows for k in x['first_candidate_subtypes'])),
        'mean_records':sum(x['trace_records'] for x in rows)/n}
    for group in ['local','valid_history']:
        counts=Counter()
        for x in rows:counts.update(x[group])
        out[group]={'counts':dict(counts),**{k:counts[k]/max(1,counts['n']) for k in ['exact','nonedge','visited','omitted','duplicate']},
            'precision':counts['tp']/max(1,counts['pred']),'recall':counts['tp']/max(1,counts['gold'])}
    return out

def audit_file(source,out,model,protocol,case,baseline=False):
    if baseline:
        contract=json.loads((source/'contract.json').read_text());files=sorted(source.glob('batch_*.json.gz'))
        fingerprint=hashlib.sha256(''.join(hashlib.sha256(p.read_bytes()).hexdigest() for p in files).encode()).hexdigest()
        rows=[]
        for p in files:
            with gzip.open(p,'rt') as f:rows.extend(json.load(f)['rows'])
    else:
        fingerprint=hashlib.sha256(source.read_bytes()).hexdigest();rows=[json.loads(x) for x in source.read_text().splitlines()];contract=None
    if out.exists():
        old=json.loads(out.read_text())
        if old.get('version')==VERSION and old['input_sha256']==fingerprint:return old
    assert len(rows)==256 and len({x['query_id'] for x in rows})==256
    byquery=[evaluate(row,protocol) for row in rows]
    result={'version':VERSION,'model':model,'protocol':protocol,'case':case,'input_sha256':fingerprint,
        'source':str(source),'contract':contract,'overall':summarize(byquery),
        'by_label':{str(k):summarize([x for x in byquery if x['label']==k]) for k in [0,1]},
        'by_distance':{str(k):summarize([x for x in byquery if x['distance']==k]) for k in sorted({x['distance'] for x in byquery})},
        'by_query':byquery}
    out.parent.mkdir(parents=True,exist_ok=True);tmp=out.with_suffix('.tmp');tmp.write_text(json.dumps(result,indent=2)+'\n');tmp.replace(out)
    print('AUDIT',model,case,'answer',round(result['overall']['correct'],4),'legal',round(result['overall']['complete_valid'],4),'actions',round(result['overall']['actions_valid'],4),flush=True)
    return result

if __name__=='__main__':
    dual=ROOT/'outputs/dual_attention_v1/launches/dual_attention_v1_s0_20261023_r1'
    for config in sorted(dual.glob('*/*/*/resolved_config.yaml')):
        d=config.parent;name=str(d.relative_to(dual));protocol=name.split('/')[0]
        for source in sorted((d/'eval').glob('predictions_test_last*.jsonl')):
            case=source.stem[len('predictions_test_last'):].lstrip('-') or 'main'
            audit_file(source,FOLDER/'audits'/name/(case+'.json'),name,protocol,case)
    for done in sorted((FOLDER/'baseline').glob('*/*/COMPLETE.json')):
        source=done.parent;protocol,case=source.parts[-2:];name='baseline/'+protocol
        audit_file(source,FOLDER/'audits'/name/(case+'.json'),name,protocol,case,True)
