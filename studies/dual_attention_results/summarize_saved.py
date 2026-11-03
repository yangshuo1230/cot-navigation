"""Read-only audit of saved dual-attention test rollouts; no model inference."""
from pathlib import Path
from types import SimpleNamespace
from collections import Counter
import json
import sys
from datetime import datetime, timezone
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from cot_navigation.config import load_config
from cot_navigation.protocols.tokenizer import Tokenizer
from cot_navigation.protocols.dfs import dfs_search
from cot_navigation.protocols.bfs import bfs_search
from cot_navigation.protocols.render import render_steps
from cot_navigation.analysis.legal_history import state_audit
OUT=ROOT/'outputs/dual_attention_v1/launches/dual_attention_v1_s0_20261023_r1'
results=[]
for config in sorted(OUT.glob('*/*/*/resolved_config.yaml')):
    d=config.parent;cfg=load_config([config]);tok=Tokenizer.from_config(cfg)
    fifo=cfg.protocol.name=='bfs_state';rows=[];errors=Counter();checks=0
    for line in (d/'eval/predictions_test_last.jsonl').read_text().splitlines():
        row=json.loads(line);prompt=row['prompt_tokens'];body=row['trace_tokens']
        qi=prompt.index(tok.id('<QUERY>'));assert qi%3==0 and len(prompt)==qi+4
        adj=[[] for _ in range(cfg.data.node_vocab)]
        nodes=[tok.node_id(i) for i in range(cfg.data.node_vocab)]
        for at in range(0,qi,3):
            assert prompt[at]==tok.id('<EDGE>')
            adj[prompt[at+1]-tok.node_lo].append(prompt[at+2]-tok.node_lo)
        ex=SimpleNamespace(nodes=nodes,adj=adj,source=prompt[qi+1]-tok.node_lo,target=prompt[qi+2]-tok.node_lo)
        # Check reconstructed graphs and the oracle against their exact references.
        steps=(bfs_search if fifo else dfs_search)(adj,ex.source,ex.target)
        label=int(steps[-1].action=='FOUND');assert label==row['label']
        gold=render_steps(steps,nodes,tok)
        assert state_audit(ex,gold,label,tok,fifo)['complete_valid']
        broken=list(gold);broken[0]=nodes[(ex.source+1)%len(nodes)]
        assert not state_audit(ex,broken,label,tok,fifo)['complete_valid']
        checks+=2
        stopped=tok.id('<STOP>') in body
        if stopped:body=body[:body.index(tok.id('<STOP>'))]
        audit=state_audit(ex,body,row['pred'],tok,fifo)
        complete=bool(audit['complete_valid'] and not stopped)
        if audit['first_error']:errors[audit['first_error']['kind']]+=1
        rows.append(dict(query_id=row['query_id'],correct=bool(row['correct']),
            complete_valid=complete,prefix_valid=audit['prefix_valid'],stopped=stopped,
            printed_records=audit['printed_records'],strict_valid_records=audit['strict_valid_records'],
            first_error=audit['first_error'],local=audit['local']))
    n=len(rows);assert n==256
    record={'job':str(d.relative_to(OUT)), 'n':n,'oracle_checks':checks,
        'complete_valid_rate':sum(x['complete_valid'] for x in rows)/n,
        'joint_valid_correct_rate':sum(x['complete_valid'] and x['correct'] for x in rows)/n,
        'correct_but_invalid_rate':sum(not x['complete_valid'] and x['correct'] for x in rows)/n,
        'prefix_valid_rate':sum(x['prefix_valid'] for x in rows)/n,
        'first_errors':dict(errors), 'mean_trace_records':sum(x['printed_records'] for x in rows)/n,
        'local_error_queries':{key:sum(x['local'].get(key,0)>0 for x in rows) for key in
            ['candidate_incorrect_records','nonedge_push_records','visited_push_records','frontier_jump_records','visited_update_error_records']},
        'tests':{},'last_sets':{},'by_query':rows}
    for tag in ['best','last']:
        report=json.loads((d/'eval'/f'free_test_{tag}.json').read_text())
        record['tests'][tag]={'step':report['checkpoint_step'],**report['overall']}
    for f in sorted((d/'eval').glob('free_test_last*.json')):
        name=f.stem[len('free_test_last'):].lstrip('-') or 'main'
        z=json.loads(f.read_text());record['last_sets'][name]=z['overall']
    metrics=[json.loads(l) for l in (d/'metrics.jsonl').read_text().splitlines()]
    updates={x['step']:x for x in metrics if 'total_loss' in x}
    final=updates[max(updates)];record['last_loss']={k:final[k] for k in ['total_loss','trace_loss','answer_loss','update_time']}
    tf=json.loads((d/'eval/teacher_forced_00140000.json').read_text());record['teacher_forced_accuracy']=tf['overall']['answer_accuracy']
    results.append(record)
    print(record['job'], 'acc',record['tests']['last']['answer_accuracy'],'legal',record['complete_valid_rate'],'joint',record['joint_valid_correct_rate'],'errors',record['first_errors'],flush=True)
output={'checked_at_utc':datetime.now(timezone.utc).isoformat(),'source':str(OUT),
        'training_steps':140000,'status':json.loads((OUT/'status.json').read_text()),'runs':results,
        'note':'Audits exact stored last-checkpoint main-test traces. Sets are order invariant; legal alternative PUSH choices accepted. Completeness requires correct FOUND/EXHAUST terminal and final label; STOP is incomplete. No new model inference.'}
Path(__file__).with_name('summary.json').write_text(json.dumps(output,indent=2)+'\n')
