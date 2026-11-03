"""Replay validation-selected short-span checkpoints on their exact test IDs.

Frozen inference only. Uses native context/STOP policy and original eval batch
size. Does not save full rollouts or activations. Distinguishes a legal walk,
a complete task-specific trace, executable search actions, and final answers.
"""
from __future__ import annotations
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import sys


def path_audit(ex,body,pred,tok):
    adj={ex.nodes[u]:{ex.nodes[v] for v in vs} for u,vs in enumerate(ex.adj)}
    src,dst=ex.nodes[ex.source],ex.nodes[ex.target]
    error=None
    if not body:error='empty_path'
    elif any(v not in adj for v in body):error='non_node_token'
    elif body[0]!=src:error='wrong_start'
    elif len(body)!=len(set(body)):error='repeated_node'
    elif any(v not in adj[u] for u,v in zip(body,body[1:])):error='non_edge'
    structural=error is None
    reaches=structural and body[-1]==dst
    complete=structural and (reaches if ex.label else not adj[body[-1]])
    if not error and not complete:error='missed_positive_goal' if ex.label else 'negative_not_at_sink'
    return dict(structural_valid=structural,complete_valid=complete,actions_valid=complete,
        first_error=None if not error else dict(kind=error),local={},goal_reached=reaches,
        goal_seen=dst in body,path_edges=len(body)-1,
        early_no=bool(ex.label and structural and not reaches and pred==0 and len(body)-1<ex.target//10-ex.source//10))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu',type=int,required=True)
    parser.add_argument('--models',nargs='+',required=True)
    args=parser.parse_args();os.environ['CUDA_VISIBLE_DEVICES']=str(args.gpu);os.environ['WANDB_MODE']='disabled'
    import torch
    torch.set_num_threads(2)
    sys.path.insert(0,str(Path.cwd()/'src'))
    from cot_navigation.analysis.scale_followup import make_adapter,generate
    from cot_navigation.analysis.legal_history import state_audit
    from cot_navigation.analysis.common import sha256,write_json
    sys.path.insert(0,str(Path.cwd()/'studies/scale_trace_audit'))
    from legacy_trace_audit import validate_nodes,validate_execution,self_test
    from cot_navigation.protocols.tokenizer import Tokenizer
    from cot_navigation.protocols.dfs import dfs_search
    from cot_navigation.protocols.bfs import bfs_search,first_expansion_order
    from cot_navigation.protocols.render import render_steps
    checks=self_test(Tokenizer(),dfs_search,bfs_search,render_steps,first_expansion_order)
    output=Path('studies/short3_trace_audit');output.mkdir(parents=True,exist_ok=True)
    for name in args.models:
        target=output/'models'/f'{name}.json'
        if target.exists() and json.loads(target.read_text()).get('complete'):continue
        arm,protocol=name.split('__')
        run=Path('outputs/fixed_short3_v1/launches/short3_n100_s0')/arm/protocol
        fingerprint=sha256(run/'checkpoints/best.pt')
        entry=dict(id=name,kind='fixed',protocol=protocol,run=str(run),checkpoint_sha256=fingerprint)
        print('START',name,flush=True)
        am=make_adapter(entry,Path.cwd(),'cuda:0')
        originals=[json.loads(l) for l in (run/'eval/predictions_test_best.jsonl').read_text().splitlines()]
        ids=[d['query_id'] for d in originals];assert len(ids)==256 and len(set(ids))==256
        original_by_id={d['query_id']:d for d in originals}
        examples=[am.example(q,'test') for q in ids]
        rows=[];saved=[];batch=am.cfg.evaluation.batch_size
        for start in range(0,len(examples),batch):
            chunk=examples[start:start+batch]
            for ex,out in zip(chunk,generate(am,[e.prompt for e in chunk],batch=batch)):
                body=out['trace_tokens'];stop=am.tok.id('<STOP>');stopped=stop in body
                if stopped:body=body[:body.index(stop)]
                adj={ex.nodes[u]:[ex.nodes[v] for v in vs] for u,vs in enumerate(ex.adj)}
                if protocol=='path':audit=path_audit(ex,body,out['pred'],am.tok)
                elif protocol.endswith('_state'):
                    audit=state_audit(ex,body,ex.label,am.tok,protocol=='bfs_state')
                    audit['structural_valid']=audit['prefix_valid']
                    audit['actions_valid']=validate_execution(body,ex.label,adj,ex.nodes[ex.source],ex.nodes[ex.target],am.tok,protocol=='bfs_state')
                else:
                    result=validate_nodes(body,ex.label,adj,ex.nodes[ex.source],ex.nodes[ex.target],am.tok,protocol=='bfs_nodes')
                    audit=dict(complete_valid=result['valid'],structural_valid=result['valid'],actions_valid=result['valid'],
                        first_error=None if result['valid'] else dict(kind=result['error']),local={})
                correct=out['pred']==ex.label;full=audit['complete_valid'] and not stopped
                gap=int(am.store.queries['distance'][ex.index])
                row=dict(query=ex.index,gap=gap,label=ex.label,pred=out['pred'],answer_correct=correct,
                    original_pred=original_by_id[ex.index]['pred'],prediction_reproduced=out['pred']==original_by_id[ex.index]['pred'],
                    complete_valid=full,joint=bool(full and correct),structural_valid=audit['structural_valid'],
                    actions_valid=bool(audit['actions_valid'] and not stopped),stop=stopped,forced_stop=out['forced_stop'],
                    first_error=audit['first_error']['kind'] if audit['first_error'] else None,
                    local=audit['local'],trace_tokens=len(body))
                if protocol=='path':row.update({k:audit[k] for k in ['goal_reached','goal_seen','path_edges','early_no']})
                rows.append(row)
                interesting=(ex.label and not full) or (correct and not full)
                if interesting and len(saved)<6:
                    center=0
                    if audit['first_error'] and 'record' in audit['first_error']:
                        semis=[-1]+[i for i,t in enumerate(body) if t==am.tok.id(';')]
                        k=audit['first_error']['record']
                        if k<len(semis):center=semis[k]+1
                    saved.append(dict(query=ex.index,source=am.tok.name(ex.nodes[ex.source]),target=am.tok.name(ex.nodes[ex.target]),
                        metrics=row,first_error_detail=audit['first_error'],
                        error_window=am.tok.decode(body[max(0,center-35):center+125]),
                        final_trace_tail=am.tok.decode(body[-70:])))
        summaries={}
        for group,condition in [('short',lambda r:r['gap']<=3),('long',lambda r:r['gap']>=4),
            ('long_yes',lambda r:r['gap']>=4 and r['label']==1),('long_no',lambda r:r['gap']>=4 and r['label']==0)]:
            chosen=[r for r in rows if condition(r)];n=len(chosen)
            rates={key:sum(bool(r[key]) for r in chosen)/n for key in ['answer_correct','complete_valid','joint','structural_valid','actions_valid','stop','forced_stop']}
            rates.update(n=n,answer_correct_n=sum(r['answer_correct'] for r in chosen),joint_n=sum(r['joint'] for r in chosen),
                correct_but_invalid_n=sum(r['answer_correct'] and not r['complete_valid'] for r in chosen),
                first_errors=dict(Counter(r['first_error'] for r in chosen if r['first_error'])),
                local_query_counts={key:sum(r['local'].get(key,0)>0 for r in chosen) for key in
                    ['nonedge_push_records','visited_push_records','frontier_jump_records','visited_update_error_records']})
            if protocol=='path':
                rates.update(goal_reached_n=sum(r['goal_reached'] for r in chosen),early_no_n=sum(r['early_no'] for r in chosen),
                    path_edge_hist=dict(sorted(Counter(r['path_edges'] for r in chosen).items())))
            summaries[group]=rates
        assert sha256(run/'checkpoints/best.pt')==fingerprint
        result=dict(complete=True,entry=entry,checkpoint=am.checkpoint,source_sha256=sha256(Path(__file__)),
            oracle_self_test_cases=checks,query_ids=ids,prediction_mismatches=sum(not r['prediction_reproduced'] for r in rows),
            summaries=summaries,by_query=rows,examples=saved,checkpoint_unchanged=True)
        write_json(target,result)
        print('DONE',name,'long',summaries['long'],'mismatches',result['prediction_mismatches'],flush=True)
        del am;torch.cuda.empty_cache()


if __name__=='__main__':main()
