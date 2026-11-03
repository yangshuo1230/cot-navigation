"""Keep short windows at interesting errors rather than saving long rollouts."""
import json
import os
from pathlib import Path
import sys


def main():
    os.environ['CUDA_VISIBLE_DEVICES']='0,1'
    os.environ['WANDB_MODE']='disabled'
    sys.path.insert(0,str(Path.cwd()/'src'))
    from cot_navigation.analysis.scale_followup import make_adapter,generate
    from cot_navigation.analysis.legal_history import state_audit
    from cot_navigation.analysis.error_detection import semantic_slots
    from cot_navigation.analysis.common import write_json
    import torch
    torch.set_num_threads(2)
    root=Path('studies/valid_history/results')
    selected=['context_l10_dfs_state','fixed_l10_dfs_state','fixed_l10_bfs_state',
              'fixed_curriculum_dfs_state','fixed_curriculum_bfs_state']
    saved=[]
    for name in selected:
        d=json.loads((root/'models'/f'{name}.json').read_text());qs=d['audit']['by_query'];ids=[]
        rules=[('nonedge_push',lambda r:r['local'].get('nonedge_push_records',0)>0),
               ('state_jump',lambda r:r['local'].get('frontier_jump_records',0)>0),
               ('correct_but_error',lambda r:r['correct'] and r['first_error'] is not None and not r['stop']),
               ('legal_but_incomplete',lambda r:r['stop'] and r['first_error'] is None)]
        reasons={}
        for label,rule in rules:
            match=next((r for r in qs if rule(r)),None)
            if match:
                q=match['query'];reasons.setdefault(q,[]).append(label)
                if q not in ids:ids.append(q)
        am=make_adapter(d['entry'],Path.cwd(),'cuda:1');exs=[am.example(q,'test') for q in ids]
        for ex,out in zip(exs,generate(am,[ex.prompt for ex in exs])):
            body=out['trace_tokens'];stop=am.tok.id('<STOP>')
            if stop in body:body=body[:body.index(stop)]
            audit=state_audit(ex,body,out['pred'],am.tok,am.protocol=='bfs_state')
            bad=None
            for pos,allowed,role in semantic_slots(ex,am.protocol,am.tok,body):
                if body[pos] not in allowed:
                    bad={'position':pos,'role':role,'actual':am.tok.name(body[pos]),
                         'expected':am.tok.decode(sorted(allowed)),
                         'window':am.tok.decode(body[max(0,pos-35):pos+65])};break
            old=next(r for r in qs if r['query']==ex.index)
            saved.append(dict(model=name,query=ex.index,reasons=reasons[ex.index],label=ex.label,pred=out['pred'],
                original_pred=old['pred'],first_error=audit['first_error'],oracle_first_error=bad,
                oracle_agrees_on_error_presence=(bad is not None)==(audit['first_error'] is not None),
                local=audit['local'],stopped=out['cot_interrupted']))
        del am;torch.cuda.empty_cache()
        print('CASES',name,flush=True)
    assert all(r['oracle_agrees_on_error_presence'] for r in saved),saved
    write_json(root/'interesting_cases.json',{'cases':saved,'independent_oracle_check':'PASS',
        'note':'Selected from original per-query diagnostics, rerun as smaller batches; full rollouts are discarded.'})


if __name__=='__main__':main()
