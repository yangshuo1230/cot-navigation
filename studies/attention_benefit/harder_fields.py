"""Recheck deep-graph candidate-list repetition with batch16 completion.

Same canonical states as attention, greedy until emitted separator, no
forced closure. Retain aggregate counts and at most two illustrative fields.
This is a local field completion, not a free-navigation rerun.
"""
import json,runpy
from pathlib import Path
from collections import Counter
import torch
P=Path(__file__).resolve().parent
H=runpy.run_path(str(P/'harder.py'));M=H['M'];D=M['D']
from cot_navigation.data.mine_natural_candidates import complete_fields
torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.15)
result={}
for arm in ('clean','error'):
    cfg,tok,model=M['load_model'](arm)
    root=dict(s.split('=',1) for s in cfg.evaluation.extra_sets)['deep']
    source=D['ContextStore'](root);cache=D['ScalePreparedRows'](source,cfg,tok)
    ids=[r['query_id'] for r in H['predictions'](D['FOLDERS']['dfs_clean'],'deep')]
    cases,_=H['bounded_cases'](source,cache,tok,ids,model.max_seq_len)
    fields=[dict(query_id=c['query_id'],prefix=c['tokens'][:p['q']+1],gold=p['eligible'],start=p['start'])
            for c in cases for p in c['probes']]
    fields.sort(key=lambda c:(len(c['prefix']),c['query_id'],c['start']))
    counts=Counter();examples=[]
    for lo in range(0,len(fields),16):
        batch=fields[lo:lo+16]
        with torch.no_grad():values=complete_fields(model,[c['prefix'] for c in batch],tok)
        for c,(pred,reason) in zip(batch,values):
            gold=c['gold'];first=bool(gold and pred and pred[0] in gold) or (not gold and not pred and reason=='closed')
            good=reason=='closed' and len(pred)==len(set(pred)) and set(pred)==set(gold)
            duplicate=len(pred)>len(set(pred))
            counts['n']+=1;counts['correct']+=int(good);counts['first_legal']+=int(first)
            counts['duplicate_fields']+=int(duplicate)
            counts['wrong_after_legal_first']+=int(first and not good)
            counts['duplicate_after_legal_first']+=int(first and not good and duplicate)
            counts['unclosed']+=int(reason!='closed')
            if first and not good and duplicate and len(examples)<2:
                examples.append(dict(query_id=c['query_id'],prefix_len=len(c['prefix']),gold=tok.decode(gold),
                                     predicted=tok.decode(pred),reason=reason))
        print(arm,lo,'/',len(fields),flush=True)
    result[arm]=dict(counts=counts,examples=examples,batch_size=16,candidate_cap=64,
                     note='helper compacts finished field rows; full navigation was not rerun')
    M['write'](P/'harder/deep_fields_batch16.json',result)
    del model;torch.cuda.empty_cache()
result['finished']=True;M['write'](P/'harder/deep_fields_batch16.json',result)
