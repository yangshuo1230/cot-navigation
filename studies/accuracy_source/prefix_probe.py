"""Candidate-set generation under identical correct prefixes.

This removes history quality as an explanation of candidate errors. Uses
preselected, shared canonical states; legal candidate reordering is accepted.
No optimizer, teacher forcing of the candidate field, or test-label forcing.
"""
import runpy, json, time
from pathlib import Path
import numpy as np
import torch
D=runpy.run_path(str(Path(__file__).with_name('run.py')))
ROOT,OUT,FOLDERS=D['ROOT'],D['OUT'],D['FOLDERS']
from cot_navigation.data.mistakes import records
from cot_navigation.data.mine_natural_candidates import complete_fields

def main(names=None,cases_override=None,output_dir=None):
 torch.set_num_threads(2)
 outdir=Path(output_dir) if output_dir else OUT
 for name in (names or ['dfs_clean','dfs_error','bfs_clean','bfs_error']):
  cfg=D['load_config']([FOLDERS[name]/'resolved_config.yaml'])
  tok=D['Tokenizer'].from_config(cfg)
  model=D['build_model'](cfg.model,len(tok),tok.pad).cuda().eval().requires_grad_(False)
  blob=torch.load(FOLDERS[name]/'checkpoints/latest.pt',map_location='cpu',weights_only=False)
  model.load_state_dict(blob['model']);del blob
  extra=dict(s.split('=',1) for s in cfg.evaluation.extra_sets)
  for case in (cases_override or (['main','deep','wide','larger'] if name.startswith('dfs') else ['main'])):
   dest=outdir/('prefix_'+name+'_'+case+'.json')
   if dest.exists():continue
   start=time.time()
   source=D['ContextStore'](cfg.data.root if case=='main' else extra[case])
   cache=D['ScalePreparedRows'](source,cfg,tok)
   suffix='' if case=='main' else '-'+case
   original=[json.loads(s) for s in (FOLDERS[name]/'eval'/('predictions_test_last'+suffix+'.jsonl')).read_text().splitlines()]
   contexts=[];canonical_checks=0
   for item in original:
    q=item['query_id'];row=cache.row(q,0)
    nodes=source.node_vocab_ids(q,tok);raw=source.adj(q)
    adj={nodes[u]:[nodes[v] for v in vs] for u,vs in enumerate(raw)}
    src=nodes[int(source.arrays['s_topo'][q])];dst=nodes[int(source.arrays['t_topo'][q])]
    ref=D['audit'](row.tokens[row.prompt_len:row.answer_index],row.label,adj,src,dst,tok,name.startswith('bfs'))
    assert ref['effective_complete'] or (ref['effective_prefix_valid'] and row.tokens[row.answer_index-1]==tok.id('<STOP>')), (name,case,q,ref)
    canonical_checks+=1
    recs=records(row,tok)
    # Exclude FOUND states: candidate content there cannot affect finding goal.
    eligible=[i for i,r in enumerate(recs) if r['action']!=[tok.id('<FOUND>')]]
    rng=np.random.default_rng(131071+q)
    chosen=sorted(rng.choice(eligible,min(4,len(eligible)),replace=False).tolist())
    for ri in chosen:
     rec=recs[ri];lo,_=rec['spans']['cands']
     contexts.append(dict(query_id=q,record=ri,label=row.label,distance=item['distance'],
       prefix=row.tokens[:lo],gold=rec['cands'],visited=len(rec['visited']),
       action=tok.decode(rec['action'])))
   contexts.sort(key=lambda r:(len(r['prefix']),r['query_id'],r['record']))
   output=[]
   for lo in range(0,len(contexts),32):
    batch=contexts[lo:lo+32]
    with torch.no_grad():values=complete_fields(model,[c['prefix'] for c in batch],tok)
    for c,(predicted,reason) in zip(batch,values):
     hit=reason=='closed' and len(predicted)==len(set(predicted)) and set(predicted)==set(c['gold'])
     output.append(dict(**{k:v for k,v in c.items() if k!='prefix'},predicted=predicted,
       prefix_len=len(c['prefix']),closed=reason=='closed',reason=reason,correct=hit,
       missing=len(set(c['gold'])-set(predicted)),extra=len(set(predicted)-set(c['gold'])),
       duplicates=len(predicted)-len(set(predicted))))
   D['write'](dest,dict(model=name,case=case,canonical_checks=canonical_checks,n=len(output),
                       correct=sum(c['correct'] for c in output),rows=output,seconds=time.time()-start))
   print(name,case,len(output),sum(c['correct'] for c in output),time.time()-start,flush=True)
  del model;torch.cuda.empty_cache()
 D['write'](outdir/'prefix_finished.json',dict(finished=True))

if __name__=='__main__':main()
