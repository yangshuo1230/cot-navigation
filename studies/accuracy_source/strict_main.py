"""Original-batch replication and fixed-shape, prefix-matched intervention."""
import os
os.environ['AUDIT_BATCH']='16'
os.environ['AUDIT_CAPTURE']='1'
import runpy,json
from pathlib import Path
import torch
D=runpy.run_path(str(Path(__file__).with_name('run.py')))
OUT=Path(__file__).resolve().parent/'strict'
OUT.mkdir(exist_ok=True)
D['run'].__globals__['OUT']=OUT

def intervention(case='main',output_dir=None,output_name=None):
 output=Path(output_dir) if output_dir else OUT
 dest=output/(output_name or globals().get('INTERVENTION_NAME','dfs_prefix_intervention.json'))
 if dest.exists():return
 cfg=D['load_config']([D['FOLDERS']['dfs_error']/'resolved_config.yaml'])
 tok=D['Tokenizer'].from_config(cfg)
 model=D['build_model'](cfg.model,len(tok),tok.pad).cuda().eval().requires_grad_(False)
 blob=torch.load(D['FOLDERS']['dfs_error']/'checkpoints/latest.pt',map_location='cpu',weights_only=False)
 model.load_state_dict(blob['model']);del blob
 saved=json.loads((output/('dfs_error_'+case+'.json')).read_text())
 cases=saved['mistake_cases'];prompts=[];prefixes=[]
 for c in cases:
  # Intervene at an action marker, never a stray MISTAKE in a state field.
  candidates=[i for i,x in enumerate(c['trace']) if x==tok.id('<MISTAKE>')
              and i>0 and c['trace'][i-1]==tok.id('|')
              and i+1<len(c['trace']) and c['trace'][i+1]==tok.id(';')]
  if globals().get('USE_RAW_MARKERS'):
   i=c['trace'].index(tok.id('<MISTAKE>'))
  else:
   assert candidates
   i=candidates[0]
  prefix=c['trace'][:i];prefixes.append(prefix);prompts.append(c['prompt']+prefix)
 # Fixed cache capacity and batch prevent other rows' changed termination
 # times from changing this row's numerical execution path between arms.
 kwargs=dict(max_new_tokens=cfg.evaluation.max_new_tokens,cot_margin=cfg.protocol.cot_margin,
             crop_cache=False,compact_finished=False)
 normal=D['generate'](model,prompts,cfg.protocol.name,tok,cfg.train.precision,**kwargs)
 def mask(_module,_args,output):
  output[...,tok.id('<MISTAKE>')]=-float('inf')
  return output
 hook=model.register_forward_hook(mask)
 blocked=D['generate'](model,prompts,cfg.protocol.name,tok,cfg.train.precision,**kwargs)
 hook.remove()
 result=[]
 for c,pre,a,b in zip(cases,prefixes,normal,blocked):
  adj={int(k):v for k,v in c['adj'].items()}
  row=dict(query_id=c['query_id'],label=c['label'],
    normal_first_is_mistake=bool(a['generated'] and a['generated'][0]==tok.id('<MISTAKE>')),
    original_pred=int(c['generated'][-1]==tok.yes) if c['generated'][-1] in (tok.yes,tok.no) else None)
  for name,g in [('normal',a),('blocked',b)]:
   row[name]=dict(pred=g['pred'],correct=not g['format_fail'] and g['pred']==c['label'],
     audit=D['audit'](pre+g['trace_tokens'],g['pred'],adj,c['source'],c['target'],tok,False),
     suffix=tok.decode(g['generated']))
  result.append(row)
 D['write'](dest,dict(n=len(cases),rows=result,
   method='Same prefix immediately before first natural MISTAKE; fixed batch and full cache for both continuations. No gold repair provided. Mask all MISTAKE logits in blocked arm.'))
 print('PREFIX INTERVENTION',[(r['query_id'],r['normal_first_is_mistake'],r['normal']['correct'],r['blocked']['correct']) for r in result],flush=True)

if __name__=='__main__':
 torch.set_num_threads(2)
 D['run']('dfs_clean',['main'])
 D['run']('dfs_error',['main'])
 intervention()
 D['write'](OUT/'finished.json',dict(finished=True))
