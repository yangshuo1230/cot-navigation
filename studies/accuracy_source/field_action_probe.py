"""Check whether wrong candidates under gold prefixes are acknowledged.

Only inspect fields previously completed by each model. Feed the complete
prefix, that model's own candidate list and its closing delimiter; read the
next action argmax. No correct candidate tokens are injected.
"""
import runpy,json,time
from pathlib import Path
import torch
D=runpy.run_path(str(Path(__file__).with_name('run.py')))
from cot_navigation.data.mistakes import records
from cot_navigation.evaluation.rollout import pad_rows,_gather_row,_autocast

def main():
 torch.set_num_threads(2)
 result={}
 for name in ['dfs_clean','dfs_error','bfs_clean','bfs_error']:
  cfg=D['load_config']([D['FOLDERS'][name]/'resolved_config.yaml']);tok=D['Tokenizer'].from_config(cfg)
  model=D['build_model'](cfg.model,len(tok),tok.pad).cuda().eval().requires_grad_(False)
  blob=torch.load(D['FOLDERS'][name]/'checkpoints/latest.pt',map_location='cpu',weights_only=False)
  model.load_state_dict(blob['model']);del blob
  extra=dict(s.split('=',1) for s in cfg.evaluation.extra_sets)
  for case in (['main','deep','wide','larger'] if name.startswith('dfs') else ['main']):
   prefix=json.loads((D['OUT']/('prefix_'+name+'_'+case+'.json')).read_text())
   source=D['ContextStore'](cfg.data.root if case=='main' else extra[case]);cache=D['ScalePreparedRows'](source,cfg,tok)
   cases=[r for r in prefix['rows'] if not r['correct'] and r['closed']]
   output=[]
   for lo in range(0,len(cases),8):
    batch=cases[lo:lo+8];prompts=[]
    for r in batch:
     canonical=cache.row(r['query_id'],0);rec=records(canonical,tok)[r['record']]
     prompts.append(canonical.tokens[:rec['spans']['cands'][0]]+r['predicted']+[tok.id('|')])
    tokens,lengths=pad_rows(prompts,tok.pad,'cuda')
    with torch.no_grad(),_autocast(torch.device('cuda'),cfg.train.precision):
     logits=model(tokens)
    selected=_gather_row(logits,lengths-1).argmax(-1).tolist()
    for r,t in zip(batch,selected):
     output.append(dict(query_id=r['query_id'],record=r['record'],empty_gold=not r['gold'],
                        next_action=tok.decode([t]),marked=t==tok.stoi.get('<MISTAKE>',-999)))
   result[name+'_'+case]=dict(n=prefix['n'],wrong=prefix['n']-prefix['correct'],
       closed_wrong=len(output),marked=sum(r['marked'] for r in output),
       empty_gold_wrong=sum(r['empty_gold'] for r in output),
       empty_gold_marked=sum(r['empty_gold'] and r['marked'] for r in output),rows=output)
   print(name,case,{k:v for k,v in result[name+'_'+case].items() if k!='rows'},flush=True)
  del model;torch.cuda.empty_cache()
 D['write'](D['OUT']/'field_action_probe.json',result)

if __name__=='__main__':main()
