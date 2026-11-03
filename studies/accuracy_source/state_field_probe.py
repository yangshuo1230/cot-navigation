"""Gold-history frontier and visited generation, matched to candidate probes."""
import runpy,json,time
from pathlib import Path
import torch
D=runpy.run_path(str(Path(__file__).with_name('run.py')))
from cot_navigation.data.mistakes import records
from cot_navigation.data.mine_natural_candidates import complete_fields

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
   contexts=[]
   for c in prefix['rows']:
    row=cache.row(c['query_id'],0);rec=records(row,tok)[c['record']]
    for field,lo,gold in [('frontier',rec['start'],rec['stack']),('visited',rec['spans']['visited'][0],rec['visited'])]:
     contexts.append(dict(query_id=c['query_id'],record=c['record'],field=field,gold=gold,prefix=row.tokens[:lo]))
   contexts.sort(key=lambda r:(len(r['prefix']),r['query_id'],r['record'],r['field']))
   output=[]
   for lo in range(0,len(contexts),32):
    batch=contexts[lo:lo+32]
    with torch.no_grad():values=complete_fields(model,[c['prefix'] for c in batch],tok,cap=256)
    for c,(predicted,reason) in zip(batch,values):
     hit=(predicted==c['gold'] if c['field']=='frontier' else len(predicted)==len(set(predicted)) and set(predicted)==set(c['gold']))
     output.append(dict(query_id=c['query_id'],record=c['record'],field=c['field'],gold_len=len(c['gold']),
       correct=bool(hit and reason=='closed'),reason=reason))
   result[name+'_'+case]={f:dict(n=sum(r['field']==f for r in output),correct=sum(r['correct'] for r in output if r['field']==f)) for f in ['frontier','visited']}
   result[name+'_'+case]['rows']=output
   print(name,case,{k:v for k,v in result[name+'_'+case].items() if k!='rows'},flush=True)
  del model;torch.cuda.empty_cache()
 D['write'](D['OUT']/'state_field_probe.json',result)

if __name__=='__main__':main()
