"""Exploratory carrier/edge-offset scan after the initial route comparison."""
import sys,json,gzip,torch,numpy as np
from pathlib import Path
ROOT=Path('/mnt/ningxuefei/yangshuo/cot-navigation-unified');sys.path.insert(0,str(ROOT/'studies/dfs_attention_lifecycle_20261030'));import analyze as a
OUT=ROOT/'studies/correction_attention_compare_20261102'
def main():
 torch.set_num_threads(4);bindings=json.loads((OUT/'models.json').read_text());plan=json.loads((a.OUT/'plan.json').read_text())['panels']['n100'];store=a.ContextStore(a.ROOT/'data/scale_v1'/a.SETS['n100'])
 for binding in bindings:
  name=binding['name'];run=Path(binding['run']);cfg=a.load_config([run/'resolved_config.yaml']);tok=a.Tokenizer.from_config(cfg);a.TOK=tok;model=a.build_model(cfg.model,len(tok),tok.pad).eval();model.requires_grad_(False);model.load_state_dict(torch.load(run/'checkpoints/best.pt',map_location='cpu',weights_only=False)['model'])
  with gzip.open(OUT/(name+'.supplement.jsonl.gz'),'wt') as f:
   for split in ['val','test']:
    for spec in plan[split]:
     row=store.render(spec['index'],'dfs_state',tok);seq=row.tokens[:-1];P=row.prompt_len;recs=a.parse(seq,P,tok);edge=np.array(seq[:P-4]).reshape(-1,3)
     with torch.inference_mode():_,ws=model(torch.tensor(seq)[None],return_attn=True)
     for e in a.events(seq,P,recs):
      if not e['phase'].startswith(('stack_init','visited_init','visited_continue','candidate_init','candidate_continue')):continue
      q,ri=e['q'],e['ri'];r=recs[ri];rt=dict(first_stack_root=[recs[0]['start']],first_visited_root=[recs[0]['a']+1],all_stack_roots=[p['start'] for p in recs[:ri+1]],last_edge_marker=[P-7],last_graph_tokens=list(range(P-7,P-4)),first_edge_marker=[0],query_marker=[P-4],previous_PUSH_argument=[],old_candidate_tail=[],old_candidate_target=[],previous_visited_target=[])
      if ri:
       prev=recs[ri-1]
       if prev['action']=='PUSH':rt['previous_PUSH_argument']=[prev['c']+2]
       rt['previous_visited_target']=[p for p in range(prev['a']+1,prev['b']) if seq[p]==seq[q+1]]
      old=[p for p in recs[:ri] if p['stack'][-1]==r['stack'][-1] and p['action']=='PUSH']
      if old:
       rt['old_candidate_tail']=list(range(old[-1]['b']+2,old[-1]['c']));rt['old_candidate_target']=[p for p in rt['old_candidate_tail'] if seq[p]==seq[q+1]]
      relevant=np.flatnonzero(edge[:,1]==r['stack'][-1])
      for offset in range(-6,7):rt['adjacency_offset_'+str(offset)]=[3*int(i)+2+offset for i in relevant]
      rt={k:[p for p in dict.fromkeys(v) if 0<=p<=q] for k,v in rt.items()};rows=np.stack([w[0,0,q,:q+1].numpy() for w in ws]);result=dict(model=name,index=spec['index'],split=split,phase=e['phase'],sub=e['sub'],q=q,routes={k:rows[:,v].sum(1).tolist() for k,v in rt.items()});f.write(json.dumps(result,separators=(',',':'))+'\n')
     f.flush();del ws;print(json.dumps(dict(model=name,index=spec['index'],split=split)),flush=True)
if __name__=='__main__':main()
