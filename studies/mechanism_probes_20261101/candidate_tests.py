"""Candidate-list diagnostics: current visited versus old same-node PUSH."""
import sys,json,torch,numpy as np
from pathlib import Path
ROOT=Path('/mnt/ningxuefei/yangshuo/cot-navigation-unified');sys.path.insert(0,str(ROOT/'studies/dfs_attention_lifecycle_20261030'));import analyze as a
OUT=ROOT/'studies/mechanism_probes_20261101'
def main():
 torch.set_num_threads(4);cfg=a.load_config([a.RUN/'resolved_config.yaml']);tok=a.Tokenizer.from_config(cfg);model=a.build_model(cfg.model,len(tok),tok.pad).eval();model.requires_grad_(False);model.load_state_dict(torch.load(a.RUN/'checkpoints/best.pt',map_location='cpu',weights_only=False)['model'])
 store=a.ContextStore(a.ROOT/'data/scale_v1'/a.SETS['n100']);panel=json.loads((a.OUT/'plan.json').read_text())['panels']['n100']['test'];results=[]
 def output(seq):
  generated=[]
  for _ in range(12):
   with torch.inference_mode():log=model(torch.tensor(seq+generated)[None])
   token=int(log[0,-1].argmax());generated.append(token)
   if token==tok.id('|') or not tok.node_lo<=token<tok.node_lo+tok.n_node_tokens:break
  return generated
 for spec in panel:
  ix=spec['index'];row=store.render(ix,'dfs_state',tok);seq=row.tokens[:-1];P=row.prompt_len;recs=a.parse(seq,P,tok);edge=np.array(seq[:P-4]).reshape(-1,3);choices={}
  for ri,r in enumerate(recs):
   if r['action']=='FOUND':continue
   prior=[p for p in recs[:ri] if p['stack'][-1]==r['stack'][-1] and p['action']=='PUSH'];neighbors=set(edge[edge[:,1]==r['stack'][-1],2].tolist());own={p['argument'] for p in prior};cross=(neighbors&set(r['visited']))-own
   category='cross_first' if cross and not prior else 'cross_revisit' if cross else 'revisit' if prior else 'fresh'
   choices.setdefault(category,[]).append((ri,r,prior,neighbors,cross))
  for category,options in choices.items():
   ri,r,prior,neighbors,cross=min(options,key=lambda x:(abs(x[0]-len(recs)/2),x[0]));prefix=seq[:r['b']+1];gold=set(r['candidates']);base=output(prefix)
   conditions=[('baseline',prefix,gold)]
   # Preserve length and syntax, change only current visited node identities.
   edited=prefix[:];source=r['visited'][0]
   for p in range(r['a']+1,r['b']):edited[p]=source
   conditions.append(('current_visited_all_source',edited,neighbors-{source}))
   if cross:
    edited=prefix[:];removed=min(cross)
    for p in range(r['a']+1,r['b']):
     if edited[p]==removed:edited[p]=source
    conditions.append(('remove_cross_node_from_current_visited',edited,neighbors-(set(r['visited'])-{removed})))
   if gold and len(r['visited'])>2:
    added=min(gold);edited=prefix[:];key=next((p for p in range(r['a']+2,r['b']) if prefix[p]!=r['stack'][-1]),None)
    if key is not None:
     edited[key]=added;newvis=set(edited[r['a']+1:r['b']]);conditions.append(('add_candidate_to_current_visited',edited,neighbors-newvis))
   if prior and len(prior[-1]['candidates'])>1:
    past=prior[-1];alternative=next(v for v in past['candidates'] if v!=past['argument']);edited=prefix[:];edited[past['c']+2]=alternative
    conditions.append(('edit_last_same_node_PUSH_argument',edited,None))
   for name,edited,expected in conditions:
    generated=base if name=='baseline' else output(edited);nodes=generated[:-1] if generated[-1]==tok.id('|') else generated
    legal=generated[-1]==tok.id('|') and len(nodes)==len(set(nodes)) and all(tok.node_lo<=v<tok.node_lo+tok.n_node_tokens for v in nodes)
    results.append(dict(index=ix,record=ri+1,category=category,condition=name,baseline=[tok.name(v) for v in base],output=[tok.name(v) for v in generated],changed=generated!=base,original_set_correct=legal and set(nodes)==gold,edited_set_correct=None if expected is None else legal and set(nodes)==expected,expected_changed=None if expected is None else expected!=gold,cross_nodes=[tok.name(v) for v in cross],note='Identity edits preserve token count but may contradict earlier DFS history; they test use of the explicit current field, not a legal alternate search.'))
   (OUT/'candidate_tests.json').write_text(json.dumps(results));print(json.dumps(dict(index=ix,category=category,record=ri+1,conditions=len(conditions))),flush=True)
if __name__=='__main__':main()
