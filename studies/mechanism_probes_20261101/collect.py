"""Query-split probes and L6 decomposition on the frozen N100 panel."""
import sys,json,torch,numpy as np,time
from pathlib import Path
ROOT=Path('/mnt/ningxuefei/yangshuo/cot-navigation-unified');sys.path.insert(0,str(ROOT/'studies/dfs_attention_lifecycle_20261030'))
import analyze as a
OUT=ROOT/'studies/mechanism_probes_20261101';OUT.mkdir(exist_ok=True)

def main():
 torch.set_num_threads(4);cfg=a.load_config([a.RUN/'resolved_config.yaml']);tok=a.Tokenizer.from_config(cfg);a.TOK=tok
 model=a.build_model(cfg.model,len(tok),tok.pad).eval();model.requires_grad_(False);model.load_state_dict(torch.load(a.RUN/'checkpoints/best.pt',map_location='cpu',weights_only=False)['model'])
 store=a.ContextStore(a.ROOT/'data/scale_v1'/a.SETS['n100']);plan=json.loads((a.OUT/'plan.json').read_text())['panels']['n100'];features=[];metadata=[];bars=[];bmeta=[];l6=[];audit=[]
 for split in ['val','test']:
  for spec in plan[split]:
   ix=spec['index'];row=store.render(ix,'dfs_state',tok);seq=row.tokens[:-1];P=row.prompt_len;recs=a.parse(seq,P,tok);cap={};hooks=[]
   def capture(name):
    def h(m,args,out):cap[name]=(out[0] if isinstance(out,tuple) else out).detach().clone()
    return h
   hooks.append(model.embed.register_forward_hook(capture('L0')))
   for li,block in enumerate(model.blocks):hooks.append(block.register_forward_hook(capture('L'+str(li+1))))
   # Components of the final block, with exact final residual reconstruction.
   hooks.append(model.blocks[5].attn.register_forward_hook(capture('A6')));hooks.append(model.blocks[5].ffn.register_forward_hook(capture('F6')))
   with torch.inference_mode():logits,ws=model(torch.tensor(seq)[None],return_attn=True)
   for h in hooks:h.remove()
   assert torch.allclose(cap['L6'],cap['L5']+cap['A6']+cap['F6'],atol=2e-5)
   # Sample stack positions uniformly within each record, then cap per-query.
   opportunities=[(ri,j,r['start']+j-1) for ri,r in enumerate(recs) for j in range(1,len(r['stack'])+1)]
   rng=np.random.default_rng(20261101+ix);sel=rng.choice(len(opportunities),size=min(160,len(opportunities)),replace=False)
   for i in sorted(sel):
    ri,j,q=opportunities[i];r=recs[ri];depth=len(r['stack']);features.append(np.stack([cap['L'+str(li)][0,q].numpy() for li in range(7)]))
    metadata.append(dict(index=ix,split=split,q=q,record=ri+1,depth=depth,emitted=j,remaining=depth-j,stop=int(j==depth),token=seq[q],relative=q-P,record_start=r['start']-P,prev_action=recs[ri-1]['action'] if ri else 'ROOT'))
   for ri,r in enumerate(recs):
    for cls,key in enumerate(['a','b','c']):
     q=r[key];bars.append(np.stack([cap['L'+str(li)][0,q].numpy() for li in range(7)]));z=ws[0][0,0,q].numpy();past=[p for p in range(q+1) if seq[p]==tok.id('|')];kinds={'prompt':[P-1],'stack':[t['a'] for t in recs[:ri+1] if t['a']<=q],'visited':[t['b'] for t in recs[:ri+1] if t['b']<=q],'candidate':[t['c'] for t in recs[:ri+1] if t['c']<=q]}
     bmeta.append(dict(index=ix,split=split,field=cls,q=q,self_attention=float(z[q]),bar_mass=float(z[past].sum()),bar_mass_excluding_self=float(z[[p for p in past if p!=q]].sum()),by_role={n:float(z[ps].sum()) for n,ps in kinds.items()}))
   # Identical output norm/head at each stage; this is a lens, not a trained decoder.
   stages={'L5':cap['L5'],'L6_attention_only':cap['L5']+cap['A6'],'L6_ffn_only':cap['L5']+cap['F6'],'L6':cap['L6']}
   with torch.inference_mode():decoded={name:model.lm_head(model.final_norm(x))[0] for name,x in stages.items()}
   for e in a.events(seq,P,recs):
    q=e['q'];gold=seq[q+1];final=int(logits[0,q].argmax());rr=dict(index=ix,split=split,phase=e['phase'],q=q,gold=tok.name(gold),final=tok.name(final),stage={})
    for name,log in decoded.items():
     p=int(log[q].argmax());rr['stage'][name]=dict(pred=tok.name(p),correct=p==gold,agrees_final=p==final,gold_logp=float(log[q].log_softmax(-1)[gold]),margin=float(log[q,gold]-torch.cat([log[q,:gold],log[q,gold+1:]]).max()))
    l6.append(rr)
   for ri,r in enumerate(recs):
    prior=[t for t in recs[:ri] if t['stack'][-1]==r['stack'][-1] and t['action']=='PUSH'];cur=r['stack'][-1];old=prior[-1] if prior else None
    simple=set(old['candidates'])-{old['argument']} if old else None
    # Excluded neighbors not previously selected FROM this node => cross-branch.
    edge=np.array(seq[:P-4]).reshape(-1,3);neighbors=set(edge[edge[:,1]==cur,2].tolist());excluded=neighbors&set(r['visited']);own={t['argument'] for t in prior};cross=excluded-own
    audit.append(dict(index=ix,split=split,record=ri+1,fresh=old is None,cross_branch=bool(cross),candidate_size=len(r['candidates']),simple_push_subtraction_matches=None if simple is None else simple==set(r['candidates']),extra_removed_count=None if simple is None else len(simple-set(r['candidates']))))
   np.savez_compressed(OUT/'probe_activations.npz',stack=np.array(features),bars=np.array(bars));(OUT/'probe_metadata.json').write_text(json.dumps(dict(stack=metadata,bars=bmeta)))
   (OUT/'layer6.json').write_text(json.dumps(l6));(OUT/'candidate_audit.json').write_text(json.dumps(audit));print(json.dumps(dict(index=ix,split=split,stack=len(features),bars=len(bars))),flush=True)
 (OUT/'COMPLETE.json').write_text(json.dumps(dict(queries=36,stack=len(features),bars=len(bars))))
if __name__=='__main__':main()
