"""Matched behavior-aligned attention on clean prefixes, 2 depths x 2 seeds."""
import sys,json,gzip,hashlib,time,torch,numpy as np
from pathlib import Path
ROOT=Path('/mnt/ningxuefei/yangshuo/cot-navigation-unified');sys.path.insert(0,str(ROOT/'studies/dfs_attention_lifecycle_20261030'));import analyze as a
OUT=ROOT/'studies/correction_attention_compare_20261102';OUT.mkdir(exist_ok=True)
BASE=a.RUN
CORR=ROOT/'outputs/correction_depth_rate/launches/dfs_depth_rate_20261031/dfs_state'
MODELS={'base6':BASE,**{f'corr{depth}_s{seed}':CORR/f'l{depth}_p30/s{seed}' for depth in [6,9] for seed in [0,1]}}

def main():
 torch.set_num_threads(4);plan=json.loads((a.OUT/'plan.json').read_text())['panels']['n100'];store=a.ContextStore(a.ROOT/'data/scale_v1'/a.SETS['n100']);basecfg=a.load_config([BASE/'resolved_config.yaml']);basetok=a.Tokenizer.from_config(basecfg);binding=[]
 for name,run in MODELS.items():
  cfg=a.load_config([run/'resolved_config.yaml']);tok=a.Tokenizer.from_config(cfg);a.TOK=tok;cp=run/'checkpoints/best.pt';raw=cp.read_bytes();fingerprint=hashlib.sha256(raw).hexdigest();state=torch.load(cp,map_location='cpu',weights_only=False)
  model=a.build_model(cfg.model,len(tok),tok.pad).eval();model.load_state_dict(state['model']);model.requires_grad_(False);layers=cfg.model.layers;del raw,state
  assert cfg.model.heads==1
  binding.append(dict(name=name,run=str(run),checkpoint_sha256=fingerprint,step=int(torch.load(cp,map_location='cpu',weights_only=False)['step']),layers=layers,heads=1))
  with gzip.open(OUT/(name+'.events.jsonl.gz'),'wt') as f:
   for split in ['val','test']:
    for spec in plan[split]:
     t0=time.time();ix=spec['index'];row=store.render(ix,'dfs_state',tok);seq=row.tokens[:-1];assert seq==store.render(ix,'dfs_state',basetok).tokens[:-1];P=row.prompt_len;recs=a.parse(seq,P,tok);roles,rids=a.metadata(seq,P,recs,tok);cap={}
     def save_x(m,args,out):cap['prelast']=out[0].clone()
     def save_a(m,args,out):cap['last_attention']=out[0].clone()
     h=model.blocks[-2].register_forward_hook(save_x);h2=model.blocks[-1].attn.register_forward_hook(save_a)
     with torch.inference_mode():logits,ws=model(torch.tensor(seq)[None],return_attn=True)
     h.remove();h2.remove()
     with torch.inference_mode():
      x=cap['prelast'];last=model.blocks[-1]
      without_attention=model.lm_head(model.final_norm(x+last.ffn(last.norm2(x))))
      without_ffn=model.lm_head(model.final_norm(x+cap['last_attention']))
      lens=model.lm_head(model.final_norm(x))
     events=a.events(seq,P,recs)
     for ri,r in enumerate(recs):
      if ri and recs[ri-1]['action']=='PUSH':events.append(dict(phase='root_write_after_PUSH',q=r['start'],target=r['start']+1,ri=ri,sub=''))
     for e in events:
      q,ri=e['q'],e['ri'];r=recs[ri];g=a.groupmask(roles,rids,ri,q);rt=a.routes(seq,P,recs,e)
      rt['history_roots']=[p['start'] for p in recs[:ri]];rt['history_first_visited']=[p['a']+1 for p in recs[:ri]]
      rt['history_selected_candidates']=[p['b']+1 for p in recs[:ri] if p['action']=='PUSH'];rt['history_arguments']=[p['c']+2 for p in recs[:ri] if p['action']=='PUSH']
      rt['history_successor_roots']=[past['start'] for k,past in enumerate(recs[:ri+1]) if k and recs[k-1]['action']=='PUSH' and recs[k-1]['visited'][-1]==seq[q]]
      rt['offset_minus3']=[q-3];rt['self']=[q]
      prior=[p for p in recs[:ri] if p['stack'][-1]==r['stack'][-1] and p['action']=='PUSH']
      if prior:rt['old_selected_candidate']=[prior[-1]['b']+1];rt['old_PUSH']=[prior[-1]['c']+1];rt['old_argument']=[prior[-1]['c']+2]
      rt={k:[p for p in dict.fromkeys(v) if 0<=p<=q] for k,v in rt.items()};rows=np.stack([w[0,0,q,:q+1].numpy() for w in ws]);assert np.max(np.abs(rows.sum(1)-1))<3e-6
      final=int(logits[0,q].argmax());gold=seq[q+1];result=dict(model=name,index=ix,split=split,phase=e['phase'],sub=e['sub'],q=q,record=ri+1,length=len(seq),next_correct=final==gold,groups=[[float(z[g==j].sum()) for j in range(len(a.GROUPS))] for z in rows],routes={k:rows[:,v].sum(1).tolist() for k,v in rt.items()},top=[[dict(position=int(k),token=tok.name(seq[k]),role=a.ROLES[int(roles[k])],record=int(rids[k])+1,weight=float(z[k])) for k in np.argsort(-z)[:3]] for z in rows],last_block=dict(before_last_agrees=int(lens[0,q].argmax())==final,no_attention_agrees=int(without_attention[0,q].argmax())==final,no_ffn_agrees=int(without_ffn[0,q].argmax())==final))
      f.write(json.dumps(result,separators=(',',':'))+'\n')
     f.flush();print(json.dumps(dict(model=name,index=ix,split=split,events=len(events),seconds=round(time.time()-t0,2))),flush=True);del ws,logits,cap,without_attention,without_ffn,lens
  (OUT/'models.json').write_text(json.dumps(binding,indent=2))
 (OUT/'COMPLETE.json').write_text(json.dumps(dict(models=5,queries_per_model=36,all_input_tokens_identical=True)))
if __name__=='__main__':main()
