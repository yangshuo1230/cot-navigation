"""L5/L6 readout, true final-component ablations, and L1 separator controls."""
import sys,json,torch,numpy as np
from pathlib import Path
ROOT=Path('/mnt/ningxuefei/yangshuo/cot-navigation-unified');sys.path.insert(0,str(ROOT/'studies/dfs_attention_lifecycle_20261030'));import analyze as a
OUT=ROOT/'studies/mechanism_probes_20261101'
def main():
 torch.set_num_threads(4);cfg=a.load_config([a.RUN/'resolved_config.yaml']);tok=a.Tokenizer.from_config(cfg);a.TOK=tok
 model=a.build_model(cfg.model,len(tok),tok.pad).eval();model.requires_grad_(False);model.load_state_dict(torch.load(a.RUN/'checkpoints/best.pt',map_location='cpu',weights_only=False)['model'])
 store=a.ContextStore(a.ROOT/'data/scale_v1'/a.SETS['n100']);panel=json.loads((a.OUT/'plan.json').read_text())['panels']['n100'];features=[];targets=[];meta=[];bar_results=[];bar_probe=np.load(OUT/'probe_weights.npz')['bars_L1']
 for split in ['val','test']:
  for spec in panel[split]:
   row=store.render(spec['index'],'dfs_state',tok);seq=row.tokens[:-1];P=row.prompt_len;recs=a.parse(seq,P,tok);cap={}
   def grab(name):
    def h(m,args,out):cap[name]=(out[0] if isinstance(out,tuple) else out).detach().clone()
    return h
   hooks=[model.blocks[4].register_forward_hook(grab('L5')),model.blocks[5].register_forward_hook(grab('L6')),model.blocks[5].attn.register_forward_hook(grab('A6'))]
   with torch.inference_mode():base,ws=model(torch.tensor(seq)[None],return_attn=True)
   for h in hooks:h.remove()
   x=cap['L5'];
   with torch.inference_mode():noattn=x+model.blocks[5].ffn(model.blocks[5].norm2(x));logno=model.lm_head(model.final_norm(noattn));logffn=model.lm_head(model.final_norm(x+cap['A6']))
   for e in a.events(seq,P,recs):
    q=e['q'];gold=seq[q+1];features.append(np.stack([cap['L5'][0,q].numpy(),cap['L6'][0,q].numpy()]));targets.append(base[0,q].numpy());meta.append(dict(index=spec['index'],split=split,phase=e['phase'],q=q,gold=gold,final=int(base[0,q].argmax()),no_attention_prediction=int(logno[0,q].argmax()),no_ffn_prediction=int(logffn[0,q].argmax())))
   del ws,cap,base
   if split=='test' and spec['band']=='long':
    r=recs[len(recs)//2]
    for field,q in [('stack_bar',r['a']),('visited_bar',r['b']),('candidate_bar',r['c'])]:
     ids=torch.tensor(seq[:q+1])[None];cap={};hooks=[model.blocks[0].attn.qkv.register_forward_hook(grab('qkv')),model.blocks[0].attn.register_forward_hook(grab('A1')),model.blocks[0].register_forward_hook(grab('L1'))]
     with torch.inference_mode():baseline,weights=model(ids,return_attn=True)
     for h in hooks:h.remove()
     v=cap['qkv'][0].view(q+1,3,cfg.model.d_model)[:,2];past=[p for p in range(q) if seq[p]==tok.id('|')];mass=float(weights[0][0,0,q,past].sum());diff=float((v[past]-v[P-1]).abs().max()) if past else 0;assert diff==0
     baseline_probe=int((np.r_[cap['L1'][0,q].numpy(),1]@bar_probe).argmax());delta=-torch.nn.functional.linear(v[P-1]*mass,model.blocks[0].attn.proj.weight)
     random=torch.tensor(np.random.default_rng(spec['index']+q).normal(size=cfg.model.d_model),dtype=torch.float32);random*=delta.norm()/random.norm()
     for condition,change in [('same_symbol_value_swap',torch.zeros_like(delta)),('zero_historical_bar_values',delta),('zero_receiver_L1_attention',-cap['A1'][0,q]),('norm_matched_random',random)]:
      changed_cap={}
      def patch(m,args,out):
       z=out[0].clone();z[0,q]+=change;return z,out[1]
      def capture(m,args,out):changed_cap['x']=out[0][0,q].clone()
      h=model.blocks[0].attn.register_forward_hook(patch);h2=model.blocks[0].register_forward_hook(capture)
      with torch.inference_mode():log,_=model(ids,return_attn=True)
      h.remove();h2.remove();assert torch.equal(log[:,:q],baseline[:,:q])
      if condition=='same_symbol_value_swap':assert torch.equal(log,baseline)
      bp=int((np.r_[changed_cap['x'].numpy(),1]@bar_probe).argmax())
      bar_results.append(dict(index=spec['index'],field=field,q=q,condition=condition,bar_value_max_difference=diff,history_bar_attention=mass,L1_probe_before=baseline_probe,L1_probe_after=bp,prediction=tok.name(int(log[0,q].argmax())),baseline_prediction=tok.name(int(baseline[0,q].argmax())),output_changed=int(log[0,q].argmax())!=int(baseline[0,q].argmax()),gold=tok.name(seq[q+1]),delta_norm=float(change.norm()),earlier_logits_unchanged=True))
   (OUT/'bar_interventions.json').write_text(json.dumps(bar_results));print(json.dumps(dict(index=spec['index'],split=split,events=len(meta))),flush=True)
 np.savez_compressed(OUT/'readout_activations.npz',features=np.array(features),logits=np.array(targets));(OUT/'readout_metadata.json').write_text(json.dumps(meta))
if __name__=='__main__':main()
