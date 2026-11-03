"""Receiver-local causal interventions on stop/continue, all six base layers.

Past positions remain clean; only the last-position computation is replayed.
Validate cached suffix replay against full forward hooks, including random controls.
"""
import sys,json,hashlib,time
from pathlib import Path
import numpy as np
import torch
ROOT=Path('/mnt/ningxuefei/yangshuo/cot-navigation-unified')
sys.path.insert(0,str(ROOT/'studies/dfs_attention_lifecycle_20261030'))
import analyze as a
OUT=ROOT/'studies/base6_field_control_20261102'
RUN_PREFIX='stack_tail.' if '--stack-tail' in sys.argv else 'candidate_empty.' if '--candidate-empty' in sys.argv else ''

def save(name,x):(OUT/(RUN_PREFIX+name+'.json')).write_text(json.dumps(x,ensure_ascii=False))
def history_hash(cache,q):
 h=hashlib.sha256()
 for tensors in [cache.k,cache.v]:
  for x in tensors:h.update(x[:,:,:q].numpy().tobytes())
 return h.hexdigest()

def capture(model,tokens):
 cache=model.new_cache(1,'cpu',torch.float32);cache.valid_length=len(tokens);cap={};hooks=[]
 for l,block in enumerate(model.blocks):
  def before(m,args,l=l):cap[l,'input']=args[0][:,-1:].clone()
  hooks.append(block.register_forward_pre_hook(before))
  for kind,module in [('attention',block.attn),('ffn',block.ffn),('residual',block)]:
   def grab(m,args,out,l=l,kind=kind):cap[l,kind]=(out[0] if isinstance(out,tuple) else out)[:,-1:].clone()
   hooks.append(module.register_forward_hook(grab))
 try:log=model(torch.tensor(tokens)[None],cache=cache,cache_mode='prefill')[0,-1].clone()
 finally:
  for h in hooks:h.remove()
 return dict(tokens=tokens,cache=cache,cap=cap,logits=log,q=len(tokens)-1)

def replay(model,state,l,kind,new):
 block=model.blocks[l];cap=state['cap']
 if kind=='attention':
  x=cap[l,'input']+new;x=x+block.ffn(block.norm2(x))
 elif kind=='ffn':x=cap[l,'input']+cap[l,'attention']+new
 else:x=new
 positions=torch.tensor([[state['q']]])
 for j in range(l+1,6):x,_=model.blocks[j](x,positions,cache=state['cache'],cache_mode='decode',layer_idx=j)
 return model.lm_head(model.final_norm(x))[0,0]

def full_patch(model,state,l,kind,new):
 module=model.blocks[l].attn if kind=='attention' else model.blocks[l].ffn if kind=='ffn' else model.blocks[l]
 def patch(m,args,out):
  z=(out[0] if isinstance(out,tuple) else out).clone();z[:,-1:]=new
  return (z,)+out[1:] if isinstance(out,tuple) else z
 h=module.register_forward_hook(patch)
 try:log=model(torch.tensor(state['tokens'])[None])[0,-1]
 finally:h.remove()
 return log

def main():
 OUT.mkdir(exist_ok=True);torch.set_num_threads(4);cfg=a.load_config([a.RUN/'resolved_config.yaml']);tok=a.Tokenizer.from_config(cfg);a.TOK=tok
 model=a.build_model(cfg.model,len(tok),tok.pad).eval();model.requires_grad_(False);cp=a.RUN/'checkpoints/best.pt';checkpoint=torch.load(cp,map_location='cpu',weights_only=False);model.load_state_dict(checkpoint['model']);assert model.n_layers==6 and model.swa_records==0
 plan=json.loads((a.OUT/'plan.json').read_text());save('binding',dict(run=str(a.RUN),sha256=hashlib.sha256(cp.read_bytes()).hexdigest(),step=checkpoint.get('step'),plan=str(a.OUT/'plan.json'),layers=6))
 store=a.ContextStore(a.ROOT/'data/scale_v1'/a.SETS['n100']);rows=[];pairs=[];checks=[];started=time.time()
 for spec in plan['panels']['n100']['test']:
  ix=spec['index'];data=store.render(ix,'dfs_state',tok);seq=data.tokens[:-1];P=data.prompt_len;recs=a.parse(seq,P,tok)
  for field in (['stack_tail'] if RUN_PREFIX=='stack_tail.' else ['candidates_empty'] if RUN_PREFIX=='candidate_empty.' else ['stack','visited','candidates']):
   if field in ['stack','stack_tail']:
    if field=='stack':
     stops=[(ri,r['start'],1) for ri,r in enumerate(recs) if len(r['stack'])==1];continues=[(ri,r['start'],1) for ri,r in enumerate(recs) if len(r['stack'])>1]
    else:
     stops=[(ri,r['a']-1,len(r['stack'])) for ri,r in enumerate(recs) if len(r['stack'])>=2];continues=[(ri,r['start']+j-1,j) for ri,r in enumerate(recs) for j in range(2,len(r['stack']))]
    choices=[(ri,q,di,d) for ri,q,j in stops for di,d,k in continues if seq[q]==seq[d] and j==k]
    if choices:
     ri,q,di,d=min(choices,key=lambda z:(abs(z[0]-len(recs)/2)+abs(z[2]-len(recs)/2),abs(z[0]-z[2]),z));prefixes=[seq[:q+1],seq[:d+1]];records=[ri,di]
   elif field=='visited':
    stops=[(ri,r['b']-1) for ri,r in enumerate(recs)];continues=[(ri,q) for ri,r in enumerate(recs) for q in range(r['a']+1,r['b']-1)]
    choices=[(ri,q,di,d) for ri,q in stops for di,d in continues if seq[q]==seq[d]]
    if choices:
     ri,q,di,d=min(choices,key=lambda z:(abs(z[0]-len(recs)/2)+abs(z[2]-len(recs)/2),abs(z[0]-z[2]),z));prefixes=[seq[:q+1],seq[:d+1]];records=[ri,di]
   elif field=='candidates_empty':
    stops=[(ri,r) for ri,r in enumerate(recs) if not r['candidates'] and r['action']!='FOUND'];continues=[(ri,r) for ri,r in enumerate(recs) if r['candidates']]
    choices=[(ri,r,di,d) for ri,r in stops for di,d in continues]
    if choices:
     ri,r,di,d=min(choices,key=lambda z:(z[1]['stack'][-1]!=z[3]['stack'][-1],abs(z[0]-len(recs)/2)+abs(z[2]-len(recs)/2),abs(z[0]-z[2]),z[0],z[2]));prefixes=[seq[:r['b']+1],seq[:d['b']+1]];records=[ri,di]
   else:
    choices=[(ri,r) for ri,r in enumerate(recs) if len(r['candidates'])>=2]
    if choices:
     ri,r=min(choices,key=lambda z:(abs(z[0]-len(recs)/2),z[0]));first=r['b']+1;last=r['c']-1;altered=seq[:last+1];altered[first],altered[last]=altered[last],altered[first];prefixes=[seq[:last+1],altered[:first+1]];records=[ri,ri]
     # A candidate permutation preserves the legal set; validate the completed
     # changed record with its correspondingly changed PUSH argument.
     ids=[int(v) for v in store.node_vocab_ids(ix,tok)];inv={v:i for i,v in enumerate(ids)}
     body=altered[P:]+[tok.id('|'),tok.id('<PUSH>'),altered[first],tok.id(';')]
     audit=a.replay_state(store.adj(ix),int(store.arrays['s_topo'][ix]),int(store.arrays['t_topo'][ix]),body,tok,inv)
     assert audit['error']=='missing_terminal' and audit['records']==ri+1,audit
   if not choices:pairs.append(dict(index=ix,field=field,available=False));continue
   assert prefixes[0][-1]==prefixes[1][-1]
   with torch.inference_mode():states=[capture(model,p) for p in prefixes]
   # Content targets are the reference next token in each prefix's own context.
   targets=[tok.id('|'),seq[states[1]['q']+1] if field!='candidates' else altered[first+1]]
   pair=dict(index=ix,field=field,available=True,visible=tok.name(prefixes[0][-1]),stop_q=states[0]['q'],continue_q=states[1]['q'],records=[r+1 for r in records],candidate_permutation_validated=field=='candidates',stack_equal_emitted_count=field in ['stack','stack_tail'],targets=[tok.name(t) for t in targets])
   pair['matched_current_top']=recs[records[0]]['stack'][-1]==recs[records[1]]['stack'][-1]
   pairs.append(pair)
   for si,state in enumerate(states):
    donor=states[1-si];label='stop' if si==0 else 'continue';base=state['logits'];q=state['q'];original=int(base.argmax());before=history_hash(state['cache'],q);prefix_hash=hashlib.sha256(np.asarray(state['tokens'],dtype=np.int64).tobytes()).hexdigest()
    def metrics(log):
     pred=int(log.argmax());cls='stop' if pred==tok.id('|') else 'node' if tok.node_lo<=pred<tok.node_lo+tok.n_node_tokens else 'invalid'
     return dict(prediction=tok.name(pred),output_class=cls,class_correct=cls==('stop' if si==0 else 'node'),exact_original_target=pred==targets[si],exact_donor_target=pred==targets[1-si],stop_margin=float(log[tok.id('|')]-log[tok.node_lo:tok.node_lo+tok.n_node_tokens].max()),stop_probability=float(log.softmax(-1)[tok.id('|')]))
    bm=metrics(base);common=dict(index=ix,band=spec['band'],field=field,direction=label+'_to_'+('continue' if si==0 else 'stop'),q=q,visible=tok.name(state['tokens'][-1]),prefix_sha256=prefix_hash,baseline=bm)
    for l in range(6):
     for kind in ['attention','ffn','residual']:
      old=state['cap'][l,kind];other=donor['cap'][l,kind];rng=np.random.default_rng(20261102+ix*100+l*7+si);random=torch.tensor(rng.normal(size=old.shape),dtype=old.dtype);random/=random.norm()
      conditions=[('sham',old),('opposite_donor',other),('donor_norm_random',old+random*(other-old).norm())]
      if kind!='residual':conditions += [('zero',torch.zeros_like(old)),('zero_norm_random',old+random*old.norm())]
      else:conditions += [('donor_half',old+.5*(other-old)),('donor_minus',old-(other-old))]
      for condition,new in conditions:
       with torch.inference_mode():log=replay(model,state,l,kind,new)
       m=metrics(log);rows.append(dict(**common,layer=l+1,component=kind,condition=condition,delta_norm=float((new-old).norm()),delta_relative_to_residual=float((new-old).norm()/state['cap'][l,'residual'].norm()),stop_margin_change=m['stop_margin']-bm['stop_margin'],flipped_to_opposite_class=m['output_class']==('node' if si==0 else 'stop'),**m))
       if condition=='sham':assert int(log.argmax())==original and (log-base).abs().max()<.003
       # Full-forward equivalence on both altered components and both directions,
       # for every receiver, plus a normative random-control check.
       if (l==2 and kind=='attention' and condition in ['zero','opposite_donor','donor_norm_random']) or (l==5 and kind=='ffn' and condition=='opposite_donor'):
        with torch.inference_mode():full=full_patch(model,state,l,kind,new)
        error=float((full-log).abs().max());assert error<.003 and int(full.argmax())==int(log.argmax()),(error,common,l,kind,condition)
        checks.append(dict(index=ix,field=field,direction=common['direction'],layer=l+1,component=kind,condition=condition,max_logit_error=error))
    assert before==history_hash(state['cache'],q)
   save('results',rows);save('pairs',pairs);save('replay_checks',checks)
   print(json.dumps(dict(index=ix,field=field,conditions=len(rows),elapsed=round(time.time()-started,1))),flush=True)
 save('COMPLETE',dict(conditions=len(rows),available_pairs=sum(p['available'] for p in pairs),selected_queries=18,full_forward_checks=len(checks),history_cache_unchanged=True,single_receiver_single_layer=True))
if __name__=='__main__':main()
