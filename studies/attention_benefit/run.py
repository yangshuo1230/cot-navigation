"""Matched-prefix attention analysis; frozen DFS models, GPU 0 only.

Extract selected attention rows from the actual SDPA Q/K projections without
changing the forward pass. Calibrate distance means on disjoint validation
graphs, separately by layer/head/probe role. Save aggregates, not matrices.
"""
import os
os.environ.setdefault('CUDA_VISIBLE_DEVICES','0')
os.environ.setdefault('WANDB_MODE','disabled')
import json,math,runpy,time
from pathlib import Path
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]
AUDIT=ROOT/'studies/accuracy_source'
D=runpy.run_path(str(AUDIT/'run.py'))
from cot_navigation.models.attention import apply_rope
from cot_navigation.data.mistakes import records
from cot_navigation.evaluation.sampling import stratified_ids
from cot_navigation.evaluation.rollout import pad_rows,_autocast,_gather_row

OUT=Path(__file__).resolve().parent
CASES=['n16','n24','n32','n36','main']
ROLES=['frontier_first','visited_first','candidate_nonempty','candidate_empty','candidate_end','action','answer']
FEATURES=['graph_mass','graph_endpoint_mass','query_mass','cot_mass','current_frontier_mass',
 'current_visited_mass','current_node_cot_mass','relevant_source_mass','relevant_dest_mass',
 'eligible_dest_mass','source_top1_correct','dest_top1_relevant','eligible_dest_top1_correct',
 'source_share','source_chance','source_selectivity','residual_graph_mass',
 'residual_source_density_contrast','residual_dest_density_contrast','residual_coverage']

def write(path,x):D['write'](path,x)

def load_model(arm):
 folder=D['FOLDERS']['dfs_'+arm]
 cfg=D['load_config']([folder/'resolved_config.yaml']);tok=D['Tokenizer'].from_config(cfg)
 model=D['build_model'](cfg.model,len(tok),tok.pad).cuda().eval().requires_grad_(False)
 checkpoint=folder/'checkpoints/latest.pt'
 blob=torch.load(checkpoint,map_location='cpu',weights_only=False)
 assert blob['step']==200000
 model.load_state_dict(blob['model'],strict=True);del blob
 return cfg,tok,model

def build_cases(source,cache,tok,ids):
 output=[]
 for idx in ids:
  idx=int(idx);row=cache.row(idx,0);recs=records(row,tok)
  candidates=[i for i,r in enumerate(recs) if r['action']!=[tok.id('<FOUND>')]]
  rng=np.random.default_rng(131071+idx)
  selected=sorted(rng.choice(candidates,min(4,len(candidates)),replace=False).tolist())
  probes=[]
  def add(role,pos,rec):
   probes.append(dict(role=ROLES.index(role),q=pos-1,start=rec['start'],
    frontier_end=rec['start']+len(rec['stack']),vis_span=rec['spans']['visited'],
    cur=rec['stack'][-1],eligible=rec['cands'],record=rec))
  for ri in selected:
   rec=recs[ri];vlo,_=rec['spans']['visited'];clo,chi=rec['spans']['cands']
   add('frontier_first',rec['start'],rec);add('visited_first',vlo,rec)
   add('candidate_nonempty' if rec['cands'] else 'candidate_empty',clo,rec)
   if rec['cands']:add('candidate_end',chi,rec)
   add('action',rec['action_start'],rec)
  if recs:add('answer',row.answer_index,recs[-1])
  tokens=row.tokens[:row.answer_index]
  assert all(0<=p['q']<len(tokens) for p in probes)
  edge_end=row.prompt_len-4
  edges=np.asarray(tokens[:edge_end],dtype=np.int64).reshape(-1,3)
  output.append(dict(query_id=idx,label=row.label,tokens=tokens,prompt_len=row.prompt_len,
                     edge_end=edge_end,edges=edges,probes=probes))
 return output

class SelectedRows:
 """QKV hooks inspect weights; they return nothing and cannot edit the model."""
 def __init__(self,model):
  self.model=model;self.handles=[];self.batch=None;self.weights={};self.enabled=True;self.checked=False
  for li,b in enumerate(model.blocks):
   self.handles.append(b.attn.qkv.register_forward_hook(self.make_hook(li,b.attn)))
 def make_hook(self,li,attn):
  def hook(module,args,out):
   if not self.enabled:return
   B,L,_=out.shape;H=attn.n_heads;dim=attn.head_dim
   q,k,_=out.view(B,L,3,H,dim).unbind(2)
   positions=torch.arange(L,device=out.device).expand(B,-1)
   q=apply_rope(q.transpose(1,2),positions,attn.rope_base)
   k=apply_rope(k.transpose(1,2),positions,attn.rope_base)
   indices=torch.tensor([[p['q'] for p in c['probes']]+[0]*(self.max_probes-len(c['probes'])) for c in self.batch],device=out.device)
   qs=torch.gather(q,2,indices[:,None,:,None].expand(B,H,self.max_probes,dim))
   keep=torch.arange(L,device=out.device)[None,None,None,:]<=indices[:,None,:,None]
   with torch.autocast(device_type='cuda',enabled=False):
    scores=torch.matmul(qs.float(),k.float().transpose(-1,-2))/math.sqrt(dim)
    weights=torch.softmax(scores.masked_fill(~keep,-torch.inf),-1)
   self.weights[li]=weights.cpu().numpy()
  return hook
 def forward(self,batch):
  self.batch=batch;self.max_probes=max(len(c['probes']) for c in batch);self.weights={}
  tokens,lengths=pad_rows([c['tokens'] for c in batch],self.model.pad_id,'cuda')
  with torch.no_grad(),_autocast(torch.device('cuda'),'bf16'):
   if not self.checked:
    self.enabled=False;standard=self.model(tokens);self.enabled=True
   logits=self.model(tokens)
  if not self.checked:
   assert torch.equal(logits,standard),'Read-only QKV extraction changed logits'
   self.checked=True
  assert all(np.allclose(w.sum(-1),1,atol=2e-6) for w in self.weights.values())
  return self.weights
 def close(self):
  for h in self.handles:h.remove()

def calibrate(model,cases,batch_size=8):
 sums=np.zeros((model.n_layers,model.n_heads,len(ROLES),model.max_seq_len),dtype=np.float64)
 counts=np.zeros((len(ROLES),model.max_seq_len),dtype=np.int64)
 reader=SelectedRows(model)
 try:
  for lo in range(0,len(cases),batch_size):
   batch=cases[lo:lo+batch_size];weights=reader.forward(batch)
   for bi,c in enumerate(batch):
    for pi,p in enumerate(c['probes']):
     role,q=p['role'],p['q'];counts[role,:q+1]+=1
     for li,w in weights.items():sums[li,:,role,:q+1]+=w[bi,:,pi,:q+1][:,::-1]
 finally:reader.close()
 return np.divide(sums,counts[None,None,:,:],out=np.zeros_like(sums),where=counts[None,None,:,:]>0),counts

def features(w,p,c,mu,count):
 q=p['q'];keys=np.arange(q+1);tokens=np.asarray(c['tokens'][:q+1])
 graph=keys<c['edge_end'];src=graph&(keys%3==1);dst=graph&(keys%3==2)
 rel_edges=np.flatnonzero(c['edges'][:,1]==p['cur'])
 rel_src=np.isin(keys,3*rel_edges+1);rel_dst=np.isin(keys,3*rel_edges+2)
 eligible=rel_dst&np.isin(tokens,p['eligible'])
 query=(keys>=c['edge_end'])&(keys<c['prompt_len']);cot=keys>=c['prompt_len']
 frontier=(keys>=p['start'])&(keys<p['frontier_end'])
 vis=(keys>=p['vis_span'][0])&(keys<p['vis_span'][1])
 node=cot&(tokens==p['cur'])
 distance=q-keys;valid=count[p['role'],distance]>0
 # Explicit indexing keeps head first under numpy advanced indexing.
 residual=w-mu[:,p['role'],:][:,distance].mean(axis=0)
 mass=lambda mask:float(w[mask].sum())
 def top(mask,target):
  positions=np.flatnonzero(mask)
  return float(target[positions[np.argmax(w[positions])]]) if target.any() and len(positions) else np.nan
 def contrast(a,b):
  a=a&valid;b=b&valid
  return float(residual[a].mean()-residual[b].mean()) if a.any() and b.any() else np.nan
 sm=mass(src);chance=rel_src.sum()/max(1,src.sum())
 share=mass(rel_src)/sm if rel_src.any() and sm>1e-12 else np.nan
 selectivity=(share-chance)/(1-chance) if chance<1 and np.isfinite(share) else np.nan
 return [mass(graph),mass(src|dst),mass(query),mass(cot),mass(frontier),mass(vis),mass(node),
  mass(rel_src),mass(rel_dst),mass(eligible),top(src,rel_src),top(dst,rel_dst),top(dst,eligible),
  share,chance if rel_src.any() else np.nan,selectivity,
  float(residual[graph&valid].sum()),contrast(rel_src,src&~rel_src),contrast(rel_dst,dst&~rel_dst),float(valid.mean())]

def observe(model,cases,mu,count,batch_size=8):
 values=np.zeros((len(cases),model.n_layers,len(ROLES),len(FEATURES)),dtype=np.float64)
 n=np.zeros_like(values,dtype=np.int64)
 reader=SelectedRows(model)
 try:
  for lo in range(0,len(cases),batch_size):
   batch=cases[lo:lo+batch_size];weights=reader.forward(batch)
   for bi,c in enumerate(batch):
    for pi,p in enumerate(c['probes']):
     for li,w in weights.items():
      v=np.asarray(features(w[bi,:,pi,:p['q']+1].mean(0),p,c,mu[li],count))
      good=np.isfinite(v);values[lo+bi,li,p['role'],good]+=v[good];n[lo+bi,li,p['role'],good]+=1
 finally:reader.close()
 return np.divide(values,n,out=np.full_like(values,np.nan),where=n>0)

class EndQueryLesion:
 """Edit only the last prefix token's attention result, before projection.

Past-token representations retain the standard SDPA path. All conditions,
including identity, use the same FP32 selected-row attention calculation.
This removes direct reads at the decision token, not all graph information
already encoded in earlier CoT tokens or token representations.
"""
 def __init__(self,model,tok):
  self.model=model;self.tok=tok;self.handles=[];self.qkv={}
  for li,block in enumerate(model.blocks):
   self.handles.append(block.attn.qkv.register_forward_hook(self.store(li)))
   self.handles.append(block.attn.proj.register_forward_pre_hook(self.replace(li,block.attn)))
 def store(self,li):
  def hook(module,args,out):self.qkv[li]=out
  return hook
 def replace(self,li,attn):
  def hook(module,args):
   out=args[0];B,L,Dm=out.shape;H=attn.n_heads;dim=attn.head_dim
   q,k,v=self.qkv[li].view(B,L,3,H,dim).unbind(2)
   indices=self.lengths-1;bi=torch.arange(B,device=out.device)
   qs=q[bi,indices].unsqueeze(2)
   qs=apply_rope(qs,indices[:,None],attn.rope_base)
   positions=torch.arange(L,device=out.device).expand(B,-1)
   ks=apply_rope(k.transpose(1,2),positions,attn.rope_base)
   keep=torch.arange(L,device=out.device)[None,:]<self.lengths[:,None]
   if self.condition!='identity':keep=keep&~self.masks[self.condition][:,:L]
   assert bool(keep.any(-1).all())
   with torch.autocast(device_type='cuda',enabled=False):
    scores=torch.matmul(qs.float(),ks.float().transpose(-1,-2))/math.sqrt(dim)
    weights=torch.softmax(scores.masked_fill(~keep[:,None,None,:],-torch.inf),-1)
    changed=torch.matmul(weights,v.transpose(1,2).float()).to(out.dtype).reshape(B,Dm)
   out=out.clone();out[bi,indices]=changed
   return (out,)
  return hook
 def prepare(self,batch):
  prompts=[c['prefix'] for c in batch];self.tokens,self.lengths=pad_rows(prompts,self.model.pad_id,'cuda')
  B,L=self.tokens.shape;self.masks={k:torch.zeros((B,L),device='cuda',dtype=torch.bool) for k in ['graph_all','relevant_edges','matched_irrelevant_edges']}
  for bi,c in enumerate(batch):
   self.masks['graph_all'][bi,:c['edge_end']]=True
   relevant=list(c['relevant_edges']);others=set(range(c['edge_end']//3))-set(relevant)
   matched=[]
   for edge in relevant:
    if not others:break
    pick=min(others,key=lambda j:(abs(j-edge),j));matched.append(pick);others.remove(pick)
   assert len(matched)==len(relevant)
   for kind,edges in [('relevant_edges',relevant),('matched_irrelevant_edges',matched)]:
    for e in edges:self.masks[kind][bi,3*e:3*e+3]=True
 def forward(self,condition):
  self.condition=condition
  with torch.no_grad(),_autocast(torch.device('cuda'),'bf16'):
   logits=self.model(self.tokens)
  return _gather_row(logits,self.lengths-1).float()
 def close(self):
  for h in self.handles:h.remove()

def lesions(model,tok,cases,batch_size=16):
 probes=[]
 for c in cases:
  for p in c['probes']:
   if ROLES[p['role']] not in ['candidate_nonempty','candidate_empty']:continue
   probes.append(dict(query_id=c['query_id'],prefix=c['tokens'][:p['q']+1],
    allowed=p['eligible'] or [tok.id('|')],empty=not p['eligible'],edge_end=c['edge_end'],
    relevant_edges=np.flatnonzero(c['edges'][:,1]==p['cur']).tolist()))
 # Same batch and sequence shapes for all intervention conditions.
 conditions=['identity','graph_all','relevant_edges','matched_irrelevant_edges']
 data=np.full((len(cases),len(conditions),2),np.nan);qindex={c['query_id']:i for i,c in enumerate(cases)}
 sums=np.zeros_like(data);counts=np.zeros_like(data);identity_drift=0;total=0;max_logit_diff=0.
 editor=EndQueryLesion(model,tok)
 try:
  for lo in range(0,len(probes),batch_size):
   batch=probes[lo:lo+batch_size];editor.prepare(batch)
   # Read standard SDPA without editing to validate the math identity path.
   for h in editor.handles:h.remove()
   with torch.no_grad(),_autocast(torch.device('cuda'),'bf16'):
    standard=_gather_row(model(editor.tokens),editor.lengths-1).float()
   editor=EndQueryLesion(model,tok);editor.prepare(batch)
   for ci,condition in enumerate(conditions):
    logits=editor.forward(condition);probs=logits.softmax(-1);pred=logits.argmax(-1).tolist()
    if condition=='identity':
     identity_drift+=int((logits.argmax(-1)!=standard.argmax(-1)).sum());total+=len(batch)
     max_logit_diff=max(max_logit_diff,float((logits-standard).abs().max()))
    for bi,c in enumerate(batch):
     qi=qindex[c['query_id']]
     sums[qi,ci]+=np.asarray([float(probs[bi,c['allowed']].sum()),float(pred[bi] in c['allowed'])])
     counts[qi,ci]+=1
 finally:editor.close()
 data=np.divide(sums,counts,out=data,where=counts>0)
 return data,dict(conditions=conditions,n_fields=len(probes),identity_argmax_drift=identity_drift,
                  identity_max_logit_difference=max_logit_diff,n_identity_checks=total)

def paired_summary(a,b,ids,gain_ids):
 delta=b-a;good=np.isfinite(delta);n=good.sum(0)
 mean=lambda x:np.divide(np.nansum(x,axis=0),np.isfinite(x).sum(0),out=np.full(x.shape[1:],np.nan),where=np.isfinite(x).sum(0)>0)
 rng=np.random.default_rng(20261024)
 boot=np.stack([mean(delta[rng.integers(0,len(ids),len(ids))]) for _ in range(500)])
 import warnings
 with warnings.catch_warnings():
  warnings.simplefilter('ignore',RuntimeWarning)
  ci=np.nanquantile(boot,[.025,.975],axis=0)
 gains=np.asarray([i in gain_ids for i in ids])
 return dict(clean=mean(a).tolist(),error=mean(b).tolist(),difference=mean(delta).tolist(),
             ci_low=ci[0].tolist(),ci_high=ci[1].tolist(),n=n.tolist(),
             gained_n=int(gains.sum()),gained_difference=mean(delta[gains]).tolist() if gains.any() else None)

def main():
 torch.set_num_threads(2)
 torch.cuda.set_per_process_memory_fraction(.15)
 settings=dict(roles=ROLES,features=FEATURES,validation_queries=256,test_queries=256,
  states_per_query=4,variant=0,batch_attention=8,batch_lesion=16,
  calibration='per model, layer, head, role and exact relative distance on validation only',
  heads_averaged_for_report=True,seed=20261024)
 write(OUT/'settings.json',settings)
 for case in CASES:
  dest=OUT/(case+'.json')
  if dest.exists():continue
  start=time.time();cfg,tok,model=load_model('clean')
  extra=dict(s.split('=',1) for s in cfg.evaluation.extra_sets)
  source=D['ContextStore'](cfg.data.root if case=='main' else extra[case])
  cache=D['ScalePreparedRows'](source,cfg,tok)
  folder=AUDIT/'strict' if case=='main' else AUDIT/'small'
  saved=[json.loads((folder/('dfs_'+arm+'_'+case+'.json')).read_text())['rows'] for arm in ['clean','error']]
  ids=[r['query_id'] for r in saved[0]];assert ids==[r['query_id'] for r in saved[1]]
  gain_ids={a['query_id'] for a,b in zip(*saved) if not a['correct'] and b['correct']}
  a=source.arrays
  val_ids=stratified_ids(source.indices('val'),a['shape_id'],a['distance'],a['label'],256,cfg.evaluation.seed)
  assert not set(ids)&set(map(int,val_ids))
  val_cases=build_cases(source,cache,tok,val_ids);test_cases=build_cases(source,cache,tok,ids)
  metrics=[];causal=[];checks=[];calibration=[]
  for arm in ['clean','error']:
   if arm=='error':
    del model;torch.cuda.empty_cache();other_cfg,other_tok,model=load_model(arm)
    assert other_tok.itos[:len(tok)]==tok.itos
   mu,count=calibrate(model,val_cases)
   observation=observe(model,test_cases,mu,count);metrics.append(observation)
   intervention,check=lesions(model,tok,test_cases);causal.append(intervention);checks.append(check)
   calibration.append(dict(max_supported_distance=[int(np.flatnonzero(c)[-1]) if c.any() else -1 for c in count],
                            probes_per_role=count[:,0].tolist()))
   print(case,arm,'done',round(time.time()-start,1),flush=True)
  result=dict(case=case,attention=paired_summary(*metrics,ids,gain_ids),
   lesion=paired_summary(*causal,ids,gain_ids),checks=checks,calibration=calibration,
   lesion_effect=paired_summary(causal[0][:,:1]-causal[0],causal[1][:,:1]-causal[1],ids,gain_ids),
   seconds=time.time()-start,query_ids=ids,gain_ids=sorted(gain_ids))
  # Cluster bootstrap uses queries, not the several states sampled per query.
  diff=metrics[1][:, :, ROLES.index('candidate_nonempty'),FEATURES.index('graph_mass')]-metrics[0][:,:,ROLES.index('candidate_nonempty'),FEATURES.index('graph_mass')]
  result['interesting_queries']=[dict(query_id=ids[i],gained=ids[i] in gain_ids,graph_attention_change=np.nanmean(diff[i]))
     for i in np.argsort(np.nan_to_num(np.nanmean(diff,axis=1),nan=0))[[0,1,-2,-1]]]
  write(dest,result)
  del model;torch.cuda.empty_cache()
 write(OUT/'finished.json',dict(finished=True))

if __name__=='__main__':main()
