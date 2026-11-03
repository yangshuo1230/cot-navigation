"""Layer-5 controls: graph attention quantity versus semantic placement.

Match each new-model decision's graph mass to the clean model on precisely
the same prefix, preserving relative weights within graph/non-graph blocks.
Shuffle whole edge-triplet weights while preserving graph mass and endpoint
token roles. Both controls affect only the candidate decision token at L5.
"""
import runpy,json,math
from pathlib import Path
import numpy as np
import torch
P=Path(__file__).resolve().parent
M=runpy.run_path(str(P/'run.py'));D=M['D'];AUDIT=M['AUDIT']
CONDITIONS=['identity','match_clean_mass','zero_graph','shuffle_edges']

class Editor:
 def __init__(self,model,batch,condition,targets):
  self.model=model;self.batch=batch;self.condition=condition;self.targets=targets
  self.tokens,self.lengths=M['pad_rows']([c['prefix'] for c in batch],model.pad_id,'cuda')
  self.qkv=None;self.before=None;self.after=None
  self.attn=model.blocks[4].attn
  self.h1=self.attn.qkv.register_forward_hook(self.store)
  self.h2=self.attn.proj.register_forward_pre_hook(self.replace)
 def store(self,module,args,out):self.qkv=out
 def replace(self,module,args):
  out=args[0];B,L,Dm=out.shape;H=self.attn.n_heads;dim=self.attn.head_dim
  q,k,v=self.qkv.view(B,L,3,H,dim).unbind(2)
  bi=torch.arange(B,device='cuda');indices=self.lengths-1
  qs=M['apply_rope'](q[bi,indices].unsqueeze(2),indices[:,None],self.attn.rope_base)
  ks=M['apply_rope'](k.transpose(1,2),torch.arange(L,device='cuda').expand(B,-1),self.attn.rope_base)
  keys=torch.arange(L,device='cuda')[None,:]
  keep=keys<self.lengths[:,None]
  ends=torch.tensor([c['edge_end'] for c in self.batch],device='cuda')
  graph=keys<ends[:,None]
  with torch.autocast(device_type='cuda',enabled=False):
   scores=torch.matmul(qs.float(),ks.float().transpose(-1,-2))/math.sqrt(dim)
   w=torch.softmax(scores.masked_fill(~keep[:,None,None,:],-torch.inf),-1)
   g=(w*graph[:,None,None,:]).sum(-1,keepdim=True)
   self.before=g.flatten().cpu().numpy()
   if self.condition=='match_clean_mass':
    target=torch.tensor(self.targets,device='cuda',dtype=torch.float32).view(B,1,1,1)
    assert bool((g>1e-12).all()) and bool((g<1-1e-12).all())
    w=w*torch.where(graph[:,None,None,:],target/g,(1-target)/(1-g))
   elif self.condition=='zero_graph':
    w=w.masked_fill(graph[:,None,None,:],0);w=w/w.sum(-1,keepdim=True)
   elif self.condition=='shuffle_edges':
    perm=torch.arange(L,device='cuda').expand(B,-1).clone()
    for i,c in enumerate(self.batch):
     E=c['edge_end']//3
     rng=np.random.default_rng(70001+c['query_id']*104729+c['record_start'])
     order=rng.permutation(E)
     positions=(3*order[:,None]+np.arange(3)[None,:]).reshape(-1)
     perm[i,:c['edge_end']]=torch.tensor(positions,device='cuda')
    w=torch.gather(w,-1,perm[:,None,None,:].expand(B,H,1,L))
   elif self.condition!='identity':raise ValueError(self.condition)
   self.after=(w*graph[:,None,None,:]).sum(-1).flatten().cpu().numpy()
   assert torch.allclose(w.sum(-1),torch.ones_like(w.sum(-1)),atol=2e-6)
   changed=torch.matmul(w,v.transpose(1,2).float()).to(out.dtype).reshape(B,Dm)
  result=out.clone();result[bi,indices]=changed
  return (result,)
 def forward(self):
  with torch.no_grad(),M['_autocast'](torch.device('cuda'),'bf16'):
   out=self.model(self.tokens)
  return M['_gather_row'](out,self.lengths-1).float()
 def close(self):self.h1.remove();self.h2.remove()

def evaluate(model,tok,cases,targets=None):
 probes=[]
 for c in cases:
  for p in c['probes']:
   if M['ROLES'][p['role']] not in ['candidate_nonempty','candidate_empty']:continue
   probes.append(dict(query_id=c['query_id'],prefix=c['tokens'][:p['q']+1],
    allowed=p['eligible'] or [tok.id('|')],empty=not p['eligible'],edge_end=c['edge_end'],record_start=p['start']))
 qindex={c['query_id']:i for i,c in enumerate(cases)}
 # Legal probability, legal argmax, and actual graph mass; split empty states.
 sums=np.zeros((len(cases),2,len(CONDITIONS),3));counts=np.zeros_like(sums)
 baseline_mass=[];checks=dict(identity_argmax_drift=0,identity_max_logit_diff=0.,max_match_error=0.,max_shuffle_mass_error=0.)
 for lo in range(0,len(probes),16):
  batch=probes[lo:lo+16]
  tokens,lengths=M['pad_rows']([c['prefix'] for c in batch],model.pad_id,'cuda')
  with torch.no_grad(),M['_autocast'](torch.device('cuda'),'bf16'):
   standard=M['_gather_row'](model(tokens),lengths-1).float()
  for ci,condition in enumerate(CONDITIONS):
   # Clean mass-matching is an identity: targets are measured in this batch.
   target=targets[lo:lo+len(batch)] if targets is not None else baseline_mass[lo:lo+len(batch)]
   editor=Editor(model,batch,condition,target)
   try:logits=editor.forward()
   finally:editor.close()
   prob=logits.softmax(-1);pred=logits.argmax(-1).tolist()
   if ci==0:
    baseline_mass.extend(editor.before.tolist())
    checks['identity_argmax_drift']+=int((logits.argmax(-1)!=standard.argmax(-1)).sum())
    checks['identity_max_logit_diff']=max(checks['identity_max_logit_diff'],float((logits-standard).abs().max()))
   if condition=='match_clean_mass':checks['max_match_error']=max(checks['max_match_error'],float(np.max(np.abs(editor.after-np.asarray(target)))))
   if condition=='shuffle_edges':checks['max_shuffle_mass_error']=max(checks['max_shuffle_mass_error'],float(np.max(np.abs(editor.after-editor.before))))
   for bi,c in enumerate(batch):
    qi=qindex[c['query_id']];state=int(c['empty'])
    sums[qi,state,ci]+=[float(prob[bi,c['allowed']].sum()),float(pred[bi] in c['allowed']),float(editor.after[bi])]
    counts[qi,state,ci]+=1
 result=np.divide(sums,counts,out=np.full_like(sums,np.nan),where=counts>0)
 return result,baseline_mass,dict(checks,n_fields=len(probes))

def main():
 torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.15)
 for case in ['n16','n32','main']:
  dest=P/('mass_'+case+'.json')
  if dest.exists():continue
  cfg,tok,model=M['load_model']('clean')
  extra=dict(s.split('=',1) for s in cfg.evaluation.extra_sets)
  source=D['ContextStore'](cfg.data.root if case=='main' else extra[case]);cache=D['ScalePreparedRows'](source,cfg,tok)
  folder=AUDIT/'strict' if case=='main' else AUDIT/'small'
  saved=[json.loads((folder/('dfs_'+a+'_'+case+'.json')).read_text())['rows'] for a in ['clean','error']]
  ids=[r['query_id'] for r in saved[0]];gains={a['query_id'] for a,b in zip(*saved) if not a['correct'] and b['correct']}
  cases=M['build_cases'](source,cache,tok,ids)
  clean,target,check_a=evaluate(model,tok,cases)
  del model;torch.cuda.empty_cache();_,_,model=M['load_model']('error')
  error,_,check_b=evaluate(model,tok,cases,targets=target)
  M['write'](dest,dict(conditions=CONDITIONS,states=['nonempty','empty'],metrics=['legal_probability','legal_argmax','graph_mass'],
     result=M['paired_summary'](clean,error,ids,gains),checks=[check_a,check_b]))
  print(case,'nonempty clean',np.nanmean(clean[:,0],axis=0).tolist(),'error',np.nanmean(error[:,0],axis=0).tolist(),flush=True)
  del model;torch.cuda.empty_cache()
 M['write'](P/'mass_finished.json',dict(finished=True))

if __name__=='__main__':main()
