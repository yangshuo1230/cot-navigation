"""Separate direct decision-token graph reads from graph reads carried by
earlier occurrences of the current node. Fixed-shape math identity controls.
These lesions do not remove all possible indirect graph-information routes.
"""
import runpy,json,math
from pathlib import Path
import numpy as np
import torch
P=Path(__file__).resolve().parent
M=runpy.run_path(str(P/'run.py'));D=M['D'];AUDIT=M['AUDIT']
CONDITIONS=['identity','candidate_graph','current_node_graph','matched_history_graph','both_graph']

class ChannelEditor(M['EndQueryLesion']):
 def prepare(self,batch):
  self.tokens,self.lengths=M['pad_rows']([c['prefix'] for c in batch],self.model.pad_id,'cuda')
  B,L=self.tokens.shape;positions=[];types=[]
  for c in batch:
   last=len(c['prefix'])-1
   current=[i for i,t in enumerate(c['prefix']) if i>=c['edge_end'] and t==c['cur']]
   available=set(range(c['edge_end'],last))-set(current)
   matched=[]
   for i in current:
    assert available
    j=min(available,key=lambda j:(abs(j-i),j));matched.append(j);available.remove(j)
   union=sorted(set(current+matched+[last]));positions.append(union)
   types.append([(i==last,i in current,i in matched) for i in union])
  S=max(map(len,positions))
  self.positions=torch.tensor([p+[p[-1]]*(S-len(p)) for p in positions],device='cuda')
  self.valid=torch.tensor([[True]*len(p)+[False]*(S-len(p)) for p in positions],device='cuda')
  self.types=torch.tensor([t+[(False,False,False)]*(S-len(t)) for t in types],device='cuda')
  self.edge_ends=torch.tensor([c['edge_end'] for c in batch],device='cuda')
 def replace(self,li,attn):
  def hook(module,args):
   out=args[0];B,L,Dm=out.shape;H=attn.n_heads;dim=attn.head_dim
   q,k,v=self.qkv[li].view(B,L,3,H,dim).unbind(2)
   indices=self.positions;S=indices.shape[1]
   qs=torch.gather(q.transpose(1,2),2,indices[:,None,:,None].expand(B,H,S,dim))
   qs=M['apply_rope'](qs,indices,attn.rope_base)
   ks=M['apply_rope'](k.transpose(1,2),torch.arange(L,device='cuda').expand(B,-1),attn.rope_base)
   keep=torch.arange(L,device='cuda')[None,None,:]<=indices[:,:,None]
   if self.condition!='identity':
    if self.condition=='candidate_graph':target=self.types[:,:,0]
    elif self.condition=='current_node_graph':target=self.types[:,:,1]
    elif self.condition=='matched_history_graph':target=self.types[:,:,2]
    elif self.condition=='both_graph':target=self.types[:,:,0]|self.types[:,:,1]
    else:raise ValueError(self.condition)
    graph=torch.arange(L,device='cuda')[None,:]<self.edge_ends[:,None]
    keep=keep&~(target[:,:,None]&graph[:,None,:])
   assert bool(keep.any(-1).all())
   with torch.autocast(device_type='cuda',enabled=False):
    scores=torch.matmul(qs.float(),ks.float().transpose(-1,-2))/math.sqrt(dim)
    weights=torch.softmax(scores.masked_fill(~keep[:,None,:,:],-torch.inf),-1)
    changed=torch.matmul(weights,v.transpose(1,2).float()).to(out.dtype).transpose(1,2).reshape(B,S,Dm)
   result=out.clone();bi=torch.arange(B,device='cuda')[:,None].expand(B,S)
   result[bi[self.valid],indices[self.valid]]=changed[self.valid]
   return (result,)
  return hook

def evaluate(model,tok,cases):
 probes=[]
 for c in cases:
  for p in c['probes']:
   if M['ROLES'][p['role']] not in ['candidate_nonempty','candidate_empty']:continue
   probes.append(dict(query_id=c['query_id'],prefix=c['tokens'][:p['q']+1],
     allowed=p['eligible'] or [tok.id('|')],edge_end=c['edge_end'],cur=p['cur']))
 index={c['query_id']:i for i,c in enumerate(cases)}
 sums=np.zeros((len(cases),len(CONDITIONS),2));counts=np.zeros_like(sums)
 drift=0;maxdiff=0.
 for lo in range(0,len(probes),16):
  batch=probes[lo:lo+16]
  tokens,lengths=M['pad_rows']([c['prefix'] for c in batch],model.pad_id,'cuda')
  with torch.no_grad(),M['_autocast'](torch.device('cuda'),'bf16'):
   standard=M['_gather_row'](model(tokens),lengths-1).float()
  editor=ChannelEditor(model,tok);editor.prepare(batch)
  try:
   for ci,condition in enumerate(CONDITIONS):
    logits=editor.forward(condition);pred=logits.argmax(-1).tolist();prob=logits.softmax(-1)
    if ci==0:
     drift+=int((logits.argmax(-1)!=standard.argmax(-1)).sum())
     maxdiff=max(maxdiff,float((logits-standard).abs().max()))
    for bi,c in enumerate(batch):
     qi=index[c['query_id']]
     sums[qi,ci]+=[float(prob[bi,c['allowed']].sum()),float(pred[bi] in c['allowed'])]
     counts[qi,ci]+=1
  finally:editor.close()
 result=np.divide(sums,counts,out=np.full_like(sums,np.nan),where=counts>0)
 return result,dict(n_fields=len(probes),identity_argmax_drift=drift,identity_max_logit_difference=maxdiff)

def main():
 torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.15)
 for case in ['n16','n32','main']:
  dest=P/('channels_'+case+'.json')
  if dest.exists():continue
  cfg,tok,model=M['load_model']('clean')
  extra=dict(s.split('=',1) for s in cfg.evaluation.extra_sets)
  source=D['ContextStore'](cfg.data.root if case=='main' else extra[case]);cache=D['ScalePreparedRows'](source,cfg,tok)
  folder=AUDIT/'strict' if case=='main' else AUDIT/'small'
  saved=[json.loads((folder/('dfs_'+a+'_'+case+'.json')).read_text())['rows'] for a in ['clean','error']]
  ids=[r['query_id'] for r in saved[0]]
  gains={a['query_id'] for a,b in zip(*saved) if not a['correct'] and b['correct']}
  cases=M['build_cases'](source,cache,tok,ids)
  values=[];checks=[]
  for arm in ['clean','error']:
   if arm=='error':
    del model;torch.cuda.empty_cache();_,_,model=M['load_model']('error')
   value,check=evaluate(model,tok,cases);values.append(value);checks.append(check)
   print(case,arm,np.nanmean(value,axis=0).tolist(),flush=True)
  M['write'](dest,dict(conditions=CONDITIONS,lesion=M['paired_summary'](*values,ids,gains),
    effect=M['paired_summary'](values[0][:,:1]-values[0],values[1][:,:1]-values[1],ids,gains),checks=checks))
  del model;torch.cuda.empty_cache()
 M['write'](P/'channels_finished.json',dict(finished=True))

if __name__=='__main__':main()
