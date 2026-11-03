"""Small common-prefix field NLL probe on CPU; no GPU use or training."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['WANDB_MODE']='disabled'
from pathlib import Path
import json,sys,time
import numpy as np
import torch
import torch.nn.functional as F
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).parent;sys.path.insert(0,str(ROOT/'src'))
from cot_navigation.config import load_config
from cot_navigation.data.dataset import ContextStore
from cot_navigation.protocols.tokenizer import Tokenizer
from cot_navigation.protocols.dfs import dfs_search
from cot_navigation.protocols.bfs import bfs_search
from cot_navigation.protocols.render import context_prompt
from cot_navigation.models.transformer import build_model
from cot_navigation.evaluation.sampling import stratified_ids
torch.set_num_threads(4)
summary=json.loads((OUT/'summary.json').read_text());first=load_config([Path(summary['runs'][0]['run'])/'resolved_config.yaml']);store=ContextStore(ROOT/first.data.root);tok=Tokenizer.from_config(first)
a=store.arrays;ids={split:stratified_ids(store.indices(split),a['shape_id'],a['distance'],a['label'],32,1729) for split in ['train','val']}
fields=['trace','frontier','visited','candidate_nodes','candidate_field','action_type','push_node','record_sep']
corpora={}
for protocol in ['dfs_state','bfs_state']:
 for split,chosen in ids.items():
  examples=[]
  for q in chosen:
   nodes=store.node_vocab_ids(q,tok);adj=store.adj(q);src,dst=int(a['s_topo'][q]),int(a['t_topo'][q]);edges=list(zip(a['edge_src'][q].tolist(),a['edge_dst'][q].tolist()))
   tokens=context_prompt(tok,nodes,edges,src,dst);positions={k:[] for k in fields};steps=(dfs_search if protocol=='dfs_state' else bfs_search)(adj,src,dst)
   assert int(steps[-1].action=='FOUND')==int(a['label'][q])
   for step in steps[:5]:
    for values,field in [(step.stack,'frontier'),(step.visited,'visited'),(step.cands,'candidate_nodes')]:
     for node in values:
      positions[field].append(len(tokens));positions['trace'].append(len(tokens))
      if field=='candidate_nodes':positions['candidate_field'].append(len(tokens))
      tokens.append(nodes[node])
     if field=='candidate_nodes':positions['candidate_field'].append(len(tokens))
     positions['trace'].append(len(tokens));tokens.append(tok.id('|'))
    positions['trace'].append(len(tokens));positions['action_type'].append(len(tokens));tokens.append(tok.id('<'+step.action+'>'))
    if step.action=='PUSH':positions['trace'].append(len(tokens));positions['push_node'].append(len(tokens));tokens.append(nodes[step.arg])
    positions['trace'].append(len(tokens));positions['record_sep'].append(len(tokens));tokens.append(tok.id(';'))
   examples.append(dict(query_id=int(q),tokens=tokens,positions=positions))
  corpora[(protocol,split)]=examples
results=[]
for run in summary['runs']:
 t0=time.monotonic();cfg=load_config([Path(run['run'])/'resolved_config.yaml']);model=build_model(cfg.model,len(tok),tok.pad).cpu().eval();model.requires_grad_(False)
 blob=torch.load(Path(run['run'])/'checkpoints/latest.pt',map_location='cpu',weights_only=False);model.load_state_dict(blob['model']);del blob
 scores={}
 for split in ids:
  rows=corpora[(run['protocol'],split)];perfield={k:[] for k in fields};token_counts={k:0 for k in fields}
  with torch.inference_mode():
   for at in range(0,len(rows),4):
    chunk=rows[at:at+4];length=max(len(x['tokens']) for x in chunk);tokens=torch.full((len(chunk),length),tok.pad,dtype=torch.long)
    for b,row in enumerate(chunk):tokens[b,:len(row['tokens'])]=torch.tensor(row['tokens'])
    logits=model(tokens);nll=F.cross_entropy(logits[:,:-1].transpose(1,2),tokens[:,1:],reduction='none')
    for b,row in enumerate(chunk):
     for field in fields:
      pos=row['positions'][field]
      if pos:perfield[field].append(float(nll[b,torch.tensor(pos)-1].mean()));token_counts[field]+=len(pos)
  scores[split]={field:{'example_mean_nll':float(np.mean(values)) if values else None,'n_examples':len(values),'n_tokens':token_counts[field]} for field,values in perfield.items()}
 result={'model':run['model'],'step':run['steps'],'scores':scores,'seconds':time.monotonic()-t0};results.append(result)
 (OUT/'field_probe_cpu.json').write_text(json.dumps({'complete':len(results)==len(summary['runs']),'models':results,
  'query_ids':{k:list(map(int,v)) for k,v in ids.items()},'method':'CPU FP32, 32 fixed train +32 fixed val N100 graphs per protocol; first up to5 canonical legal search records only; per-example field means. Candidate_field includes closing separator. Prefix probe, not full-trace validation loss or exact training-mixture loss.'},indent=2)+'\n')
 print(run['model'],{s:{f:round(scores[s][f]['example_mean_nll'],5) for f in ['trace','candidate_field','visited','frontier','action_type']} for s in ids},'seconds',round(result['seconds'],1),flush=True)
 del model
