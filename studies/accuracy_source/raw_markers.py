"""Uncensored marker census and positive-path evidence.

Semantic replay stops after an illegal action, but raw MISTAKE tokens later
in that trace still matter to answer causality. Never equate trusted-prefix
marker counts with all generated markers. Match strict hashes, replaying
only original minibatches whose batch-128 trace differs.
"""
import os,runpy,json,time,hashlib
from pathlib import Path
import torch
D=runpy.run_path(str(Path(__file__).with_name('run.py')))
OUT=D['OUT']

def witness(trace,adj,src,dst,tok,fifo=False):
 frontier=[(src,True)];seen_goal=src==dst;start=0
 for end,t in enumerate(trace):
  if t!=tok.id(';'):continue
  rec=trace[start:end];start=end+1;fields=[[]]
  for x in rec:
   if x==tok.id('|'):fields.append([])
   else:fields[-1].append(x)
  if len(fields)!=4 or not frontier:break
  action=fields[-1];cur,reachable=frontier[0] if fifo else frontier[-1]
  if action==[tok.stoi.get('<MISTAKE>',-999)]:continue
  if len(action)==2 and action[0]==tok.id('<PUSH>'):
   v=action[1];good=reachable and v in adj.get(cur,[])
   frontier.append((v,good));seen_goal |= good and v==dst
  elif action==[tok.id('<POP>')]:frontier.pop(0 if fifo else -1)
  elif action in ([tok.id('<FOUND>')],[tok.id('<EXHAUST>')]):
   seen_goal |= reachable and cur==dst
   break
  else:break
 return bool(seen_goal)

def main():
 torch.set_num_threads(2)
 name='dfs_error';folder=D['FOLDERS'][name]
 cfg=D['load_config']([folder/'resolved_config.yaml']);tok=D['Tokenizer'].from_config(cfg)
 model=D['build_model'](cfg.model,len(tok),tok.pad).cuda().eval().requires_grad_(False)
 blob=torch.load(folder/'checkpoints/latest.pt',map_location='cpu',weights_only=False)
 model.load_state_dict(blob['model']);del blob
 source=D['ContextStore'](cfg.data.root);cache=D['ScalePreparedRows'](source,cfg,tok)
 old=[json.loads(s) for s in (folder/'eval/predictions_test_last.jsonl').read_text().splitlines()]
 census={};payload={}
 def batch_run(batch):
  rows=[cache.row(x['query_id'],0) for x in batch]
  generated=D['generate'](model,[r.tokens[:r.prompt_len] for r in rows],cfg.protocol.name,tok,cfg.train.precision,
    max_new_tokens=cfg.evaluation.max_new_tokens,cot_margin=cfg.protocol.cot_margin)
  for q,row,g in zip(batch,rows,generated):
   idx=q['query_id'];nodes=source.node_vocab_ids(idx,tok);raw=source.adj(idx)
   adj={nodes[u]:[nodes[v] for v in vs] for u,vs in enumerate(raw)}
   src,dst=row.tokens[row.prompt_len-3:row.prompt_len-1]
   trace=g['trace_tokens'];marker=tok.id('<MISTAKE>')
   positions=[i for i,t in enumerate(trace) if t==marker]
   stats=dict(query_id=idx,raw_mistake_tokens=len(positions),
      raw_mistake_action_tokens=sum(i>0 and trace[i-1]==tok.id('|') and i+1<len(trace) and trace[i+1]==tok.id(';') for i in positions),
      positive_witness=witness(trace,adj,src,dst,tok),
      trace_hash=hashlib.sha256(bytes(str(trace),'ascii')).hexdigest())
   census[idx]=stats
   if positions:
    payload[idx]=dict(query_id=idx,prompt=row.tokens[:row.prompt_len],generated=g['generated'],trace=trace,
                     label=row.label,source=src,target=dst,adj=adj)
   elif idx in payload:del payload[idx]
 for lo in range(0,len(old),128):batch_run(old[lo:lo+128])
 strict=OUT/'strict/dfs_error_main.json'
 # The original-batch worker runs concurrently on GPU 0.
 while not strict.exists():time.sleep(2)
 saved=json.loads(strict.read_text());target={r['query_id']:r for r in saved['rows']}
 mismatch_groups=[]
 for lo in range(0,len(old),16):
  batch=old[lo:lo+16]
  if any(census[r['query_id']]['trace_hash']!=target[r['query_id']]['trace_hash'] for r in batch):
   mismatch_groups.append(lo//16);batch_run(batch)
   print('REPLAY ORIGINAL BATCH',lo//16,flush=True)
 assert all(census[q]['trace_hash']==target[q]['trace_hash'] for q in target)
 for r in saved['rows']:r.update(census[r['query_id']])
 saved['mistake_cases']=list(payload.values())
 D['write'](strict,saved)
 clean_path=OUT/'strict/dfs_clean_main.json';clean=json.loads(clean_path.read_text())
 for r in clean['rows']:r.update(raw_mistake_tokens=0,raw_mistake_action_tokens=0)
 D['write'](clean_path,clean)
 result=dict(rows=list(census.values()),n=len(census),raw_marker_queries=len(payload),
             raw_marker_tokens=sum(c['raw_mistake_tokens'] for c in census.values()),
             replayed_original_minibatches=mismatch_groups,all_strict_trace_hashes_matched=True)
 D['write'](OUT/'strict/raw_metadata.json',result)
 print('RAW MARKERS',result['raw_marker_queries'],result['raw_marker_tokens'],flush=True)
 # Re-run the matched-prefix intervention on ALL raw marker cases.
 del model;torch.cuda.empty_cache()
 module=runpy.run_path(str(OUT/'strict_main.py'))
 module['intervention'].__globals__['INTERVENTION_NAME']='dfs_prefix_intervention_raw.json'
 module['intervention'].__globals__['USE_RAW_MARKERS']=True
 module['intervention']()
 D['write'](OUT/'strict/raw_finished.json',dict(finished=True))

if __name__=='__main__':main()
