"""Paired, frozen-checkpoint attribution audit. GPU 0 only; no training.

Replay actions against the actual graph, independently of displayed fields.
Report literal state errors separately from errors explicitly rolled back by
MISTAKE. Do not interpret annotation errors after an illegal action as fresh
independent errors. Keep compact per-query metrics and only selected traces.
"""
import os
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')
os.environ.setdefault('WANDB_MODE', 'disabled')
import sys, json, hashlib, time
from pathlib import Path
from collections import Counter
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'src'))
from cot_navigation.config import load_config
from cot_navigation.protocols.tokenizer import Tokenizer
from cot_navigation.models.transformer import build_model
from cot_navigation.data.dataset import ContextStore
from cot_navigation.data.scale_prepared import ScalePreparedRows
from cot_navigation.evaluation.rollout import generate

OUT = Path(__file__).resolve().parent
BASE = ROOT/'outputs'
FOLDERS = {
 'dfs_clean': BASE/'curriculum_loss_swa_v1/launches/curriculum_dfs_loss_swa_s0/context/dfs_state/full_l16',
 'dfs_error': BASE/'natural_correction_v1/launches/natural_correction_s0/context/dfs_state/p30_l16',
 'bfs_clean': BASE/'scale_v1/launches/scale_v1_s0/context/curriculum/bfs_state',
 'bfs_error': BASE/'natural_correction_v1/launches/natural_correction_s0/context/bfs_state/p50_l16',
}

def write(path, data):
 tmp = path.with_suffix('.tmp')
 tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2)+'\n')
 tmp.replace(path)

def audit(trace, pred, adj, src, dst, tok, fifo):
 """All checks use actual execution state; legal neighbor orders may vary."""
 frontier, visited = [src], {src}
 terminal = None
 pending = False
 result = dict(records=0, checked_records=0, normal_records=0,
               raw_bad_records=0, state_bad_records=0, candidate_bad=0,
               candidate_missing=0, candidate_extra=0, candidate_duplicate=0,
               candidate_bad_nonterminal=0, frontier_bad=0, visited_bad=0,
               action_bad=0, mistake_records=0, true_marks=0, false_marks=0,
               repairs=0, repaired_then_complete=False, first_error=None,
               first_error_record=None, effective_prefix_valid=True,
               execution_prefix_valid=True, incomplete_record=False)
 effective_ok, execution_ok = True, True
 start = 0
 stop = tok.stoi.get('<STOP>')
 body = list(trace)
 if body and body[-1] == stop: body.pop()
 for end, token in enumerate(body):
  if token != tok.id(';'): continue
  rec = body[start:end]; start = end+1
  result['records'] += 1
  fields = [[]]
  for x in rec:
   if x == tok.id('|'): fields.append([])
   else: fields[-1].append(x)
  if not execution_ok: continue
  result['checked_records'] += 1
  if len(fields) != 4 or terminal or not frontier:
   if result['first_error'] is None:
    result['first_error'] = 'malformed_or_after_terminal'
    result['first_error_record'] = result['records']
   result['raw_bad_records'] += 1
   effective_ok = execution_ok = False
   continue
  shown, vis, candidates, action = fields
  cur = frontier[0] if fifo else frontier[-1]
  eligible = set(adj.get(cur, [])) - visited
  wrong = []
  if shown != frontier: wrong.append('frontier')
  if len(vis) != len(set(vis)) or set(vis) != visited: wrong.append('visited')
  cand_bad = len(candidates) != len(set(candidates)) or set(candidates) != eligible
  if cand_bad: wrong.append('candidate')
  if wrong:
   result['raw_bad_records'] += 1
   result['state_bad_records'] += 1
   result['frontier_bad'] += int('frontier' in wrong)
   result['visited_bad'] += int('visited' in wrong)
   result['candidate_bad'] += int(cand_bad)
   result['candidate_bad_nonterminal'] += int(cand_bad and cur != dst)
   result['candidate_missing'] += int(bool(eligible-set(candidates)))
   result['candidate_extra'] += int(bool(set(candidates)-eligible))
   result['candidate_duplicate'] += int(len(candidates)!=len(set(candidates)))
   if result['first_error'] is None:
    result['first_error'] = '+'.join(wrong)
    result['first_error_record'] = result['records']
  if action == [tok.stoi.get('<MISTAKE>', -999)]:
   result['mistake_records'] += 1
   result['true_marks'] += int(bool(wrong))
   result['false_marks'] += int(not wrong)
   pending = pending or bool(wrong)
   # Cancel this displayed record; no execution state is changed.
   continue
  result['normal_records'] += 1
  expected = 'FOUND' if cur == dst else 'PUSH' if eligible else 'POP' if len(frontier)>1 else 'EXHAUST'
  wanted = tok.id('<'+expected+'>')
  legal = (len(action)==(2 if expected=='PUSH' else 1) and action[0]==wanted)
  if legal and expected=='PUSH': legal = action[1] in eligible
  priority_ok = expected!='PUSH' or (bool(candidates) and len(action)==2 and action[1]==candidates[0])
  if not legal:
   result['action_bad'] += 1
   if not wrong: result['raw_bad_records'] += 1
   if result['first_error'] is None:
    result['first_error'] = 'action'
    result['first_error_record'] = result['records']
   execution_ok = False
  if wrong or not legal or not priority_ok: effective_ok = False
  if pending and not wrong and legal and priority_ok:
   result['repairs'] += 1
   pending = False
  if legal:
   if expected=='PUSH': frontier.append(action[1]); visited.add(action[1])
   elif expected=='POP': frontier.pop(0 if fifo else -1)
   else: terminal = expected
 if start != len(body):
  result['incomplete_record'] = True
  effective_ok = execution_ok = False
 if pending: effective_ok = False
 complete = (terminal=='FOUND' and pred==1) or (terminal=='EXHAUST' and pred==0)
 result.update(effective_prefix_valid=effective_ok,
               execution_prefix_valid=execution_ok,
               effective_complete=bool(effective_ok and complete),
               execution_complete=bool(execution_ok and complete),
               literal_complete=bool(effective_ok and complete and not result['raw_bad_records']),
               terminal=terminal,
               repaired_then_complete=bool(result['repairs'] and effective_ok and complete))
 return result

def self_test():
 cfg=load_config([FOLDERS['dfs_error']/'resolved_config.yaml'])
 t=Tokenizer.from_config(cfg)
 a,b,c=[t.node_id(i) for i in range(3)]
 adj={a:[b],b:[],c:[]}
 def rec(s,v,cs,op):return s+[t.id('|')]+v+[t.id('|')]+cs+[t.id('|')]+op+[t.id(';')]
 first=rec([a],[a],[b],[t.id('<PUSH>'),b])
 end=rec([a,b],[a,b],[],[t.id('<FOUND>')])
 wrong=rec([a],[a],[c],[t.id('<MISTAKE>')])
 x=audit(first+end,1,adj,a,b,t,False)
 assert x['literal_complete'] and x['execution_complete']
 x=audit(wrong+first+end,1,adj,a,b,t,False)
 assert x['effective_complete'] and not x['literal_complete'] and x['repairs']==1
 bad=rec([a],[a],[c],[t.id('<PUSH>'),b])
 x=audit(bad+end,1,adj,a,b,t,False)
 assert x['execution_complete'] and not x['effective_complete']
 bad=rec([a],[a],[c],[t.id('<PUSH>'),c])
 x=audit(bad+end,1,adj,a,b,t,False)
 assert not x['execution_complete']
 return 4

def run(model_name, cases, block=False):
 folder=FOLDERS[model_name]
 cfg=load_config([folder/'resolved_config.yaml']);tok=Tokenizer.from_config(cfg)
 ckpt=folder/'checkpoints/latest.pt'
 stat=ckpt.stat()
 name=model_name+('_blocked' if block else '')
 pending=[c for c in cases if not (OUT/(name+'_'+c+'.json')).exists()]
 if not pending:return
 torch.manual_seed(0)
 model=build_model(cfg.model,len(tok),tok.pad).cuda().eval().requires_grad_(False)
 blob=torch.load(ckpt,map_location='cpu',weights_only=False)
 assert blob['step']==200000
 model.load_state_dict(blob['model'],strict=True);del blob
 hook=None
 if block:
  # Same weights, prompt, precision and decoder; only exclude MISTAKE logits.
  def mask(_module,_args,output):
   output[...,tok.id('<MISTAKE>')]=-float('inf')
   return output
  hook=model.register_forward_hook(mask)
 extra=dict(x.split('=',1) for x in cfg.evaluation.extra_sets)
 for case in pending:
  start=time.time()
  suffix='' if case=='main' else '-'+case
  saved=folder/'eval'/('predictions_test_last'+suffix+'.jsonl')
  original=[json.loads(x) for x in saved.read_text().splitlines()]
  source=ContextStore(cfg.data.root if case=='main' else extra[case])
  cache=ScalePreparedRows(source,cfg,tok)
  rows=[];examples=[];mistake_cases=[]
  # A common larger batch for all arms improves single-GPU throughput.
  # Check every answer against the historical batch-16 result; report drift.
  analysis_batch=int(os.environ.get('AUDIT_BATCH','128'))
  for offset in range(0,len(original),analysis_batch):
   batch=original[offset:offset+analysis_batch]
   records=[cache.row(int(r['query_id']),0) for r in batch]
   generated=generate(model,[r.tokens[:r.prompt_len] for r in records],cfg.protocol.name,tok,cfg.train.precision,
                      max_new_tokens=cfg.evaluation.max_new_tokens,cot_margin=cfg.protocol.cot_margin)
   for old,row,g in zip(batch,records,generated):
    idx=int(old['query_id']);nodes=source.node_vocab_ids(idx,tok);raw=source.adj(idx)
    adj={nodes[u]:[nodes[v] for v in vs] for u,vs in enumerate(raw)}
    src=nodes[int(source.arrays['s_topo'][idx])];dst=nodes[int(source.arrays['t_topo'][idx])]
    seen={src};todo=[src]
    while todo:
     for v in adj[todo.pop()]:
      if v not in seen:seen.add(v);todo.append(v)
    assert int(dst in seen)==row.label==old['label']
    metrics=audit(g['trace_tokens'],g['pred'],adj,src,dst,tok,cfg.protocol.name=='bfs_state')
    correct=not g['format_fail'] and g['pred']==row.label
    match=(g['pred']==old['pred'] and correct==old['correct'])
    item=dict(query_id=idx,label=row.label,pred=g['pred'],correct=correct,
              distance=old['distance'],matches_saved=match,interrupted=g['cot_interrupted'],
              forced_stop=g['forced_stop'],trace_length=len(g['trace_tokens']),
              trace_hash=hashlib.sha256(bytes(str(g['trace_tokens']),'ascii')).hexdigest(),**metrics)
    rows.append(item)
    if os.environ.get('AUDIT_CAPTURE')=='1' and metrics.get('raw_mistake_tokens',metrics['mistake_records']):
     mistake_cases.append(dict(query_id=idx,prompt=row.tokens[:row.prompt_len],
                               generated=g['generated'],trace=g['trace_tokens'],label=row.label,
                               source=src,target=dst,adj=adj))
    if (metrics['repairs'] or not match or (correct and not metrics['execution_complete'])) and len(examples)<5:
     examples.append(dict(**item,prompt=tok.decode(row.tokens[:row.prompt_len]),trace=tok.decode(g['generated'])))
   print(json.dumps(dict(job=name,case=case,done=len(rows),seconds=round(time.time()-start,1))),flush=True)
  # Store full compact metrics, not all activations / full generated traces.
  report=dict(model=name,case=case,checkpoint=str(ckpt),checkpoint_step=200000,
              checkpoint_size=stat.st_size,checkpoint_mtime_ns=stat.st_mtime_ns,
              source=str(source.root),batch_size=analysis_batch,
              seconds=time.time()-start,n=len(rows),rows=rows,examples=examples,mistake_cases=mistake_cases)
  write(OUT/(name+'_'+case+'.json'),report)
  assert ckpt.stat().st_mtime_ns==stat.st_mtime_ns
 if hook:hook.remove()
 del model
 torch.cuda.empty_cache()

if __name__=='__main__':
 torch.set_num_threads(2)
 print('Validator self-tests:',self_test(),flush=True)
 # Main, depth, width and node-count extrapolation; identical 256 saved IDs.
 for m in ['dfs_clean','dfs_error']:
  run(m,['main','deep','wide','larger'])
 run('dfs_error',['main','deep','wide','larger'],True)
 for m in ['bfs_clean','bfs_error']:
  run(m,['main'])
 run('bfs_error',['main'],True)
 write(OUT/'finished.json',dict(finished=True,cases=15))
