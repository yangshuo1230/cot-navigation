"""Validation-selected routes, held-out query-balanced comparisons, no pooling seeds."""
import gzip,json,collections
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent
GROUPS=['graph_EDGE','graph_source','graph_destination','query_source','query_target','prompt_bar','current_stack','current_visited','current_candidates','current_bars','current_action','history_stack','history_visited','history_candidates','history_bars','history_action','history_semicolon','self','other']
FEATURES=[('stack初始化','stack_init_PUSH','prompt_bar',None),('visited初始化','visited_init','prompt_bar',None),('root写入','root_write_after_PUSH','previous_push_argument',None),('POP后stack恢复','stack_continue','parent_matched_first_visited_carrier','POP'),('visited续写/root','visited_continue','history_roots',None),('visited续写/同名visited','visited_continue','previous_same_field_target',None),('首次候选/图+3','candidate_init_fresh','current_adjacency_destination_shift3',None),('首次候选/当前visited','candidate_init_fresh','@current_visited',None),('回访候选/旧PUSH参数','candidate_init_revisit','previous_same_current_PUSH_pair',None),('PUSH选节点','push_argument','current_candidates',None),('POP动作','action_POP','current_stack',None),('FOUND动作','action_FOUND','current_stack',None),('stack结束','stack_end','@current_stack',None),('visited结束','visited_end','@current_visited',None),('候选结束/bar','candidate_end','current_visited_bar',None),('YES答案','answer_YES','terminal_action',None),('NO答案','answer_NO','terminal_action',None)]
def load(name):
 with gzip.open(ROOT/(name+'.events.jsonl.gz'),'rt') as f:return [json.loads(x) for x in f]
def byquery(rows,fn):
 groups=collections.defaultdict(list)
 for r in rows:groups[r['index']].append(fn(r))
 return {q:np.mean(v,axis=0) for q,v in groups.items()}
def summary(rows,fn):
 d=byquery(rows,fn);values=np.stack(list(d.values()));rng=np.random.default_rng(20261102);boot=values[rng.integers(len(values),size=(600,len(values)))].mean(1)
 return dict(mean=values.mean(0).tolist(),low=np.percentile(boot,2.5,axis=0).tolist(),high=np.percentile(boot,97.5,axis=0).tolist(),n_queries=len(d),n_events=len(rows))
def selector(rows,phase,sub):return [r for r in rows if r['phase']==phase and (sub is None or r['sub']==sub)]
def vector(r,route,layers):return np.array(r['groups'])[:,GROUPS.index(route[1:])] if route.startswith('@') else r['routes'].get(route,[0.]*layers)
def main():
 models=json.loads((ROOT/'models.json').read_text());allrows={m['name']:load(m['name']) for m in models};out={};selected={}
 for model in models:
  name=model['name'];layers=model['layers'];rows=allrows[name];val=[r for r in rows if r['split']=='val'];test=[r for r in rows if r['split']=='test'];features={};selected[name]={}
  for title,phase,route,sub in FEATURES:
   va=selector(val,phase,sub);te=selector(test,phase,sub)
   if not va or not te:continue
   vs=summary(va,lambda r:vector(r,route,layers));ts=summary(te,lambda r:vector(r,route,layers));layer=int(np.argmax(vs['mean']))
   features[title]=dict(phase=phase,route=route,sub=sub,layer=layer+1,relative_depth=(layer+1)/layers,val=vs,test=ts)
   selected[name][title]=dict(layer=layer+1,route=route,phase=phase,sub=sub)
  phases={}
  for phase in sorted(set(r['phase'] for r in test)):
   rs=[r for r in test if r['phase']==phase];phases[phase]=dict(groups=summary(rs,lambda r:r['groups']),accuracy=summary(rs,lambda r:r['next_correct']),last_block={c:summary(rs,lambda r:r['last_block'][c]) for c in ['before_last_agrees','no_attention_agrees','no_ffn_agrees']})
  out[name]=dict(features=features,phases=phases)
 # Discover a monotone layer alignment from val group profiles, evaluate on test.
 alignments={}
 for left,right in [('base6','corr6_s0'),('corr6_s0','corr6_s1'),('corr9_s0','corr9_s1'),('corr6_s0','corr9_s0'),('corr6_s1','corr9_s1')]:
  def profiles(name,split):
   rows=[r for r in allrows[name] if r['split']==split];phases=sorted(set(r['phase'] for r in rows));return np.concatenate([np.mean(list(byquery([r for r in rows if r['phase']==p],lambda r:r['groups']).values()),axis=0) for p in phases],axis=1)
  a,b=profiles(left,'val'),profiles(right,'val');cos=a@b.T/(np.linalg.norm(a,axis=1)[:,None]*np.linalg.norm(b,axis=1)[None]);cost=1-cos;n,m=cost.shape;dp=np.full((n,m),np.inf);parent={};dp[0,0]=cost[0,0]
  for i in range(n):
   for j in range(m):
    if i==j==0:continue
    candidates=[(dp[ii,jj],ii,jj) for ii,jj in [(i-1,j),(i,j-1),(i-1,j-1)] if ii>=0 and jj>=0];v,ii,jj=min(candidates);dp[i,j]=v+cost[i,j];parent[i,j]=(ii,jj)
  path=[];i,j=n-1,m-1
  while True:
   path.append((i,j))
   if i==j==0:break
   i,j=parent[i,j]
  path=path[::-1];aa,bb=profiles(left,'test'),profiles(right,'test');check=aa@bb.T/(np.linalg.norm(aa,axis=1)[:,None]*np.linalg.norm(bb,axis=1)[None]);alignments[left+'__'+right]=dict(path=[[i+1,j+1] for i,j in path],validation_similarity=float(np.mean([cos[i,j] for i,j in path])),test_similarity=float(np.mean([check[i,j] for i,j in path])),test_matrix=check.tolist())
 result=dict(models=models,features=FEATURES,results=out,alignments=alignments,scope='Same 18 val/18 test query IDs and input tokens; clean correct reference prefixes. Route layers selected using val only. Seeds remain separate; bootstrap resamples queries. Raw attention, not causal contribution.')
 (ROOT/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8');(ROOT/'selected_routes.json').write_text(json.dumps(selected,ensure_ascii=False,indent=2),encoding='utf8')
 print(json.dumps({name:{title:dict(layer=f['layer'],mean=f['test']['mean'][f['layer']-1]) for title,f in item['features'].items()} for name,item in out.items()},ensure_ascii=False))
if __name__=='__main__':main()
