"""CPU-only error-profile comparison from exact batch-16 frozen rollouts.

No new generation. Separate query incidence from record counts, retain the
semantic replayer's censoring rule, and classify extraneous candidates on
shared canonical prefixes using the real graph.
"""
import json,runpy
from pathlib import Path
from collections import Counter
import numpy as np
P=Path(__file__).resolve().parent
D=runpy.run_path(str(P/'run.py'))
from cot_navigation.data.mistakes import records

def describe(rows):
 keys=['candidate_bad','candidate_missing','candidate_extra','candidate_duplicate','frontier_bad','visited_bad','action_bad']
 first=[r['first_error_record'] for r in rows if r['first_error_record'] is not None]
 bad=[r for r in rows if r['candidate_bad']]
 return dict(n=len(rows),query_counts={k:sum(bool(r[k]) for r in rows) for k in keys},
   record_counts={k:sum(r[k] for r in rows) for k in keys},
   checked_records=sum(r['checked_records'] for r in rows),
   first_error_types=dict(Counter(r['first_error'] or 'none' for r in rows)),
   first_error_median=float(np.median(first)),first_error_mean=float(np.mean(first)),
   candidate_error_correct=sum(r['correct'] for r in bad),candidate_error_wrong=sum(not r['correct'] for r in bad))

def main():
 cfg=D['load_config']([D['FOLDERS']['dfs_error']/'resolved_config.yaml'])
 tok=D['Tokenizer'].from_config(cfg)
 extra=dict(s.split('=',1) for s in cfg.evaluation.extra_sets)
 result={}
 for case in ['n16','n24','n32','n36','main']:
  folder=P/'strict' if case=='main' else P/'small'
  prefix_folder=P if case=='main' else P/'small'
  rr={a:json.loads((folder/('dfs_'+a+'_'+case+'.json')).read_text())['rows'] for a in ['clean','error']}
  assert all(r['matches_saved'] for rows in rr.values() for r in rows)
  aa,bb=[{r['query_id']:r for r in rr[a]} for a in ['clean','error']]
  both=[(aa[q]['first_error_record'],bb[q]['first_error_record']) for q in aa if aa[q]['first_error_record'] is not None and bb[q]['first_error_record'] is not None]
  profiles={a:describe(rows) for a,rows in rr.items()}
  profiles['paired_first_error']=dict(n=len(both),later=sum(y>x for x,y in both),earlier=sum(y<x for x,y in both),same=sum(y==x for x,y in both))
  source=D['ContextStore'](cfg.data.root if case=='main' else extra[case])
  cache=D['ScalePreparedRows'](source,cfg,tok)
  profiles['prefix']={}
  for arm in ['clean','error']:
   data=json.loads((prefix_folder/('prefix_dfs_'+arm+'_'+case+'.json')).read_text())
   patterns=Counter();classes=Counter();field_classes=Counter();wrong=[r for r in data['rows'] if not r['correct']]
   for r in wrong:
    m,e=bool(r['missing']),bool(r['extra'])
    key='missing+extra' if m and e else 'missing_only' if m else 'extra_only' if e else 'duplicate_or_format'
    patterns[key]+=1
    idx=r['query_id'];canonical=cache.row(idx,0);rec=records(canonical,tok)[r['record']]
    assert rec['cands']==r['gold']
    nodes=source.node_vocab_ids(idx,tok);graph_nodes=set(nodes);raw=source.adj(idx)
    adj={nodes[u]:set(nodes[v] for v in vs) for u,vs in enumerate(raw)}
    cur=rec['stack'][-1];visited=set(rec['visited'])
    assert set(r['gold'])==adj[cur]-visited
    present=set()
    for t in set(r['predicted'])-set(r['gold']):
     if t not in graph_nodes:kind='outside_graph'
     elif t not in adj[cur]:kind='graph_node_not_successor'
     else:
      assert t in visited
      kind='successor_already_visited'
     classes[kind]+=1;present.add(kind)
    for kind in present:field_classes[kind]+=1
   profiles['prefix'][arm]=dict(n=data['n'],wrong=len(wrong),patterns=dict(patterns),
    extra_node_instances=dict(classes),fields_with_extra_class=dict(field_classes),
    mean_missing_per_wrong=float(np.mean([r['missing'] for r in wrong])),
    mean_extra_per_wrong=float(np.mean([r['extra'] for r in wrong])))
  result[case]=profiles
 D['write'](P/'error_profile.json',result)
 lines=['# CoT 错误分布与干净基线对比','',
 'DFS：干净 λ16 对比 30% 自然错误训练 λ16，均为 last 200k；五组测试各256条，按原 batch=16 的轨迹与答案核验。只分析已有输出，无新增 GPU 生成或训练。','',
 '## 查询级错误发生率','',
 '下表为发生该类错误的查询数量，均除以256；类型可重叠。字段错误只在实际动作仍可独立重放时统计，非法动作之后不把错误继续累加为独立观测。','',
 '|图|候选错误：基线→新模型|漏候选|多候选|非法动作|栈错误|visited错误|','|---|---:|---:|---:|---:|---:|---:|']
 for case,x in result.items():
  keys=['candidate_bad','candidate_missing','candidate_extra','action_bad','frontier_bad','visited_bad']
  lines.append('|'+case.upper()+'|'+'|'.join(f"{x['clean']['query_counts'][k]} → {x['error']['query_counts'][k]}" for k in keys)+'|')
 lines+=['','## 候选错误记录并非统一减少','',
 '保留原始错误记录计数，包含随后被 MISTAKE 撤销的记录。恢复/继续执行使更多后续记录进入可审计区间，因此记录总数及其分母受轨迹长短、纠错和截断影响，不是无偏的同状态错误概率。','',
 '|图|错误候选记录数：基线→新模型|错误候选/可检查记录|每条出错查询的候选错误次数|','|---|---:|---:|---:|']
 for case,x in result.items():
  a,b=[x[k] for k in ['clean','error']]
  n,m=[z['record_counts']['candidate_bad'] for z in [a,b]]
  lines.append(f"|{case.upper()}|{n} → {m}|{100*n/a['checked_records']:.2f}% → {100*m/b['checked_records']:.2f}%|{n/a['query_counts']['candidate_bad']:.2f} → {m/b['query_counts']['candidate_bad']:.2f}|")
 lines+=['','## 首次出错位置','',
 '位置以输出搜索记录序号计，不等同于相同搜索状态；模型可能采取不同合法路径。配对仅取两模型都出现首次错误的同一查询。','',
 '|图|首次错误记录序号中位数：基线→新模型|配对查询数|新模型更晚 / 更早 / 相同|','|---|---:|---:|---:|']
 for case,x in result.items():
  a,b=x['clean'],x['error'];p=x['paired_first_error']
  lines.append(f"|{case.upper()}|{a['first_error_median']:g} → {b['first_error_median']:g}|{p['n']}|{p['later']} / {p['earlier']} / {p['same']}|")
 lines+=['','## 正确历史下的候选错误类型','',
 '以下是两模型共享的正确搜索前缀；候选集合自行生成。漏项与多项同时发生占主要部分，不能仅凭此认定模型读取了某个特定错误节点的邻接表。','',
 '|图|候选错误字段数：基线→新模型|同时漏项+多项|仅漏项|仅多项|','|---|---:|---:|---:|---:|']
 for case,x in result.items():
  a,b=[x['prefix'][k] for k in ['clean','error']]
  vals=[f"{a['wrong']} → {b['wrong']}"]+[f"{a['patterns'].get(k,0)} → {b['patterns'].get(k,0)}" for k in ['missing+extra','missing_only','extra_only']]
  lines.append('|'+case.upper()+'|'+'|'.join(vals)+'|')
 lines+=['','## 错误候选节点来自哪里','',
 '按真实图节点和正确历史区分：图外节点、图内但不是当前节点后继、真实后继但已经 visited。以下统计独特多余节点的出现次数，另有 JSON 保存包含这些类型的字段数。','',
 '|图|图外节点：基线→新模型|图内非后继|已 visited 的后继|','|---|---:|---:|---:|']
 for case,x in result.items():
  a,b=[x['prefix'][k]['extra_node_instances'] for k in ['clean','error']]
  lines.append('|'+case.upper()+'|'+'|'.join(f"{a.get(k,0)} → {b.get(k,0)}" for k in ['outside_graph','graph_node_not_successor','successor_already_visited'])+'|')
 lines+=['','## 候选出错与最终答案','',
 '以下是出现过候选错误的查询中，最后答错的数量。条件查询集合不同，不是对同一错误前缀的容错因果实验，也不能直接解释为纠错成功率。','',
 '|图|基线答错/有候选错误|新模型答错/有候选错误|','|---|---:|---:|']
 for case,x in result.items():
  a,b=[x[k] for k in ['clean','error']]
  lines.append(f"|{case.upper()}|{a['candidate_error_wrong']}/{a['query_counts']['candidate_bad']}|{b['candidate_error_wrong']}/{b['query_counts']['candidate_bad']}|")
 lines+=['','本分析基于单训练 seed 的历史模型；训练难例重采样和词表/初始化等混杂因素尚未独立消融。','']
 (P/'ERROR_PROFILE.md').write_text('\n'.join(lines),encoding='utf-8')
 print('\n'.join(lines))

if __name__=='__main__':main()
