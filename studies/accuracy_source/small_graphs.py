"""Extend the frozen DFS attribution study to small graphs, original batch 16."""
import os
os.environ['AUDIT_BATCH']='16'
os.environ['AUDIT_CAPTURE']='1'
import runpy,json
from pathlib import Path
import torch
P=Path(__file__).resolve().parent
D=runpy.run_path(str(P/'run.py'))
OUT=P/'small';OUT.mkdir(exist_ok=True)
CASES=['n16','n24','n32','n36']
W=runpy.run_path(str(P/'raw_markers.py'))['witness']
old_audit=D['audit']

def raw_audit(trace,pred,adj,src,dst,tok,fifo):
 r=old_audit(trace,pred,adj,src,dst,tok,fifo)
 marker=tok.stoi.get('<MISTAKE>',-999)
 r['raw_mistake_tokens']=trace.count(marker)
 r['positive_witness']=W(trace,adj,src,dst,tok,fifo)
 return r

def summarize():
 S=runpy.run_path(str(P/'summarize.py'))
 result={}
 lines=['# 小图准确率提升来源：DFS，last 200k，λ16','',
 '干净训练与 30% 自然错误训练，相同查询和原 batch=16。候选顺序允许变化；合法性定义、可信前缀截断规则与主报告一致。MISTAKE 使用完整轨迹统计，不因审计遇到非法动作而漏记后面的标记。','',
 '|图|干净准确率|错误训练准确率|错→对 / 对→错|干净完整合法|新模型完整合法|新增正确中有 MISTAKE|',
 '|---|---:|---:|---:|---:|---:|---:|']
 for case in CASES:
  a,b=[json.loads((OUT/(n+'_'+case+'.json')).read_text()) for n in ['dfs_clean','dfs_error']]
  assert all(r['matches_saved'] for d in [a,b] for r in d['rows'])
  aa,bb=map(S['aggregate'],[a,b]);pair=S['paired'](a,b)
  iv=json.loads((OUT/('intervention_'+case+'.json')).read_text())
  pa,pb=[json.loads((OUT/('prefix_'+n+'_'+case+'.json')).read_text()) for n in ['dfs_clean','dfs_error']]
  assert pa['n']==pb['n']
  ids_a={(r['query_id'],r['record']):r['gold'] for r in pa['rows']}
  ids_b={(r['query_id'],r['record']):r['gold'] for r in pb['rows']}
  assert ids_a==ids_b
  orig=json.loads((D['FOLDERS']['dfs_error']/'eval'/('free_test_last-'+case+'.json')).read_text())['mistake_markers']
  assert bb['raw_marker_queries']==round(orig['query_rate']*orig['n'])
  assert sum(r['raw_mistake_tokens'] for r in b['rows'])==orig['events']
  result[case]=dict(clean=aa,error=bb,paired=pair,intervention=iv,
                   prefix=dict(n=pa['n'],clean_correct=pa['correct'],error_correct=pb['correct']))
  gain,loss=pair.get('gain',{'n':0,'new_marked':0}),pair.get('loss',{'n':0})
  lines.append(f"|{case.upper()}|{100*aa['correct']/256:.2f}%|{100*bb['correct']/256:.2f}%|{gain['n']} / {loss['n']}|{aa['effective_complete']}/256|{bb['effective_complete']}/256|{gain['new_marked']}/{gain['n']}|")
 lines+=['','## 相同正确前缀下的候选生成','','|图|字段数|干净正确率|错误训练正确率|','|---|---:|---:|---:|']
 for case,x in result.items():
  p=x['prefix'];lines.append(f"|{case.upper()}|{p['n']}|{100*p['clean_correct']/p['n']:.2f}%|{100*p['error_correct']/p['n']:.2f}%|")
 lines+=['','## 首次 MISTAKE 前缀的配对续写','','正常/禁止两分支固定 batch 和完整缓存形状。不给正确修复；核对正常分支能否复现原预测。','',
 '|图|标记查询数|正常续写答对|禁止后答对|禁止后变错 / 变对|正常首 token 复现 MISTAKE|正常最终预测复现|',
 '|---|---:|---:|---:|---:|---:|---:|']
 for case,x in result.items():
  rs=x['intervention']['rows'];n=len(rs)
  hurt=sum(r['normal']['correct'] and not r['blocked']['correct'] for r in rs)
  helped=sum(not r['normal']['correct'] and r['blocked']['correct'] for r in rs)
  first=sum(r['normal_first_is_mistake'] for r in rs)
  final=sum(r['normal']['pred']==r['original_pred'] for r in rs)
  lines.append(f"|{case.upper()}|{n}|{sum(r['normal']['correct'] for r in rs)}|{sum(r['blocked']['correct'] for r in rs)}|{hurt} / {helped}|{first}/{n}|{final}/{n}|")
 lines+=['','## 新增正确查询与合法修复','','|图|新增正确|完整合法|正例中有路径见证|可信前缀局部修复查询|修复后仍完整合法结束|',
 '|---|---:|---:|---:|---:|---:|']
 for case,x in result.items():
  g=x['paired']['gain'];e=x['error']
  lines.append(f"|{case.upper()}|{g['n']}|{g['new_effective']}|{g['new_positive_witness']}/{g['n_positive']}|{e['query_counts']['repairs']}|{e['effective_complete']-e['literal_complete']}|")
 lines+=['','所有数值均为单训练 seed 的历史 checkpoint。初始化词表和难例重采样等混杂因素仍存在；本分析识别推理行为，不独立证明训练数据的因果贡献。','']
 D['write'](OUT/'summary.json',result)
 (OUT/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
 print('\n'.join(lines),flush=True)

if __name__=='__main__':
 torch.set_num_threads(2)
 D['run'].__globals__.update(OUT=OUT,audit=raw_audit)
 for name in ['dfs_clean','dfs_error']:D['run'](name,CASES)
 prefix=runpy.run_path(str(P/'prefix_probe.py'))
 prefix['main'](names=['dfs_clean','dfs_error'],cases_override=CASES,output_dir=OUT)
 iv=runpy.run_path(str(P/'strict_main.py'))
 iv['intervention'].__globals__['USE_RAW_MARKERS']=True
 for case in CASES:
  iv['intervention'](case=case,output_dir=OUT,output_name='intervention_'+case+'.json')
 summarize()
 D['write'](OUT/'finished.json',dict(finished=True))
