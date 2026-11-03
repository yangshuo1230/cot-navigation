"""Reproducible query-balanced summaries and consistency checks."""
import json,collections
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent
def read(name):return json.loads((ROOT/(name+'.json')).read_text(encoding='utf8'))
def query_mean(rows,fn):
 groups=collections.defaultdict(list)
 for r in rows:groups[r['index']].append(fn(r))
 return float(np.mean([np.mean(v) for v in groups.values()]))
def main():
 probes=read('probe_results');candidates=read('candidate_tests');push=read('push_carrier');length=read('length_intervention');bars=read('bar_interventions');raw=read('layer6');meta=read('readout_metadata');cal=read('readout_probe_results');audit=read('candidate_audit');pm=read('probe_metadata')
 for kind in ['stack','bars']:
  p=probes[kind];assert not set(p['refit_queries'])&set(p['test_queries']);assert not set(p['training_queries'])&set(p['development_queries']);assert len(p['refit_queries'])==18 and len(p['test_queries'])==18
 assert len(length)==48 and all(r['baseline_correct'] and r['earlier_logits_unchanged'] for r in length)
 assert all(np.max(np.abs(np.array(r['decoded_after'])-r['decoded_before']))<1e-5 for r in length if r['condition'] in ['sham','norm_matched_null'])
 assert all(r['bar_value_max_difference']==0 and r['earlier_logits_unchanged'] for r in bars)
 assert all(not r['output_changed'] for r in bars if r['condition']=='same_symbol_value_swap')
 bases={(r['index'],r['record'],r['category']):r['original_set_correct'] for r in candidates if r['condition']=='baseline'}
 candidate_summary={}
 for condition in sorted(set(r['condition'] for r in candidates)):
  rs=[r for r in candidates if r['condition']==condition];eligible=[r for r in rs if bases[r['index'],r['record'],r['category']] and r['expected_changed']]
  candidate_summary[condition]=dict(n=len(rs),changed=sum(r['changed'] for r in rs),baseline_correct_and_target_changed=len(eligible),changed_when_target_changed=sum(r['changed'] for r in eligible))
 repeats=[r for r in audit if r['split']=='test' and r['simple_push_subtraction_matches'] is not None]
 assert len(repeats)==344 and all(r['simple_push_subtraction_matches'] for r in repeats)
 stages={};lookup={(r['index'],r['phase'],r['q']):r for r in raw}
 for phase in sorted(set(r['phase'] for r in meta)):
  rs=[r for r in meta if r['split']=='test' and r['phase']==phase]
  stages[phase]=dict(n_queries=len(set(r['index'] for r in rs)),n_events=len(rs),raw_L5_agreement=query_mean(rs,lambda r:lookup[r['index'],phase,r['q']]['stage']['L5']['agrees_final']),calibrated_L5_agreement=cal['L5']['test_phases'][phase]['agrees_final'],no_L6_attention_agreement=query_mean(rs,lambda r:r['no_attention_prediction']==r['final']),no_L6_ffn_agreement=query_mean(rs,lambda r:r['no_ffn_prediction']==r['final']))
 push_summary={c:dict(n=len([r for r in push if r['condition']==c]),set_correct=sum(r['set_correct'] for r in push if r['condition']==c)) for c in sorted(set(r['condition'] for r in push))}
 out=dict(reference_queries=36,probe_training_queries=18,probe_test_queries=18,stack_examples=len(pm['stack']),bar_examples=len(pm['bars']),candidate=candidate_summary,repeat_audit=dict(n=len(repeats),tail_matches=sum(r['simple_push_subtraction_matches'] for r in repeats)),push=push_summary,length={c:dict(n=len([r for r in length if r['condition']==c]),switches=sum(r['expected_class_switch'] for r in length if r['condition']==c)) for c in sorted(set(r['condition'] for r in length))},stages=stages)
 (ROOT/'summary.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf8');(ROOT/'validation.json').write_text(json.dumps(dict(passed=True,query_splits_disjoint=True,stack_probe_examples=len(pm['stack']),bar_probe_examples=len(pm['bars']),candidate_conditions=len(candidates),push_conditions=len(push),length_conditions=len(length),bar_conditions=len(bars),stage_events=len(meta)),indent=2))
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 plt.rcParams['font.sans-serif']=['Microsoft YaHei'];plt.rcParams['axes.unicode_minus']=False
 fig,ax=plt.subplots(1,2,figsize=(11,4),layout='constrained')
 ss=[r for r in probes['stack']['results'] if r['features'] in [f'L{i}' for i in range(7)]]
 for i,name in enumerate(['目标长度','已输出数量','剩余数量']):ax[0].plot(range(7),[r['test']['r2'][i] for r in ss],marker='o',label=name)
 ax[0].axhline(probes['stack']['results'][0]['test']['r2'][0],ls='--',c='gray',label='位置控制：目标长度');ax[0].set(ylim=(-.6,1.05),xlabel='表示层',ylabel='独立查询测试 R²',title='stack：线性可读性 ≠ 因果控制');ax[0].legend(fontsize=9)
 bb=[r for r in probes['bars']['results'] if r['features'] in [f'L{i}' for i in range(7)]];ax[1].plot(range(7),[r['test']['accuracy'] for r in bb],marker='o',label='字段身份（query均衡）');ax[1].axhline(1/3,ls='--',c='gray',label='随机水平');ax[1].set(ylim=(0,1.05),xlabel='表示层',ylabel='测试准确率',title='同一个 | 的三种字段身份');ax[1].legend(fontsize=9);fig.savefig(ROOT/'probe_curves.png',dpi=150);plt.close(fig)
 print(json.dumps(dict(validated=True,queries=36,candidate_conditions=len(candidates),length_conditions=len(length),bar_conditions=len(bars),push_conditions=len(push))))
if __name__=='__main__':main()
