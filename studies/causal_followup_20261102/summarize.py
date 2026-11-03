import json,collections
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent
def read(name):return json.loads((ROOT/(name+'.json')).read_text(encoding='utf8'))
def main():
 fresh=read('fresh_candidates');length=read('length_layers');donor=read('length_donor');bars=read('bar_count');pairs=read('paired_validation');selection=read('fresh_selection')
 assert len(length)==420 and len(donor)==300 and len(bars)==324 and len(fresh)==221
 assert all(r['baseline_correct'] and r['earlier_logits_unchanged'] for r in length+donor)
 for r in length:
  expected=(r['dose'] if r['receiver']=='stop' else -r['dose']) if r['condition']=='length' else 0
  assert abs(r['probe_shift'][0]-expected)<2e-4 and abs(r['probe_shift'][1])<2e-4
 for r in donor:
  if r['condition'] in ['sham','complement_donor','matched_null']:assert max(abs(v) for v in r['probe_shift'][:2])<2e-4
 assert pairs['passed'] and len(pairs['results'])==13 and all(r['legal_through_current_record'] for r in pairs['results'])
 assert sum(r['available'] for r in selection)==13 and len(selection)==18
 assert all(r['earlier_logits_unchanged'] for r in bars if r['mode']=='virtual')
 for r in bars:
  if 'far' in r['condition']:assert r['nearest_bar_distance_before']==r['nearest_bar_distance_after']
  if r['condition'].startswith('preserve_count'):assert r['count_delta']==0
  if r['condition']=='sham':assert not r['output_changed'] and r['delta_norm']==0
 fs={c:dict(n=len([r for r in fresh if r['condition']==c]),expected_correct=sum(r['expected_correct'] for r in fresh if r['condition']==c),original_correct=sum(r['original_correct'] for r in fresh if r['condition']==c),excluded_returned=sum(r['excluded_returned'] for r in fresh if r['condition']==c)) for c in sorted(set(r['condition'] for r in fresh))}
 ls={str(l):{str(d):{c:sum(r['class_switch'] for r in length if r['layer']==l and r['dose']==d and r['condition']==c) for c in ['length','null']} for d in [1,2,4]} for l in range(2,7)}
 ds={str(l):{c:sum(r['class_switch'] for r in donor if r['layer']==l and r['condition']==c) for c in ['full_donor','depth_only_donor','complement_donor','matched_null']} for l in range(2,7)}
 bs={}
 for c in sorted(set(r['condition'] for r in bars)):
  rs=[r for r in bars if r['condition']==c];bs[c]=dict(n=len(rs),count_delta=rs[0]['count_delta'],output_changed=sum(r['output_changed'] for r in rs),L1_role_changed=sum(r['probe_before']['1']!=r['probe_after']['1'] for r in rs),L2_role_changed=sum(r['probe_before']['2']!=r['probe_after']['2'] for r in rs),L4_role_changed=sum(r['probe_before']['4']!=r['probe_after']['4'] for r in rs),follows_count_mod3_L4=sum(r['probe_after']['4']==r['expected_global_count_role'] for r in rs))
 out=dict(fresh=fs,length=ls,donor=ds,bars=bs,n_fresh_queries=13,n_length_queries=6,n_bar_queries=6)
 (ROOT/'summary.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf8')
 validation=dict(passed=True,paired_legal_states=13,fresh_conditions=len(fresh),length_conditions=len(length),donor_conditions=len(donor),bar_conditions=len(bars),length_shifts_verified=True,shams_identity=True,virtual_interventions_leave_history_unchanged=True,far_bar_interventions_preserve_nearest_distance=True)
 (ROOT/'validation.json').write_text(json.dumps(validation,indent=2))
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 plt.rcParams['font.sans-serif']=['Microsoft YaHei'];plt.rcParams['axes.unicode_minus']=False
 fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
 colors=['#227f9c','#e2a64d','#89a3ad','#b9c3cb']
 for i,(c,label) in enumerate([('full_donor','完整自然表示'),('depth_only_donor','仅probe长度分量'),('complement_donor','去掉长度/进度分量'),('matched_null','同幅度随机对照')]):axes[0].plot(range(2,7),[ds[str(l)][c]/12 for l in range(2,7)],marker='o',label=label,c=colors[i])
 axes[0].set(xlabel='干预层',ylabel='停止/继续切换比例',ylim=(-.05,1.05),title='相同起点、相同已输出数量=1');axes[0].legend(fontsize=9)
 names=['基线','仅改当前visited','仅改旧选中候选','改选中候选+参数','改未选中候选+参数'];conds=['baseline','explicit_visited_only','historical_selected_candidate_only','historical_selected_candidate_argument_only','historical_unselected_candidate_argument_only']
 values=[fs[c]['excluded_returned'] for c in conds];axes[1].barh(names,values,color=['#aebec8','#aebec8','#438ea4','#207f94','#aebec8']);axes[1].set(xlim=(0,14),xlabel='被排除节点重新输出的案例数 / 13',title='首次展开：历史选择见证的作用');axes[1].invert_yaxis()
 for i,v in enumerate(values):axes[1].text(v+.1,i,str(v),va='center')
 fig.savefig(ROOT/'causal_summary.png',dpi=160);plt.close(fig);print(json.dumps(validation))
if __name__=='__main__':main()
