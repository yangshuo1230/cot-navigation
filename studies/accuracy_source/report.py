"""One key figure and a reviewable report from completed frozen-model audits."""
import json,runpy
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path(__file__).resolve().parent
S=runpy.run_path(str(P/'summarize.py'))

def main():
 S['main']()
 broad=json.loads((P/'summary.json').read_text())
 strict=[json.loads((P/'strict'/(n+'_main.json')).read_text()) for n in ['dfs_clean','dfs_error']]
 clean,error=strict
 assert all(r['matches_saved'] for d in strict for r in d['rows']), 'Original batch replication did not match all saved answers'
 ca,ea=map(S['aggregate'],strict)
 pair=S['paired'](clean,error)
 intervention=json.loads((P/'strict/dfs_prefix_intervention_raw.json').read_text())
 meta=json.loads((P/'training_audit.json').read_text())
 stored=json.loads((P/'paired_saved.json').read_text())
 result=dict(primary=dict(clean=ca,error=ea,paired=pair),prefix_intervention=intervention,
             cross_scale=broad,limitations=['one training seed','historical initialization differs due to extra MISTAKE vocabulary','hard-example resampling changes query distribution','cross-scale free generation uses batch 128 and exhibits BF16 drift; original batch 16 primary replicated exactly'])
 (P/'attribution.json').write_text(json.dumps(result,indent=2)+'\n')
 pct=lambda n,d=256:f'{100*n/d:.2f}%'
 gain=pair.get('gain',{'n':0});loss=pair.get('loss',{'n':0})
 iv=intervention['rows']
 first=sum(x['normal_first_is_mistake'] for x in iv)
 hurt=sum(x['normal']['correct'] and not x['blocked']['correct'] for x in iv)
 helped=sum(not x['normal']['correct'] and x['blocked']['correct'] for x in iv)
 originally_reproduced=sum(x['normal']['pred']==x['original_pred'] for x in iv)
 lines=['# 准确率提升来源审计','',
 '## 结论与边界','',
 f"主测试准确率提高 {(ea['correct']-ca['correct'])/256*100:.2f} 个百分点；完整状态合法轨迹增加 {ea['effective_complete']-ca['effective_complete']} 条，但新模型经过错误记录撤销后仍完整合法结束的轨迹只有 {ea['effective_complete']-ea['literal_complete']} 条。改善答案的 {gain['n']} 条中，有 {gain['n']-gain['new_marked']} 条没有生成 MISTAKE。因此，不能将整体增益概括为在线纠错提高合法率；普通生成改善、部分仍不完全合法的成功，以及训练查询重采样均需区分。",'',
 '本报告检验推理过程中的行为来源，不把历史模型之间的差异直接当成错误数据的独立因果效应。重点是 DFS 30% 自然错误、λ16，与干净 DFS λ16 的 last（200k 步）；BFS 对照为历史 λ1，仅作参考。没有新训练、修改 checkpoint 或创建 W&B run。','',
 '## 主测试：按原 batch=16 精确复现','',
 '全部 512 条模型预测均与原保存结果一致；使用相同 256 个查询，独立图可达性 oracle 核对标签。','',
 '|指标|干净 DFS λ16|错误 DFS 30% λ16|','|---|---:|---:|',
 f"|答案正确|{ca['correct']}/256（{pct(ca['correct'])}）|{ea['correct']}/256（{pct(ea['correct'])}）|",
 f"|完整状态与动作均合法，且终止一致|{ca['effective_complete']}/256|{ea['effective_complete']}/256|",
 f"|忽略显示字段后的完整合法动作搜索|{ca['execution_complete']}/256|{ea['execution_complete']}/256|",
 f"|出现候选错误的查询|{ca['query_counts']['candidate_bad']}/256|{ea['query_counts']['candidate_bad']}/256|",
 f"|生成 MISTAKE 的查询（完整轨迹，不截断统计）|0|{ea['raw_marker_queries']}/256|",
 f"|可信前缀中的 MISTAKE 查询|0|{ea['query_counts']['mistake_records']}/256|",
 f"|观察到局部合法修复的查询|0|{ea['query_counts']['repairs']}/256|",
 f"|中断轨迹后回答|{ca['interrupted']}/256|{ea['interrupted']}/256|",'',
 '合法性独立于特定金标准遍历顺序：候选按集合检查，visited 按集合及无重复检查，栈/队列按实际动作检查。MISTAKE 撤销该记录、不更新执行状态；后续正确记录可计为有效修复。严格合法要求正常记录的状态正确、动作与候选优先级一致；动作搜索另行忽略显示字段，允许合法邻居优先级。遇到非法动作后不再把后续字段错误当独立样本统计。','',
 '完整合法搜索与正确终止本身蕴含正确答案，因此不能把其与答案正确的相关性当成因果证据。状态字段不完全合法的正例也可能保留有效可达路径，不等同于模型猜答案。','',
 '## 配对查询：哪些答案发生了改善','',
 f"从错变对 {gain['n']} 条，从对变错 {loss['n']} 条，净增 {gain['n']-loss['n']} 条。",
 f"改善的 {gain['n']} 条中：{gain['new_marked']} 条出现 MISTAKE，{gain['new_repaired']} 条出现局部合法修复；{gain['new_effective']} 条变为完整状态合法轨迹，{gain['new_execution']} 条具备完整合法动作搜索。",'',
 f"改善的 {gain['n_positive']} 条正例中，{gain['new_positive_witness']} 条能从实际 PUSH 边重建可达见证；这个检查允许提前放弃分支、忽略显示字段，但不允许把图中不存在的边当成见证。它用于区分不完整合法搜索与仍找到有效正例路径。",'',
 '新增正确查询的行为分解：','',
 '|类别|数量|','|---|---:|',
 f"|完整状态与搜索动作合法|{gain['new_effective']}|",
 f"|状态/搜索不完全合法，但正例仍有有效 PUSH 路径见证|{sum(r['label']==1 and r['new_positive_witness'] and not r['new_effective'] for r in gain['examples'])}|",
 f"|正例答案正确，但本次审计未重建有效 PUSH 路径|{sum(r['label']==1 and not r['new_positive_witness'] for r in gain['examples'])}|",
 f"|负例答案正确，但没有完整合法穷尽搜索|{sum(r['label']==0 and not r['new_execution'] for r in gain['examples'])}|",'',
 '|查询组|数量|新模型有 MISTAKE|新模型有局部合法修复|新模型完整状态合法|新模型完整动作合法|',
 '|---|---:|---:|---:|---:|---:|']
 for name,desc in [('gain','从错变对'),('loss','从对变错'),('both_correct','两者都对'),('both_wrong','两者都错')]:
  x=pair[name]
  lines.append(f"|{desc}|{x['n']}|{x['new_marked']}|{x['new_repaired']}|{x['new_effective']}|{x['new_execution']}|")
 lines+=['',f"原主测试配对 McNemar 精确双侧 p={stored['dfs_last']['mcnemar_exact_p']:.3f}，未校正多重比较；它只反映这批测试查询的不确定性，不替代多训练 seed。",'',
 '## 显式纠错的因果干预','',
 f"取原 batch=16 自由生成中实际出现动作 MISTAKE 的 {len(iv)} 条查询，固定在其首次 MISTAKE 之前的完全相同前缀，分别继续正常生成或禁止 MISTAKE。两分支采用相同固定 batch、完整 KV 容量、不移除完成行，避免别的样本改变结束时间导致数值路径变化。不给模型正确修复记录。",
 f"重新 prefill 后，{first}/{len(iv)} 条正常分支首 token 仍为 MISTAKE；{originally_reproduced}/{len(iv)} 条正常分支最终预测与原连续 decode 一致。正常分支答对 {sum(x['normal']['correct'] for x in iv)}/{len(iv)}，禁止后答对 {sum(x['blocked']['correct'] for x in iv)}/{len(iv)}；禁止后变错 {hurt} 条、变对 {helped} 条。",'',
 '该干预检验这些已触发纠错的前缀是否依赖显式标记，不能识别训练期间错误数据对普通生成能力的贡献。它也不能直接把重启续写分支的差值外推成原全部 256 条准确率的精确因果差值。逐查询记录含原预测、正常/禁止预测、修复合法性与后续文本。','',
 '## 排除历史错误：相同正确前缀下生成候选集合','',
 '每个查询固定参考轨迹 variant=0，最多抽取 4 个相同搜索状态；排除 FOUND 状态。给两模型相同正确历史、当前 frontier 与 visited，让模型自行输出候选直到实际生成分隔符。允许候选重排，成员缺失、多出、重复及未正常闭合均判错。全部参考轨迹先通过独立合法性审计。','',
 '|图|字段数|干净 λ16 正确率|错误 30% λ16 正确率|','|---|---:|---:|---:|']
 for case,desc in [('main','N100 主测试'),('deep','N100 深图'),('wide','N100 宽图'),('larger','N144')]:
  x=broad['dfs_'+case]['clean_prefix'];lines.append(f"|{desc}|{x['n']}|{pct(x['clean_correct'],x['n'])}|{pct(x['error_correct'],x['n'])}|")
 lines+=['','主测试上的候选生成有改善，但深图与 N144 不符合全面提高的解释。这些字段来自同一图的多个状态，不是相互独立的训练 seed。另有 field_action_probe.json 记录错误字段之后的下一动作，避免把所有错误字段都解释为有意产生并准备修复。','',
 '## 正确历史下的状态字段生成','',
 '在上述完全相同的搜索状态，分别只提供当前 frontier 字段之前或 visited 字段之前的正确前缀，让模型自行生成该字段直到分隔符。frontier 要求顺序正确，visited 接受无重复的同一集合。','',
 '|图|栈：干净→错误训练|visited：干净→错误训练|','|---|---:|---:|']
 states=json.loads((P/'state_field_probe.json').read_text())
 for case,desc in [('main','N100 主测试'),('deep','N100 深图'),('wide','N100 宽图'),('larger','N144')]:
  a,b=[states['dfs_'+n+'_'+case] for n in ['clean','error']]
  vals=[f"{pct(a[f]['correct'],a[f]['n'])} → {pct(b[f]['correct'],b[f]['n'])}" for f in ['frontier','visited']]
  lines.append('|'+desc+'|'+'|'.join(vals)+'|')
 lines+=['','深图上 visited 生成改善而候选集合生成下降，说明状态能力变化是分项的；不能用单一“CoT 合法率提高”解释所有规模的表现。','',
 '## 跨形状自由生成审计：探索性结果','',
 '此部分统一 batch=128 以加速，BF16 数值变化导致部分预测或轨迹与历史 batch=16 不同；以下准确率属于本次重新生成，不与历史数字混用。主测试最终结论以精确复现的 batch=16 为准。','',
 '|图|干净答案正确数|错误训练答案正确数|干净完整状态合法数|错误训练完整状态合法数|干净/新模型预测不复现数|',
 '|---|---:|---:|---:|---:|---:|']
 for case,desc in [('main','N100'),('deep','N100 深图'),('wide','N100 宽图'),('larger','N144')]:
  x=broad['dfs_'+case]['aggregates'];a=x['dfs_clean_'+case];b=x['dfs_error_'+case]
  lines.append(f"|{desc}|{a['correct']}|{b['correct']}|{a['effective_complete']}|{b['effective_complete']}|{a['saved_answer_mismatches']}/{b['saved_answer_mismatches']}|")
 lines+=['','宽松的全程 MISTAKE 禁用也已完成，但可信前缀外还可能生成 MISTAKE，批次压缩也可能改变数值路径。不能根据可信前缀没有标记就把全部轨迹变化归为数值漂移，也不能把原始准确率差值都当成干预因果效应。正式干预解释使用上述完整轨迹标记普查与固定形状、相同前缀实验。','',
 '## 数据与训练混杂因素','',
 '已核验全部自然错误池的 train/val/test 元数据与原始 source split 一致，训练加载仅取 train；本次未发现该路径的留出数据混入。DFS 的有效模型架构、优化配置、轨迹权重和课程相同，主要额外差异包括 MISTAKE 词表、错误数据管线和周期自由生成验证由 16 增至 64。不同词表会改变相同 seed 的随机初始化过程；因此历史模型不是严格配对初始权重。','',
 '更关键的是，错误数据并非仅给原来抽到的查询加错误，而是按相同图规模、相同标签，从旧模型出错的查询池中替换抽样；还优先采样复杂错误。这同时改变了训练查询分布，构成难例重采样。','',
 '|DFS 图规模|错误池负例查询数|错误池正例查询数|正例原训练平均层距离|错误分支正例期望层距离|',
 '|---|---:|---:|---:|---:|']
 for s in meta['dfs']['sources']:
  a=s['by_label']['0'];b=s['by_label']['1'];name=s['root'].split('/')[-1]
  lines.append(f"|{name}|{a['pool_size']}|{b['pool_size']}|{b['baseline_distance_mean']:.2f}|{b['pool_expected_distance_mean']:.2f}|")
 last=meta['dfs']['sources'][-1]['by_label'];neg,pos=last['0'],last['1']
 lines+=['',f"还检查了额外错误记录是否增加训练截断：在 N100 错误池的同一批查询、按实际复杂度采样权重计算，正例的 STOP 比例插入前后均为 {100*pos['pool_expected_original_stop']:.2f}%；负例由 {100*neg['pool_expected_original_stop']:.2f}% 变为 {100*neg['pool_expected_augmented_stop']:.2f}%。这是错误分支内的条件分布，不是全训练集比例；未发现错误插入造成明显额外正例截断来解释主测试正例提升。",'',
 '30% 错误分支保留标签，因此这不等同于简单增加正例比例；改变的是标签内部的查询、距离、轨迹以及重复频次。错误记录还重复了 frontier/visited 等内容，且 full loss 对错误候选 token 本身也施加生成监督。这些训练变化没有在当前实验中独立消融。要识别纠错训练的效果，需要同词表/初始化，并增加“同一难例重采样、但去除错误插入”的干净对照；本次没有额外 launch 训练。','',
 '## 结果文件','',
 '- attribution.json：主结果、配对改善查询与干预记录。',
 '- strict/*.json：按原 batch 精确复现的主测试、受控前缀干预。',
 '- prefix_*.json：相同正确前缀的候选集合生成。',
 '- paired_saved.json：全部九个规模/形状、best/last 的历史逐查询配对。',
 '- training_audit.json：有效配置差异、错误池采样分布与 split 核验。',
 '- summary.json：大批量跨形状审计，含数值漂移标记。','']
 (P/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
 # A single compact scientific figure; no activation dump or image per case.
 fig,axes=plt.subplots(1,3,figsize=(15,4.4))
 colors=['#4c78a8','#f58518'];width=.35
 labels=['Answer accuracy','Valid full states','Valid action search']
 for j,a in enumerate([ca,ea]):
  values=[100*a[k]/256 for k in ['correct','effective_complete','execution_complete']]
  axes[0].bar(np.arange(3)+(j-.5)*width,values,width,color=colors[j],label=['Clean','Error-trained'][j])
 axes[0].set_xticks(range(3),labels,rotation=18);axes[0].set_ylim(0,105);axes[0].set_ylabel('% of 256 queries');axes[0].set_title('DFS main: original batch replicated');axes[0].legend()
 for j,k in enumerate(['clean_correct','error_correct']):
  values=[100*broad['dfs_'+c]['clean_prefix'][k]/broad['dfs_'+c]['clean_prefix']['n'] for c in ['main','deep','wide','larger']]
  axes[1].bar(np.arange(4)+(j-.5)*width,values,width,color=colors[j])
 axes[1].set_xticks(range(4),['Main','Deep','Wide','Larger']);axes[1].set_ylim(0,105);axes[1].set_ylabel('Candidate-set accuracy (%)');axes[1].set_title('Identical correct history')
 vals=[gain['n']-gain['new_marked'],gain['new_marked']-gain['new_repaired'],gain['new_repaired']]
 axes[2].bar(range(3),vals,color=['#54a24b','#eeca3b','#b279a2']);axes[2].set_xticks(range(3),['No MISTAKE','Marked, no repair','Repaired'],rotation=18);axes[2].set_ylabel('Newly correct queries');axes[2].set_title('What happened on paired gains?')
 for i,v in enumerate(vals):axes[2].text(i,v+.15,str(v),ha='center')
 fig.tight_layout();fig.savefig(P/'key_results.png',dpi=180);plt.close(fig)
 print('Report and single figure written; all primary answers replicated.')

if __name__=='__main__':main()
