"""Compact evidence report; completed and error-censored traces stay separate."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent
protocols=['dfs_nodes','bfs_nodes','dfs_state','bfs_state']
runs=[json.loads((HERE/f'{s}_{p}.json').read_text()) for s in ('simple','hard') for p in protocols]
assert all(r['version']==2 for r in runs)
def pct(x):return '—' if x is None else f'{100*x:.1f}%'

lines=['# Fixed N100 DFS/BFS 训练轨迹复用检查','',
       '检查简单图 all_hops 与困难图，各自验证选择的 best checkpoint，覆盖 DFS/BFS 节点版和全状态版共8个模型。冻结推理在GPU0进行，没有训练、修改checkpoint或上传W&B。', '',
       '主要结论：存在整条和片段复用，但全文匹配率与正确参考搜索的自然重合接近，不能据此认定在抄训练轨迹。四个全状态模型都生成了训练库没有的合法frontier＋visited组合，比例分别为simple DFS13.1%、simple BFS23.0%、hard DFS22.3%、hard BFS51.0%。这些是合法记录出现次数，不是独立或去重状态数。由此可以排除纯整条复印/完整状态精确查表；不能排除局部记忆与组合，也没有证明通用搜索算法。','',
       '## 方法','',
       '- 每张图按查询跨度×正负标签固定抽取最多4条，simple64条、hard63条；同图各协议查询相同。',
       '- 从best checkpoint的采样器epoch/当前游标/缓冲pending恢复实际曝光查询，计数严格等于queries_seen；测试查询ID及起终点对均与已曝光训练查询不重叠。',
       '- 收集这些训练查询全部4个离线随机搜索变体，形成候选库。训练每次随机选择其中一个变体，不是无限在线重新打乱。逐变体累计曝光未保存，因此库是实际见过轨迹的保守超集；未匹配可排除整条原样复制，匹配只表示有复制可能。',
       '- 比较不含输入和YES/NO的CoT正文：逐token完全相同、首次扩展节点序列相同、完整节点序列为某训练序列前缀、状态记录前缀重合及8节点连续片段重合。',
       '- 对照1：同一测试查询的4条离线合法随机搜索，均不属于训练查询；对照2：3套全图固定邻居优先级产生的合法搜索。对照面板与完成生成的查询配对；不是独立模型或显著性检验。',
       '- 全状态额外检查语义状态：frontier的精确顺序＋visited集合。不含候选/动作、忽略visited打印顺序，因此新状态不只是表面排序差异。排除前两条记录，只统计独立重放确认合法的记录；比较整个训练候选库。',
       '- 原上下文/STOP预算保留；首次语义错误即停止并标记censored，不补造后续轨迹或最终答案。整段匹配主表仅统计已输出答案且未STOP中断的轨迹；分母与全抽样数分开。',
       '- 匹配不要求答案相同；主问题是轨迹复用。终止FOUND/EXHAUST保留在全文匹配中，在记录前缀比例中移除，避免目标相关终止造成机械不匹配。',
       '- BF16 greedy、batch64；小数值差异可能改变邻居选择，不声称逐轨迹复现旧batch32审计。', '',
       '## 正文完全相同：模型与合法搜索对照','',
       '|图|协议|完成/面板|完整合法/面板|模型全文匹配|合法随机对照全文匹配|模型扩展序列匹配|合法随机对照扩展序列匹配|',
       '|---|---|---:|---:|---:|---:|---:|---:|']
for r in runs:
    g=r['generated_complete'];ref=r['matched_reference']['heldout_random_reference']
    lines.append(f"|{r['setting']}|{r['protocol']}|{r['complete_count']}/{len(r['panel_ids'])}|{r['semantic_valid_count']}/{len(r['panel_ids'])}|{pct(g['exact_tokens'])}|{pct(ref['exact_tokens'])}|{pct(g['exact_expansions'])}|{pct(ref['exact_expansions'])}|")
lines += ['', '全文重合不等于记忆：相同起点的不可达查询，在同一邻居顺序下会执行同一整棵可达子图搜索；目标不同仍可完全相同。正例state全文又隐含起点、终点和FOUND，因此查询留出本身就限制了其整条匹配，不能只凭该指标判定机制。', '',
          '## 排除短序列及正负样本混合','',
          '|图|协议|≥8扩展节点的完成输出数|其中全文匹配|随机参考 ≥8节点全文匹配|正例全文匹配|负例全文匹配|',
          '|---|---|---:|---:|---:|---:|---:|']
for r in runs:
    g=r['generated_long8'];ref=r['reference_long8']
    lines.append(f"|{r['setting']}|{r['protocol']}|{g['n']}|{pct(g.get('exact_tokens'))}|{pct(ref.get('exact_tokens'))}|{pct(r['by_label']['1'].get('exact_tokens'))}|{pct(r['by_label']['0'].get('exact_tokens'))}|")
lines += ['', '随机参考长度筛选不保证与模型长度逐一相同；该表是长轨迹诊断，不能据此直接估计因果复制比例。', '',
          '## 节点版补查WALK库','',
          '节点版还可能直接复用随机游走。补充重放同一批输出，核对长度和是否完成不变；best checkpoint采样器证明Stage B已经实际抽到WALK池中的每个条目。下表仅比较整条节点序列，不把任意连续子片段计为整个WALK匹配。','',
          '|图|协议|WALK池条目数|整条匹配WALK|匹配CoT或WALK任一库|','|---|---|---:|---:|---:|']
for r in runs:
    if r['protocol'].endswith('_nodes'):
        w=r['walk_supplement']
        lines.append(f"|{r['setting']}|{r['protocol']}|{w['walk_items']}|{pct(w['exact_walk_rate'])}|{pct(w['cot_or_walk_rate'])}|")
lines += ['', '因此节点版的未匹配CoT输出不能一概称为训练中从未见过；少量会匹配WALK。即使合并两个库，仍有多数完成输出不匹配任何整条训练序列。全状态格式无法与WALK正文整条相同；WALK仍可能提供局部图知识。', '',
          '## 前缀和片段复用','',
          '|图|协议|模型整个扩展序列是训练前缀|合法随机对照|全图固定优先级对照|模型记录最长前缀比例均值|8节点片段覆盖|',
          '|---|---|---:|---:|---:|---:|---:|']
for r in runs:
    g=r['generated_complete'];refs=r['matched_reference']
    lines.append(f"|{r['setting']}|{r['protocol']}|{pct(g['whole_node_prefix'])}|{pct(refs['heldout_random_reference']['whole_node_prefix'])}|{pct(refs['global_priority_reference']['whole_node_prefix'])}|{pct(g['record_prefix_fraction'])}|{pct(g['ngram8_seen_fraction'])}|")
lines += ['', '节点前缀完全重合包含短查询；不能与长轨迹全文匹配混用。固定图相同起点、不同目标的合法DFS/BFS自然共用搜索前缀，目标到达时的截断也改变前缀长度。', '',
          '## 是否出现训练中没有的合法语义状态','',
          '|图|协议|合法新状态数 / 合法状态数（排除每条轨迹前两步）|比例|同查询合法随机参考比例|',
          '|---|---|---:|---:|---:|']
for r in runs:
    if not r['protocol'].endswith('_state'):continue
    g=r['generated_all'];total=g['novel_valid_state_fraction_denominator'];fraction=g['novel_valid_state_fraction']
    novel=round(fraction*total) if total else 0
    ref=r['references']['heldout_random_reference']['novel_valid_state_fraction']
    lines.append(f"|{r['setting']}|{r['protocol']}|{novel}/{total}|{pct(fraction)}|{pct(ref)}|")
lines += ['', '这项证据比“某条token序列没出现过”更强：visited按集合规范化后，该frontier＋visited组合仍未出现在任何训练候选轨迹中，模型却生成了合法动作。它反驳“只按完整状态精确查表、没有任何组合”的极端解释；仍不能排除局部子程序、子树记忆、因子化检索或其他shortcut，也不证明模型掌握了通用搜索算法。', '',
          '## 匹配实例','']
for r in runs:
    for e in r['examples'][:2]:
        lines.append(f"- {r['setting']} {r['protocol']}：{e['source']} → {e['target']}，标签{e['label']}，全文匹配={e['exact_tokens']}，训练匹配目标={e['matched_training_target']}，训练匹配标签={e['matched_training_label']}；扩展前缀 {', '.join(e['expansion_preview'][:16])}。")
lines += ['', '## 结论边界','',
          '不能把训练轨迹重合直接解释为抄答案。应同时看合法搜索基线、长轨迹、正负样本以及新语义状态。即使发现合法的新组合，也只排除纯整条复印和完整状态精确查表，无法从行为重合检查中区分所有记忆与算法机制。下一步若要因果区分，应强制给出偏离模型常用顺序、但图搜索合法且训练中未出现的历史，检查能否持续更新状态和完成搜索。', '',
          '所有逐查询结果与少量实例保存在8个JSON。没有保存生成全文、大型激活或新checkpoint。']
(HERE/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
summary=[{k:v for k,v in r.items() if k not in ('rows','examples')} for r in runs]
(HERE/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')

plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(2,2,figsize=(11,7),layout='constrained')
x=np.arange(4);width=.26
for row,setting in enumerate(('simple','hard')):
    group=[r for r in runs if r['setting']==setting]
    for col,(metric,title) in enumerate([('exact_tokens','Whole CoT body matches a training variant'),('whole_node_prefix','Expansion sequence is a training prefix')]):
        ax=axes[row,col]
        for offset,(kind,color,label) in enumerate([('model','#d95f02','Model'),('heldout_random_reference','#1b9e77','Legal random reference'),('global_priority_reference','#7570b3','Global-priority reference')]):
            values=[100*(r['generated_complete'] if kind=='model' else r['matched_reference'][kind])[metric] for r in group]
            ax.bar(x+(offset-1)*width,values,width,color=color,label=label)
        ax.set(xticks=x,xticklabels=['DFS node','BFS node','DFS state','BFS state'],ylim=(0,100),ylabel='Completed trajectories (%)',title=f'{setting.capitalize()} N100: {title}')
        ax.grid(axis='y',alpha=.15)
axes[0,0].legend(fontsize=8)
fig.suptitle('Training overlap needs an algorithm-generated baseline',fontsize=14)
fig.savefig(HERE/'trace_reuse.png',dpi=180);fig.savefig(HERE/'trace_reuse.pdf');plt.close(fig)
print('\n'.join(lines[20:40]))
