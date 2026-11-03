from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent
d=json.loads((HERE/'results_snapshot_20261024.json').read_text())
diagnostic=json.loads((HERE/'transfer_diagnostic_20261024.json').read_text())
fig,axes=plt.subplots(2,1,figsize=(9,6),sharex=True,constrained_layout=True)
for g in sorted(d['grokking'],key=lambda x:x['id'],reverse=True):
    p=.1 if g['id'].startswith('d0.1_') else 0.
    x=[r['step']/1e6 for r in g['curves']]
    axes[0].plot(x,[r['monitor']['balanced_accuracy']*100 for r in g['curves']],label=f'Dropout {p:g}')
    axes[1].plot(x,[r['monitor']['yes_recall']*100 for r in g['curves']],label=f'Dropout {p:g}')
for ax in axes:
    ax.axhline(95,color='#666666',ls='--',lw=1,label='Gate: 95%')
    ax.set_ylim(0,103);ax.grid(alpha=.2);ax.legend(loc='upper right')
axes[0].axhline(100,color='#333333',ls=':',lw=1)
axes[0].set_ylabel('Monitor balanced accuracy (%)')
axes[0].set_title('One-hop QUERY generalization: both runs remain in phase 1\nTraining panel accuracy is 100% at every recorded evaluation')
axes[1].set_ylabel('Monitor YES recall (%)')
axes[1].set_xlabel('Optimizer updates (millions)')
fig.savefig(HERE/'grokking_curves_20261024.png',dpi=170)
lines=['# 2026-10-24：训练结果核查','',
       '只读核查，未停止训练、未修改判据。36个固定任务全部完成，GPU6/7仍在one_hop阶段，无失败任务。',
       '', '## Grokking快照','',
       '| 设置 | 冻结检查点步数 | 完整训练集准确率 | 监控BA | YES recall | NO recall |',
       '|---|---:|---:|---:|---:|---:|']
for row in diagnostic['results']:
    t=row['monitor']
    lines.append(f"| {row['id']} | {row['checkpoint_step']:,} | {row['full_train']['accuracy']*100:.1f}% | {t['balanced_accuracy']*100:.2f}% | {t['yes_recall']*100:.2f}% | {t['no_recall']*100:.2f}% |")
lines+=['', '所有1382条训练QUERY均正确，非仅128条面板。监控集460条：64正例、396负例，因此普通accuracy会掩盖正例失败，主看BA和两类recall。',
        '', '## 是否只是尚未达到连续三次条件？','',
        '不是。两组每次已记录评估都未达到BA95%和两类recall95%；不存在已达标却未切换的问题。',
        '无dropout组历史最佳BA66.25%（120k）；dropout0.1组最佳74.29%（200k）。此后整体下降。训练面板从20k起一直100%。',
        '', '## 知识是否学会：冻结GPU0诊断','',
        '读取当时latest的完整字节、记录SHA256，在空闲GPU0运行，不更改6/7卡。两组的131个非空后继源节点，后继集合exact match@真实出度均100%，全部320条边recall@真实出度均100%；真实后继概率质量分别约99.81%/99.85%。',
        '监控集64条正例边在WALK接口全部可恢复；同一冻结模型通过QUERY只识别9/64（无dropout）、7/64（dropout0.1）。',
        '这直接区分了知识访问与查询泛化：在测试接口和当前诊断下，边事实可提取，未见QUERY的关系判断仍失败。不能由此断言所有内部知识都只存于某个参数区域，也不能证明无限训练永远不会发生grokking。',
        '', '## 36个固定任务','',
        '| 协议 | 暴露条件 | 最终BA均值 | YES recall均值 | NO recall均值 |',
        '|---|---|---:|---:|---:|']
for row in d['fixed_summary']:
    lines.append(f"| {row['protocol']} | {row['arm']} | {row['final_ba_mean']*100:.2f}% | {row['yes_recall_mean']*100:.2f}% | {row['no_recall_mean']*100:.2f}% |")
lines+=['', '每行是3张图×2初始化的6次描述性平均，不把6次当6张独立图。最终集每模型256条，正负各128。训练面板均100%。R3明确披露完整正例witness，但没有使长查询性能接近满分；额外暴露并未自动转为QUERY可用能力。因此这些低分尚不能直接解释为纯粹的组合/搜索上限。',
        '', '图：grokking_curves_20261024.png。原始快照：results_snapshot_20261024.json；冻结诊断：transfer_diagnostic_20261024.json。',
        '', '当前建议：优先研究知识—QUERY接口为何分离；没有出现随长训练持续接近阈值的趋势。保留原训练等待后续指示，没有改动GPU6/7任务。']
(HERE/'RESULTS_20261024.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print('Saved RESULTS_20261024.md and grokking_curves_20261024.png')
