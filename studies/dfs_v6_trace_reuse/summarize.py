"""One key figure and compact report; retain every seed in the source JSON."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

root=Path(__file__).resolve().parent
d=json.loads((root/'results.json').read_text())
assert len(d['seeds'])==4 and d['audit_version']==2
names=['short_test','long_AA_test','long_TA_test','long_AT_test','long_TT_test']
labels=['短程','长程 A→A','长程 T→A','长程 A→T','长程 T→T']
summary={}
for name in names:
    p=[s['panels'][name] for s in d['seeds']]
    out={}
    for group in ['generated','legal_generated','reference']:
        out[group]={}
        for key in p[0][group]:
            values=[v[group][key] for v in p if key in v[group]]
            if values:out[group][key]=float(np.mean(values))
    out['legal_counts_by_seed']=[int(v['legal_generated']['n']) for v in p]
    out['native_count_differences']=[v['original_count_differences'] for v in p]
    summary[name]=out
(root/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')

font=Path('C:/Windows/Fonts/msyh.ttc')
if font.exists():
    font_manager.fontManager.addfont(str(font))
    plt.rcParams['font.family']=font_manager.FontProperties(fname=str(font)).get_name()
plt.rcParams.update({'font.size':10,'axes.unicode_minus':False,'pdf.fonttype':42})
fig,axs=plt.subplots(1,2,figsize=(13,6.3))
fig.subplots_adjust(left=.07,right=.98,top=.72,bottom=.31,wspace=.2)
x=np.arange(5);w=.36
panels=[('整条节点序列与训练轨迹完全相同',
         [(lambda a:a['generated']['exact'],'模型输出','#2878b5'),
          (lambda a:a['reference']['exact'],'合法随机参考','#9aa8b4')]),
        ('仅看完整合法的模型输出',
         [(lambda a:1-a['legal_generated']['whole_subsequence'],'不能由单条训练轨迹删减得到','#e59b43'),
          (lambda a:a['legal_generated']['novel_state_and_not_subsequence'],'且出现未见栈＋visited 组合','#327d70')])]
for ax,(title,series) in zip(axs,panels):
    for offset,(fn,label,color) in zip([-w/2,w/2],series):
        vals=[100*fn(summary[n]) for n in names]
        ax.bar(x+offset,vals,w*.95,color=color,label=label)
        for xx,yy in zip(x+offset,vals):ax.text(xx,yy+1.3,f'{yy:.1f}',ha='center',fontsize=9)
    ax.set_title(title,pad=15,fontsize=12)
    ax.set_xticks(x,labels,rotation=15);ax.set_ylim(0,105)
    ax.set_yticks([0,25,50,75,100]);ax.set_axisbelow(True);ax.grid(axis='y',alpha=.15)
    ax.spines[['top','right']].set_visible(False)
    ax.legend(loc='upper right',frameon=False,fontsize=9)
axs[0].set_ylabel('百分比（4个初始化种子均值）')
fig.text(.04,.95,'DFS nodes：能否用单条训练轨迹的复用解释？',fontsize=19,weight='bold',color='#17324d')
fig.text(.04,.87,'v6 · N80 固定图 · final 50k · 仅比较实际曝光轨迹 · 每seed 8,910条去重训练节点序列',fontsize=11,color='#526171')
fig.text(.04,.12,'右图只统计完整合法轨迹，避免把非法输出的新颖性当成能力。训练状态库含隐式回退及终止状态。\n'
         '“不能删减得到”排除单条训练轨迹的任意子序列匹配；仍不排除多片段拼接、局部记忆或因子化检索。',fontsize=10,color='#526171',linespacing=1.7)
fig.text(.04,.025,'本次重新自由生成，BF16与历史汇总有少量差异；参考轨迹每query 8种合法顺序。A/T为长程端点角色。',fontsize=9,color='#697783')
for ext in ['png','pdf']:fig.savefig(root/f'trace_reuse.{ext}',dpi=200,facecolor='white')
plt.close(fig)

lines=['# v6 DFS nodes：实际训练轨迹复用审计','',
       '范围：N80 graph0、4初始化、固定50k；全部5个测试面板共602查询/seed，2408条原生自由生成。',
       '完整合法性用真实图独立重放；生成不使用图约束/修复；冻结推理GPU0，无训练、checkpoint或activation文件。',
       '每seed实际曝光12064条query/variant，去重后8910条训练节点序列。QUERY文本不参与匹配，以免换目标就被误计为新轨迹。',
       '所有训练变体以exposure_counts>0筛选；不把未实际抽到的变体加入训练库。','',
       '|面板|生成整条完全匹配|参考整条完全匹配|合法输出数，四seed|合法输出非单条删减|合法输出有新状态|同时非单条删减＋新状态|',
       '|---|---:|---:|---|---:|---:|---:|']
for name in names:
    a=summary[name];g=a['generated'];l=a['legal_generated'];r=a['reference']
    lines.append(f"|{name}|{g['exact']:.1%}|{r['exact']:.1%}|{a['legal_counts_by_seed']}|{1-l['whole_subsequence']:.1%}|{l['has_novel_state']:.1%}|{l['novel_state_and_not_subsequence']:.1%}|")
lines+=['','## 连续片段与局部复用（全部生成）','',
        '|面板|整条是训练前缀|整条是某训练连续片段|整条是某训练任意子序列|8-token窗口已见比例|16-token窗口已见比例|',
        '|---|---:|---:|---:|---:|---:|']
for name in names:
    g=summary[name]['generated']
    lines.append('|'+name+'|'+'|'.join(f'{g[k]:.1%}' for k in ['whole_prefix','whole_contiguous','whole_subsequence','window8_coverage','window16_coverage'])+'|')
lines+=['','## 解释边界','',
        '- 完整合法、但不是任何单条训练轨迹的子序列，是反对“只挑一条旧轨迹再截短/删节点”的反例。',
        '- 新状态以有序栈＋visited集合定义，忽略visited打印顺序及query目标。训练状态库包含所有可重建的隐式POP与终止状态，避免把已见的回退/终止组合误称为新状态。测试只看第3次展开及以后、未到目标的状态。',
        '- 新状态只排除对完整状态的精确查表，仍允许对子栈、节点、片段或状态因子的记忆和组合。',
        '- 合法参考也会与训练匹配；仅凭“出现已见轨迹”不能归因为模型在抄训练集。参考与模型长度/分支偏好不同，匹配率差异不作因果解释。',
        '- 这不是跨图泛化证据，也没有撤销此前发现的100%测试正例发现树关系语义覆盖。',
        '- 所有比例为4初始化宏平均；参考每查询8变体。合法输出统计条件于本次生成成功，不能作为全部查询成功率。',
        '- 原生BF16重放与final.json有少量计数差异。完整差异逐seed逐面板保存在results.json；报告依据本次实际轨迹，不声称逐token复现。','',
        '## 可检查的长程反例','']
for s in d['seeds']:
    examples=[e for e in s['examples'] if e['panel']=='long_TT_test']
    if not examples:continue
    e=examples[0]
    lines += [f"### seed{s['seed']}，TT query{e['query']}，{e['source']} → {e['target']}，label={e['label']}",
              '', '完整合法，且不是任何单条实际训练节点轨迹的任意子序列。',
              '', '生成：`'+' '.join(e['trace'])+'`',
              '', '未见状态的stack：`'+' '.join(e['novel_stack'])+'`',
              '', '未见状态的visited集合：`'+' '.join(e['novel_visited'])+'`','']
(root/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print('\n'.join(lines[:16]))
