"""One key figure: role maps, distance-centered maps, phase target curves."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent
DATA={s:json.loads((HERE/f'{s}.json').read_text()) for s in ('simple','hard')}
PHASES=[('source_copy','Source copied','#2675b9'),('middle','Middle','#d87929'),('last_before_goal','Last node before goal','#31915c')]
roles=['Syntax','Q source','Q target','Separator','History','Current']
fig,axes=plt.subplots(3,2,figsize=(12,11.5),layout='constrained')
for col,(setting,d) in enumerate(DATA.items()):
    group=d['groups']['mixed_all'];raw=np.array(group['raw_query_balanced'])*100;centered=np.array(group['corrected_query_balanced'])
    for row,values,cmap,vmin,vmax in [(0,raw,'Blues',0,100),(1,centered,'RdBu_r',-.9,.9)]:
        ax=axes[row,col];im=ax.imshow(values,cmap=cmap,vmin=vmin,vmax=vmax,aspect='auto')
        for i in range(6):
            for j in range(6):
                value=values[i,j];label=f'{value:.1f}' if row==0 else f'{value:+.2f}'
                ax.text(j,i,label,ha='center',va='center',fontsize=9,color='white' if (value>55 if row==0 else abs(value)>.5) else 'black')
        ax.set_xticks(range(6),roles,rotation=20,ha='right',fontsize=9);ax.set_yticks(range(6),[f'L{i}' for i in range(1,7)])
        ax.set_title(setting.capitalize()+(' / raw attention (%)' if row==0 else ' / distance-centered attention'),fontsize=12)
        fig.colorbar(im,ax=ax,fraction=.04,pad=.02)
    ax=axes[2,col]
    for phase,label,color in PHASES:
        group=d['groups'][phase];values=np.array(group['raw_query_balanced'])[:,2]*100
        ax.plot(range(1,7),values,'o-',color=color,label=f"{label} (n={group['queries']})",lw=1.8,ms=5)
    ax.set_xticks(range(1,7));ax.set_ylim(-2,104);ax.set_xlabel('Attention layer');ax.set_ylabel('Attention to query target (%)')
    ax.set_title(setting.capitalize()+' / target attention across trajectory phases',fontsize=11)
    ax.grid(alpha=.16);ax.spines[['top','right']].set_visible(False);ax.legend(fontsize=8,loc='lower right')
fig.suptitle('Fixed DAG PATH: attention while predicting the next path node\nRole maps: mixed branches, query-balanced; phase curves: legal successful trajectories only',fontsize=13)
fig.savefig(HERE/'path_attention_patterns.png',dpi=190);fig.savefig(HERE/'path_attention_patterns.pdf');plt.close(fig)

lines=['# PATH 生成过程的 attention pattern','',
    '结果：两张旧N100固定图PATH均显示目标条件读取，但层位置不同。简单图混合分叉的L2/L3几乎只读查询目标；困难图L1主要self、L2主要分隔符、L4/L6主要目标，L5在源节点复制与路径中段之间切换读取位置。描述性距离校正后目标峰值仍然存在。','',
    '## 范围、接收位置与校正','',
    '- 旧N100简单/困难图PATH验证best，未分析N160限制暴露模型；checkpoint SHA256沿用并核验。每层1个attention head，因此无需跨head平均。',
    '- 接收者是已生成的当前路径节点token，正在预测下一个节点；不是观察目标token自身的注意力，也不包含最终YES/NO。',
    '- 六类key位置：WORLD/QUERY等语法、查询起点、查询目标、分隔符、较早路径token、当前token自身。按位置分组，不是按token身份，复制起点和查询起点虽然ID相同仍属不同位置。',
    '- 主role图只取原test正例合法前缀、尚未到目标、同时存在有用/无用后继的mixed分叉；简单1171次决策/376条查询、困难1807次/298条。各查询内部先平均，再对查询等权；不是对长轨迹的所有决策直接平均。',
    '- 距离baseline仅使用原train查询的真实因果token对，不含padding和未来YES/NO。按层及相对距离d取平均mu_l(d)，校正值为A_l(i,j)-mu_l(i-j)，再按位置类别求和。校正值有正负、不必和为1，也不是概率；不能保证消除了全部位置混杂。',
    '- 轨迹阶段：成功路径的第一节点（复制起点）、中间节点、到目标前的最后一个节点。每条查询每阶段取一个位置，简单374、困难296；中间/末尾是依赖最终轨迹的回顾性锚点，短路径的锚点可能重复。它们不是所有查询、失败查询的阶段统计。',
    '- 冻结CPU FP32原生SDPA。由实际norm1输入重建QK+RoPE softmax，验证权重与V相乘再投影的输出对齐原生attention delta；没有启用会改变计算路径的return_attn模式。无训练和activation存盘。','',
    '## mixed分叉：原始注意力质量','',
    '|图|层|语法|查询起点|查询目标|分隔符|历史路径|当前自身|','|---|---|---:|---:|---:|---:|---:|---:|']
for setting,d in DATA.items():
    raw=np.array(d['groups']['mixed_all']['raw_query_balanced'])
    for layer,values in enumerate(raw,1):lines.append(f'|{setting}|L{layer}|'+'|'.join(f'{100*v:.1f}%' for v in values)+'|')
lines+=['','## mixed分叉：目标注意力的距离校正值','',
    '|图|L1|L2|L3|L4|L5|L6|','|---|---:|---:|---:|---:|---:|---:|']
for setting,d in DATA.items():
    values=np.array(d['groups']['mixed_all']['corrected_query_balanced'])[:,2]
    lines.append('|'+setting+'|'+'|'.join(f'{v:+.3f}' for v in values)+'|')
lines+=['','困难图L1/L5虽然原始self质量很高，但相对距离0的训练均值也很高，校正后当前自身只有约+0.01/+0.02；不能把raw self峰值全部视为任务特异性路由。简单图L6历史质量36.2%，校正后约-0.08，也不能只凭raw总质量声称超常关注整段历史。','',
    '## 随生成阶段变化：目标读取','',
    '|图/阶段|L1|L2|L3|L4|L5|L6|','|---|---:|---:|---:|---:|---:|---:|']
for setting,d in DATA.items():
    for phase,label,color in PHASES:
        values=np.array(d['groups'][phase]['raw_query_balanced'])[:,2]
        lines.append(f'|{setting}/{phase}|'+'|'.join(f'{100*v:.1f}%' for v in values)+'|')
lines+=['','困难图L5在复制起点时，查询起点/当前自身注意力分别66.5%/32.3%；路径中段为3.4%/93.6%，到目标前为1.3%/95.8%。这是读取位置变化，不能据此断言它在刷新哪种特定关系表。',
    '简单图L6读取历史路径的质量在源节点复制时为0，中段44.2%、目标前41.8%；困难图对应为0、2.1%、4.1%。不同阶段历史token数量也不同，主图同时提供距离校正作为描述性对照。','',
    '## 如何理解信息流','',
    '简单图与“前层形成当前节点相关表示，L2/L3从目标位置读取条件信息，再逐步影响下一节点排序”相容。困难图更像分阶段从当前节点、查询汇总位置和目标位置搬运信息。',
    '困难图L2关注分隔符97.5%，不意味着读取无意义标点：该位置在上一层可能已整合查询信息。读取目标也不自动等于读取祖先集合。低历史权重不排除较早信息经value/残差间接传递，或图知识存在参数中。',
    '当前节点低self attention也不表示忽略当前节点，因为当前embedding/此前状态一直通过residual skip保留。反过来高self attention不等于新做了图搜索。',
    '这些模式与当前位置+目标条件的局部导航策略相容，但不能单独区分路径检索、片段拼接或通用搜索，也不构成某条attention边的因果必要性证明。此前FFN/attention分支消融提供的是额外证据，未在本次重新执行。','',
    '## 核验与文件','']
for setting,d in DATA.items():lines.append(f"- {setting}：重建attention value输出与原生delta最大差异{d['checks']['reconstructed_attention_value_delta_max_error']:.6g}；mixed查询/决策支持量与前次概率实验一致。")
lines+=['','文件：simple.json/hard.json含query等权和决策等权两种聚合、全部阶段原始/校正矩阵；唯一关键图path_attention_patterns.png/.pdf。run.py和summarize.py可在repo根目录复现。','']
(HERE/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
print(HERE/'REPORT.md')
