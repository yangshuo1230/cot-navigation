"""Plot existing class-separated readouts and CoT progress, no new fitting.

Keep query-only vs both-endpoint-heldout tests separate; preserve changing
sample counts in fixed-k progress. Relative progress uses future trace length
retrospectively, not an online stopping rule. No model/data changes.
"""
from pathlib import Path
import json,csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
BEFORE={s:json.loads((ROOT/f'path_before_trace/{s}.json').read_text()) for s in ('simple','hard')}
OLD={s:json.loads((ROOT/f'fixed100_layer_probes/{s}_path.json').read_text()) for s in ('simple','hard')}
BLUE='#2675b9';ORANGE='#d87929';GREEN='#31915c';GRAY='#888888'
LINEAR=[('prompt_separator','Prompt\nseparator'),('prompt_target','Prompt\ntarget'),('prompt_joint','Prompt\njoint'),
        ('endpoint_embeddings','Input\nembedding'),('endpoint_onehot','Input\none-hot'),('copied_source_separator','Source\ncopied'),('final','Final\nmodel')]
MLP=[('prompt_separator','Prompt\nseparator'),('prompt_joint','Prompt\njoint'),('endpoint_embeddings','Input\nembedding'),('random','Random\nendpoints')]

fig,axes=plt.subplots(3,2,figsize=(13,11.4),layout='constrained',gridspec_kw={'height_ratios':[1,1,.95]})
class_rows=[]
for col,(setting,d) in enumerate(BEFORE.items()):
    title='Simple' if setting=='simple' else 'Hard';sample=d['features']['prompt_separator']['metrics'][-1]
    counts=(sample['n_positive'],sample['n']-sample['n_positive'])
    ax=axes[0,col];x=np.arange(len(LINEAR))
    for delta,key,label,color in [(-.18,'yes_recall','Reachable / positive',BLUE),(.18,'no_recall','Unreachable / negative',ORANGE)]:
        values=[]
        for name,display in LINEAR:
            m=d['native_final_answer'] if name=='final' else d['features'][name]['metrics'][-1]
            values.append(m[key]*100)
            class_rows.append(dict(graph=setting,experiment='linear_query_holdout',feature=name,**{key:m[key]}))
        bars=ax.bar(x+delta,values,.35,color=color,label=label)
        ax.bar_label(bars,fmt='%.1f',fontsize=7,padding=2)
    ax.set_title(f'{title} N100 / linear query holdout\nn+={counts[0]}, n-={counts[1]}',fontsize=11)
    ax.set_xticks(x,[display for name,display in LINEAR],fontsize=8);ax.legend(fontsize=8,loc='lower left')
    ax=axes[1,col];x=np.arange(len(MLP))
    for delta,key,label,color in [(-.18,'yes_recall','Reachable / positive',BLUE),(.18,'no_recall','Unreachable / negative',ORANGE)]:
        means=[];sd=[]
        for name,display in MLP:
            rows=[v for k,items in d['nonlinear_readouts'].items() if k.startswith('random_endpoints') for v in items] if name=='random' else d['nonlinear_readouts'][name]
            a=np.array([v['test'][key]*100 for v in rows]);means.append(a.mean());sd.append(a.std())
        bars=ax.bar(x+delta,means,.35,color=color,yerr=sd,capsize=3,label=label)
        ax.bar_label(bars,fmt='%.1f',fontsize=8,padding=7)
    ax.set_title(f'{title} / small MLP, query holdout only\nMean; error bars = initialization / random-vector SD',fontsize=10)
    ax.set_xticks(x,[display for name,display in MLP],fontsize=9)
    for row in (0,1):
        a=axes[row,col];a.set_ylim(0,113);a.set_yticks(range(0,101,20));a.set_ylabel('Class accuracy (%)')
        a.grid(axis='y',alpha=.15);a.set_axisbelow(True);a.spines[['top','right']].set_visible(False)
    # Separate node-heldout sensitivity panel, with class-specific support.
    ax=axes[2,col];entries=[e for e in d['node_heldout'] if 'skipped' not in e]
    matrix=[];labels=[]
    for entry in entries:
        a=[]
        for name in ('prompt_separator','prompt_joint','copied_source_separator'):
            m=entry['features'][name]['metrics'][-1];a.extend([m['yes_recall']*100,m['no_recall']*100])
        matrix.append(a);n=entry['counts'][-1];pos=entry['positive_counts'][-1]
        labels.append(f"seed {entry['seed']}\nn+={pos}, n-={n-pos}")
    matrix=np.array(matrix);im=ax.imshow(matrix,vmin=0,vmax=100,cmap='Blues',aspect='auto')
    for r in range(len(entries)):
        for c in range(6):ax.text(c,r,f'{matrix[r,c]:.1f}',ha='center',va='center',fontsize=10,color='white' if matrix[r,c]>=60 else 'black')
    ax.set_xticks(range(6),['Sep +','Sep -','Joint +','Joint -','Copy +','Copy -'],fontsize=9)
    ax.set_yticks(range(len(labels)),labels,fontsize=9)
    ax.set_title(f'{title} / linear query + BOTH-endpoint node holdout',fontsize=10)
    for spine in ax.spines.values():spine.set_visible(False)
fig.suptitle('PATH: positive and negative answer-readout accuracy\nPrompt = before any trajectory token; Copy = first copied source only; node-heldout tests are separate',fontsize=13)
fig.savefig(HERE/'path_class_accuracy.png',dpi=190);fig.savefig(HERE/'path_class_accuracy.pdf');plt.close(fig)

FIXED=['prompt','k1','k2','k4','k8'];RELATIVE=['prompt','p25','p50','p75','end']
fig,axes=plt.subplots(2,2,figsize=(12.5,8.1),layout='constrained',sharey=True)
progress=[]
for col,(setting,d) in enumerate(OLD.items()):
    answers=d['answers']['linear_readout'];title='Simple' if setting=='simple' else 'Hard'
    for row,phases in enumerate((FIXED,RELATIVE)):
        ax=axes[row,col];values=[answers[phase]['residual']['layers'][-1]['all'] for phase in phases]
        controls=[answers[phase]['position_goal_control']['layers'][-1]['all'] for phase in phases]
        for key,label,color,style in [('yes_recall','Positive accuracy',BLUE,'-'),('no_recall','Negative accuracy',ORANGE,'-'),('balanced_accuracy','Balanced accuracy',GREEN,'--')]:
            ys=[100*m[key] for m in values];ax.plot(range(5),ys,'o',ls=style,color=color,lw=1.8,ms=4,label=label)
        ax.plot(range(5),[100*m['balanced_accuracy'] for m in controls],':',color=GRAY,lw=1.6,label='Position / goal-visible control BA')
        if row==0:
            labels=[]
            for phase,m in zip(phases,values):
                k='0' if phase=='prompt' else phase[1:]
                labels.append(f"{k}\n+{m['n_positive']}/-{m['n']-m['n_positive']}")
            ax.set_xticks(range(5),labels,fontsize=9)
            ax.set_xlabel('Generated trace tokens k / positive and negative sample counts')
            ax.set_title(title+' / fixed token counts (changing query cohorts)',fontsize=11)
        else:
            ax.set_xticks(range(5),['Prompt','25%','50%','75%','End'])
            ax.set_xlabel('Retrospective fraction of final generated trace length')
            m=values[0]
            assert all((v['n'],v['n_positive'])==(m['n'],m['n_positive']) for v in values)
            ax.set_title(f"{title} / relative progress (same queries, n+={m['n_positive']}, n-={m['n']-m['n_positive']})",fontsize=10)
        for phase,m,control in zip(phases,values,controls):
            progress.append(dict(graph=setting,panel='fixed' if row==0 else 'relative',phase=phase,**m,control_BA=control['balanced_accuracy']))
        ax.set_ylim(40,104);ax.set_ylabel('Test accuracy / BA (%)');ax.grid(alpha=.16)
        ax.spines[['top','right']].set_visible(False);ax.legend(fontsize=8,loc='lower right')
fig.suptitle('PATH: answer decoding as the original CoT is generated\nL6 frozen-state linear probes, fitted separately per anchor; all prefixes exclude the future YES/NO token',fontsize=12)
fig.savefig(HERE/'path_accuracy_cot_progress.png',dpi=190);fig.savefig(HERE/'path_accuracy_cot_progress.pdf');plt.close(fig)

with (HERE/'progress.csv').open('w',newline='',encoding='utf-8') as stream:
    writer=csv.DictWriter(stream,fieldnames=list(progress[0]));writer.writeheader();writer.writerows(progress)
lines=['# PATH 正负例读出与 CoT 进度图','',
    '本次仅根据已完成的实验结果画图，没有新训练或新采样。图一来自path_before_trace的CPU FP32复测；图二来自fixed100_layer_probes原BF16阶段读出，两者prompt分数有约0.2–0.5个百分点数值/阈值差异，不混拼同一条曲线。','',
    '## 图一：正负例分开','',
    '- 上排：原查询留出、节点可重复的线性探针。Prompt为零轨迹token，Copy只复制起点，尚未沿边导航；Final是完整自由生成后的原模型答案，不是probe。',
    '- 中排：小MLP查询留出实验；三次初始化均值和标准差，随机输入还包含三个向量seed。不具有MLP节点留出泛化证据，也不是完全同参数量比较（输入维数不同）。',
    '- 下排：双端点节点60/20/20留出的线性探针，query split也留出。每折样本很少；每行给出正/负样本数，不能与上两排当同一测试分布。+为可达、-为不可达，数值为各类召回。',
    '- 主模型始终见过全部节点；留出仅针对外部probe训练。','',
    '## 图二：答案随CoT进度的可读出性','',
    '- 这里画的是冻结L6状态上额外线性探针的答案读出，不是原模型在每一步实际输出YES/NO的准确率。每个锚点独立拟合，原query train/val/test划分，val选正则与阈值，没有节点留出。',
    '- 上排k=0/1/2/4/8：仅含模型原轨迹至少有k个token的查询，样本池随k变化。k=1是复制起点，还没有走一条边。横轴按所取锚点等间隔展示，不表示相等token时间间隔。',
    '- 下排：固定相同测试查询，取ceil(最终轨迹长度×25/50/75%)及末尾token，prompt为0。百分比依赖最终长度，仅用于回顾性分析，不是在线停止规则；短轨迹多个锚点可能落在同一token。',
    '- 所有前缀都是原模型自由生成，不替换成正确参考路径、不修正错误；不包含未来YES/NO答案token。',
    '- 灰线是位置与目标可见性控制的BA（生成token数、绝对位置、目标是否出现、当前是否目标），不是答案探针；它在结尾也可接近99%，因此结尾高分不能单独证明推理过程。',
    '- 这两图仅旧N100简单/困难图PATH各一个验证最佳checkpoint，不包括N160限制暴露模型。','',
    '## 原进度数据','',
    '|图|锚点|正例n|负例n|正例准确率|负例准确率|BA|位置/目标控制BA|','|---|---|---:|---:|---:|---:|---:|---:|']
for setting in OLD:
    for phase in FIXED+RELATIVE[1:]:
        r=OLD[setting]['answers']['linear_readout'][phase];m=r['residual']['layers'][-1]['all'];c=r['position_goal_control']['layers'][-1]['all']
        lines.append(f"|{setting}|{phase}|{m['n_positive']}|{m['n']-m['n_positive']}|{100*m['yes_recall']:.1f}%|{100*m['no_recall']:.1f}%|{100*m['balanced_accuracy']:.1f}%|{100*c['balanced_accuracy']:.1f}%|")
lines+=['','来源：原始每查询生成与checkpoint manifest见fixed100_layer_probes；class拆分见path_before_trace。复现plot.py即可，不需要加载模型。','']
(HERE/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
print(HERE/'REPORT.md')
