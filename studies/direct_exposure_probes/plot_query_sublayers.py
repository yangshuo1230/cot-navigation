"""Plot the completed Direct attention/FFN answer-readout experiment."""
from pathlib import Path
import json, csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent
data={s:json.loads((HERE/f'{s}_query_sublayers.json').read_text()) for s in ('simple','hard')}
fig,axes=plt.subplots(1,2,figsize=(13,4.7),layout='constrained',sharey=True)
ticks=['E']+[f'{i}{stage}' for i in range(1,7) for stage in ('A','F')]
for setting,label,color in [('simple','Simple DAG, N100','#2675b9'),('hard','Hard DAG, N100','#d87929')]:
    for ax,metric,title in zip(axes,('accuracy','balanced_accuracy'),('Query accuracy','Balanced query accuracy')):
        values=np.array([m[metric]*100 for m in data[setting]['metrics']])
        other=np.array([m[metric]*100 for m in data['hard' if setting=='simple' else 'simple']['metrics']])
        ax.plot(range(13),values,'o-',ms=4,lw=2,color=color,label=label)
        if metric=='balanced_accuracy':
            for i,v in enumerate(values):
                if setting=='hard' and abs(v-other[i])<.01:continue
                ax.annotate(f'{v:.1f}',(i,v),xytext=(0,9 if v>=other[i] else -13),textcoords='offset points',ha='center',fontsize=7,color=color)
        ax.set_title(title)
for ax in axes:
    ax.set_xticks(range(13),ticks);ax.set_ylim(40,106)
    ax.set_xlabel('A: after attention residual; F: after FFN residual')
    ax.grid(axis='y',alpha=.2);ax.spines[['top','right']].set_visible(False)
    for boundary in np.arange(2.5,12.5,2):ax.axvline(boundary,color='gray',ls=':',lw=.6,alpha=.45)
    ax.legend(fontsize=9,loc='lower right')
axes[0].set_ylabel('Test score (%)');axes[1].axhline(50,color='gray',ls='--',lw=.8)
fig.suptitle('Direct: answer readout before / after each sublayer\nPrompt separator state; original query holdout; separate validation-selected linear probe at every stage',fontsize=12)
fig.savefig(HERE/'direct_query_accuracy_sublayers.png',dpi=200)
fig.savefig(HERE/'direct_query_accuracy_sublayers.pdf');plt.close(fig)
lines=['# Direct 查询答案：attention / FFN 子层读出','',
    '使用旧 N100 简单图/困难图验证最佳 Direct checkpoint，在原始 QUERY 的分隔符位置提取 E 与每层 attention/FFN 残差相加后的状态。复用原查询划分、类别平衡 ridge 与验证选阈值方法。每个位置独立拟合，不是截断模型生成，也不是子层因果贡献。没有额外节点留出。','',
    '|位置|简单图 Accuracy|简单图 BA|困难图 Accuracy|困难图 BA|','|---|---:|---:|---:|---:|']
for i in range(13):
    a=data['simple']['metrics'][i];b=data['hard']['metrics'][i]
    lines.append('|'+a['stage']+'|'+'|'.join(f'{100*v:.1f}%' for v in (a['accuracy'],a['balanced_accuracy'],b['accuracy'],b['balanced_accuracy']))+'|')
lines+=['','## 检查与解释','',
    '提取 activation 的观察 hook 对原生 logits 的差异为 0。模型前向沿用旧分析的 GPU1 BF16，probe 在 GPU0；原始标签、查询划分与 checkpoint 哈希均复用并核验。打乱标签对照、正负召回、AUC、正则和阈值保存在两个 JSON。', '',
    '同一层 attention 到 FFN 的 BA 差值，只表示重新拟合的线性答案读出变化，不是 FFN 增加了多少真实模型准确率；更不能直接解释成完成一跳推理。','',
    '与上一张图 FFN 端点的差异：']
for setting,d in data.items():
    delta=d['verification']['previous_FFN_readout_comparisons']
    maximum=max(abs(v[k]) for v in delta for k in ('accuracy_delta','balanced_accuracy_delta'))
    lines.append(f'- {setting}：Accuracy/BA 最大差异 {100*maximum:.4f} 个百分点。')
(HERE/'QUERY_SUBLAYERS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print('\n'.join(lines))
