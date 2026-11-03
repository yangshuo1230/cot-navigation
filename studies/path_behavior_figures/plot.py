"""Two final figures and report from verified frozen PATH statistics."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent
DATA={s:json.loads((HERE/f'{s}.json').read_text()) for s in ('simple','hard')}

fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained',sharey=True)
for ax,(setting,d) in zip(axes,DATA.items()):
    rows=d['behavior']['by_hops'];h=[r['hops'] for r in rows]
    for key,label,color,marker in [('answer_accuracy','Final answer correct','#2675b9','o'),
                                   ('legal_goal_rate','Whole path legal + ends at goal','#d87929','s'),
                                   ('random_goal_probability','Uniform-successor random walk','#888888','.')]:
        values=np.array([r[key] for r in rows])*100
        ax.plot(h,values,marker=marker,color=color,lw=1.8,ms=5,label=label)
    ax.set_title(('Simple' if setting=='simple' else 'Hard')+f" fixed DAG / N100, M{d['manifest']['n_edges']}")
    ax.set_xticks(h,[f"{r['hops']}\nn={r['n']}" for r in rows],fontsize=8)
    ax.set_xlabel('Shortest directed query distance (hops) / test query count')
    ax.set_ylim(0,106);ax.grid(alpha=.18);ax.spines[['top','right']].set_visible(False)
    ax.legend(loc='center right',fontsize=8)
axes[0].set_ylabel('Success rate (%)')
fig.suptitle('Fixed DAG PATH: answer accuracy and valid goal-reaching path by query hops\nReachable held-out queries only; original greedy generation; no answer requirement for path score',fontsize=12)
fig.savefig(HERE/'path_accuracy_by_hops.png',dpi=200);fig.savefig(HERE/'path_accuracy_by_hops.pdf');plt.close(fig)

ticks=['E']+[f'{i}{kind}' for i in range(1,7) for kind in ('A','F')]
fig,axes=plt.subplots(2,2,figsize=(12,7.2),layout='constrained',sharex=True,sharey=True)
for col,(setting,d) in enumerate(DATA.items()):
    p=d['probabilities']['mixed_all'];first=d['probabilities']['mixed_first_step']
    top,bottom=axes[:,col]
    for key,label,color in [('successor_full_vocab','All true successors','#2675b9'),('useful_full_vocab','Goal-reaching successors','#d87929')]:
        top.plot(range(13),np.array(p[key])*100,'o-',ms=4,lw=1.8,color=color,label=label)
    top.set_title(('Simple' if setting=='simple' else 'Hard')+f" / {p['decisions']} mixed decisions, {p['queries']} queries",fontsize=11)
    top.set_ylabel('Full-vocabulary probability mass (%)')
    top.legend(loc='lower right',fontsize=8)
    bottom.plot(range(13),np.array(p['useful_given_successor'])*100,'o-',ms=4,lw=1.8,color='#d87929',label='P(useful | true successor): all prefixes')
    bottom.plot(range(13),np.array(first['useful_given_successor'])*100,'s--',ms=3.5,lw=1.4,color='#31915c',label=f"First step only (n={first['queries']})")
    bottom.axhline(100*p['uniform_useful'],color='#888888',ls=':',lw=1.5,label='Uniform-successor baseline: all prefixes')
    bottom.axhline(100*first['uniform_useful'],color='#31915c',ls=':',lw=1,alpha=.65,label='Uniform baseline: first step')
    bottom.set_ylabel('Useful probability within successors (%)')
    bottom.set_xlabel('A: after attention residual; F: after FFN residual')
    bottom.legend(loc='lower right',fontsize=7)
for ax in axes.flat:
    ax.set_xticks(range(13),ticks,fontsize=8);ax.set_ylim(-3,106)
    ax.grid(axis='y',alpha=.18);ax.spines[['top','right']].set_visible(False)
    for boundary in np.arange(2.5,12.5,2):ax.axvline(boundary,color='gray',ls=':',lw=.6,alpha=.4)
fig.suptitle('PATH: useful-successor probability across attention / FFN stages\nMixed branches in legal, pre-goal prefixes; query-balanced means; intermediate stages are uncalibrated logit lenses',fontsize=12)
fig.savefig(HERE/'path_useful_probability_layers.png',dpi=200);fig.savefig(HERE/'path_useful_probability_layers.pdf');plt.close(fig)

lines=['# Fixed DAG PATH：按跳数的行为与逐层有用后继概率','',
    '## 范围和定义','',
    '- 原 N100 简单/困难固定图 PATH 验证最佳 checkpoint，分别 step20000/10000，图边数150/200。每类只有一个图/模型，训练设置不同，不能把差异只归因于图复杂度。未分析最新限制暴露模型。',
    '- 使用原测试集自由贪心生成；跳数用独立BFS重新核对，为真实最短有向距离。跳数图仅含可达正例，因此答案准确率等于该桶YES recall。负例不可定义有限最短跳数，单独列出。',
    '- 路径成功要求从正确起点出发、整条轨迹每一步都是真实边、最后一个节点恰好等于目标；不要求随后答案正确。此前“合法前缀曾经到达目标”是更宽松的指标，本图不使用该定义。',
    '- 图中均匀随机游走从真实后继等概率选择，目标吸收，DAG上精确动态规划得出成功概率；不加额外训练。图上贪心成功比例不是温度1采样到达概率。',
    '- 有用后继为真实直接后继中仍能到达目标者，包含目标自身。概率图只取正例、自由生成的合法前缀、到达目标之前，且同时含有用与无用后继的mixed分叉；当前模型已走错的轨迹不纠正。每条查询先平均，再跨查询等权平均，避免长路径占权重。',
    '- 所有13个位置使用原final_norm和LM head计算softmax，不训练新probe。A为attention残差相加后，F为FFN残差相加后；前12个是未校准logit lens，只有L6F是模型实际下一token分布。',
    '- 完整词表概率质量：P(Succ(u))及P(Succ(u)∩ReachTo(t))；限定后继概率为两者之比。采用logsumexp避免中间层概率极小时的下溢。限定后继概率高而合法后继总质量极低时，不代表模型实际会生成有用节点。','',
    '## 按查询跳数','',
    '|图|最短跳数|n|答案准确率|整条路径合法且结束于目标|随机游走到达率|','|---|---:|---:|---:|---:|---:|']
for setting,d in DATA.items():
    for r in d['behavior']['by_hops']:
        lines.append(f"|{setting}|{r['hops']}|{r['n']}|{100*r['answer_accuracy']:.1f}%|{100*r['legal_goal_rate']:.1f}%|{100*r['random_goal_probability']:.1f}%|")
    n=d['behavior']['negative'];pos=d['behavior']['positive']
    lines.append(f"\n{setting}：可达正例n={pos['n']}，答案准确率{100*pos['answer_accuracy']:.1f}%，严格路径到达率{100*pos['legal_goal_rate']:.1f}%；不可达负例n={n['n']}，答案准确率{100*n['answer_accuracy']:.1f}%。\n")
lines+=['','高跳数桶有些只有1–3条，100%不能解释为稳定泛化；不同桶不是同难度随机重复，且原训练可能披露过完整witness。不能从曲线推出跳数越多推理越容易。','',
    '## 逐层概率：全部mixed前缀、查询等权','',
    '|图|位置|合法后继完整词表质量|有用后继完整词表质量|后继内有用概率|','|---|---|---:|---:|---:|']
for setting,d in DATA.items():
    p=d['probabilities']['mixed_all']
    for i,name in enumerate(d['layers']):
        lines.append(f"|{setting}|{name}|"+'|'.join(f'{100*p[key][i]:.1f}%' for key in ('successor_full_vocab','useful_full_vocab','useful_given_successor'))+'|')
lines+=['','首步mixed查询的对照曲线单独画出，避免全部结果只来自模型已经走对的中间前缀。E位置对后继的完整词表质量通常极低，因此只看条件概率容易误判。','',
    '这与之前报告的“后继内最高分节点是否有用”的top-1选择率不同；例如最后层接近99%的top-1选择率，不等于给有用节点分配99%的概率。也与按每次决策直接平均不同，JSON同时保留决策等权作为辅助口径。', '',
    '## 验证与复现','',
    '每个checkpoint校验SHA256；可达闭包逐起点与BFS核对，跳数再独立核对；最后层head结果与原生logits逐决策对齐，最后层各概率与上一轮PATH行为分析对齐误差小于1e-5。仅冻结CPU FP32前向，无新训练、探针、W&B或checkpoint/activation存盘。','']
for setting,d in DATA.items():lines.append(f"- {setting}：{json.dumps(d['checks'],ensure_ascii=False)}")
lines+=['','服务器复现：repo根目录执行 `.venv/bin/python studies/path_behavior_figures/run.py` 和 `plot.py`。','']
(HERE/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
print(HERE/'REPORT.md')
