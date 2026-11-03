"""One figure and a report separating decodability from already-computed answers."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent
DATA={s:json.loads((HERE/f'{s}.json').read_text()) for s in ('simple','hard')}
FEATURES=[('prompt_separator','Before trace: separator','#2675b9','-'),
          ('prompt_target','Before trace: target','#d87929','-'),
          ('prompt_joint','Before trace: source + target + separator','#31915c','-'),
          ('copied_source_separator','After source copy: no edge traversed','#9467bd','--')]
ticks=['E']+[f'{i}{stage}' for i in range(1,7) for stage in ('A','F')]
fig,axes=plt.subplots(1,2,figsize=(12.5,4.8),layout='constrained',sharey=True)
for ax,(setting,d) in zip(axes,DATA.items()):
    for name,label,color,style in FEATURES:
        values=[m['balanced_accuracy']*100 for m in d['features'][name]['metrics']]
        ax.plot(range(13),values,marker='o',ms=3,lw=1.6,color=color,ls=style,label=label)
    ax.axhline(100*d['features']['endpoint_onehot']['metrics'][0]['balanced_accuracy'],color='gray',ls=':',lw=1.5,label='Input endpoint one-hot baseline')
    ax.set_title(('Simple' if setting=='simple' else 'Hard')+' fixed DAG / PATH N100')
    ax.set_xticks(range(13),ticks,fontsize=8);ax.set_ylim(40,102);ax.grid(alpha=.18)
    ax.set_xlabel('A: after attention residual; F: after FFN residual')
    ax.spines[['top','right']].set_visible(False);ax.legend(fontsize=7,loc='lower right')
axes[0].set_ylabel('Test balanced accuracy (%)')
fig.suptitle('Reachability readout before PATH generates any trajectory\nFrozen model; original query holdout; separate validation-selected affine decoder for each state',fontsize=12)
fig.savefig(HERE/'path_before_trace.png',dpi=200);fig.savefig(HERE/'path_before_trace.pdf');plt.close(fig)

lines=['# PATH 在生成轨迹前是否已经知道可达性？','',
    '结论：生成前状态包含可达性相关信息，但现有证据不支持“模型已经基本算出最终答案，PATH只是在复述”。分隔符的线性/小MLP读出都只有约74%/72%；将输入端点及多个位置状态交给额外解码器可取得高分，但固定随机节点身份向量也能支持高分，不能把解码器新增的关系计算归给原模型。','',
    '## 范围与方法','',
    '- 原N100简单/困难固定图PATH验证最佳checkpoint，SHA256复核。不是最新N160限制暴露模型。',
    '- 真正生成前：输入仅WORLD（若有）、QUERY、起点、目标、分隔符，不含轨迹、答案或其他提示；在分隔符、目标以及起点/目标/分隔符拼接上取E和每层A/F状态。',
    '- 对照阶段只增加自然第一个输出——起点的复制。全部4500个查询的原生第一token均为起点，尚未沿任何边移动；它有额外一次自回归计算，不算“零轨迹token”。',
    '- 类别平衡仿射ridge，每位置独立拟合；仅query-train拟合，query-val的AUROC选正则、BA选阈值，原test报告。固定高斯节点向量三seed，与训练embedding端点拼接同维。one-hot输入只能实现起点偏置+目标偏置，不能直接表示配对交互。',
    '- 主指标为balanced accuracy，不能把多数类准确率当能力。模型最终实际答案另按同一BA口径汇总。',
    '- 同时做probe双端点节点留出60/20/20、与原query split取交集，三个seed；原主模型训练仍见过全部节点。此严格面板每折test只有18–40条。',
    '- 补充小MLP：32隐藏GELU层、三个初始化，train-only标准化、平衡BCE，验证AUROC早停和验证BA阈值。仅query留出，未做MLP节点留出；输入维数128/256/384不同，因此是信息访问诊断，不是完全同参数量对照。', '',
    '## 生成前与复制起点后：最后一层线性读出','',
    '|输入/状态|简单图BA|困难图BA|','|---|---:|---:|']
for name,label in [('prompt_separator','生成前：分隔符'),('prompt_target','生成前：目标'),('prompt_joint','生成前：起点+目标+分隔符'),
                    ('endpoint_embeddings','输入：起终点训练embedding'),('endpoint_onehot','输入：起终点one-hot'),('copied_source_separator','仅复制起点后')]:
    values=[d['features'][name]['metrics'][-1]['balanced_accuracy'] for d in DATA.values()]
    lines.append('|'+label+'|'+'|'.join(f'{100*v:.1f}%' for v in values)+'|')
values=[(d['native_final_answer']['yes_recall']+d['native_final_answer']['no_recall'])/2 for d in DATA.values()]
lines.append('|原模型完整自由生成最终答案|'+'|'.join(f'{100*v:.1f}%' for v in values)+'|')
lines+=['','生成前分隔符的正例/负例召回：']
for setting,d in DATA.items():
    m=d['features']['prompt_separator']['metrics'][-1]
    lines.append(f"- {setting}：YES recall {100*m['yes_recall']:.1f}%，NO recall {100*m['no_recall']:.1f}%，AUROC {m['auc']:.3f}。")
lines+=['','不能把分隔符线性读出较低直接解释为没有答案信息。原输出头此处的正常任务是复制起点，不是回答YES/NO；各层独立探针分数也不是原模型截断后的任务准确率。', '',
    '## 小型非线性解码器','',
    '|信息来源|简单图BA均值±初始化SD|困难图BA均值±初始化SD|','|---|---:|---:|']
for name,label in [('prompt_separator','生成前分隔符'),('prompt_joint','生成前起点+目标+分隔符'),('endpoint_embeddings','仅输入端点embedding'),('random','仅固定随机端点向量')]:
    values=[]
    for d in DATA.values():
        rows=[v for key,items in d['nonlinear_readouts'].items() if key.startswith('random_endpoints') for v in items] if name=='random' else d['nonlinear_readouts'][name]
        a=np.array([v['test']['balanced_accuracy'] for v in rows]);values.append(f'{100*a.mean():.1f}% ± {100*a.std():.1f}pp')
    lines.append('|'+label+'|'+'|'.join(values)+'|')
lines+=['','随机向量结果说明：主模型的图编码不是取得这类query-holdout高分的必要条件，额外训练的非线性解码器能从已见节点身份学习固定图查询规律。它不是原模型在生成前已经做完关系判断的证据。MLP只看query留出，节点身份可重复，初始化SD不是置信区间。','',
    '## 全部四问都在test的交叉关系对照','',
    '寻找同两起点、两目标构成的四问，标签满足对角相同、非对角相反。只保留四问均属于原test的面板，图标签选取，不看预测。端点可达频率的加性分类器无法四问全对。配对共享查询，不能按独立样本推断。','',
    '|图|四问组数|独立查询|生成前分隔符四问全对|生成前联合状态四问全对|复制起点后四问全对|','|---|---:|---:|---:|---:|---:|']
for setting,d in DATA.items():
    panel=d['xor_panel'];values=[d['features'][name]['xor_all_four_correct'][-1] for name in ('prompt_separator','prompt_joint','copied_source_separator')]
    lines.append(f"|{setting}|{panel['squares']}|{panel['unique_queries']}|"+'|'.join(f'{100*v:.1f}%' for v in values)+'|')
lines+=['','这个表仅针对线性解码器；困难图只有7组，不能作稳健统计结论。','',
    '## 双端点留出线性探针对照','',
    '|图/seed|train/val/test数|生成前分隔符BA|生成前联合状态BA|输入embedding BA|随机端点BA|复制起点后BA|','|---|---|---:|---:|---:|---:|---:|']
for setting,d in DATA.items():
    for record in d['node_heldout']:
        if 'skipped' in record:continue
        names=('prompt_separator','prompt_joint','endpoint_embeddings','random_endpoints_201','copied_source_separator')
        values=[record['features'][name]['metrics'][-1]['balanced_accuracy'] for name in names]
        lines.append(f"|{setting}/{record['seed']}|{'/'.join(map(str,record['counts']))}|"+'|'.join(f'{100*v:.1f}%' for v in values)+'|')
lines+=['','严格面板支持量较少，val只有18–34条，阈值选择及不同节点分布会导致波动；主模型见过所有节点，不能称作未见图或主模型未见节点泛化。','',
    '## 原输出头与核验','']
for setting,d in DATA.items():
    raw=d['native_prompt'][-1]
    lines.append(f"- {setting}：生成前最后层实际argmax为YES/NO的比例{100*raw['native_argmax_is_yesno']:.1f}%，YES+NO完整词表质量{100*raw['full_vocab_yesno_mass']:.6f}%；强制只在YES/NO之间比较的BA为{100*raw['forced_yesno']['balanced_accuracy']:.1f}%。这不是模型训练过的直接作答接口，不能作为信息不存在的证据。")
    lines.append(f"- 核验：{json.dumps(d['checks'],ensure_ascii=False)}")
lines+=['','最后层输出头与原生logits一致，复制未来token前后的prompt状态满足因果不变性；重用原checkpoint、原test与旧probe方法，FP32与旧BF16分数小幅差异记录在JSON。', '',
    '冻结CPU前向，只有外部probe训练使用GPU0。未停止其他训练，未保存新模型/探针checkpoint或activation，不上传W&B。复现run.py、nonlinear.py、summarize.py；后两者从已有最终结果继续。','']
(HERE/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
print(HERE/'REPORT.md')
