"""Summarize necessity and natural interchange without overclaiming graph storage."""
import json
from pathlib import Path
import statistics

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
ds = [json.loads((HERE / f'seed{s}.json').read_text()) for s in range(4)]
def avg(values):
    values = np.asarray(values, dtype=float)
    return {'mean': values.mean(0).tolist(), 'min': values.min(0).tolist(), 'max': values.max(0).tolist()}

summary = {'baseline': {key: avg([d['baseline'][key] for d in ds])
                       for key in ('legal_next', 'useful_next', 'uniform_baseline')},
           'ablations': {}, 'patches': {}, 'pair_balanced_ablations': {}}
for name in ds[0]['ablations']:
    summary['ablations'][name] = {key: avg([d['ablations'][name]['all'][key] for d in ds])
                                  for key in ('legal_next', 'useful_next')}
for role in ds[0]['patches']:
    summary['patches'][role] = {'stages': [p['stage'] for p in ds[0]['patches'][role]],
        **{key: avg([[p[key] for p in d['patches'][role]] for d in ds]) for key in ('donor_useful', 'recipient_useful', 'legal')}}
for name in ds[0]['pair_balanced_ablations']:
    summary['pair_balanced_ablations'][name] = {key: avg([d['pair_balanced_ablations'][name][key] for d in ds])
                                               for key in ('recipient_useful', 'legal')}
(HERE / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')

lines = ['# PATH v5：下一步选边计算的因果检查', '',
    '本轮用已有4个50k PATH checkpoint，不重新训练。范围是一张80节点图、原测试查询自然生成中的合法前缀，且当前位置距目标至少2跳、有用和无用后继同时存在。观察下一token，不是完整查询/轨迹的闭环成功率。', '',
    '## 当前可支持的回路线索', '',
    '当前位置第1层FFN是生成合法后继的关键节点变换；目标信息经attention（尤其第2层，并在第3–4层继续整合）进入当前位置残差，后续层将其转为目标相关的分支选择。这个描述得到模块消融与自然目标状态替换共同支持。', '',
    '尚未识别出明确的“邻接表/可达闭包”存储坐标，也没有证明网络在线执行DFS、完整路径规划或递归可达性计算。第1层FFN被移除会破坏合法性，不等于图事实必然储存在FFN权重；它也可能在处理embedding中已有的结构。', '',
    '## 模块/attention边干预', '',
    '只修改最后当前位置的更新，其余位置正常计算：FFN/attention消融将对应残差增量置0，不是把整个残差清零。单边干预重新计算该query row的SDPA，禁止读取一个key并重新归一化；额外对照包含起点key、分隔符key以及不删任何key的sham重算。FP32、禁用TF32；每条件后撤销hook。', '',
    '|干预|合法后继argmax|有用后继argmax|', '|---|---:|---:|']
base = summary['baseline']
lines.append(f"|原模型|{100*base['legal_next']['mean']:.2f}%|{100*base['useful_next']['mean']:.2f}%|")
conditions = [('zero_ffn_0', '去掉当前位置L1 FFN增量'), ('zero_attention_1', '去掉当前位置L2 attention增量'),
              ('drop_target_1', 'L2当前位置不读目标key'), ('drop_target_all', '所有层当前位置不读目标key'),
              ('drop_separator_all', '所有层当前位置不读分隔符key'), ('drop_source_all', '所有层当前位置不读起点key'),
              ('zero_attention_all', '去掉所有层当前位置attention增量'), ('zero_ffn_all', '去掉所有层当前位置FFN增量')]
for name, label in conditions:
    v = summary['ablations'][name]
    lines.append(f"|{label}|{100*v['legal_next']['mean']:.2f}%|{100*v['useful_next']['mean']:.2f}%|")
sham_error = max(v.get('max_logit_error_vs_native', 0) for d in ds for k, v in d['ablations'].items() if k.startswith('sham'))
lines += ['', f'sham重算与原始输出最大logit差异为{sham_error:.7g}。去掉目标key比去掉起点/分隔符key影响大，支持直接目标读取的重要性。注意，删key会重分配注意力并产生非自然混合状态；不能把合法性下降直接等同于目标位置储存了邻接表，也不能仅凭消融识别纯粹的语义功能。', '',
          '## 自然目标状态替换：更强的定向证据', '',
          '寻找同起点、相同完整路径前缀、相同目标层级、不同目标且有用后继集合不相交的原生状态。没有为了得到好结果而要求原始选择正确。每seed有46–58对（每对两个方向），原始模型恰好两侧都选对；每方向只替换一个位置、一个子层的隐藏状态，下游正常重算，输入token和其他位置保持原目标。', '',
          '|被替换的位置/阶段|下一步转向供体目标的有用后继|', '|---|---:|']
stage_names = ['E'] + [s for i in range(1, 7) for s in (f'L{i}A', f'L{i}F')]
for role, ix in [('target', 1), ('target', 2), ('target', 4), ('current', 0), ('current', 1),
                 ('current', 3), ('current', 5), ('current', 7), ('current', 9)]:
    values = summary['patches'][role]
    stage = values['stages'][ix]
    label = {'target': '目标位置', 'current': '当前位置'}[role]
    lines.append(f"|{label} {stage_names[stage]} 后|{100*values['donor_useful']['mean'][ix]:.2f}%|")
lines += ['', '目标位置L1F替换后约98%改走新目标；替换目标位置到L4F已不改变当前选择。当前位置L1A/F交换几乎无效，L2A开始约78%转向供体目标，L4A达到约99.5%，之后接近100%。与消融一起支持目标信息通过L2–L4 attention传入当前位置。', '',
          '这里的“从某层能转向”意味着该层状态足以影响后续选择，不是信息首次出现层：早期交换可能被后续重新读取原目标覆盖。L6F当前位置交换等于直接换输出头输入，是阳性sanity control，不是深层规划的独立证据。', '',
          '替换分隔符或输入起点位置均未产生供体方向转向。起点位置在目标之前，因果掩码下两目标查询的该状态本应相同；同当前位置embedding交换也应无效。目标embedding交换和L6F当前位置交换均逐元素重现供体logits，末层目标/分隔符交换为noop，全部控制通过。', '',
          '## 一个重要的shortcut对照', '',
          '去掉全部当前位置attention后，普通测试状态仍有约74%有用选择率；此时当前位置只有自身embedding和tokenwise FFN，已经不能随输入目标改变。原因是测试分布中某些默认分支对许多目标都好，均匀随机后继不是充分的非目标策略对照。', '',
          '|相同前缀、互斥目标的配对面板|两方向等权的原目标有用选择率|', '|---|---:|', '|原模型|100.00%|']
for name, label in [('zero_attention_all', '去掉全部当前位置attention'), ('drop_target_all', '所有层不读目标key'),
                    ('drop_separator_all', '所有层不读分隔符key')]:
    value = summary['pair_balanced_ablations'][name]['recipient_useful']['mean']
    lines.append(f'|{label}|{100*value:.2f}%|')
lines += ['', '全部attention置零时，每对不同目标的logits逐元素相同，验证目标依赖确实消失；互斥有用集合下任意固定下一节点策略的两方向平均最多50%。这是与均匀随机基线不同的控制。配对复用query和前缀，不当作独立图样本推断。', '',
          '## 可以怎样进一步区分计算实现', '',
          '候选机制是局部后继评分A(u,v)与目标条件可达关系R(v,t)结合；现有因果结果只定位处理/传递位置，尚未验证这两个语义对象的具体坐标。下一轮应在这些位置进行结构方向的定向替换，并与等范数随机方向及身份保持对照比较；或编辑一条图事实后验证所有受影响查询是否按新图变化。', '',
          '固定图上预存R或预存(u,t)策略表也能完成目标相关导航。若要主张在线搜索，需要额外的图变化/局部事实变化设置，并确保模型获得变化信息；不能改变外部真图却不告诉模型，再把失败解释为不会推理。DFS state在完成训练后还可做visited/frontier合法历史干预；本轮不把PATH解释为DFS执行器。', '',
          '## 复现与资源', '',
          '所有前向在空闲GPU0，未干扰其他GPU上的DFS训练；未保存activation、模型权重或新增W&B run。自然BF16生成与FP32单步分析的下一token argmax差异约0.42%–1.26%，记录在seed JSON；因果实验全部在同一FP32口径内比较，不把二者混为完全同一数值实验。仅一张固定图、4初始化，结论范围不能扩展为通用模型算法。', '',
          'run.py产生seed0–3.json；summarize.py生成summary.json和唯一关键图key_results.png/.pdf。']
(HERE / 'REPORT.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')

fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.5), layout='constrained')
x = np.arange(1, 7)
for kind, label, color in [('zero_ffn', 'Zero current FFN', '#3973a9'), ('zero_attention', 'Zero current attention', '#dc8134')]:
    for key, linestyle, suffix in [('legal_next', '-', 'legal'), ('useful_next', '--', 'useful')]:
        axes[0].plot(x, [100*summary['ablations'][f'{kind}_{i}'][key]['mean'] for i in range(6)],
                     color=color, ls=linestyle, marker='o', ms=4, label=f'{label}: {suffix}')
axes[0].axhline(100*base['useful_next']['mean'], color='#9aa1ab', ls=':', label='Baseline useful')
axes[0].set(title='Necessity of current-position increments', xlabel='Ablated layer', ylabel='Next-token rate (%)', ylim=(0, 104))
for role, label, color in [('target', 'Patch target position', '#dc8134'), ('current', 'Patch current position', '#26936b'),
                          ('separator', 'Patch separator position', '#7d8794')]:
    v = summary['patches'][role]
    axes[1].plot(v['stages'], np.asarray(v['donor_useful']['mean'])*100, marker='o', color=color, ms=3.5, label=label)
axes[1].set_xticks(range(13), stage_names, rotation=45, ha='right', fontsize=8)
axes[1].set(title='Natural donor target: direction transfer', xlabel='E=embedding; A=attention; F=FFN', ylabel='Choice useful for donor target (%)', ylim=(0, 104))
conditions = [('baseline', 'Baseline'), ('zero_attention_all', 'No current\nattention'),
              ('drop_target_all', 'No target\nkey reads'), ('drop_separator_all', 'No separator\nkey reads')]
for j, (key, label) in enumerate(conditions):
    for offset, metric, color, legend in [(-.18, 'recipient_useful', '#26936b', 'Useful for requested goal'),
                                         (.18, 'legal', '#3973a9', 'Legal successor')]:
        value = 1. if key == 'baseline' else summary['pair_balanced_ablations'][key][metric]['mean']
        axes[2].bar(j+offset, value*100, .34, color=color, label=legend if j == 0 else None)
axes[2].axhline(50, color='#aaaaaa', ls=':', label='Target-blind maximum')
axes[2].set_xticks(range(4), [label for key, label in conditions], fontsize=9)
axes[2].set(title='Same-prefix mutually exclusive target pairs', ylabel='Directed-pair mean (%)', ylim=(0, 104))
for ax in axes:
    ax.grid(axis='y', alpha=.16)
    ax.set_axisbelow(True)
    ax.legend(fontsize=7.4, loc='lower right')
fig.suptitle('Fixed80 PATH, last50k: four initialization means; next-decision causal interventions', fontsize=12)
fig.savefig(HERE / 'key_results.png', dpi=180)
fig.savefig(HERE / 'key_results.pdf')
print('baseline', summary['baseline'])
print('pair no-attention useful', summary['pair_balanced_ablations']['zero_attention_all']['recipient_useful'])
