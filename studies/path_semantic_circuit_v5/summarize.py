"""Aggregate initialization means; repeated probe/control seeds are not samples."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ds = [json.loads((HERE / f'seed{s}.json').read_text()) for s in range(4)]
coarse = json.loads((HERE.parent / 'path_causal_circuit_v5/summary.json').read_text())

def aggregate(values):
    x = np.asarray(values, float)
    return dict(mean=x.mean(0).tolist(), min=x.min(0).tolist(), max=x.max(0).tolist(),
                by_initialization=x.tolist())

summary = {'pairs_per_seed': [d['pairs'] for d in ds],
           'unique_queries_per_seed': [d['unique_queries'] for d in ds],
           'checkpoints': [d['checkpoint_sha256'] for d in ds],
           'maximum_untie_logit_error': max(d['untied_baseline_max_logit_error'] for d in ds)}
for side in ('goal', 'head'):
    summary[side] = {}
    for kind in ('effects', 'controls'):
        summary[side][kind] = {}
        for key, entry in ds[0][side][kind].items():
            summary[side][kind][key] = {metric: aggregate([d[side][kind][key][metric] for d in ds])
                                       for metric in entry if metric != 'evaluations'}

def mean(side, key, metric='donor_useful', kind='effects'):
    return summary[side][kind][key][metric]['mean']

def pct(side, key, metric='donor_useful', kind='effects'):
    return f'{100*mean(side, key, metric, kind):.2f}%'

(HERE / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
lines = ['# PATH：图关系子空间的因果干预', '',
    '使用同一张80节点固定DAG上的4个50k PATH初始化，不重新训练主模型。每seed分别有48、46、50、58对自然查询，起点、完整已生成路径前缀、目标层级相同，但目标不同，且两目标的有用直接后继集合不相交。当前位置距目标至少两跳。每对两个方向等权，每个初始化先内部平均，再对4个初始化等权平均。重复探针划分/随机方向并非独立图样本。', '',
    '## 最主要的新结果', '',
    'L1 FFN之后，目标位置中由祖先探针定义的16维方向，对下一步定向选择有可重复的因果作用；但它们不足以承担整个目标表征的功能。候选后继输出行中的后代方向编辑，作用较弱。这支持目标侧的祖先相关特征参与选边，尚未证明模型逐个候选执行显式集合包含判断。', '',
    '## 如何做编辑', '',
    '输入embedding与输出头原本绑权重。先复制输出头并解除共享，所有原始logits完全不变（最大误差0）。之后目标输入侧编辑不改输出头，输出行编辑不改输入embedding或中间activation。模型权重与探针仅在内存中使用，不保存新checkpoint。', '',
    '对每个被编辑节点对，重新做48训练/16验证/16测试节点划分，两个被编辑节点都只在测试集。三个划分种子17/23/31；按验证MSE选择ridge正则。输出标签为80节点的严格祖先、全部可达后代、直接后继。输出候选列未留出；主模型训练时已经见过所有节点。', '',
    '目标E使用原始embedding。目标L1 attention/FFN使用原模型训练查询中的目标状态均值拟合方向，且拟合上下文的起点限制在探针训练节点；部分早层目标没有上下文，因此实际拟合节点数小于48/16，详见seed JSON。编辑时替换的是当前自然上下文的真实状态，不是这个均值。', '',
    '将ridge矩阵在输入坐标中的前8/16个左奇异向量记作U，P=UUᵀ。主干预是 h′=h+P(h_donor−h)。其余方向保持不变，下游全部重新计算；不是只改一层而冻结后层。输出行实验对两个合法候选的行做互换式投影编辑。秩8、16、系数1在查看结果前固定。', '',
    '对照包括：同范数、祖先子空间外的随机子空间投影；同范数、子空间内的随机方向；互补方向；8维半剂量；全向量替换；以及先去除另一关系前r维方向再拟合的关系方向。后者仅与那r维正交，不能称为纯祖先/纯后代。', '',
    '## 目标位置编辑：下一步转向供体目标的有用后继', '',
    '原查询的有用集合与供体集合不相交，因此未编辑时转向供体比例为0；本表是下一token的定向变化率，不是完整轨迹或查询准确率。', '',
    '|目标位置|祖先16维|同范数外部随机|同空间随机|后代16维|全向量替换|',
    '|---|---:|---:|---:|---:|---:|']
for stage, label in [(0, 'Embedding'), (1, 'L1 attention 后'), (2, 'L1 FFN 后')]:
    prefix = f'stage{stage}/'
    lines.append(f"|{label}|{pct('goal', prefix+'ancestor/r16/semantic')}|"
                 f"{pct('goal', prefix+'ancestor/r16/random_orth_normmatched')}|"
                 f"{pct('goal', prefix+'ancestor/r16/random_inspace_normmatched')}|"
                 f"{pct('goal', prefix+'descendant/r16/semantic')}|"
                 f"{pct('goal', f'stage{stage}_full_swap', kind='controls')}|")
key = 'stage2/ancestor/r16/semantic'
v = summary['goal']['effects'][key]['donor_useful']
lines += ['', f"L1 FFN后祖先16维转向率均值{100*v['mean']:.2f}%，4个初始化范围{100*v['min']:.2f}–{100*v['max']:.2f}%。编辑后合法后继率{pct('goal', key, 'legal')}。因此这个效果主要是合法分支间的定向变化，而非输出整体损坏。", '',
    f"同位置8维祖先方向转向率为{pct('goal', 'stage2/ancestor/r8/semantic')}；去除后代前16维后，祖先相关方向仍有{pct('goal', 'stage2/ancestor_unique/r16/semantic')}；反向去除祖先方向的后代编辑为{pct('goal', 'stage2/descendant_unique/r16/semantic')}。", '',
    f"祖先16维编辑平均传递了完整供体差异在祖先探针读出方向上{pct('goal', key, 'probe_fraction_of_full_donor_change')}的投影；定义为 Δy_edit·Δy_full/||Δy_full||²，并非真实祖先集合恢复率。对应编辑范数为完整供体状态差范数的{pct('goal', key, 'edit_norm_ratio')}，不能称为微小扰动。", '',
    '关键限制：探针方向包含身份及其他相关特征；改变探针读出并不保证只改变祖先关系。24%的行为转向证明该方向有因果作用，但不证明全部作用经由祖先这一语义变量，也不能由剩余不转向样本推断祖先未被使用。', '',
    '## 输出行编辑：候选的可达后代是否直接用于打分', '',
    '保持全部隐藏状态不变，只编辑两个候选节点的输出头行；方向解码器同样不见这两个候选节点。用保存的最终状态计算精确logit变化，并与显式编辑输出头的前向结果核对。', '',
    '|输出行方向（16维）|转向供体|同范数外部随机|', '|---|---:|---:|']
for relation, label in [('ancestor', '祖先'), ('descendant', '可达后代'), ('successor', '直接后继')]:
    lines.append(f"|{label}|{pct('head', relation+'/r16/semantic')}|{pct('head', relation+'/r16/random_orth_normmatched')}|")
lines += ['', '这些输出侧效应较弱。全行互换100%转向只是输出标签重命名，不能作为发现规划计算的证据。额外选择相同前缀、同目标深度、两个候选具有相同可达标签的其他目标，记录选边变化与有用率，详见summary；这些对照的覆盖不完整。', '',
    '本实验没有编辑所有层中候选的内部表征，因而无法排除模型在内部使用后代信息；只能说明这些留出节点探针定义的低维输出行方向，没有表现出主要的充分作用。', '',
    '## 与模块干预结合后的回路', '',
    '此前同checkpoint的模块干预显示：当前位置L1 FFN增量置零，合法率从99.58%降至50.80%；当前位置L2 attention增量置零，有用率从98.39%降至82.65%。禁止当前位置读取目标key影响显著，禁止读取起点或分隔符影响很小。', '',
    '自然状态互换显示：目标L1F整体互换约98%转向供体；当前位置L1互换几乎不转向，L2A约78%、L4A约99.5%。这定位了目标信息进入当前位置、并逐步决定选边的通路。早层状态替换也可能被后续原始信息覆盖，因此不能把转向率上升直接解释为信息首次出现。', '',
    '最稳妥的描述：当前位置的早层FFN提供对合法输出重要的节点变换；目标侧在L1F后存在因果相关的祖先探针方向；目标信息通过后续attention进入当前位置，并在L2–L4形成目标条件选边状态。', '',
    '尚不能把它写成已证实的“L1枚举全部后继 → 后层逐个检查后代集合 → 求交”。邻接/可达表查找、预存策略及其他混合编码仍可能实现同一回路。本轮结果也不等于证明在线DFS或通用图搜索。', '',
    '## 资源与复现', '',
    '仅GPU0执行冻结前向与临时ridge拟合；未干扰主训练。未保存activation、探针权重或新模型checkpoint。已有粗粒度干预见相邻path_causal_circuit_v5目录。run.py产生4个seed JSON，summarize.py生成本报告、summary及一张关键图。']
(HERE / 'REPORT.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')

fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.5), layout='constrained')
for relation, label, color in [('ancestor', 'Ancestor directions', '#26936b'),
                               ('descendant', 'Descendant directions', '#3973a9'),
                               ('successor', 'Direct-successor directions', '#dc8134')]:
    values = [summary['goal']['effects'][f'stage{s}/{relation}/r16/semantic']['donor_useful'] for s in range(3)]
    y = np.array([x['mean'] for x in values])*100
    low = np.array([x['min'] for x in values])*100
    high = np.array([x['max'] for x in values])*100
    axes[0].plot(range(3), y, '-o', label=label, color=color)
    axes[0].fill_between(range(3), low, high, color=color, alpha=.12)
axes[0].plot(range(3), [100*mean('goal', f'stage{s}/ancestor/r16/random_orth_normmatched') for s in range(3)],
             '--o', color='#888888', label='Norm-matched random (ancestor)')
axes[0].set_xticks(range(3), ['Embedding', 'L1 attention', 'L1 FFN'])
axes[0].set(title='Target-position subspace swaps (rank 16)', ylabel='Donor-useful next choice (%)', ylim=(0, 40))

conditions = [('ancestor/r16/semantic', 'Ancestor\nrank 16'),
              ('ancestor_unique/r16/semantic', 'Ancestor after\nremoving Desc-16'),
              ('descendant_unique/r16/semantic', 'Descendant after\nremoving Anc-16'),
              ('ancestor/r16/random_inspace_normmatched', 'Random within\nancestor space'),
              ('ancestor/r16/random_orth_normmatched', 'Random outside\nancestor space')]
for i, (key, label) in enumerate(conditions):
    vals = np.array([d['goal']['effects']['stage2/'+key]['donor_useful'] for d in ds])*100
    axes[1].bar(i, vals.mean(), color=['#26936b', '#71aa8d', '#3973a9', '#a5a9b0', '#c0c3c9'][i])
    axes[1].scatter(i+np.linspace(-.12,.12,4), vals, s=14, color='#333333', zorder=3)
axes[1].set_xticks(range(len(conditions)), [label for _, label in conditions], rotation=25, ha='right', fontsize=8)
axes[1].set(title='Target L1 FFN: specificity controls', ylabel='Donor-useful next choice (%)', ylim=(0, 40))

for role, label, color in [('target', 'Whole target state', '#dc8134'), ('current', 'Whole current state', '#26936b')]:
    v = coarse['patches'][role]
    axes[2].plot(v['stages'], np.asarray(v['donor_useful']['mean'])*100, '-o', ms=3, color=color, label=label)
axes[2].set_xticks(range(13), ['E']+[f'L{i}{s}' for i in range(1,7) for s in ('A','F')], rotation=50, fontsize=8)
axes[2].set(title='Whole-state route localization (earlier run)', ylabel='Donor-useful next choice (%)', ylim=(0,105))
for ax in axes:
    ax.grid(axis='y', alpha=.18)
    ax.set_axisbelow(True)
    if ax is not axes[1]:
        ax.legend(fontsize=7.5)
fig.suptitle('Fixed80 PATH, four 50k initializations: causal target dependence and partial semantic mediation', fontsize=12)
fig.savefig(HERE / 'key_results.png', dpi=180)
fig.savefig(HERE / 'key_results.pdf')
print('Wrote summary, report and one key figure; target L1F ancestor16:', summary['goal']['effects']['stage2/ancestor/r16/semantic']['donor_useful'])
