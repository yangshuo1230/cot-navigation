"""Render the one key figure and report from final saved measurements."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
DATA = {g: json.loads((HERE/f'{g}.json').read_text()) for g in ('simple', 'hard')}

def values(g, role, relation, name, fixed=False):
    field = 'native_post_ffn_decoder_fixed' if fixed else 'fresh_decoder'
    return [r[field][name]['auc'] for r in DATA[g]['roles'][role][relation]]

def mean(g, role, relation, name, fixed=False): return float(np.mean(values(g, role, relation, name, fixed)))
def stat(g, role, relation, name):
    v = values(g, role, relation, name)
    return f'{np.mean(v):.3f} ± {np.std(v):.3f}'

plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
fig, axes = plt.subplots(2, 2, figsize=(13, 8.5), layout='constrained')
states = ['embedding', 'after_attention', 'ffn_input_norm2', 'ffn_delta', 'post_ffn']
labels = ['Embedding', 'After\nattention', 'FFN input\n(LayerNorm)', 'FFN delta\nonly', 'After FFN']
for j, g in enumerate(('simple', 'hard')):
    x = np.arange(len(states))+(j-.5)*.35
    v = [values(g, 'source', 'descendants', k) for k in states]
    axes[0, 0].bar(x, np.mean(v, 1), .35, yerr=np.std(v, 1), capsize=3, label=g.title())
axes[0, 0].set(xticks=np.arange(len(states)), xticklabels=labels, ylim=(.45, 1.03), ylabel='Heldout-node AUROC', title='Source: reachable descendants through L1')
axes[0, 0].legend()
for j, g in enumerate(('simple', 'hard')):
    keys = ['post_ffn', 'after_attention', 'no_attention_post_ffn']
    vals = [values(g, 'source', 'descendants', k) for k in keys]
    rand = [v for seed in (201, 303, 407) for v in values(g, 'source', 'descendants', f'random_ffn_post_{seed}')]
    y = [np.mean(v) for v in vals]+[np.mean(rand)]
    err = [np.std(v) for v in vals]+[np.std(rand)]
    axes[0, 1].bar(np.arange(4)+(j-.5)*.35, y, .35, yerr=err, capsize=3, label=g.title())
axes[0, 1].set(xticks=np.arange(4), xticklabels=['Native L1', 'FFN1 off', 'Attention1 off\nFFN kept', 'Random FFN1\n(norm matched)'], ylim=(.45, 1.03), ylabel='Fresh heldout-node probe AUROC', title='Source: ablations and randomized FFN control')
for j, g in enumerate(('simple', 'hard')):
    fun = DATA[g]['functional']
    y = [fun['source']['native']['constrained_ba'], fun['source']['ffn_off_at_this_position']['constrained_ba'],
         fun['source']['attention_off_at_this_position']['constrained_ba'], fun['target']['ffn_off_at_this_position']['constrained_ba'],
         fun['target']['attention_off_at_this_position']['constrained_ba']]
    axes[1, 0].bar(np.arange(5)+(j-.5)*.35, y, .35, label=g.title())
axes[1, 0].set(xticks=np.arange(5), xticklabels=['Native', 'Source\nFFN1 off', 'Source\nattention1 off', 'Goal\nFFN1 off', 'Goal\nattention1 off'], ylim=(.45, 1.03), ylabel='YES/NO balanced accuracy', title='Functional effects: one position, L1 only')
for j, (role, relation, metric) in enumerate([('target', 'ancestors', 'Ancestor probe AUROC'), ('target', 'ancestors', 'Answer balanced accuracy')]):
    if j == 0:
        y = [mean('simple', role, relation, 'post_ffn'), mean('simple', role, relation, 'after_attention')]
    else:
        y = [DATA['simple']['functional']['target']['native']['constrained_ba'], DATA['simple']['functional']['target']['ffn_off_at_this_position']['constrained_ba']]
    axes[1, 1].bar(np.arange(2)+(j-.5)*.35, y, .35, label=metric)
axes[1, 1].set(xticks=[0, 1], xticklabels=['Native', 'Goal FFN1 off'], ylim=(.45, 1.03), ylabel='Metric value (different metrics)', title='Simple graph: decodability and behavior dissociate')
axes[1, 1].legend(fontsize=9)
fig.savefig(HERE/'l1_ablation.png', dpi=170); plt.close(fig)

lines = ['# L1 attention / FFN 消融与图关系探针', '',
'完成两个 N100 Fixed DAG Direct 验证最佳 checkpoint 的分析。结论：困难图起点可达后代的线性可读性由 attention 和 FFN 两步共同增强；FFN 输出增量自身承载强信号，不能把全部提升归于 FFN。简单图原 embedding 后代信号已强，L1 不带来净提升。尚未定位 YES/NO 的特定祖先/后代因果中介。', '',
'## 设置与检查', '',
'沿用原 checkpoint、原 query train/val/test 和焦点节点划分 seed17/31/47，60/20/20节点划分与查询划分取交集。每个seed分别训练/验证/测试，不合并不同解码器的分数。原模型见过全部节点，留出的是新探针的节点身份。实际测试焦点约13–18个，三个划分的标准差不是跨图置信区间。', '',
'冻结 native FP32 CPU 前向；只在 GPU0 拟合仿射 ridge，验证 MSE 选正则。不训练主模型，不保存 activation、probe/model checkpoint 或 W&B run。全新代码在 studies/l1_graph_ablation；未修改旧实验。checkpoint SHA256 已核对。手动重构 L1 与原 native block residual 的最大差异为0；完整状态 no-op patch 的答案 margin 误差有<1e-3断言。', '',
'同时测试祖先、直接后继、严格可达后代，均不含自身。固定一个候选节点，在不同留出焦点之间算 AUROC，排除拓扑上不可能成立的候选、当前焦点和另一个查询端点。同一焦点重复查询先平均，避免重复加权；该指标不等于准确率或完整集合恢复。每个条件重新拟合的探针评估线性可读性，另保存同一个原生L1后探针在各条件上的读出，用于检查读出坐标变化。', '',
'## 起点可达后代的逐子层读出', '',
'| 表示 | 简单图 AUROC | 困难图 AUROC |', '|---|---:|---:|']
for key, label in [('embedding', '原始 embedding'), ('norm1_embedding', '仅 norm1(embedding)'),
                   ('attention_delta', 'attention 输出增量'), ('after_attention', 'attention 后残差（FFN 消融）'),
                   ('ffn_input_norm2', 'FFN 的实际 LayerNorm 输入'), ('ffn_delta', 'FFN 输出增量'),
                   ('post_ffn', '完整 L1 后残差'), ('no_attention_ffn_delta', '关闭 attention 时的 FFN 增量'),
                   ('no_attention_post_ffn', '关闭 attention，保留 FFN 的 L1 残差')]:
    lines.append(f'| {label} | {stat("simple", "source", "descendants", key)} | {stat("hard", "source", "descendants", key)} |')
lines += ['',
'困难图：E0.701 → attention后0.872 → norm2后0.858 → FFN后0.948；FFN增量单独0.951。关闭attention、保留原FFN时只有0.764。可读性增强需要这两个训练后分支协同。起点看不到后面的目标token，不能从该目标直接拷贝答案；但是attention仍会读取自己及前面的协议token，所以这里不是无条件的原embedding→FFN。', '',
'简单图：E0.923 → attention后0.872 → norm2后0.901 → FFN后0.900。没有净增强，部分恢复发生在LayerNorm步骤；不能把归一化带来的读出变化算成FFN权重学得的新知识。', '',
'三种独立随机FFN保持norm2与attention不变，up/down矩阵分别匹配训练矩阵 Frobenius 范数。只在内存计算替代输出，不修改原参数。下列范围是每个随机权重seed跨三个节点划分的平均AUROC范围，不是置信区间。', '',
'| 随机 FFN 对照 | 简单 | 困难 |', '|---|---:|---:|']
for suffix, label in [('delta', '仅随机FFN增量'), ('post', '随机FFN后的残差')]:
    vals = {g: [mean(g, 'source', 'descendants', f'random_ffn_{suffix}_{seed}') for seed in (201, 303, 407)] for g in ('simple', 'hard')}
    lines.append('| '+label+' | '+' | '.join(f'{min(vals[g]):.3f}–{max(vals[g]):.3f}' for g in ('simple', 'hard'))+' |')
lines += ['',
'困难图训练FFN后0.948，高于三个匹配范数随机FFN后的0.821–0.877；随机非线性也可能保留/改变原信息，故该对照支持训练权重的特定作用，但不唯一定位知识存储。', '',
'## 祖先和直接后继对照', '',
'以下为与可达性判定相对应的目标祖先、起点直接后继；所有关系的两个位置、全部特征与支持数均保存于JSON。', '',
'| 图/位置/关系 | E | attention后 | FFN输入 | FFN增量 | FFN后 |', '|---|---:|---:|---:|---:|---:|']
for g in ('simple', 'hard'):
    for role, relation in [('target', 'ancestors'), ('source', 'successors')]:
        lines.append(f'| {g}/{role}/{relation} | '+' | '.join(f'{mean(g, role, relation, key):.3f}' for key in ['embedding', 'after_attention', 'ffn_input_norm2', 'ffn_delta', 'post_ffn'])+' |')
lines += ['', '## 单位置 L1 消融对最终答案的影响', '',
'在该位置L1后patch已计算的反事实状态，其余位置L1保持原状态，后续层重新前向。去FFN等价于用attention后残差替换；去attention后保留FFN则替换为 e+FFN(norm2(e))。这是单位置/单层消融，不是删除全部层或全部位置。原输入token不变。', '',
'| 条件 | 简单图 YES/NO BA | 困难图 YES/NO BA |', '|---|---:|---:|']
for role, condition, label in [('source','native','原模型'), ('source','ffn_off_at_this_position','仅去起点FFN1'),
                              ('source','attention_off_at_this_position','仅去起点attention1，FFN保留'),
                              ('target','ffn_off_at_this_position','仅去目标FFN1'), ('target','attention_off_at_this_position','仅去目标attention1，FFN保留')]:
    lines.append('| '+label+' | '+' | '.join(f'{DATA[g]["functional"][role][condition]["constrained_ba"]:.3f}' for g in ('simple','hard'))+' |')
lines += ['',
'所有条件的原生输出仍100%是YES/NO，因此这里的关系损害不能归因于非法token。简单图测试1091查询，困难图555查询。困难图起点FFN1删除后BA0.976→0.878，说明该分支对答案重要；但整个分支删除不具有后代概念选择性，不能证明损害由后代信息减少所中介。', '',
'## 两个值得记录的反例/限制', '',
'1. 简单图目标FFN1删除后，祖先新探针AUROC从0.852升到0.896，但答案BA从0.931降到0.604。保持一个概念可读不保证保持原模型可用的计算形式；反之，probe读出变弱也不证明概念未被使用。', '',
'2. 困难图同一个原生FFN后训练的后代探针，应用到FFN前残差仍有0.920，应用到原embedding还有0.836（分别重新拟合探针时为0.872、0.701）。这些都是线性读出，说明有限训练节点下的探针估计/正则与训练表示选择会影响得分，不能把某个探针分数当作表示信息量的上界。固定探针跨表示应用本身也有分布偏移，不把差异当作纯语义因果效应。', '',
'## 能够支持的结论', '',
'可以说：困难固定图的起点L1 FFN输出包含较强的可达后代线性可读信号，并进一步增强attention已变换表示的可读性；这一增强在原查询、探针留出节点上测到。不能说全部提升仅由FFN产生，也不能说后代事实一定存于FFN而不在embedding。更不能从本实验确定最终YES/NO必经后代方向。每图只有一个Direct checkpoint，不能外推PATH/DFS/BFS或其他训练seed。', '',
'![L1子层探针与消融](l1_ablation.png)', '',
'复现：从服务器repo root运行 `.venv/bin/python studies/l1_graph_ablation/run.py`，然后运行同目录 `report.py`。冻结推理CPU、闭式probe拟合GPU0。只生成两个结果JSON、一张图和本报告，未写中间activation/权重。', '']
(HERE/'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')
print('Wrote REPORT.md and l1_ablation.png')
