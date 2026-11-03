"""One figure and a report for the fixed-attention control."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
DATA = {g: json.loads((HERE/f'{g}.json').read_text()) for g in ('simple', 'hard')}
CONDITIONS = ['dynamic', 'train_mean_fixed', 'self_only', 'uniform_visible', 'mean_fixed_affine_norm', 'attention_off']
LABELS = ['Dynamic', 'Train-mean\nfixed', 'Self only', 'Uniform', 'Fixed +\naffine norm', 'Attention off']

def vals(g, key, relation='descendants'):
    return [r['fresh_decoder'][key]['auc'] for r in DATA[g]['relations'][relation]]
def avg(g, key, relation='descendants'): return float(np.mean(vals(g, key, relation)))

plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
fig, axes = plt.subplots(2, 2, figsize=(13, 8), layout='constrained')
for row, g in enumerate(('simple', 'hard')):
    x = np.arange(len(CONDITIONS))
    for j, (stage, name) in enumerate([('after_attention', 'Before FFN'), ('post_ffn', 'After FFN')]):
        v = [vals(g, c+'/'+stage) for c in CONDITIONS]
        axes[row, 0].bar(x+(j-.5)*.36, np.mean(v, 1), .36, yerr=np.std(v, 1), capsize=2, label=name)
    axes[row, 0].axhline(avg(g, 'embedding'), ls=':', color='gray', label='Fresh probe of raw embedding')
    axes[row, 0].set(xticks=x, xticklabels=LABELS, ylim=(.45, 1.03), ylabel='Heldout-node descendant AUROC', title=g.title()+': source-node readouts')
    axes[row, 1].bar(x, [DATA[g]['functional']['results'][c]['constrained_ba'] for c in CONDITIONS], color='tab:green')
    axes[row, 1].set(xticks=x, xticklabels=LABELS, ylim=(.45, 1.03), ylabel='YES/NO balanced accuracy', title=g.title()+': source attention1 intervention only')
    if row == 0: axes[row, 0].legend(fontsize=8)
fig.savefig(HERE/'fixed_attention.png', dpi=170); plt.close(fig)

lines = ['# 起点 attention 动态权重、固定混合与仿射变换对照', '',
'结论：困难图起点attention1近乎只关注自身，动态权重对当前后代probe提升和测试答案并非必要；固定权重、甚至只保留自身，均复现原表现。固定attention再把norm1替换成训练统计量得到的仿射函数，仍保留大部分提升。简单图固定协议/自身混合也保持表现，但只看自身会伤害答案。这里不涉及后续层或目标位置attention是否需要动态权重。', '',
'## 范围与方法', '',
'使用此前两个N100 Fixed DAG Direct验证最佳checkpoint，未训练/修改主模型。冻结CPU FP32 native推理，闭式仿射probe拟合仅GPU0，不保存activation/probe/model checkpoint或W&B run。checkpoint hash、RoPE后QK重构与native注意力输出一致、原状态patch no-op均有断言。', '',
'只改L1起点attention的输出，再通过原FFN构造该位置L1残差；其他位置L1状态保持原样，后续层重算。源位置看不到后面的目标。简单可见WORLD、QUERY、自身；困难可见QUERY、自身。均值权重按训练来源节点等权平均，不按重复query加权。', '',
'probe继续用原query train/val/test与60/20/20焦点节点划分取交集，seed17/31/47；每个probe划分的固定权重及norm统计量仅使用该划分的训练节点/训练查询，不使用留出节点。真实答案全测试面板则用原query-train的90个起点估计均值，与probe面板的训练支持不同。每seed各自评估固定候选跨焦点的AUROC，不跨折拼接分数；排除自身、另一端点和拓扑上不可能满足关系的候选。', '',
'## 原起点 attention 权重', '',
'按90个不同起点等权统计：', '',
'| 图 | WORLD | QUERY | 自身 | 自身权重范围 |', '|---|---:|---:|---:|---:|']
for g in ('simple', 'hard'):
    w = DATA[g]['native_weights']
    prefix = f'{100*w["mean"][0]:.4f}%' if g == 'simple' else '无此token'
    query = w['mean'][-2]; self_ = w['mean'][-1]
    lines.append(f'| {g} | {prefix} | {100*query:.5f}% | {100*self_:.5f}% | {100*w["minimum"][-1]:.4f}%–{100*w["maximum"][-1]:.4f}% |')
lines += ['',
'困难图的起点动态attention实际上已接近self-only。此前“softmax可能产生非线性门控”在架构上正确，但不是这个起点checkpoint的主要实测解释。简单图权重确实随节点变化较多，不过变化不一定对本测试行为必要。', '',
'## 可达后代线性探针', '',
'每个条件重新训练probe；表内为三个节点划分AUROC均值，不是准确率或完整集合恢复。标准差、支持数及固定原生decoder的跨条件结果保留在JSON。', '',
'| 条件 | 简单：attention后 | 简单：FFN后 | 困难：attention后 | 困难：FFN后 |', '|---|---:|---:|---:|---:|']
pretty = {'dynamic':'原动态权重', 'train_mean_fixed':'训练均值固定权重', 'self_only':'只关注自身', 'uniform_visible':'可见位置均匀权重', 'attention_off':'attention输出置零',
          'mean_fixed_affine_norm':'固定权重＋仿射norm1近似', 'self_only_affine_norm':'self-only＋仿射norm1近似'}
for c in CONDITIONS+['self_only_affine_norm']:
    lines.append('| '+pretty[c]+' | '+' | '.join(f'{avg(g, c+"/"+stage):.3f}' for g in ('simple','hard') for stage in ('after_attention','post_ffn'))+' |')
lines += ['',
'原embedding后代probe：简单0.923、困难0.701；只做原norm1：简单0.914、困难0.707。困难固定混合/自投影仍能达到attention后0.872、FFN后0.948，动态权重不是提升所必需。', '',
'## 真实答案与预测一致性', '',
'只干预L1起点，不改变目标/分隔符在L1的原生状态，完整后续网络重算。简单测试1091条、困难555条。所有条件原生输出均为YES/NO。', '',
'| 条件 | 简单 YES/NO BA | 困难 YES/NO BA | 简单预测改变比例 | 困难预测改变比例 |', '|---|---:|---:|---:|---:|']
for c in CONDITIONS+['self_only_affine_norm']:
    f = {g: DATA[g]['functional']['results'][c] for g in ('simple','hard')}
    lines.append('| '+pretty[c]+' | '+' | '.join(f'{f[g]["constrained_ba"]:.3f}' for g in ('simple','hard'))+' | '+' | '.join(f'{100*f[g]["prediction_disagreement"]:.2f}%' for g in ('simple','hard'))+' |')
lines += ['',
'困难图固定均值和self-only的测试预测与原模型逐条一致，不只是总体BA碰巧相同。简单图固定均值或均匀权重仍保持BA，self-only却明显下降：前面的固定协议值与自身值的组合/幅度很重要，不需要因此认定动态门控必需。', '',
'均匀权重是一个反例：困难图FFN后后代probe仍0.946，答案BA却降到0.713。概念仍可被重新训练的probe读出，不保证原后续网络沿用它。', '',
'## 严格仿射条件：提升不是新增的理论线性可表示性', '',
'固定attention混合权重，再把起点norm1的逐节点均值和标准差换成训练节点上的常数，保留已训练的gamma/beta、V/O矩阵、固定协议值和residual。此时attention后的表示严格为 h=Ae+b；原FFN和后续LayerNorm/GELU仍保留，因此没有把整个模型变成线性网络。', '',
'该A在所有节点划分与两张图中均为128维满秩。最大仿射重构误差<2e-5。困难图仅这一可逆仿射变换，就让新拟合probe从原E的0.701变为0.850；FFN后0.947、全测试答案BA0.974。', '',
'可逆仿射变换不扩大所有仿射解码器能表达的函数类。它会改变有限样本、坐标标准化及ridge正则下的学习条件与隐式先验；A本身来自训练后的模型参数，不能当作无知识的随机坐标变换，也不能据此定位图事实只在embedding或只在A。', '',
'为了验证这一点，把仿射条件的探针权重解析折回原E：若 s=(Ae+b)ᵀw+c，则原E上使用 Aᵀw 和新偏置即可获得相同分数。实际已验证最大分数误差<1e-3，不使用测试标签重新拟合。原E上折回探针后代AUROC：']
for g in ('simple','hard'):
    values = [r['transported_raw_embedding_probe']['auc'] for r in DATA[g]['relations']['descendants']]
    errors = [r['transport_max_score_error'] for r in DATA[g]['relations']['descendants']]
    lines.append(f'- {g}：{np.mean(values):.3f}，最大分数重构误差 {max(errors):.2g}。')
lines += ['',
'所以“attention提高可解码性”在这里应该解释为当前训练流程下的probe性能改善，不能当作attention新生成了原E中不存在的线性信息。尤其困难起点，主要机制是接近固定的self V/O变换与表示重整，动态权重非线性没有得到必要性支持。', '',
'## 验证与边界', '',
'- native动态条件的E/attention后/FFN后probe与上一轮原数据对照，划分和指标不变。',
'- 均值及常数norm统计在probe train内估计；折回权重无需test标签；每个条件都重算FFN与后续层。',
'- 数据只覆盖各一张固定图、一个Direct checkpoint；不推到PATH、其他位置/层或其他seed。',
'- 阴性消融不排除冗余路线；正面readout不证明后代方向是答案中介。', '',
'![固定attention对照](fixed_attention.png)', '',
'复现：repo root 执行 `.venv/bin/python studies/fixed_attention/run.py`，再执行同目录 report.py。两个JSON、一个PNG和本报告；无中间激活或权重文件。', '']
(HERE/'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')
print('Wrote REPORT.md and fixed_attention.png')
