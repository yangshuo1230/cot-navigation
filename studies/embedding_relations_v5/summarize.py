"""Summarize node-heldout decoding with frequency/layer controls and set recovery."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

HERE = Path(__file__).resolve().parent
ds = [json.loads((HERE / f'seed{s}.json').read_text()) for s in range(4)]
tasks = ['ancestors', 'reachable_descendants', 'direct_successors']
names = ['祖先集合', '可达后代集合', '直接后继集合']
feature_groups = {
    'embedding': ['embedding'],
    'random_identity': [f'random_identity_{i}' for i in (101, 202, 303)],
    'layer_oracle': ['layer_onehot_oracle'],
    'candidate_prior': ['candidate_prior_only'],
    'layer_shuffled_train': [f'within_layer_shuffled_train_{i}' for i in (101, 202, 303)]}
metrics = {
    'column_auc': ('full_candidate_column_auc', 'auc', None),
    'feasible_column_auc': ('feasible_column_auc', 'auc', None),
    'full_row_auc': ('full_row_ranking', 'auc', 'informative_rows'),
    'feasible_row_auc': ('feasible_row_ranking', 'auc', 'informative_rows'),
    'full_recall': ('full_row_ranking', 'recall_at_true_k', 'informative_rows'),
    'feasible_recall': ('feasible_row_ranking', 'recall_at_true_k', 'informative_rows'),
    'feasible_uniform_recall': ('feasible_row_ranking', 'uniform_recall', 'informative_rows'),
    'feasible_score_margin': ('feasible_row_ranking', 'mean_positive_minus_negative_score', 'informative_rows'),
    'exact_set': ('full_set_recovery', 'exact_set', 'n_nodes'),
    'nonempty_exact_set': ('full_set_recovery', 'nonempty_exact_set', 'nonempty_nodes'),
    'macro_f1': ('full_set_recovery', 'macro_f1', 'n_nodes')}


def aggregate(task, features, group, key, weight):
    per_model = []
    for d in ds:
        assignments = []
        for node_seed in (17, 23, 31):
            control_seeds = []
            for feature in features:
                entries = [r['features'][feature][task][group] for r in d['folds'] if r['node_seed'] == node_seed]
                entries = [r for r in entries if r[key] is not None]
                values = [r[key] for r in entries]
                weights = [r[weight] for r in entries] if weight else np.ones(len(entries))
                control_seeds.append(float(np.average(values, weights=weights)))
            assignments.append(float(np.mean(control_seeds)))
        per_model.append(float(np.mean(assignments)))
    return {'mean': float(np.mean(per_model)), 'min': min(per_model), 'max': max(per_model), 'per_model': per_model}


summary = {}
for task in tasks:
    summary[task] = {}
    for label, features in feature_groups.items():
        summary[task][label] = {name: aggregate(task, features, *spec) for name, spec in metrics.items()}
(HERE / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')

lines = ['# PATH v5 raw embedding：节点留出集合解码', '',
    '对象：当前一张80节点固定图上4个PATH初始化的最后50k checkpoint，仅取每个实际图节点的一条原始128维embedding，无QUERY、上下文或FFN。', '',
    '## 划分和解码器', '',
    '- 每个节点划分seed17/23/31做5fold轮换：48个焦点节点训练、16验证、16测试，每个节点在每个轮换中恰好测试一次。分配按层打散，所有fold均覆盖10层。4模型×3划分×5fold是重复评估，唯一节点仍只有80个、图只有1张。',
    '- 测试焦点embedding和它的集合标签从不用于探针拟合、标准化、选ridge或阈值。80个输出候选身份固定，训练节点的标签集合可以包含测试焦点节点作为候选；这是输入焦点节点留出，不是输入和输出身份同时未见。原主模型见过全部节点。',
    '- 仿射ridge128→80，FP64闭式dual解；train-only均值/标准差，val MSE从1e-4到100共7档选择每任务一个正则。非线性解码器未测试。阈值另由val集合micro-F1选择，test不调参；线性分数不称概率。',
    '- 祖先j满足j→i，可达后代j满足i→j，直接后继j满足边i→j；均排除自身。BFS与独立拓扑DP逐对一致。原LM的embed与输出头绑权，本次仍只读取embedding矩阵。', '',
    '## 控制与指标', '',
    '- 固定128维高斯随机身份向量3seed；oracle层级one-hot；仅训练候选节点的出现频率（完全不输入焦点）；在同层内打乱训练节点的标签集合3seed，保持层级/候选频率，val/test仍用真实标签。',
    '- 固定候选跨焦点AUROC：每列候选分别在本fold的16个测试焦点上区分有/无关系，再列平均，最后fold平均。不能把不同fold预测分数跨行拼接算列AUROC。',
    '- 焦点内AUROC和Top-k Recall：区分一个焦点集合中哪些候选正确。主诊断排除明显不可能的层级：祖先只比更浅层、后代只比更深层、直接后继只比下一层。这个mask只用于评分，不是给原模型的输入。',
    '- Top-k的k是真实集合大小，是排名诊断而非自主集合输出。空集合/候选全正的行排除排名并记录数量。完整集合恢复则不使用层级mask、只排除自身，评价val选阈值后的全体候选；单报非空集合，避免只会判空也拿分。', '',
    '## 主结果（模型均值，内部先按节点划分聚合）', '',
    '|集合|固定候选AUROC|可行候选内AUROC|可行候选内Top-k召回|非空集合完整恢复|',
    '|---|---:|---:|---:|---:|']
for task, name in zip(tasks, names):
    v = summary[task]['embedding']
    lines.append(f"|{name}|{v['column_auc']['mean']:.3f}|{v['feasible_row_auc']['mean']:.3f}|"
                 f"{100*v['feasible_recall']['mean']:.2f}%|{100*v['nonempty_exact_set']['mean']:.2f}%|")
lines += ['', '## 排除层级捷径后的AUROC对照', '',
          '|集合|embedding|随机身份|仅层级|仅候选频率|同层打乱训练标签|', '|---|---:|---:|---:|---:|---:|']
for task, name in zip(tasks, names):
    lines.append(f'|{name}|' + '|'.join(f"{summary[task][f]['feasible_row_auc']['mean']:.3f}" for f in feature_groups) + '|')
lines += ['', '## 固定候选跨焦点AUROC对照', '',
          '|集合|embedding|随机身份|仅层级|仅候选频率|同层打乱训练标签|', '|---|---:|---:|---:|---:|---:|']
for task, name in zip(tasks, names):
    lines.append(f'|{name}|' + '|'.join(f"{summary[task][f]['column_auc']['mean']:.3f}" for f in feature_groups) + '|')
lines += ['', '## 应怎样解释', '',
          '祖先具有较明确的焦点相关线性信号：跨焦点AUROC及层级约束内排名均超过随机身份、层级和候选频率控制。但这仍是部分可读出性，不是整张祖先表被完整恢复。', '',
          '可达后代也有跨焦点线性信号，超过随机身份；不过具体集合成员的行内排名与候选频率先验非常接近。因此高行内AUROC不能直接归因于embedding包含完整后代集合，提升中有较强的候选共性和层级成分。', '',
          '直接后继有弱的节点特异线性信号：在下一层候选中优于随机身份/层级/候选频率，但全图Top-k召回和完整集合恢复均有限。固定候选AUROC甚至低于层级oracle，所以只报告该分数会高估邻接表编码。', '',
          '三类集合都不能称为已被该探针完整解码。完整恢复低也不能证明embedding没有信息：48个训练节点、有限数据、ridge归纳偏置、阈值选择及可能的非线性编码都限制这个结论；可解码性也不等于模型实际使用这些方向。', '',
          '## 复现与文件', '',
          'run.py读取4个固定checkpoint，所有拟合GPU0、无主模型前向或更新；没有保存activation、探针权重、模型checkpoint或W&B run。seed0–3.json保存每fold节点ID、正则、阈值、指标及有效样本数；summary.json聚合；key_results.png/.pdf为同一关键图。', '',
          '行排名在一个节点分配的5个测试fold间按有效焦点数加权，随后平均3个分配及4模型；完整恢复按节点数/非空节点数加权；列AUROC在fold内宏平均，再平均fold，不混拼不可校准的跨fold分数。']
(HERE / 'REPORT.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')

font = Path('C:/Windows/Fonts/msyh.ttc')
if font.exists():
    font_manager.fontManager.addfont(str(font))
    plt.rcParams['font.family'] = font_manager.FontProperties(fname=str(font)).get_name()
plt.rcParams['axes.unicode_minus'] = False
styles = [('embedding', '真实 embedding', '#3973a9'), ('random_identity', '随机身份向量', '#94a3b8'),
          ('layer_oracle', '仅层级', '#dc8134'), ('candidate_prior', '仅候选频率', '#786f6a'),
          ('layer_shuffled_train', '同层打乱训练标签', '#9259aa')]
fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.4))
plots = [('column_auc', '固定候选跨留出焦点 AUROC', False, (.4, 1.02)),
         ('feasible_row_auc', '排除不可能层级后的集合内 AUROC', False, (.4, 1.02)),
         ('feasible_recall', '可行候选内 Top-k 召回（k 为真实集合大小）', True, (0, 102)),
         ('nonempty_exact_set', '非空集合完整恢复率：全候选、验证集选阈值', True, (0, 10))]
x = np.arange(3)
for ax, (metric, title, percent, ylim) in zip(axes.ravel(), plots):
    factor = 100 if percent else 1
    for j, (feature, label, color) in enumerate(styles):
        means = np.array([summary[t][feature][metric]['mean'] for t in tasks])*factor
        low = means - np.array([summary[t][feature][metric]['min'] for t in tasks])*factor
        high = np.array([summary[t][feature][metric]['max'] for t in tasks])*factor - means
        bars = ax.bar(x + (j-2)*.15, means, width=.145, label=label, color=color,
                      yerr=np.array([low, high]), capsize=2, error_kw={'lw': .8})
        if metric == 'nonempty_exact_set' and feature == 'embedding':
            ax.bar_label(bars, fmt='%.1f', padding=4, fontsize=9)
    ax.set_xticks(x, names, fontsize=10)
    ax.set_title(title, fontsize=11)
    ax.set_ylim(*ylim)
    ax.set_ylabel('比例（%）' if percent else 'AUROC')
    if not percent:
        ax.axhline(.5, color='#b8bec5', ls=':', lw=1)
    ax.grid(axis='y', alpha=.15)
    ax.set_axisbelow(True)
    ax.spines[['top', 'right']].set_visible(False)
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(.5, .945), ncol=5, frameon=False, fontsize=10)
fig.suptitle('PATH 原始 embedding 的结构解码：焦点节点严格留出 · 4模型 · 3种五折划分', fontsize=15, y=.985)
fig.text(.5, .025, '每折 48训练 /16验证 /16测试；主模型见过全部节点。误差线为模型均值范围，非置信区间；未输入查询或上下文。',
         ha='center', fontsize=9.5, color='#475569')
fig.subplots_adjust(top=.86, bottom=.105, left=.065, right=.985, hspace=.38, wspace=.22)
fig.savefig(HERE / 'key_results.png', dpi=190)
fig.savefig(HERE / 'key_results.pdf')
for task in tasks:
    print(task, {f: {m: round(summary[task][f][m]['mean'], 4) for m in ('column_auc', 'feasible_row_auc', 'feasible_recall', 'nonempty_exact_set')}
                 for f in feature_groups})
