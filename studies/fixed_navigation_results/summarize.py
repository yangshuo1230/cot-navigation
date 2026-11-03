"""Aggregate seeds without treating repeated queries as independent graph draws."""
from collections import Counter
import json
from pathlib import Path
import statistics as stats

HERE = Path(__file__).resolve().parent
data = json.loads((HERE / 'collected.json').read_text())
runs = data['runs']
panels = list(runs[0]['final']['panels'])
fields = ['yes_recall', 'no_recall', 'ba', 'positive_valid_goal_rate']
labels = dict(short_test='短程1–3跳', long_AA_test='长程AA：端点已见、查询留出',
              long_TA_test='长程TA：起点角色留出', long_AT_test='长程AT：目标角色留出',
              long_TT_test='长程TT：两端角色留出')
summary = {}
for name in panels:
    metrics = [r['final']['panels'][name]['metrics']['all'] for r in runs]
    entry = {'n_per_seed': metrics[0]['n'], 'positive_n_per_seed': metrics[0]['n1'],
             'negative_n_per_seed': metrics[0]['n0']}
    for field in fields:
        values = [m[field] for m in metrics]
        entry[field] = {'mean': stats.mean(values), 'min': min(values), 'max': max(values),
                        'sample_sd': stats.stdev(values)}
    entry['legal_trace_rate_mean'] = stats.mean(m['legal'] / m['n'] for m in metrics)
    errors = Counter()
    for r in runs:
        errors.update(r['final']['panels'][name]['errors'])
    entry['errors_across_seed_query_evaluations'] = dict(errors)
    entry['witness_subgroups'] = {}
    for subgroup in ('witness_seen', 'witness_unseen'):
        ms = [r['final']['panels'][name]['metrics'].get(subgroup) for r in runs]
        if all(ms):
            entry['witness_subgroups'][subgroup] = {
                'n_per_seed': ms[0]['n'],
                'positive_answer_accuracy_mean': stats.mean(m['yes_recall'] for m in ms),
                'valid_goal_rate_mean': stats.mean(m['positive_valid_goal_rate'] for m in ms)}
    summary[name] = entry

tt_seed = [{"seed": r['seed'], **r['final']['panels']['long_TT_test']['metrics']['all']} for r in runs]
tt_gaps = {}
for gap in ('gap4-5', 'gap6-7', 'gap8-9'):
    ms = [r['final']['panels']['long_TT_test']['metrics'][gap] for r in runs]
    tt_gaps[gap] = {'positive_n': ms[0]['n1'], 'negative_n': ms[0].get('n0', 0),
                    'yes_recall': stats.mean(m['yes_recall'] for m in ms),
                    'no_recall': stats.mean(m['no_recall'] for m in ms) if ms[0]['no_recall'] is not None else None,
                    'valid_goal': stats.mean(m['positive_valid_goal_rate'] for m in ms)}
result = {'scope': data['scope'], 'panels': summary, 'TT_by_seed': tt_seed, 'TT_by_gap': tt_gaps}
(HERE / 'summary.json').write_text(json.dumps(result, indent=2), encoding='utf-8')

lines = ['# PATH joint-training pilot: 4 initialization seeds', '',
    '全部4个seed完成50k步，无失败；仅一张80节点图，不能用初始化重复代替图重复。模型6层、1头、hidden128、FFN512；随机初始化、单阶段，每步48短QUERY+16长QUERY+16完整ADJ。最后固定50k checkpoint，未按test挑选。', '',
    '## 最终测试：每seed先统计，再平均', '',
    '|面板|正/负query数（每seed）|正例答案|负例答案|BA|正例合法到达|',
    '|---|---:|---:|---:|---:|---:|']
for name, v in summary.items():
    lines.append(f"|{labels[name]}|{v['positive_n_per_seed']}/{v['negative_n_per_seed']}|" +
                 '|'.join(f"{100*v[f]['mean']:.2f}%" for f in fields) + '|')
lines += ['', '合法到达要求从正确起点开始、所有输出节点及每一条边合法、最后一个路径节点为目标；此指标不额外要求最终YES也正确，答案与轨迹分别计分。合法路径本身不能证明NO。', '',
          '## 双端留出：初始化差异', '', '|seed|正例答案|负例答案|BA|正例合法到达|', '|---|---:|---:|---:|---:|']
for row in tt_seed:
    lines.append(f"|{row['seed']}|" + '|'.join(f"{100*row[f]:.2f}%" for f in fields) + '|')
lines += ['', '## 双端留出：长度分组', '', '|距桶|正/负数|正例答案|负例答案|正例合法到达|', '|---|---:|---:|---:|---:|']
for gap, m in tt_gaps.items():
    no = f"{100*m['no_recall']:.2f}%" if m['no_recall'] is not None else 'NA'
    lines.append(f"|{gap}|{m['positive_n']}/{m['negative_n']}|{100*m['yes_recall']:.2f}%|{no}|{100*m['valid_goal']:.2f}%|")
lines += ['', '8–9跳TT只有12个唯一正例、没有负例，不能以该桶准确率声称同时泛化了长程不可达判断。AA只有7个负例，估计同样较粗。', '',
          '## 实际暴露与解释边界', '',
          '四个seed的实际训练计数相同：每个模型320万次QUERY（1508个不同query，4398个不同query/variant），80万次完整ADJ样本。短程留出query仍可能从其他训练轨迹获得完整路径，这是预期的查询留出定义。', '',
          'TT的57个正例中，8个在其他训练轨迹的子路径中暴露过完整witness，49个未暴露。四个seed均为此划分。未暴露组的合法到达率平均' +
          f"{100*summary['long_TT_test']['witness_subgroups']['witness_unseen']['valid_goal_rate_mean']:.2f}%，" +
          '正例答案准确率平均' + f"{100*summary['long_TT_test']['witness_subgroups']['witness_unseen']['positive_answer_accuracy_mean']:.2f}%。" +
          '已暴露组合法到达均100%。未暴露组在已披露关系宏图中39个需2段、10个需3段；这支持有限未见路径组合，不足以证明通用在线搜索。该分组不是随机化因果对照。', '',
          'TT的4×57次正例执行共有14次未能合法到达：9次进入不能到目标的分支、3次非法token/边、2次仍处于有用前缀却错误停止。AT目标角色留出更弱：488次正例执行有43次未合法到达，其中32次错误分支、7次有用前缀错误停止、2次越过目标、2次非法token/边。这里计数是seed×query执行，不是新的独立query。', '',
          '## 学习过程', '',
          '约5k步时四个seed的固定训练诊断面板（短/长各最多64条）答案BA和正例合法到达均100%，不是完整训练池评估。全图80节点的ADJ最终完整集合恢复率均100%。最终答案CE约1e-6至2e-5，轨迹CE约0.161至0.163；多合法变体让轨迹loss非零，不能据此判定任务欠拟合。', '',
          '留出端点的验证表现并不随步数单调提升。10k时VV验证BA跨seed平均约95.9%，50k约91.9%；VV合法到达平均约90.8%下降至86.0%。这是后期泛化退化/波动的信号，不能据此预言更长训练一定改善。未用验证最好点重选测试checkpoint，20k checkpoint可供后续预先指定对照。', '',
          '相较旧R0/R3，图规模、监督长度、输出协议与混合设置都发生改变，不能把提升归因于某一个改动。当前结果证明这一80节点设置可学，下一步若扩展，应冻结规则并在多张160节点图上复验。', '',
          '完整原始汇总collected.json；聚合summary.json；唯一关键图results.png（另有PDF同图）。本次只读已有结果，没有重新训练、加载权重或生成激活文件。']
(HERE / 'REPORT.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.3), layout='constrained')
x = np.arange(len(panels))
for j, (field, label, color) in enumerate([
    ('yes_recall', 'Positive answer', '#3973a9'), ('no_recall', 'Negative answer', '#dc8a39'),
    ('positive_valid_goal_rate', 'Positive legal goal path', '#38936a')]):
    means = [100*summary[p][field]['mean'] for p in panels]
    axes[0].bar(x + (j-1)*.24, means, width=.23, label=label, color=color)
axes[0].set_xticks(x, ['Short', 'Long AA', 'Long TA', 'Long AT', 'Long TT'])
axes[0].set_ylim(0, 105)
axes[0].set_ylabel('Rate (%)')
axes[0].set_title('Final step50k test: mean over4 initializations')
axes[0].legend(loc='lower left', fontsize=8)
for name, label, color in [('short_val', 'Short query holdout', '#3973a9'),
                            ('long_AA_val', 'Long AA query holdout', '#dc8a39'),
                            ('long_AV_val', 'Long AV target holdout', '#8d62a8'),
                            ('long_VV_val', 'Long VV both-endpoint holdout', '#38936a')]:
    steps = [r['step']/1000 for r in runs[0]['validation_curve']]
    values = np.array([[100*c['panels'][name]['ba'] for c in r['validation_curve']] for r in runs])
    axes[1].plot(steps, values.mean(0), label=label, color=color)
    if name == 'long_VV_val':
        axes[1].fill_between(steps, values.min(0), values.max(0), color=color, alpha=.15,
                             label='VV min-max over initializations')
axes[1].set(xlabel='Optimizer updates (thousands)', ylabel='Validation balanced accuracy (%)',
            ylim=(40, 102), title='Validation: longer training is not uniformly better')
axes[1].legend(loc='lower right', fontsize=7.5)
for ax in axes:
    ax.grid(axis='y', alpha=.18)
    ax.set_axisbelow(True)
fig.savefig(HERE / 'results.png', dpi=180)
fig.savefig(HERE / 'results.pdf')
print(json.dumps(summary, indent=2))
