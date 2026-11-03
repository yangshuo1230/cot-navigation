"""Plot saved final evaluations by exact query distance, without new inference."""
import json
from pathlib import Path
import statistics

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

HERE = Path(__file__).resolve().parent
data = json.loads((HERE / 'collected.json').read_text())
runs = data['runs']
font = Path('C:/Windows/Fonts/msyh.ttc')
if font.exists():
    font_manager.fontManager.addfont(str(font))
    plt.rcParams['font.family'] = font_manager.FontProperties(fname=str(font)).get_name()
plt.rcParams['axes.unicode_minus'] = False

settings = [
    ('short_test', '短程：仅查询留出', '#64748b'),
    ('long_AA_test', '长程 AA：端点已见', '#3973b7'),
    ('long_TA_test', '长程 TA：起点角色留出', '#26936b'),
    ('long_AT_test', '长程 AT：目标角色留出', '#dd8135'),
    ('long_TT_test', '长程 TT：双端角色留出', '#9259aa'),
]
metrics = [('yes_recall', '正例答案准确率'), ('no_recall', '负例答案准确率'),
           ('positive_valid_goal_rate', '正例路径合法且到达目标')]
export = {'checkpoint': 50000, 'graph_nodes': 80, 'model_seeds': [r['seed'] for r in runs],
          'notes': ['Negative examples have no finite shortest path; x is positive layer gap.',
                    'Thin lines are individual initialization seeds; thick lines are arithmetic means.',
                    'Missing classes are null, not zero or perfect accuracy.'], 'panels': {}}

fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.05))
for panel, label, color in settings:
    export['panels'][panel] = {}
    for hop in range(1, 10):
        cells = [r['final']['panels'][panel]['metrics'].get(f'gap{hop}') for r in runs]
        if not any(cells):
            continue
        assert all(cells), 'Distance support changed across seeds'
        assert len({(m.get('n1', 0), m.get('n0', 0)) for m in cells}) == 1
        record = {'positive_n_per_seed': cells[0].get('n1', 0),
                  'negative_n_per_seed': cells[0].get('n0', 0), 'metrics': {}}
        for field, _ in metrics:
            values = [m[field] for m in cells]
            record['metrics'][field] = {
                'mean': statistics.mean(values) if all(v is not None for v in values) else None,
                'by_seed': values}
        export['panels'][panel][str(hop)] = record
    hops = np.array([int(h) for h in export['panels'][panel]])
    for ax, (field, title) in zip(axes, metrics):
        means = []
        for hop in hops:
            value = export['panels'][panel][str(hop)]['metrics'][field]['mean']
            means.append(100 * value if value is not None else np.nan)
        for seed_index in range(len(runs)):
            values = [export['panels'][panel][str(h)]['metrics'][field]['by_seed'][seed_index] for h in hops]
            ax.plot(hops, [100 * v if v is not None else np.nan for v in values],
                    color=color, alpha=.18, linewidth=.85, zorder=1)
        ax.plot(hops, means, color=color, marker='o', markersize=4.5,
                linewidth=2.3, label=label, zorder=3)

for ax, (_, title) in zip(axes, metrics):
    ax.set_title(title, fontsize=13, pad=10)
    ax.set(xlim=(.7, 9.3), ylim=(0, 104), xticks=range(1, 10),
           yticks=range(0, 101, 20), xlabel='查询跳数', ylabel='比例（%）')
    ax.axvline(3.5, color='#94a3b8', linestyle='--', linewidth=.9)
    ax.grid(axis='y', alpha=.18)
    ax.spines[['top', 'right']].set_visible(False)
    ax.text(2, 5, '短程', color='#64748b', ha='center', fontsize=9)
    ax.text(6.5, 5, '长程', color='#64748b', ha='center', fontsize=9)
axes[1].axvspan(8.65, 9.3, color='#f1f5f9', zorder=0)
axes[1].text(9, 47, '无\n负\n例', color='#64748b', ha='center', fontsize=10)
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(.5, .075),
           ncol=5, frameon=False, fontsize=9.3, columnspacing=1.25)
fig.suptitle('PATH：查询跳数与最终答案、合法到达率', fontsize=16, y=.98)
fig.text(.5, .905, '同一张 80 节点图 · 最后 50k 步 checkpoint · 粗线：4 个初始化均值；细线：各初始化',
         ha='center', fontsize=10.5, color='#475569')
fig.text(.5, .032, '负例按正向层间距分桶；缺失类别留空。9 跳各组均无负例，AT/TT 在 8 跳也无负例。',
         ha='center', fontsize=9.5, color='#475569')
fig.subplots_adjust(left=.055, right=.988, top=.82, bottom=.24, wspace=.25)
fig.savefig(HERE / 'results_by_hops.png', dpi=200)
fig.savefig(HERE / 'results_by_hops.pdf')
(HERE / 'results_by_hops.json').write_text(json.dumps(export, ensure_ascii=False, indent=2), encoding='utf-8')
print('Saved results_by_hops.png/.pdf/.json')
