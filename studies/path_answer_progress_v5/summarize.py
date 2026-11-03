"""Summarize answer accessibility along PATH; all plotted metrics are probes."""
import json
from pathlib import Path
import statistics as st

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

HERE = Path(__file__).resolve().parent
ds = [json.loads((HERE / f'seed{i}.json').read_text()) for i in range(4)]
assert all(d['checks']['features_from_prefix_only_forwards'] for d in ds)
assert all(all(d['checks']['native_replay_matches_final'].values()) for d in ds)
assert all(d['checks']['source_copy_fraction'] == 1 for d in ds)
font = Path('C:/Windows/Fonts/msyh.ttc')
if font.exists():
    font_manager.fontManager.addfont(str(font))
    plt.rcParams['font.family'] = font_manager.FontProperties(fname=str(font)).get_name()
plt.rcParams['axes.unicode_minus'] = False
FIELDS = ['yes_recall', 'no_recall', 'ba', 'auroc']
STAGES = ['生成前', '复制起点', '25%', '50%', '75%', '轨迹末尾']
FIXED = ['生成前', '复制起点', '续写1节点', '续写2节点', '续写3节点', '续写4节点']


def collect(experiment, panel='all', scheme='query_only', feature='state_L6'):
    curves = []
    counts = []
    for d in ds:
        names = [scheme] if scheme == 'query_only' else [f'{scheme}_{fold}' for fold in (17, 23, 31)]
        subcurves = []
        for name in names:
            feats = [feature] if feature != 'random_endpoints' else [f'random_endpoints_{seed}' for seed in (101, 202, 303)]
            for f in feats:
                entry = d['experiments'][experiment]['schemes'][name]['features'][f]
                if 'skipped' in entry or panel not in entry['metrics']:
                    continue
                rows = entry['metrics'][panel]
                if any(m['ba'] is None for m in rows):
                    continue
                npos = {m['n_positive'] for m in rows}
                nneg = {m['n_negative'] for m in rows}
                assert len(npos) == len(nneg) == 1, 'Changing cohorts across anchors'
                counts.append({'seed': d['seed'], 'scheme': name, 'positive': next(iter(npos)), 'negative': next(iter(nneg))})
                subcurves.append([[r[field] for field in FIELDS] for r in rows])
        curves.append(np.asarray(subcurves).mean(0))
    values = np.asarray(curves)
    return {'fields': FIELDS, 'mean': values.mean(0).tolist(),
            'min': values.min(0).tolist(), 'max': values.max(0).tolist(), 'counts': counts,
            'per_model': values.tolist()}


summary = {'relative': {}, 'fixed_long_cohort': {}, 'strict': {}}
for panel in ['all', 'short_test', 'long_AA_test', 'long_TA_test', 'long_AT_test', 'long_TT_test']:
    summary['relative'][panel] = {f: collect('relative', panel, feature=f)
                                 for f in ('state_L6', 'position_only', 'position_goal', 'endpoint_embeddings', 'random_endpoints')}
for f in ('state_L6', 'position_only', 'position_goal', 'endpoint_embeddings', 'random_endpoints'):
    summary['fixed_long_cohort'][f] = collect('fixed_long_cohort', feature=f)
for scheme in ('target_node_heldout', 'both_node_heldout'):
    summary['strict'][scheme] = {f: collect('relative', scheme=scheme, feature=f)
                               for f in ('state_L6', 'position_only', 'position_goal', 'endpoint_embeddings', 'random_endpoints')}
summary['shuffled_BA'] = np.mean([[v['ba'] for v in d['experiments']['relative']['shuffled_labels']] for d in ds], axis=0).tolist()
(HERE / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')

BLUE, ORANGE, GREEN, GRAY, PURPLE = '#3973a9', '#dc8134', '#26936b', '#7d8794', '#9259aa'
fig, axes = plt.subplots(2, 2, figsize=(13.8, 9.5))
def line(ax, curve, field, label, color, style='-', shade=False):
    field_id = FIELDS.index(field)
    yy = np.asarray(curve['mean'])[:, field_id] * 100
    if len(yy) == 1:
        ax.axhline(yy[0], color=color, ls=style, label=label, lw=1.5)
    else:
        ax.plot(range(len(yy)), yy, marker='o', ms=4, lw=1.9, color=color, ls=style, label=label)
        if shade:
            ax.fill_between(range(len(yy)), 100*np.asarray(curve['min'])[:, field_id],
                            100*np.asarray(curve['max'])[:, field_id], color=color, alpha=.1)

for ax, panel, title in [(axes[0, 0], 'all', '全部测试：相同602条查询'),
                         (axes[0, 1], 'long_TT_test', '长程双端角色留出：相同84条查询')]:
    s = summary['relative'][panel]
    line(ax, s['state_L6'], 'yes_recall', '正例答案读出', BLUE)
    line(ax, s['state_L6'], 'no_recall', '负例答案读出', ORANGE)
    line(ax, s['state_L6'], 'ba', '答案探针 BA', GREEN, shade=True)
    line(ax, s['position_goal'], 'ba', '位置＋目标可见性对照 BA', GRAY, '--')
    ax.set_title(title, fontsize=12)
    ax.set_xticks(range(6), STAGES, fontsize=10)
    ax.set_xlabel('回顾性相对轨迹进度（不输入未来答案）')

ax = axes[1, 0]
s = summary['fixed_long_cohort']
npos, nneg = s['state_L6']['counts'][0]['positive'], s['state_L6']['counts'][0]['negative']
assert {(v['positive'], v['negative']) for v in s['state_L6']['counts']} == {(npos, nneg)}
line(ax, s['state_L6'], 'yes_recall', '正例答案读出', BLUE)
line(ax, s['state_L6'], 'no_recall', '负例答案读出', ORANGE)
line(ax, s['state_L6'], 'ba', '答案探针 BA', GREEN, shade=True)
line(ax, s['position_goal'], 'ba', '位置＋目标可见性对照 BA', GRAY, '--')
ax.set_title(f'固定长轨迹子集：相同{npos+nneg}条查询（+{npos}/−{nneg}）', fontsize=12)
ax.set_xticks(range(6), FIXED, fontsize=10)
ax.set_xlabel('复制起点后增加的生成节点数；每个位置查询集合不变')

ax = axes[1, 1]
s = summary['strict']['both_node_heldout']
line(ax, s['state_L6'], 'ba', '答案探针 BA', GREEN, shade=True)
line(ax, s['position_goal'], 'ba', '位置＋目标可见性对照 BA', GRAY, '--')
line(ax, s['random_endpoints'], 'ba', '固定随机端点输入 BA', PURPLE, ':')
ax.set(title='外部探针额外双端点节点留出：3个划分均值', xlabel='回顾性相对轨迹进度')
ax.set_xticks(range(6), STAGES, fontsize=10)
for ax in axes.ravel():
    ax.set(ylim=(0, 104), ylabel='类别准确率 / BA（%）')
    ax.axhline(50, color='#d1d5db', ls=':', lw=.8, zorder=0)
    ax.grid(axis='y', alpha=.15)
    ax.spines[['top', 'right']].set_visible(False)
    ax.legend(fontsize=9, loc='lower right')
fig.suptitle('PATH 展开过程中的答案可读出性：4个初始化 · 50k checkpoint', fontsize=16, y=.98)
fig.text(.5, .025, '冻结 L6 状态上的独立线性探针，并非原模型被截断后直接回答。阴影为初始化最小—最大范围；复制起点尚未沿边移动。',
         ha='center', fontsize=10, color='#475569')
fig.subplots_adjust(left=.065, right=.985, top=.915, bottom=.10, hspace=.43, wspace=.22)
fig.savefig(HERE / 'answer_progress.png', dpi=190)
fig.savefig(HERE / 'answer_progress.pdf')

lines = ['# PATH v5：答案读出随生成轨迹展开', '',
    '本轮使用同一张80节点图的4个50k checkpoint。仅冻结前向与GPU0外部线性探针，没有继续训练原模型，未保存激活、探针权重、模型checkpoint或新W&B run。', '',
    '## 上一张后继选择图的总结', '',
    '1. 距目标至少2跳且存在有用/无用混合后继时，原生生成选择有用后继98.4%，均匀后继基线50.9%；不是只复制目标。',
    '2. 原输出头的真实后继概率从L1F约7%逐步升至L6F约99%，有用后继条件偏好也逐层提高；目前更符合共同增强的现象，不能确定为候选检索/目标筛选两个独立阶段。',
    '3. 有用后继探针在额外目标节点留出时AUROC从E约0.71升至L6F约0.95，随机目标身份对照约0.63。仅query留出时随机身份也能取得高分，不能忽略探针查表。', '',
    '## 新实验的读数含义', '',
    '- 输入QUERY s t |，沿每个checkpoint自己生成的原轨迹取前缀。包含错误轨迹，不替换为oracle路径、不按最终正确筛选。所有前缀排除YES/NO；独立prefix-only forward，没有把完整未来轨迹喂入后再取过去位置。',
    '- 只用最后层FFN后的128维残差，仿射ridge读出可达性。每个锚点独立拟合；train-only标准化、类别平衡拟合，原query-val AUROC选正则(.001/.01/.1/1/10)，val BA选阈值；test只汇报。概率阈值操作的是未校准线性分数，不是原模型输出概率。',
    '- 图上准确率是外部探针可读出性，不是原模型在任意前缀被要求直接回答的能力。原输出头在中间主要被训练去生成下一个节点。',
    '- 相对进度按ceil(最终生成轨迹token数×比例)取点，属于回顾性分析；短轨迹可能多个点重合。位置对照仅用已见token数及其平方；位置＋目标对照再加入当前是否目标、目标是否已在可见轨迹中出现。均不输入未来长度或图oracle标签。', '',
    '## 相同测试查询的相对进度', '',
    '|锚点|全部正例准确率|全部负例准确率|全部BA|位置对照BA|位置＋目标对照BA|长程TT BA|',
    '|---|---:|---:|---:|---:|---:|---:|']
main = summary['relative']['all']
tt = summary['relative']['long_TT_test']['state_L6']['mean']
for phase, label in enumerate(STAGES):
    r = main['state_L6']['mean'][phase]
    lines.append(f'|{label}|' + '|'.join(f'{100*v:.2f}%' for v in
        (r[0], r[1], r[2], main['position_only']['mean'][phase][2],
         main['position_goal']['mean'][phase][2], tt[phase][2])) + '|')
lines += ['', '全部测试602个唯一query（390正/212负）；TT为84个（57正/27负）。每阶段集合一致，先每模型统计再跨4初始化平均。所有模型第一个生成token都是复制起点。', '',
          '终点高分不能单独证明计算机制：位置＋目标可见性对照在末尾同样很高，甚至高于答案探针。中途的相对进度还含已见长度与标签分布的关联，所以另做固定步数对照。', '',
          '## 相同查询、固定绝对续写步数', '',
          f'固定一组所有4个模型均生成至少5个轨迹token、原query层间距≥4的查询；每模型每锚点相同{npos+nneg}个test（{npos}正/{nneg}负）。这是按未来长度选择的长轨迹敏感性子集，不是全部任务，也不按答案正确与否筛选。主模型自然生成，续写节点不强制沿合法边。', '',
          '|锚点|正例准确率|负例准确率|BA|位置＋目标对照BA|', '|---|---:|---:|---:|---:|']
fixed = summary['fixed_long_cohort']
for phase, label in enumerate(FIXED):
    r = fixed['state_L6']['mean'][phase]
    lines.append(f'|{label}|' + '|'.join(f'{100*v:.2f}%' for v in
        (r[0], r[1], r[2], fixed['position_goal']['mean'][phase][2])) + '|')
lines += ['', '这组对照中，前几步位置完全相同，位置/目标对照没有靠长度直接区分类别；若activation读出上升，支持答案信息随真实前缀更易线性访问。但每锚点独立训练探针，仍不是CoT对最终表现的因果消融。', '',
          '## 探针节点留出与随机身份对照', '',
          '额外按每层4/2/2将节点划为probe train/val/test，分别只留目标或同时留两端，与原query split交集；重复17/23/31三个划分。主模型本身已通过ADJ/短程训练见过这些节点。严格双端probe每折test只有31/41/18条，val43/31/40条，阈值波动必须保留。', '',
          '|锚点|probe留目标BA|probe留两端BA|留两端的位置＋目标BA|', '|---|---:|---:|---:|']
strict_target = summary['strict']['target_node_heldout']
strict_both = summary['strict']['both_node_heldout']
for phase, label in enumerate(STAGES):
    lines.append(f'|{label}|' + '|'.join(f'{100*v:.2f}%' for v in
        (strict_target['state_L6']['mean'][phase][2], strict_both['state_L6']['mean'][phase][2],
         strict_both['position_goal']['mean'][phase][2])) + '|')
random_ba = strict_both['random_endpoints']['mean'][0][2]
embedding_ba = strict_both['endpoint_embeddings']['mean'][0][2]
lines += ['', f'留两端时固定随机端点向量的BA平均{100*random_ba:.2f}%，训练embedding端点拼接为{100*embedding_ba:.2f}%；均为额外解码器，维度256，大于单位置128，不是同容量比较。随机向量三seed控制只作为身份查表诊断。',
          '仅query留出时训练/随机端点可取得更高读出，见summary.json；不把这些新增外部计算归给原模型。打乱标签控制的BA约为' + ', '.join(f'{100*v:.1f}' for v in summary['shuffled_BA']) + '%。', '',
          '## 核验与解释限制', '',
          '20个最终测试面板原生答案计数全部与此前保存结果一致；每个模型第一token复制起点比例100%。FP32因果前缀核验最大差异记录在seed JSON。BF16不同GEMM形状可造成约0.1量级logit误差，因此所有用于拟合的特征都单独前向当前可见前缀，不从含未来轨迹的整段前向提取。', '',
          '结果可回答“答案是否随PATH更易解码”，不能仅据曲线证明模型在线搜索、CoT对答案必要、或轨迹前完全没有非线性答案信息。生成前低线性读出也可能是信息不在线性可访问方向；独立探针之间的变化不等同于同一个固定解码器改进。只有一张图，4初始化并非4个图seed。', '',
          'run.py可复跑；seed0–3.json存拟合参数和关键指标；fixed_cohort.json固定共同子集；summary.json聚合；answer_progress.png/.pdf为同一结果图。']
(HERE / 'REPORT.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
print('ALL BA:', [round(v[2]*100, 2) for v in summary['relative']['all']['state_L6']['mean']])
print('TT BA:', [round(v[2]*100, 2) for v in tt])
print('FIXED BA:', [round(v[2]*100, 2) for v in fixed['state_L6']['mean']])
print('FIXED counts:', npos, nneg)
print('BOTH BA:', [round(v[2]*100, 2) for v in strict_both['state_L6']['mean']])
