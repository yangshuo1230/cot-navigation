"""One reusable summary and one key figure for v5 successor-selection analysis."""
import json
from pathlib import Path
import statistics as st

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
runs = [json.loads((HERE / f'seed{s}.json').read_text()) for s in range(4)]
names = runs[0]['stage_names']
assert len({d['checkpoint_sha256'] for d in runs}) == 4
assert all(d['observation_hook_logit_error'] == 0 for d in runs)
assert all(v['all_answer_and_goal_counts_match'] for d in runs for v in d['native_replay'].values())
folds = (17, 23, 31)
randoms = (101, 202, 303)


def aggregate(values):
    arr = np.asarray(values, dtype=float)
    return {'mean': arr.mean(0).tolist(), 'min': arr.min(0).tolist(), 'max': arr.max(0).tolist()}


def probe_curve(scheme, task, metric, feature='residual', nonlocal_only=False):
    key = 'nonlocal_stages' if nonlocal_only else 'stages'
    values = []
    for d in runs:
        schemes = [scheme] if scheme == 'query_only' else [f'{scheme}_{f}' for f in folds]
        per_fold = []
        for s in schemes:
            fs = [feature] if not feature.startswith('random_') else [f'{feature}_{r}' for r in randoms]
            per_fold.extend([[layer[metric] for layer in d['probes'][s]['features'][f][task][key]] for f in fs])
        values.append(np.asarray(per_fold).mean(0))
    return aggregate(values)


summary = {'stages': names, 'model_seeds': [0, 1, 2, 3], 'probe_node_split_seeds': list(folds),
           'random_identity_seeds': list(randoms), 'native_behavior': {}, 'lens_nonlocal_mixed': {}, 'probes': {}}
for group in runs[0]['behavior']:
    summary['native_behavior'][group] = aggregate([d['behavior'][group]['native_argmax_useful'] for d in runs])
for metric in ('successor_full_vocab_mass', 'successor_node_conditional_mass', 'useful_given_successor_mass',
               'useful_choice_given_successors', 'uniform_useful_baseline'):
    summary['lens_nonlocal_mixed'][metric] = aggregate([
        [s[metric] for s in d['lenses']['nonlocal_mixed_test']['stages']] for d in runs])
for scheme in ('query_only', 'current_heldout', 'target_heldout'):
    summary['probes'][scheme] = {
        'successor_recall': probe_curve(scheme, 'successor', 'recall_at_true_k'),
        'useful_nonlocal_auc': probe_curve(scheme, 'useful', 'auc', nonlocal_only=True),
        'useful_nonlocal_top1': probe_curve(scheme, 'useful', 'top1_useful', nonlocal_only=True),
        'useful_nonlocal_score_margin': probe_curve(scheme, 'useful', 'positive_minus_negative_score', nonlocal_only=True),
        'random_current_successor_recall': probe_curve(scheme, 'successor', 'recall_at_true_k', 'random_current'),
        'random_target_useful_nonlocal_auc': probe_curve(scheme, 'useful', 'auc', 'random_target', True),
        'target_embedding_useful_nonlocal_auc': probe_curve(scheme, 'useful', 'auc', 'target_embedding', True)}
(HERE / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')

lines = ['# PATH v5：后继选择性与分层读出', '',
    '结论：模型在混合分叉上有强烈的目标相关选择性，且排除一步直接输出目标后依然成立。分层输出头的后继概率与有用偏好逐层提高，L4–L6 FFN提升明显；此模型没有复现旧模型“L1输出头已几乎全指向后继”的模式。现有结果不能把信息出现层等同于必要计算层，也不能证明在线搜索。', '',
    '## 范围和核验', '',
    '- 一张80节点、160边固定图；4个初始化，全部固定50k latest checkpoint，不按分析指标挑模型。每模型重做2712条唯一原train/val/test QUERY的原生BF16贪心生成。',
    '- 20个最终测试面板的正负答案及合法到达计数全部与已保存final.json一致。观察hook对原生输出logit的差异为0；最后FFN的logit lens逐元素等于原头输出。',
    '- 收集合法前缀的当前位置，排除目标到达位置、无后继的终止位置，以及非法边之后的历史；首次错误选择仍纳入。混合分叉必须同时存在有用与无用真实后继，有用指该后继等于目标或能到目标。',
    '- 主结果另外要求当前位置距目标至少2层（可达时等于剩余跳数），因此有用选择不是简单把目标token复制出来。主样本每seed953–958个决策、356–359个QUERY；先在每个QUERY内部平均，再平均QUERY，最后平均初始化。', '',
    '## 实际生成的选择性', '', '|位置组|实际下一token选中有用后继|', '|---|---:|']
behavior_names = {'all_mixed': '全部混合分叉', 'first_mixed': '第一步混合分叉',
                  'nonlocal_mixed': '混合分叉，距目标至少2跳',
                  'first_nonlocal_mixed': '第一步混合分叉，距目标至少2跳',
                  'short_test': '短程测试', 'long_AA_test': '长程AA', 'long_TA_test': '长程TA',
                  'long_AT_test': '长程AT', 'long_TT_test': '长程TT'}
for name, v in summary['native_behavior'].items():
    lines.append(f"|{behavior_names[name]}|{100*v['mean']:.2f}%|")
baseline = summary['lens_nonlocal_mixed']['uniform_useful_baseline']['mean'][0]
lines += ['', f'距目标至少2跳的均匀后继基线为{100*baseline:.2f}%。这里仅统计当下仍可到目标且有混合选择的位置，不是整条路径的成功率；累积小概率错误会降低完整到达率。真实生成使用全词表argmax，无邻接掩码或oracle替换。', '',
          '## 分层原输出头观察', '',
          'A=attention残差之后，F=FFN残差之后。使用模型原final_norm与绑权LM head，无额外训练；早期层未校准。后继概率在完整270 token词表中计算；另外两个指标先限定真实后继，是诊断分数，不能冒充受限制解码或真实行为。仅在上述非局部混合分叉统计。', '',
          '|位置|完整词表后继概率|后继内有用概率|后继内argmax有用率|', '|---|---:|---:|---:|']
for i, name in enumerate(names):
    lines.append('|' + name + '|' + '|'.join(f"{100*summary['lens_nonlocal_mixed'][m]['mean'][i]:.2f}%" for m in
        ('successor_full_vocab_mass', 'useful_given_successor_mass', 'useful_choice_given_successors')) + '|')
lines += ['', 'L1F已提升后继评分，但平均后继概率仅约7%，到L4F约47%、L5F约81%、L6F约99%；有用后继条件偏好也在同一过程中增强。不能指定“L1已完整取出候选，某一层再独立筛选”的明确两阶段机制。早期原输出头质量低也不等于该层没有信息。', '',
          '## 仿射线性探针：包含随机身份和节点留出', '',
          '每个E/A/F位置各拟合一个128→80的仿射ridge，标签为完整后继集合或完整有用后继集合，不是只拟合正确token。对有用性严格在真实后继内评估AUROC、Top-|true| recall、top1及正负分数差，避免大量非邻接节点抬高分数。分数不是概率。', '',
          '归一化仅用train，lambda在{.001,.01,.1,1,10}中按val MSE选，每位置一个lambda，test不选参数；拟合和评估均使QUERY等权。原始QUERY train/val/test严格分开。额外当前节点或目标节点holdout每层按4/2/2划分，并重复17/23/31三个划分seed。探针节点留出不等于主模型没有见过该节点。', '',
          '|设置/指标|E|L1F|L2F|L3F|L4F|L5F|L6F|', '|---|---:|---:|---:|---:|---:|---:|']
probe_rows = [('仅query留出，后继Recall', 'query_only', 'successor_recall'),
              ('再留当前节点，后继Recall', 'current_heldout', 'successor_recall'),
              ('仅query留出，有用AUROC', 'query_only', 'useful_nonlocal_auc'),
              ('再留目标节点，有用AUROC', 'target_heldout', 'useful_nonlocal_auc')]
for label, scheme, metric in probe_rows:
    curve = summary['probes'][scheme][metric]['mean']
    lines.append('|' + label + '|' + '|'.join(f'{curve[i]:.3f}' for i in (0, 2, 4, 6, 8, 10, 12)) + '|')
qrand = summary['probes']['query_only']['random_current_successor_recall']['mean'][0]
qtarget = summary['probes']['query_only']['random_target_useful_nonlocal_auc']['mean'][0]
nrand = summary['probes']['current_heldout']['random_current_successor_recall']['mean'][0]
trand = summary['probes']['target_heldout']['random_target_useful_nonlocal_auc']['mean'][0]
temb = summary['probes']['target_heldout']['target_embedding_useful_nonlocal_auc']['mean'][0]
margin = summary['probes']['target_heldout']['useful_nonlocal_score_margin']['mean'][-1]
lines += ['', f'身份对照：query留出时，未训练高斯当前节点向量的后继Recall为{qrand:.3f}，随机目标向量的有用AUROC为{qtarget:.3f}，表明探针能直接拟合身份关系。',
          f'加节点留出后：随机当前节点后继Recall为{nrand:.3f}；随机目标有用AUROC为{trand:.3f}（不是自动等于0.5，含数据/候选结构偏差）；真实目标embedding对照为{temb:.3f}。最后层有用探针正例分数减无用后继分数的均值为{margin:.4f}，方向正确但不作概率解释。', '',
          'E位置只含当前节点embedding，数学上没有输入目标信息；即便有用AUROC高于0.5，也只能反映节点/分支偏好与测试分布相关性，不能据此说E含当前目标。关键证据是上下文化的后层读出在留出目标时明显优于这些对照。', '',
          '## 同起点、同目标深度，改变目标', '',
          '测试中有6对距目标至少2跳、目标不同且有用后继集合不相交的配对，涉及4个起点、10个独立QUERY。四个初始化都在所有配对中随目标更换分支，且两侧均选对。配对共用QUERY，不能当作24个独立图/查询证据。', '']
world = json.loads((HERE.parent / 'fixed_navigation_v5/data/n80_g0.json').read_text())
def node(i):
    return f"V{world['node_names'][i]}"
for p in runs[0]['matched_targets_nonlocal']['examples'][:2]:
    lines.append(f"- 起点{node(p['source'])}，目标{node(p['target_a'])}时有用后继"
                 f"{[node(i) for i in p['useful_a']]}、选{node(p['next_a'])}；改目标"
                 f"{node(p['target_b'])}后有用后继{[node(i) for i in p['useful_b']]}、选{node(p['next_b'])}。")
lines += ['', '## 限制与复现', '',
          '这是目标相关分支选择和可解码性证据，不是必要性/因果中介证明，也未排除固定图可达性记忆、片段拼接或目标条件策略表。当前只有一个图seed。尚未在本轮做激活/参数干预。', '',
          '前缀完整前向与KV逐token生成存在BF16微小差异，next-token argmax不一致约0.02%–0.07%；真实行为和配对使用原生生成，不用logit lens代替。模型权重冻结，探针在GPU0闭式求解，未保存激活、探针权重、模型checkpoint或新W&B run。', '',
          'run.py复跑4个checkpoint；seed0–3.json保留关键分组、正则与少量实例；summary.json聚合；key_results.png/.pdf是同一关键图。']
(HERE / 'REPORT.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')

fig, axes = plt.subplots(1, 3, figsize=(17, 4.7), layout='constrained')
x = np.arange(13)
def draw(ax, data, label, color, percent=False):
    scale = 100 if percent else 1
    y = np.asarray(data['mean']) * scale
    if len(y) == 1:
        ax.axhline(y[0], color=color, ls='--', lw=1.6, label=label)
    else:
        ax.plot(x, y, color=color, marker='o', ms=3.3, lw=1.8, label=label)
        ax.fill_between(x, np.asarray(data['min'])*scale, np.asarray(data['max'])*scale, alpha=.10, color=color)
draw(axes[0], summary['lens_nonlocal_mixed']['successor_full_vocab_mass'], 'Full-vocab successor mass', '#3973a9', True)
draw(axes[0], summary['lens_nonlocal_mixed']['useful_given_successor_mass'], 'Useful mass / successor mass', '#31936b', True)
axes[0].axhline(100*baseline, color='#7b8794', ls='--', label='Uniform legal successor')
axes[0].set(title='Original head: legality and useful preference', ylim=(0, 103), ylabel='Probability (%)')
draw(axes[1], summary['probes']['query_only']['successor_recall'], 'QUERY holdout', '#3973a9', True)
draw(axes[1], summary['probes']['current_heldout']['successor_recall'], '+ Current-node holdout', '#c98233', True)
draw(axes[1], summary['probes']['current_heldout']['random_current_successor_recall'], 'Random current ID; node held out', '#7b8794', True)
axes[1].set(title='Linear probe: successor-set recall', ylim=(0, 103), ylabel='Top-|neighbors| recall (%)')
draw(axes[2], summary['probes']['query_only']['useful_nonlocal_auc'], 'QUERY holdout', '#3973a9')
draw(axes[2], summary['probes']['target_heldout']['useful_nonlocal_auc'], '+ Target-node holdout', '#31936b')
draw(axes[2], summary['probes']['query_only']['random_target_useful_nonlocal_auc'], 'Random target ID; QUERY holdout', '#9e6aaf')
draw(axes[2], summary['probes']['target_heldout']['random_target_useful_nonlocal_auc'], 'Random target ID; node held out', '#7b8794')
axes[2].axhline(.5, color='#cccccc', ls=':', lw=1)
axes[2].set(title='Linear probe: useful vs useless successors', ylim=(.4, 1.02), ylabel='Within-successor AUROC')
for ax in axes:
    ax.set_xticks(x, names, rotation=45, ha='right', fontsize=8)
    ax.set_xlabel('E = embedding; A = after attention; F = after FFN')
    ax.grid(axis='y', alpha=.16)
    ax.legend(loc='lower right', fontsize=7.5)
fig.suptitle('Fixed80 PATH, step50k: 4 initializations; useful selection excludes final one-hop-to-goal steps', fontsize=12)
fig.savefig(HERE / 'key_results.png', dpi=180)
fig.savefig(HERE / 'key_results.pdf')
print(json.dumps(summary, indent=2))
