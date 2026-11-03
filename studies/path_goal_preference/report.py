"""Produce one compact figure and a self-contained audit report from final results."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
data = {s: json.loads((HERE / f'{s}.json').read_text()) for s in ('simple', 'hard')}
percent = lambda x: f'{100*x:.1f}%'
checks = {}
for setting, d in data.items():
    p, sampling = d['greedy']['positive'], d['sampling']
    assert p['legal_goal_hit'] <= p['n']
    assert d['greedy']['negative']['legal_goal_hit'] == 0
    assert sum(sampling['stops'].values()) == sampling['trajectories']
    assert sampling['successes'] == sampling['stops']['goal']
    assert sampling['CPU_GPU_logit_max_error'] < 1e-3
    assert sampling['cache_decode_logit_max_error'] < 1e-3
    mismatches = d['preference']['mismatch_details']
    assert all(m['saved_useful'] == m['replay_useful'] for m in mismatches)
    for g in d['preference']['groups'].values():
        assert 0 <= g['useful_probability'] <= g['successor_probability'] + 1e-6 <= 1.00001
    checks[setting] = dict(replay_different_argmax=len(mismatches),
        changed_useful_membership=0,
        largest_logit_gap=max(m['logit_gap'] for m in mismatches))

fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), constrained_layout=True)
x = np.arange(2)
for offset, key, label, color in [(-.26, 'random', 'Uniform successor walk', '#a6a6a6'),
                                (0, 'sampling', 'PATH sampling (T=1)', '#69a8d5'),
                                (.26, 'greedy', 'PATH greedy', '#276498')]:
    values = [d['greedy']['positive']['exact_random_goal_rate'] if key == 'random'
              else d['sampling']['success_rate'] if key == 'sampling'
              else d['greedy']['positive']['legal_goal_rate'] for d in data.values()]
    bars = axes[0].bar(x+offset, np.array(values)*100, .24, label=label, color=color)
    axes[0].bar_label(bars, fmt='%.1f', fontsize=8)
for offset, key, label, color in [(-.26, 'random_useful_probability', 'Uniform successor', '#a6a6a6'),
                                (0, 'useful_conditional', 'PATH probability | successor', '#69a8d5'),
                                (.26, 'actual_useful', 'PATH greedy choice', '#276498')]:
    values = [d['preference']['groups']['first_step_mixed'][key]*100 for d in data.values()]
    bars = axes[1].bar(x+offset, values, .24, label=label, color=color)
    axes[1].bar_label(bars, fmt='%.1f', fontsize=8)
for ax, title in zip(axes, ['Legal goal hitting: reachable test queries',
                           'Useful successor: first mixed branch at source']):
    ax.set_xticks(x, ['Simple N100', 'Hard N100'])
    ax.set_ylim(0, 119); ax.set_ylabel('Percent'); ax.set_title(title, fontsize=10)
    ax.legend(loc='upper center', fontsize=7, frameon=False)
    ax.spines[['top', 'right']].set_visible(False)
fig.savefig(HERE/'key_results.png', dpi=180)
plt.close(fig)

lines = ['# Fixed DAG PATH：到达目标与有用后继偏好', '',
         '结论：两张固定图上的 PATH 都明显优于均匀随机游走，并在真正有选择的分叉处偏好能通往目标的后继。这个结果是行为证据，不能单独证明模型执行了通用搜索或排除固定图关系记忆。', '',
         '## 范围与定义', '',
         '- 仅使用原验证集选出的 N100 PATH best checkpoint；简单图 150 边、step 20000，困难图 200 边、step 10000。具体路径和 SHA256 见 JSON manifest。没有新训练。',
         '- 仅以原测试集正例比较目标到达率：简单图 386 条、困难图 300 条。原模型训练时见过全部节点和 WALK；测试留出查询对，未留出节点。',
         '- 有用后继：当前节点的直接后继中，等于目标或能沿真实边到达目标的节点。仅把有用和无用后继同时存在的状态称为 mixed。',
         '- 合法到达：从正确起点出发，始终沿真实有向边，并至少一次抵达目标。额外报告整条轨迹合法且终点恰为目标，二者都不等同于最终 YES/NO 正确。',
         '- 随机基线：每步从真实直接后继均匀选择，目标吸收；在 DAG 上用反向动态规划精确计算到达概率，没有蒙特卡洛误差。可达闭包已逐节点与 BFS 交叉检查。', '',
         '## 到达结果', '',
         '|指标|简单图|困难图|', '|---|---:|---:|']
for label, getter in [
    ('原始贪心轨迹沿合法边到达目标', lambda d: f"{d['greedy']['positive']['legal_goal_hit']}/{d['greedy']['positive']['n']} ({percent(d['greedy']['positive']['legal_goal_rate'])})"),
    ('整条轨迹合法且结束于目标', lambda d: f"{d['greedy']['positive']['complete_legal_end_goal']}/{d['greedy']['positive']['n']}"),
    ('均匀随机游走精确到达率', lambda d: percent(d['sampling']['exact_random_rate'])),
    ('PATH 温度 1 采样到达率', lambda d: f"{percent(d['sampling']['success_rate'])} ({d['sampling']['successes']}/{d['sampling']['trajectories']})"),
    ('正例最终答案准确率', lambda d: percent(d['greedy']['positive']['answer_accuracy'])),
    ('负例最终答案准确率', lambda d: percent(d['greedy']['negative']['answer_accuracy']))]:
    lines.append('|'+label+'|'+'|'.join(getter(d) for d in data.values())+'|')
lines += ['', '采样从 QUERY 加已给出的起点 token 开始，每个正例独立采样 16 次，完整词表、温度 1、不加邻接掩码、不重试。遇到非法边、提前输出协议符或进入再也不能到达目标的分支即记失败；达到目标即成功，不要求随后输出正确答案。图只用于评分，未参与生成。两张图的采样标准误分别约 0.31、0.45 个百分点，仅表示给定测试集的采样误差。', '',
          '## 后继选择偏好', '',
          '首先看源节点的第一步，避免只统计模型已走对的中途前缀。下表只含 mixed 查询：', '',
          '|指标|简单图|困难图|', '|---|---:|---:|']
for label, key in [('查询数','queries'), ('贪心选择有用后继','actual_useful'),
                   ('完整词表中有用后继概率质量','useful_probability'),
                   ('完整词表中全部合法后继概率质量','successor_probability'),
                   ('限定在合法后继内的有用概率质量','useful_conditional'),
                   ('合法后继均匀选择基线','random_useful_probability')]:
    lines.append('|'+label+'|'+'|'.join(str(d['preference']['groups']['first_step_mixed'][key]) if key=='queries'
                                      else percent(d['preference']['groups']['first_step_mixed'][key]) for d in data.values())+'|')
lines += ['', '再看原始自由生成轨迹中所有合法前缀的 mixed 状态：简单图 1171 次决策/376 条查询，困难图 1807 次/298 条查询。每条查询先取平均、再跨查询平均：合法后继内的有用概率分别为 95.7%、97.5%，均匀基线分别为 43.1%、50.0%。按决策直接平均则为 97.2%、98.5%；长且正确的轨迹会贡献更多决策，因此主报告使用查询等权。', '',
          '## 同起点、更换目标', '',
          '在原测试查询里寻找同起点、第一步均为 mixed、且两目标有用后继集合互不相交的配对。', '',
          '|指标|简单图|困难图|', '|---|---:|---:|']
for label, key in [('配对数','pairs'),('独立查询数','unique_queries'),('不同起点数','sources'),
                   ('两查询均选中各自有用后继','both_useful'),('更换目标后下一节点改变','changes'),
                   ('独立查询选择有用后继','unique_query_useful')]:
    lines.append('|'+label+'|'+'|'.join(str(d['preference']['goal_pairs']['all'][key]) if key in ('pairs','unique_queries','sources')
                                       else percent(d['preference']['goal_pairs']['all'][key]) for d in data.values())+'|')
lines += ['', '困难图的 25 对只有 30 条不同查询，且错误查询会被反复配对，不能把配对当作独立样本。简单图同目标深度、同生成层级的子集还有 31 对，其中 28 对两侧都选对；困难图没有满足这两个额外匹配条件的配对，不能据此排除其深度等结构特征的贡献。', '',
          '实例（有用集合由真实图计算）：', '',
          '- 简单图，同起点 V154：目标 V167 时有用后继为 {V67}，实际选 V67；目标 V204 时为 {V150}，实际选 V150。',
          '- 困难图，同起点 V139：目标 V183 时有用后继为 {V50,V32,V247}，实际选 V50；目标 V57 时为 {V134}，实际选 V134。',
          '- 失败也存在：困难图 V236→V65，有用后继 {V65}，模型却选择 V93，最终输出 NO；简单图 V85→V159，有用后继 {V165}，模型选择 V147，但最终仍输出 YES。后者说明答案正确不能代替路径正确。', '',
          '## 核验与解释边界', '',
          '实际贪心选择取自保存的自由生成；概率用原 checkpoint FP32 重放同一前缀。简单图 6 次、困难图 20 次 argmax 与保存的下一 token 不同，均未改变有用/无用分类；没有把 FP32 重放选择冒充原始轨迹。CPU/GPU logits 和 KV-cache/完整前向的最大误差均小于 1.2e-5。', '',
          '两图各只有一个固定图 seed，且训练设置不同，因此不能把两图差别只归因于图复杂度。高到达率和更换目标后改变分支说明模型存在目标相关的定向选择；仍可由固定图可达关系记忆实现，不足以证明显式搜索、排除 shortcut，或说明 PATH 相比 Direct 增加了什么能力。', '',
          '文件：`run.py` 可复跑；`simple.json`、`hard.json` 保存汇总、跳数分组、失败实例与 checkpoint 标识；`key_results.png` 是唯一结果图。未保存新 checkpoint 或 activation。', '']
(HERE/'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')
print(json.dumps(dict(verification=checks, report=str(HERE/'REPORT.md')), indent=2))
