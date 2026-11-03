"""Compact report and two figures; no tensors or auxiliary plot formats."""
from collections import defaultdict
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
FIELDS = ('frontier', 'visited', 'candidates')


def mean(rows, key):
    return sum(float(r[key]) for r in rows) / len(rows) if rows else None


def main():
    runs = []
    checks = dict(self_patches=0, final_layer_past_patches=0, same_token_cases=0, semantic_matches=0)
    flat, table, findings = [], [], []
    for folder in ('results', 'current_record'):
        paths = sorted((ROOT / folder).glob('*.json'))
        assert len(paths) == 4, (folder, len(paths))
        for p in paths:
            data = json.loads(p.read_text())
            data['readout'] = folder
            runs.append(data)
            for c in data['cases']:
                f = c['field']
                assert c['recipient_state'][f] == c['donor_state'][f]
                assert c['recipient_split'] == 'test' and c['donor_split'] == 'train'
                for g in FIELDS:
                    if g == f:
                        continue
                    a, b = c['recipient_state'][g], c['donor_state'][g]
                    assert a != b if g == 'frontier' else set(a.split()) != set(b.split())
                checks['same_token_cases'] += 1
                checks['semantic_matches'] += 1
            for r in data['observations']:
                if r['source'] == 'self':
                    assert r['probability_max_abs_delta'] == 0 and not r['continuation_changed']
                    checks['self_patches'] += 1
                negative = (r['layer'] == 6 and r['source'] in ('cross_example', 'no_history') and
                            r['group'] in (('nodes', 'bar', 'operator', 'nodes_bar') if folder == 'results' else ('nodes',)))
                if negative:
                    assert r['probability_max_abs_delta'] == 0 and not r['continuation_changed']
                    checks['final_layer_past_patches'] += 1
                flat.append(dict(model=data['model']['id'], readout=folder, **r))
            # Pre-specified illustrative setting: post-block 4, nodes+bar.
            for source in ('no_history', 'cross_example'):
                rows = [r for r in data['observations'] if r['layer'] == 4 and r['group'] == 'nodes_bar' and r['source'] == source]
                table.append(dict(model=data['model']['id'], readout=folder, source=source, n=len(rows),
                    changed=sum(r['continuation_changed'] for r in rows),
                    legal_control=sum(r['control_action_legal'] for r in rows), legal_patch=sum(r['action_legal'] for r in rows),
                    parseable_patch=sum(r['parseable'] for r in rows)))
            interesting = [r for r in data['observations'] if r['source'] == 'cross_example' and r['layer'] < 6 and r['continuation_changed']]
            def directional(r):
                return max((r[f + '_donor_similarity_shift'] for f in FIELDS
                            if r.get(f + '_is_generated', True)), default=0)
            interesting.sort(key=directional, reverse=True)
            for r in interesting[:2]:
                findings.append(dict(model=data['model']['id'], readout=folder,
                                     observation=r, case=data['cases'][r['case']]))

    columns = sorted(set().union(*(r.keys() for r in flat)))
    with (ROOT / 'statistics.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, columns)
        writer.writeheader(); writer.writerows(flat)
    (ROOT / 'verification.json').write_text(json.dumps(dict(status='passed', **checks), indent=2) + '\n')
    (ROOT / 'findings.json').write_text(json.dumps(findings, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

    for folder, groups in [('results', ['nodes', 'bar', 'operator', 'semicolon', 'nodes_bar', 'all']),
                           ('current_record', ['nodes', 'bar', 'nodes_bar'])]:
        selected = [d for d in runs if d['readout'] == folder]
        fig, axes = plt.subplots(2, 4, figsize=(14, 5.7), constrained_layout=True)
        for col, data in enumerate(selected):
            for row, source in enumerate(('no_history', 'cross_example')):
                ax = axes[row, col]
                matrix = np.full((len(groups), 3), np.nan)
                for a, group in enumerate(groups):
                    for b, layer in enumerate((2, 4, 6)):
                        cells = [r for r in data['observations'] if r['source'] == source and r['group'] == group and r['layer'] == layer]
                        if cells:
                            matrix[a, b] = mean(cells, 'continuation_changed')
                            ax.text(b, a, f'{100 * matrix[a,b]:.0f}%\n(n={len(cells)})', ha='center', va='center', fontsize=8)
                im = ax.imshow(matrix, vmin=0, vmax=1, cmap='YlOrRd', aspect='auto')
                ax.set_xticks(range(3), ['L2', 'L4', 'L6'])
                ax.set_yticks(range(len(groups)), groups if col == 0 else [])
                ax.set_title(data['model']['id'].replace('_l10_', ' '), fontsize=10)
                if col == 0:
                    ax.set_ylabel(source)
        fig.colorbar(im, ax=axes, label='Paired one-record continuation changed', shrink=.7)
        fig.suptitle(('Next-record readout (current record clamped)' if folder == 'results' else
                      'Current-record readout (after matched field)') + '\nExploratory native pairs; different groups may have different support', fontsize=11)
        fig.savefig(ROOT / f'{folder}.png', dpi=160)
        plt.close(fig)

    lines = ['# 同 token 隐藏状态干预：首轮方法验证', '',
             '四个 N=100 完整状态模型（新图/固定图 × DFS/BFS），各自验证最佳 checkpoint；固定图仅 seed 0。冻结推理使用物理 GPU1，没有新训练、checkpoint、W&B run 或激活文件。', '',
             '本轮完成两种读取：字段后立刻续写当前记录，以及固定当前记录后续写下一条。节点、字段分隔符、动作操作符、记录结束符分别做，另做节点+分隔符及全部组合；PUSH 参数不干预。操作符仅覆盖非终止 PUSH/POP。详细定义见 DESIGN.md。', '',
             '**核心边界：token 相同而输出变化，支持这些位置的隐藏表示有因果作用；不能据此断言隐藏状态比显式文本更重要。** 隐藏状态本身就是文本进入后续计算的媒介。', '',
             '## 配对覆盖', '', '| 模型 | frontier | visited | candidates |', '|---|---:|---:|---:|']
    for d in runs[:4]:
        lines.append('| ' + d['model']['id'] + ' | ' + ' | '.join(str(d['coverage'][f]['selected']) for f in FIELDS) + ' |')
    lines += ['', '每格至多四个不同 recipient 查询；目标字段逐 token 相同，另外两个字段必须语义不同。候选/visited 用集合差异，栈/队列用有序序列差异。新图 visited 只有两个可用配对；固定图 BFS frontier 在本轮有界检索中没有严格配对，不等于穷举全池或数学上不存在。不同组还会因操作符身份或空字段而缺格。', '',
              '## 局部响应示例：第4层，节点+字段分隔符', '',
              '“改变”指整个被生成记录与同 batch 未干预 twin 不同。“动作合法”按 recipient 原状态，允许合法候选替代；并非最终答案准确率。', '',
              '| 读取 | 模型 | donor | 改变/配对 | 动作合法：对照→干预 |', '|---|---|---|---:|---:|']
    for t in table:
        lines.append(f"| {t['readout']} | {t['model']} | {t['source']} | {t['changed']}/{t['n']} | {t['legal_control']}/{t['n']} → {t['legal_patch']}/{t['n']} |")
    lines += ['', '这组预先指定的读取下，no_history 对当前动作的破坏明显多于 cross_example，但这不是语义重要性的公平排名：两种 donor 的历史长度、语法熟悉度与激活分布不同。固定其余正确文本后，下一记录的变化通常更少，是后续文本可能纠正局部扰动的线索，尚不能确定其机制。', '', '一个候选线索：新图 DFS query 50523，用 donor 38161 在相同栈 token <V250> 上的第2层节点表示替换后，下一记录候选偏向 donor 的空候选，动作也与 donor 的 POP 一致；同时 recipient 状态正确性下降。这与 donor 影响相容，也与一般破坏相容，不能单独当作规划状态成功移植的证明。配对信息与指标在 findings.json。']
    lines += ['', '## 验证与异常记录', '',
              f"Self-patch 共 {checks['self_patches']} 条，概率差严格为0，续写完全相同。最后一层过去位置负对照共 {checks['final_layer_past_patches']} 条，同样严格无影响。9 项 CPU 单元测试覆盖因果方向、cache 重算及当前记录读取等。配对前缀已经逐记录用图语义重放验证，checkpoint 在每模型运行前后校验 SHA256。", '',
              '最后一层当前末尾 token 可以直接改变 next-token logits；这与最后一层过去位置无通路并不矛盾。图中必须区分节点/过去分隔符与当前末尾分隔符，不能按整层是否有效下结论。', '',
              'findings.json 保存每个模型/读取最多两个变化 case 的完整匹配字段、其他字段、参考后续和配对响应指标。优先列出朝 donor 字段靠近的案例，属于异常线索，不能用筛选后的比例作总体效应。statistics.csv 保留全部统计，包括格式失败、截断、recipient/donor 相似度变化。', '',
              '尚未完成的关键控制：更多查询/图 seed；历史长度精确匹配；同状态不同历史 donor；范数匹配/随机 donor；替换强度扫描；共同 cohort 上的组合效应；长程最终答案。当前 early/shallow 小样本和不同 donor 历史长度限制了外推。不能把一般输出破坏解释成语义迁移，也不能只看成功变化的样本。', '',
              '![下一记录](results.png)', '', '![当前记录](current_record.png)', '']
    (ROOT / 'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps(dict(runs=len(runs), observations=len(flat), verification=checks)))


if __name__ == '__main__':
    main()
