"""Summaries with query/graph clusters and donor-correct transfer controls."""
from __future__ import annotations

from collections import defaultdict, Counter
import csv
import gzip
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
FIELDS = ('frontier', 'visited', 'candidates')
GROUPS = ('nodes', 'bar', 'operator', 'semicolon', 'nodes_bar', 'all')


def load(path):
    with gzip.open(path, 'rt', encoding='utf-8') as f:
        return json.load(f)


def exact_record(row):
    # Deliberately strict: different legal neighbor/display orders are not
    # accepted as reference-correct. No duplicate-set false positives.
    return row['parseable'] and row['action_exact'] and all(row[f + '_exact'] for f in FIELDS)


def exact_control(row):
    return row['control_parseable'] and row['action_control_exact'] and all(row[f + '_control_exact'] for f in FIELDS)


def clustered_rate(rows, key, seed=21):
    """Pairs on the same base graph/query are one bootstrap sampling unit."""
    grouped = defaultdict(lambda: [0, 0])
    for row in rows:
        grouped[row['unit']][0] += int(row[key]); grouped[row['unit']][1] += 1
    if not grouped:
        return dict(n=0, units=0, value=None, lower=None, upper=None)
    a = np.asarray(list(grouped.values()), dtype=float)
    rng = np.random.default_rng(seed)
    picked = rng.integers(len(a), size=(500, len(a)))
    counts = a[picked].sum(axis=1)
    values = counts[:, 0] / counts[:, 1]
    return dict(n=len(rows), units=len(a), value=float(a[:, 0].sum() / a[:, 1].sum()),
                lower=float(np.quantile(values, .025)), upper=float(np.quantile(values, .975)))


def summarize(rows, donors):
    n = len(rows)
    result = dict(n=n, units=len({r['unit'] for r in rows}),
        changed=sum(r['continuation_changed'] for r in rows),
        action_changed=sum(r['action_changed'] for r in rows),
        parse_fail=sum(not r['parseable'] for r in rows),
        cap=sum(r['hit_cap'] for r in rows),
        legal_control=sum(r['control_action_legal'] for r in rows),
        legal_patch=sum(r['action_legal'] for r in rows),
        complete_reference_control=sum(exact_control(r) for r in rows),
        complete_reference_patch=sum(exact_record(r) for r in rows))
    result['change_rate'] = clustered_rate(rows, 'continuation_changed')
    for target in FIELDS + ('action',):
        eligible, transferred = [], 0
        for row in rows:
            donor = donors[row['case']]
            differing = row[target + '_target_differs']
            generated = row.get(target + '_is_generated', True)
            if differing and generated and exact_control(row) and exact_record(donor):
                transfer = (row['action_donor_exact'] if target == 'action' else row[target + '_donor_similarity'] == 1)
                eligible.append(dict(unit=row['unit'], transfer=bool(transfer)))
                transferred += int(transfer)
        result[target + '_transfer_eligible'] = len(eligible)
        result[target + '_transfer_count'] = transferred
        result[target + '_transfer_rate'] = clustered_rate(eligible, 'transfer')
    return result


def main():
    paths = sorted((ROOT / 'results').glob('*.json.gz'))
    assert len(paths) == 8, f'Expected four models x two readouts, found {len(paths)}'
    datasets = {p.name[:-8]: load(p) for p in (ROOT / 'datasets').glob('*.json.gz')}
    assert len(datasets) == 4
    runs, flat, checks = [], [], Counter()
    cells, primary, directional, baselines, examples = [], [], [], [], []
    for path in paths:
        run = load(path); model = run['model']['id']; readout = run['readout']
        data = datasets[model]
        obs = run['observations']
        donors = {r['case']: r for r in obs if r['source'] == 'donor_control'}
        assert len(donors) == len(data['records'])
        for row in obs:
            if row['source'] == 'self':
                assert not row['continuation_changed'] and row['probability_max_abs_delta'] == 0
                checks['self'] += 1
            if row['source'] == 'cross_example' and readout == 'current_record' and row['cohort'] == 'shared_history':
                assert not row['action_target_differs']
                assert all(not row[f + '_target_differs'] for f in FIELDS)
        checks['disconnected_final_layer'] += run['checks']['disconnected_final_layer']
        groups = defaultdict(list)
        for row in obs:
            if row['source'] not in ('no_history', 'cross_example'):
                continue
            flat.append(dict(model=model, readout=readout, **row))
            key = tuple(row[k] for k in ('cohort', 'field', 'layer', 'group', 'source'))
            groups[key].append(row)
        for key, rows in groups.items():
            cohort, field, layer, group, source = key
            cells.append(dict(model=model, readout=readout, cohort=cohort, field=field, layer=layer,
                              group=group, source=source, **summarize(rows, donors)))
        for cohort in ('exact_field', 'shared_history'):
            controls = [r for r in obs if r['cohort'] == cohort and r['source'] == 'none']
            baselines.append(dict(model=model, readout=readout, cohort=cohort, n=len(controls),
                recipient_correct=sum(exact_record(r) for r in controls),
                donor_correct=sum(exact_record(donors[r['case']]) for r in controls),
                both_correct=sum(exact_record(r) and exact_record(donors[r['case']]) for r in controls)))
            for source in ('no_history', 'cross_example'):
                for group in GROUPS:
                    rows = [r for r in obs if r['cohort'] == cohort and r['source'] == source and r['layer'] == 4 and r['group'] == group]
                    if not rows:
                        continue
                    summary = dict(model=model, readout=readout, cohort=cohort, source=source, group=group,
                                   **summarize(rows, donors))
                    primary.append(summary)
                    if source == 'cross_example':
                        for target in FIELDS + ('action',):
                            n = summary[target + '_transfer_eligible']; k = summary[target + '_transfer_count']
                            if n:
                                directional.append(dict(model=model, readout=readout, cohort=cohort, group=group,
                                    target=target, eligible=n, transferred=k, rate=k / n))
        if readout == 'next_record':
            for detail in run['examples']:
                if detail['source'] != 'cross_example' or detail['layer'] != 4:
                    continue
                matching = [r for r in obs if r['case'] == detail['case'] and r['layer'] == detail['layer'] and
                    r['group'] == detail['group'] and r['source'] == detail['source']]
                if matching and exact_control(matching[0]) and exact_record(donors[detail['case']]):
                    examples.append(dict(model=model, dataset_case=data['records'][detail['case']], **detail))
        runs.append(run)
    verification = dict(status='passed', models=4, readouts=8,
        matched_pairs=sum(len(d['records']) for d in datasets.values()),
        conditional_observations=len(flat), **checks)
    (ROOT / 'verification.json').write_text(json.dumps(verification, indent=2) + '\n')
    (ROOT / 'summary.json').write_text(json.dumps(dict(primary=primary, cells=cells, directional=directional, baselines=baselines),
        ensure_ascii=False, separators=(',', ':')) + '\n', encoding='utf-8')
    # Prefer the controlled same-history example, then at most a few other
    # cases. This is an illustrative selection, never a rate estimate.
    examples.sort(key=lambda x: (x['dataset_case']['cohort'] != 'shared_history', not x['action_donor_exact'], x['group'] != 'operator'))
    distinct_examples, seen_examples = [], set()
    for example in examples:
        key = (example['model'], example['case'])
        if key not in seen_examples:
            distinct_examples.append(example); seen_examples.add(key)
    (ROOT / 'key_examples.json').write_text(json.dumps(distinct_examples[:6], ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    # One compressed table, not duplicated large activation/trace artifacts.
    with gzip.open(ROOT / 'statistics.csv.gz', 'wt', encoding='utf-8', newline='') as f:
        columns = sorted(set().union(*(r.keys() for r in flat)))
        writer = csv.DictWriter(f, columns); writer.writeheader(); writer.writerows(flat)
    with (ROOT / 'directional_transfer.csv').open('w', encoding='utf-8', newline='') as f:
        columns = ['model', 'readout', 'cohort', 'group', 'target', 'eligible', 'transferred', 'rate']
        writer = csv.DictWriter(f, columns); writer.writeheader(); writer.writerows(directional)

    models = sorted(datasets)
    for cohort in ('exact_field', 'shared_history'):
        fig, axes = plt.subplots(2, 4, figsize=(14, 6.5), constrained_layout=True)
        for col, model in enumerate(models):
            for a, source in enumerate(('no_history', 'cross_example')):
                ax = axes[a, col]
                matrix = np.full((6, 2), np.nan)
                for gi, group in enumerate(GROUPS):
                    for li, layer in enumerate((2, 4)):
                        relevant = [r for run in runs if run['model']['id'] == model and run['readout'] == 'next_record'
                            for r in run['observations'] if r['cohort'] == cohort and r['source'] == source and r['group'] == group and r['layer'] == layer]
                        if relevant:
                            changed = sum(r['continuation_changed'] for r in relevant)
                            matrix[gi, li] = changed / len(relevant)
                            ax.text(li, gi, f'{100 * matrix[gi,li]:.0f}%\n(n={len(relevant)})', ha='center', va='center', fontsize=8)
                im = ax.imshow(matrix, vmin=0, vmax=1, cmap='YlOrRd', aspect='auto')
                ax.set_xticks([0, 1], ['L2', 'L4'])
                ax.set_yticks(range(6), GROUPS if col == 0 else [])
                ax.set_title(model.replace('_l10_', ' '), fontsize=10)
                if col == 0:
                    ax.set_ylabel(source)
        fig.colorbar(im, ax=axes, label='Paired next-record continuation changed', shrink=.7)
        fig.suptitle(cohort + ': next-record readout\nCounts are paired interventions, not independent graph samples', fontsize=11)
        fig.savefig(ROOT / f'{cohort}.png', dpi=160); plt.close(fig)

    lines = ['# 扩大合法配对的 activation patch 实验', '',
        '保持原四个 N=100 完整状态 checkpoint，使用所有数据 split 与合法人工构造。这里只研究局部因果响应，结果不是独立测试集泛化准确率，也没有重新训练模型。', '',
        '## 配对规模', '', '| 模型 | 精确字段配对：栈/visited/候选 | 共同历史配对：栈/visited/候选 |', '|---|---|---|']
    for model in models:
        data = datasets[model]
        values = [' / '.join(str(data['coverage'][f'{cohort}/{field}']['selected']) for field in FIELDS) for cohort in ('exact_field', 'shared_history')]
        lines.append('| ' + model + ' | ' + ' | '.join(values) + ' |')
    lines += ['', f"共 {verification['matched_pairs']} 对；两种读取重复使用这些配对。L2/L4全量，L6与self-patch稀疏校验。每格以 visited 大小1–3、4–7、8–15、16+分层，具体覆盖写入 datasets。", '',
        '同一基础单位在不同字段/配对类中可能重复。每格分层覆盖并不总能完全均衡，具体以数据清单为准。', '',
        'exact_field 要求一个字段逐 token 一致，其他两个字段语义不同。新图通过完整、单射的节点重命名实现；固定图保留原图，用新查询与合法邻居优先级匹配。shared_history 保证当前整条 CoT 前缀相同且等长：新图只改后续未出现节点的名称，固定图只改合法查询目标。两类不能混为一种实验。', '',
        '固定图BFS缺少exact_field/frontier是结构性限制：相邻层DAG上的合法BFS满足 CAND = successors(queue[0]) - set(queue)，同队列不可能有不同候选集合。这里保留为空，不伪造配对。固定图recipe中的base_split=train只是读取原图的锚点，实际source/target已另行生成，不意味着这些查询全属于原训练集。', '',
        '## 未干预控制的支持度', '',
        '| 模型 | 配对类 | recipient完整参考正确 | donor完整参考正确 | 两者都正确 |', '|---|---|---:|---:|---:|']
    for b in baselines:
        if b['readout'] == 'next_record':
            lines.append(f"| {b['model']} | {b['cohort']} | {b['recipient_correct']}/{b['n']} | {b['donor_correct']}/{b['n']} | {b['both_correct']}/{b['n']} |")
    for b in baselines:
        if b['readout'] == 'next_record' and b['recipient_correct'] == 0:
            lines += ['', f"{b['model']} / {b['cohort']} 下一记录的recipient完整参考正确数为0，故该格没有合格方向性迁移分母；不能用低响应推断不存在相关能力。"]
    lines += ['',
        '## 第4层跨 donor 的下一记录响应', '',
        '| 模型 | 配对类 | 干预 | 续写改变 | 动作改变 | 未解析 | 动作合法：对照→干预 |', '|---|---|---|---:|---:|---:|---:|']
    for p in primary:
        if p['readout'] == 'next_record' and p['source'] == 'cross_example' and p['group'] in ('nodes_bar', 'operator', 'semicolon', 'all'):
            n = p['n']
            lines.append(f"| {p['model']} | {p['cohort']} | {p['group']} | {p['changed']}/{n} | {p['action_changed']}/{n} | {p['parse_fail']}/{n} | {p['legal_control']}/{n} → {p['legal_patch']}/{n} |")
    lines += ['', '动作改变要求对照与干预都能解析；未解析不计动作改变，单独列出。动作合法是相对于 recipient 参考状态判断，不能代替整条生成记录合法率。不同组支持的case不同，不按不同分母直接排名组合效应。', '',
        '## 有方向的迁移', '',
        'directional_transfer.csv 只保留参考后续确实不同、该字段实际生成、recipient配对控制和donor控制均完整复现参考记录的样本。字段迁移要求完全匹配donor对应集合/序列；动作要求精确匹配donor动作及参数。条件较严格，合法但不同显示优先级可能被排除，分母小应如实解释。', '',
        '| 模型 | 配对类 | 干预 | 目标 | 迁移 / 合格对照 |', '|---|---|---|---|---:|']
    for d in directional:
        if d['readout'] == 'next_record' and d['group'] in ('operator', 'semicolon', 'all') and d['transferred']:
            lines.append(f"| {d['model']} | {d['cohort']} | {d['group']} | {d['target']} | {d['transferred']}/{d['eligible']} |")
    lookup = {(p['model'], p['readout'], p['cohort'], p['source'], p['group']): p for p in primary}
    s = lookup['context_l10_dfs', 'next_record', 'shared_history', 'cross_example', 'operator']
    null = lookup['context_l10_dfs', 'next_record', 'shared_history', 'no_history', 'operator']
    lines += ['', f"较干净的同历史新图DFS实验：L4操作符替换后，指定donor动作迁移 {s['action_transfer_count']}/{s['action_transfer_eligible']}，候选集合迁移 {s['candidates_transfer_count']}/{s['candidates_transfer_eligible']}；无历史替换的对应计数为 {null['action_transfer_count']}/{null['action_transfer_eligible']} 和 {null['candidates_transfer_count']}/{null['candidates_transfer_eligible']}。前者的输入图不同，但查询名称、整段CoT文本及位置一致，支持操作符表示携带特定后续信息，而不只是一般生成破坏。", '',
        'key_examples.json保存少量两边控制均完整正确的实际续写、参考状态及可复现数据索引。案例是定向挑出的说明例子，不用于估计迁移比例。', '',
        '这些计数仍需和同token随机donor、范数控制等比较，不能单凭匹配donor就排除偶然错误。summary.json中的置信区间按recipient基础图/查询unit聚类bootstrap，donor复用的相关性未完整建模；固定图依然只有seed0，区间不表示跨图seed的不确定性。', '',
        '## 当前记录控制', '',
        'shared_history 两边当前完整记录相同，字段后立即续写当前记录的正确输出应相同。如果干预改变这条记录，这是局部扰动或不必要的信息干扰，不能算向不同正确后续迁移。exact_field当前记录可有不同正确后续。两种读取及所有层、字段、失败统计保留在压缩统计中。', '',
        '## 校验与边界', '',
        f"每条生成轨迹经独立状态机重放，配对加载时校验前缀哈希；checkpoint运行前后校验。self-patch {checks['self']}条、末层过去位置负对照 {checks['disconnected_final_layer']}条均严格无变化。", '',
        '同一基础图的命名变体共享unit，不被当作独立图。新图基础来源包含train/val/test；固定图为同一已知世界上的额外合法查询。新图BFS沿用验证最佳完整状态checkpoint，训练step仅5000，控制表现不足时不能把缺少迁移归结为该架构没有相关信息。', '',
        '没有导航模型训练、checkpoint、W&B或激活/KV文件。这里只续写一条记录，未测最终YES/NO、长程恢复或完整规划。', '',
        '![精确字段配对](exact_field.png)', '', '![共同历史配对](shared_history.png)', '']
    (ROOT / 'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps(verification))


if __name__ == '__main__':
    main()
