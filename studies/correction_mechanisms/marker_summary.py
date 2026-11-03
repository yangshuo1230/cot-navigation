"""CPU-only marker census from frozen natural annotations; never regenerate traces.

Whole repair counts trajectories, immediate repair counts events. Raw token
counts are reported only when retained tokens match the original trace hash.
"""
import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def read(path):
    with gzip.open(path, 'rt') as f:
        return json.load(f)


def aggregate(manifest_path):
    from cot_navigation.config import load_config
    from cot_navigation.protocols.tokenizer import Tokenizer

    manifest = json.loads(manifest_path.read_text())
    base = ROOT / manifest['output']
    assert not (base / 'DEPRECATED.json').exists()
    summary = json.loads((base / 'summary.json').read_text())
    assert summary['complete']
    result = dict(cohort=manifest['cohort'], complete=True, groups={}, sources={},
        definitions=dict(
            auditable_marks='MISTAKE in complete scored records before illegal-action censoring',
            true_marks='cancelled record contains a verified semantic error',
            immediate_repairs='true mark followed immediately by a non-cancelled, error-free legal record',
            whole_repaired='trajectory with at least one immediate repair, complete legal CoT and correct answer',
            raw_marks='all MISTAKE tokens; null unless original full tokens retained and hash verified'))
    grouped = defaultdict(list)
    for model in manifest['models']:
        tok = Tokenizer.from_config(load_config([ROOT / model['run'] / 'resolved_config.yaml']))
        corpus_path = base / 'probe_corpus' / (model['id'] + '__test.json.gz')
        corpus = read(corpus_path) if corpus_path.exists() else None
        if corpus:
            assert corpus['complete']
            tokens = {r['query']: r['trace'] for r in corpus['rows']}
        else:
            tokens = {}
        for task in manifest['tasks']:
            path = base / 'results' / f"{model['id']}__{task}.json.gz"
            data = read(path)
            assert data['complete'] and data['identity']['model']['id'] == model['id']
            result['sources'][str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
            c = Counter(queries=0, auditable_marks=0, true_marks=0, false_marks=0,
                immediate_repairs=0, whole_repaired=0, error_records=0, raw_covered=0, raw_known=0)
            for t in data['rows']:
                rr = t['records']
                true = sum(r['true_cancel'] for r in rr)
                false = sum(r['false_cancel'] for r in rr)
                repairs = sum(r['true_cancel'] and i+1 < len(rr) and
                    not rr[i+1]['bad'] and rr[i+1]['action_legal'] and
                    not rr[i+1]['cancel'] for i, r in enumerate(rr))
                assert true + false == sum(r['cancel'] for r in rr)
                c.update(queries=1, auditable_marks=true+false, true_marks=true,
                    false_marks=false, immediate_repairs=repairs,
                    whole_repaired=int(repairs > 0 and t['trace_valid'] and t['correct']),
                    error_records=sum(r['bad'] for r in rr))
                if task == 'main' and t['query'] in tokens:
                    trace = tokens[t['query']]
                    assert hashlib.sha256(np.asarray(trace, np.int32).tobytes()).hexdigest() == t['trace_sha256']
                    c.update(raw_covered=1, raw_known=trace.count(tok.id('<MISTAKE>')))
            assert c['queries'] == 256
            grouped[task, model['group']].append(dict(seed=model['seed'], **c))
    for (task, group), rows in grouped.items():
        totals = {k: sum(r[k] for r in rows) for k in rows[0] if k != 'seed'}
        totals['raw_marks'] = totals['raw_known'] if totals['raw_covered'] == totals['queries'] else None
        totals['per_seed'] = rows
        # Independent record-audit definition; shared replay counts remain in the source summary.
        totals['shared_replay_true_marks'] = summary['tasks'][task][group]['true_marks']
        totals['shared_replay_repairs'] = summary['tasks'][task][group]['repairs']
        result['groups'].setdefault(task, {})[group] = totals
    out = ROOT / 'outputs/analysis/correction_mechanisms' / manifest['cohort']
    (out / 'marker_counts.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print('MARKER SUMMARY', len(result['sources']), 'frozen files; no inference')


def plot(data, manifest, out):
    # Reuse the same styles, Chinese font, model labels and graph ordering.
    from report import plt, COLORS, TASKS, LABELS, label, save
    depths = sorted({m['layers'] for m in manifest['models']})
    groups = {d: sorted({m['group'] for m in manifest['models'] if m['layers'] == d},
                       key=lambda g: int(g.split('_p')[1])) for d in depths}
    specs = [('auditable_marks', '可核验 MISTAKE 数'),
             ('true_marks', '标对错误记录数'),
             ('unmarked_errors', '未标出错误记录数'),
             ('immediate_repairs', '下一条修复成功次数'),
             ('whole_repaired', '整条修复成功轨迹数')]

    def value(task, group, metric):
        row = data['groups'][task][group]
        if metric == 'unmarked_errors':
            # Same complete-record audit for both terms; count records, not error fields.
            assert 0 <= row['true_marks'] <= row['error_records']
            return row['error_records'] - row['true_marks']
        return row[metric]

    # One model group and graph per panel; colors identify the five metrics.
    panels = [(depth, g) for depth in depths for g in groups[depth]
              if int(g.split('_p')[1]) > 0]
    metric_colors = ['#277DA1', '#459B76', '#D65A4A', '#E9A23B', '#8864AA']
    short_labels = ['MISTAKE', '标对', '未标出', '下一条\n修复', '整条\n修复']
    fig, aa = plt.subplots(len(panels), len(TASKS), figsize=(21, 3*len(panels)+2), squeeze=False)
    fig.subplots_adjust(top=.94, bottom=.13, hspace=.65, wspace=.3)
    for row, (depth, g) in enumerate(panels):
        for col, task in enumerate(TASKS):
            ax = aa[row, col]
            values = [value(task, g, metric) for metric, _ in specs]
            for x, ((_, title), count, color) in enumerate(zip(specs, values, metric_colors)):
                ax.bar(x, count, width=.65, color=color, label=title)
                ax.text(x, count, str(count), ha='center', va='bottom', fontsize=9)
            if row == 0:
                ax.set_title(LABELS[col].replace('\n', ' '), fontsize=13)
            ax.set_xticks(range(len(specs)), short_labels, fontsize=9)
            if col == 0:
                ax.set_ylabel(f'{depth}层 / {label(g)}\n次数 / 轨迹数', fontsize=10)
            # Same graph column uses a common scale across all model settings.
            maximum = max(value(task, group, metric)
                          for _, group in panels for metric, _ in specs)
            ax.set_ylim(0, max(1, maximum)*1.18)
            ax.spines[['top', 'right']].set_visible(False)
            ax.grid(axis='y', alpha=.15)
    save(fig, out, '08_natural_mistake_repairs', '自然生成：MISTAKE 标注、漏标与修复',
         '每模型每图 256 个可达查询；每组两种子合计 512 条。仅核验非法动作删失前的完整记录。\n'
         '未标出＝可核验错误记录总数－标对记录数；同一记录多个字段错误只计一条；不可核验后缀不计入。\n'
         '下一条修复要求：标对后紧接的记录合法、无错误且不是 MISTAKE；整条修复按轨迹去重，且要求最终合法并答对。')
    fig, aa = plt.subplots(1, len(depths), figsize=(13, 5), squeeze=False)
    fig.subplots_adjust(top=.8, bottom=.3)
    for col, depth in enumerate(depths):
        ax = aa[0, col]
        for gi, g in enumerate(groups[depth]):
            r = data['groups']['main'][g]
            assert r['raw_marks'] is not None
            values = [r['raw_marks'], r['auditable_marks'], r['true_marks'], r['immediate_repairs'], r['whole_repaired']]
            x = np.arange(5) + (gi-1)*.24
            ax.bar(x, values, width=.23, color=COLORS[int(g.split('_p')[1])], label=label(g))
            for xx, y in zip(x, values): ax.text(xx, y, str(y), ha='center', va='bottom', fontsize=8)
        ax.set_xticks(range(5), ['原始全部\nMISTAKE', '可核验\nMISTAKE', '标对次数', '下一条\n修复次数', '整条修复\n轨迹数'])
        ax.set_title(f'{depth}层 / 普通 N100')
        ax.set_ylabel('次数 / 轨迹数');ax.spines[['top', 'right']].set_visible(False);ax.margins(y=.2)
    save(fig, out, '09_main_raw_mistake_counts', '普通 N100：原始 MISTAKE 全量计数与可核验计数',
         '仅普通 N100 保留完整原始 token，逐轨迹哈希核对；其他图原始全量未知，不以可核验计数代替。\n'
         '每组两种子合计 512 条；整条修复按轨迹去重；完整计数、误报及每种子结果见 marker_counts.json。')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, required=True)
    aggregate(p.parse_args().manifest)
