"""Aggregate compact probe results; no feature dumps or navigation training."""
from pathlib import Path
import argparse
import csv
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def values(d, architecture, population, head, key='auroc'):
    return [f[population][head][key] for f in d['fits']
            if f['architecture'] == architecture and f[population] is not None
            and f[population][head][key] is not None]


def stat(v):
    return (float(np.mean(v)), float(np.std(v))) if len(v) else (None, None)


def formatted(v):
    m, s = stat(v)
    return 'N/A' if m is None else f'{m:.3f} ± {s:.3f}'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('output', type=Path)
    args = parser.parse_args(); out = args.output
    selection = json.loads((out/'selection.json').read_text())['models']
    models = [json.loads((out/'models'/f'{e["id"]}.json').read_text()) for e in selection]
    rows = []
    for d in models:
        for population in ['synthetic', 'natural']:
            for arch in ['linear', 'mlp']:
                for head, name in enumerate(d['heads']):
                    mean, std = stat(values(d, arch, population, head))
                    rows.append(dict(model=d['entry']['id'], population=population, architecture=arch,
                                     head=name, auroc_mean=mean, auroc_seed_std=std))
    with (out/'summary.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    fig, axes = plt.subplots(1, 2, figsize=(14, 9), sharey=True)
    styles = [('linear', 2, 'Linear L6', '#4477AA', -.24),
              ('mlp', 2, 'MLP L6', '#CC6677', -.08),
              ('linear', 3, 'Linear identity control', '#228833', .08),
              ('mlp', 3, 'MLP identity control', '#AA3377', .24)]
    for ax, population in zip(axes, ['synthetic', 'natural']):
        for arch, head, label, color, offset in styles:
            for i, d in enumerate(models):
                mean, std = stat(values(d, arch, population, head))
                if mean is not None:
                    ax.errorbar(mean, i+offset, xerr=std, fmt='o', color=color,
                                markersize=4, capsize=2, label=label if i == 0 else None)
        ax.axvline(.5, color='gray', linestyle='--', linewidth=1)
        ax.set_xlim(-.02, 1.02); ax.set_xlabel('AUROC (error bars: probe initialization SD)')
        ax.set_title('Synthetic held-out queries' if population == 'synthetic' else 'Natural first error / legal repair')
        ax.grid(axis='x', alpha=.2)
    axes[0].set_yticks(range(len(models)),
        [d['entry']['id']+f" (natural n={d['datasets']['natural']['pairs']})" for d in models])
    axes[0].invert_yaxis(); axes[0].legend(fontsize=8, loc='lower left')
    fig.tight_layout(); fig.savefig(out/'error_detection_comparison.png', dpi=150); plt.close(fig)
    context = [d for d in models if d['entry']['kind'] == 'context_graph']
    fig, axes = plt.subplots(len(context), 2, figsize=(10, 3*len(context)), squeeze=False)
    for i, d in enumerate(context):
        for j, population in enumerate(['synthetic', 'natural']):
            ax = axes[i, j]
            for arch, color in [('linear', '#4477AA'), ('mlp', '#CC6677')]:
                stats = [stat(values(d, arch, population, h)) for h in range(4)]
                if stats[0][0] is not None:
                    ax.errorbar(range(4), [a for a,b in stats], yerr=[b for a,b in stats],
                                marker='o', color=color, label=arch, capsize=2)
            ax.axhline(.5, color='gray', linestyle='--', linewidth=1)
            ax.set_xticks(range(4), d['heads']); ax.set_ylim(.25, 1.01)
            ax.set_title(d['entry']['id']+' / '+population); ax.set_ylabel('AUROC'); ax.legend()
    fig.tight_layout(); fig.savefig(out/'context_readout_depth.png', dpi=150); plt.close(fig)
    lines = ['# 最后一个已输入 token 的错误识别：扩大人工数据与 MLP', '',
        r'读取前缀 $x_{\le t-1}$ 最后输入位置的 L4–L6 residual，判断 $x_{t-1}$ 是否违规。'
        '此时 $x_t$ 尚未采样，输入中没有 $x_t$、占位符或额外探针 token。'
        '这不是在生成 $x_{t-1}$ 前预判它是否会出错。', '',
        '导航模型与验证集选出的 best checkpoint 全部冻结。复用 Path/DFS/BFS 的两种任务，'
        '固定图保留三个 seed；另外保留三个固定图 BFS-state 对照。Direct/Filler 不适用轨迹错误识别。', '',
        '## 数据与标签', '',
        '- 人工训练目标每个 checkpoint 8,192 对，即最多 16,384 个二分类样本；验证 2,048 对，测试 4,096 对。'
        '新图分别使用 4,096 / 1,024 / 2,048 个原始查询；固定图使用已有 split 中的查询，多次取不同位置/替换。',
        '- 每对样本共享完全相同的合法历史，只在最后一个 token 不同：一个合法，一个违法。'
        '合法候选随机选取；仅监督首次局部违规，忽略错误后的连锁偏离及最终 YES/NO/EOS。',
        '- 人工错误包括图内错误节点、集合/边界处多余节点、过早输出分隔符、错误动作。'
        '按字段轮流抽样。Path 的人工训练只有图内节点错误，没有额外构造图外节点或语法错误。',
        '- 整个查询的所有改写均留在原始 train/val/test split；固定图中大量改写不等于大量独立查询或图。',
        '- 自然错误完全不参与训练或调参：对原有 256 个分层测试查询（固定图取最多可用数）贪心生成到原生上下文上限，'
        '取合法替换集合非空、可单 token 修复的首次语义错误，与只修复该 token 的合法前缀配对。'
        '它测量真实错误前缀与反事实修复的区分，'
        '不是自然连续 token 流中的错误率估计。',
        '- oracle 允许任意合法路径、集合序列化顺序和合法 PUSH。BFS-nodes 使用距离层级约束，'
        '不能检测全部违反具体队列顺序的情况。有限 oracle 不代表完备的导航正确性判定。', '',
        '## 探针与对照', '',
        '每层分别训练 affine 与单隐层 MLP（64 hidden、GELU、dropout 0.1）；相同数据、三个初始化 seed。'
        '训练集标准化，AdamW，最多 1,000 次更新，验证 BCE 分别选择各层最优更新。'
        '导航模型无梯度；多改写查询按逆样本数加权。没有保存探针权重、activation 缓存或新导航 checkpoint。', '',
        '每种架构都有匹配的 token 身份对照：最后输入 token 的冻结 embedding，加位置、字段、是否图内和是否目标节点。'
        '对照没有上下文或边列表，但知道采样字段，因此是一个较强的捷径基线。'
        '主指标 AUROC，同时保存正确/错误两类的概率、balanced accuracy、配对排序、字段和类型分数。', '',
        '人工训练正负先验为 1:1，sigmoid 输出不是自然生成中错误发生率的校准估计。'
        '本轮相对旧探针同时改变了数据来源、数量和读出约定，因此不能把跨轮差异单独归因于 MLP；'
        '本轮内部 Linear/MLP 的数据、读出和划分一致。', '',
        '## 结果', '',
        '以下均为 L6，均值 ± 三次探针初始化的标准差；它不表示测试样本不确定性。'
        '自然错误样本少时，不能根据微小差异得出稳定结论。', '',
        '| checkpoint | 人工训练对/查询 | 自然错误对 | 人工 Linear | 人工 MLP | 自然 Linear | 自然 MLP | 自然 MLP 身份对照 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    findings = []
    for d in models:
        train, natural = d['datasets']['train'], d['datasets']['natural']
        cells = [formatted(values(d, arch, population, head)) for population, arch, head in
                 [('synthetic','linear',2), ('synthetic','mlp',2), ('natural','linear',2),
                  ('natural','mlp',2), ('natural','mlp',3)]]
        lines.append(f"| {d['entry']['id']} | {train['pairs']}/{train['unique_queries']} | {natural['pairs']} | "+' | '.join(cells)+' |')
        syn_l = stat(values(d,'linear','synthetic',2))[0]
        syn_m = stat(values(d,'mlp','synthetic',2))[0]
        nat_l = stat(values(d,'linear','natural',2))[0]
        nat_m = stat(values(d,'mlp','natural',2))[0]
        control = stat(values(d,'mlp','natural',3))[0]
        findings.append(dict(model=d['entry']['id'], synthetic_mlp_minus_linear=syn_m-syn_l,
                             natural_mlp_minus_linear=None if nat_m is None else nat_m-nat_l,
                             natural_mlp_minus_identity=None if nat_m is None else nat_m-control,
                             natural_pairs=natural['pairs']))
    lines += ['', '自然样本少于 10 对（例如部分固定图 Path）只保留诊断数值，不作能力结论。'
              '人工训练与验证使用相同的合成策略，最优更新也按人工验证分布选择；'
              '自然测试的低分不能单独归因于探针容量不足。', '',
              '## 新图：仅图内错误节点子集', '',
              '下面排除过早分隔符、额外节点和错误动作。该子集仍含起点/状态抄写等局部一致性检查，'
              '不能把它的分数直接等同于可达性推理或完整搜索能力。', '',
              '| checkpoint | Linear L6 | MLP L6 | MLP 身份对照 | 测试错误对 |',
              '|---|---:|---:|---:|---:|']
    for d in context:
        cells = []
        for arch, head in [('linear',2), ('mlp',2), ('mlp',3)]:
            v = [f['synthetic'][head]['by_kind']['in_graph_node']['auroc'] for f in d['fits']
                 if f['architecture'] == arch]
            cells.append(formatted(v))
        n = d['datasets']['test']['categories'].get('in_graph_node', 0)
        lines.append('| '+d['entry']['id']+' | '+' | '.join(cells)+f' | {n} |')
    lines += ['', '## 人工到自然错误的分布落差', '']
    for d in context:
        if d['entry']['family'] not in {'dfs','bfs'}:
            continue
        vals = {}
        for population in ['synthetic','natural']:
            vals[population] = [f[population][2]['by_field']['candidates']['auroc']
                for f in d['fits'] if f['architecture']=='mlp' and f[population] is not None
                and 'candidates' in f[population][2]['by_field']]
        count = d['datasets']['natural']['fields'].get('candidates', 0)
        total = d['datasets']['natural']['pairs']
        lines.append(f"- {d['entry']['id']}：自然首次错误 {count}/{total} 在 candidates 字段。"
                     f"只看该字段，MLP L6 的人工 AUROC 为 {formatted(vals['synthetic'])}，"
                     f"自然 AUROC 为 {formatted(vals['natural'])}。这说明落差不只是字段比例变化。")
    for d in models:
        n = d['datasets']['natural']['pairs']
        v = values(d, 'mlp', 'natural', 2)
        if n >= 10 and v and np.mean(v) < .45:
            lines.append(f"- 异常：{d['entry']['id']} 的自然测试有 {n} 对，"
                         f"MLP L6 AUROC 为 {formatted(v)}，低于随机水平。"
                         "保留原训练标签方向，未根据测试集反转分数。这是泛化失败，不能解释为可靠错误识别。")
    lines += ['', '人工测试上的非线性可读性与自然错误泛化必须分别判断。'
              '特别是 Path 的自然错误类型包含人工节点替换训练未覆盖的格式/图外错误；'
              'MLP 更擅长人工分布不保证更擅长这些错误。'
              '分数高于身份对照也仅支持可从 activation 读出错误相关信息，不能证明模型主动监控错误或具备自我纠正机制。', '',
              '关键图：`error_detection_comparison.png`、`context_readout_depth.png`。'
              '所有模型的训练曲线、各层、字段、类型及少量异常前缀记录在 `models/*.json`；'
              '`findings.json` 保存跨分布变化，`summary.csv` 保存可比较表格。']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    (out/'findings.json').write_text(json.dumps(findings, indent=2)+'\n')
    print(json.dumps(findings, indent=2))


if __name__ == '__main__':
    main()
