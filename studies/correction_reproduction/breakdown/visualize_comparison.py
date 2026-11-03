"""Plot existing positive-only replay summaries; no model loading or inference."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from report import TASKS, NAMES, ARMS, LABELS, COLORS, causes, CNAMES

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'comparison_figures'
FOOT = '仅 DFS 正样本；每任务相同256个查询 × 4个seed = 1024次模型评估（非1024张独立图）。'


def save(fig, name, note=''):
    fig.text(.5, .012, FOOT + ('\n' + note if note else ''), ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .09, 1, .94))
    for ext in ['png', 'svg']:
        fig.savefig(OUT / f'{name}.{ext}', dpi=170)
    plt.close(fig)


def grouped(d, name, title, metric, ylabel, percent=True, ylim=None, note=''):
    fig, ax = plt.subplots(figsize=(13, 5.7))
    x = np.arange(len(TASKS))
    for i, (arm, label, color) in enumerate(zip(ARMS, LABELS, COLORS)):
        pools = [d['tasks'][t]['groups'][arm]['pooled'] for t in TASKS]
        values = np.array([metric(p) for p in pools], dtype=float)
        seeds = [[metric(s) * (100 / s['n'] if percent else 1)
                  for s in d['tasks'][t]['groups'][arm]['seeds']] for t in TASKS]
        if percent:
            values = values / np.array([p['n'] for p in pools]) * 100
        # Count bars pool four seeds; scatter seed counts scaled to the same denominator.
        seed_values = np.asarray(seeds) * (1 if percent else 4)
        pos = x + (i - 1) * .25
        bars = ax.bar(pos, values, .23, color=color, label=label, alpha=.85)
        ax.scatter(np.repeat(pos, 4), seed_values.ravel(), color='black', s=10, alpha=.35, zorder=3)
        for bar, val in zip(bars, values):
            ax.annotate(f'{val:.1f}%' if percent else f'{int(val)}',
                        (bar.get_x()+bar.get_width()/2, val), xytext=(0, 5),
                        textcoords='offset points', ha='center', fontsize=9)
    ax.set(xticks=x, xticklabels=NAMES, ylabel=ylabel, title=title)
    if ylim: ax.set_ylim(*ylim)
    ax.grid(axis='y', alpha=.2); ax.set_axisbelow(True)
    ax.legend(frameon=False, ncol=3, loc='upper right')
    save(fig, name, note + ' 黑点表示4个seed；计数图黑点为单seed计数×4，仅用于呈现波动。')


def main():
    OUT.mkdir(exist_ok=True)
    plt.rcParams.update({'font.sans-serif': ['Microsoft YaHei', 'Noto Sans CJK SC', 'DejaVu Sans'],
                         'axes.unicode_minus': False, 'font.size': 11, 'svg.fonttype': 'none'})
    d = json.loads((ROOT / 'summary.json').read_text(encoding='utf-8'))
    assert d['finished'] and d['positive_only']
    for t in TASKS:
        for arm in ARMS:
            p = d['tasks'][t]['groups'][arm]['pooled']
            assert p['n'] == 1024
            assert p['correct'] == sum(v for k, v in p['outcomes'].items() if k != 'wrong_answer')
            assert p['trace_valid'] == p['outcomes']['correct_clean_valid'] + p['outcomes']['correct_repaired_valid']
    grouped(d, '01_accuracy', '最终回答准确率', lambda p: p['correct'], '正样本回答准确率（%）', ylim=(0, 110))
    grouped(d, '02_legality', 'CoT 完整合法率：允许 MISTAKE 合法撤销并重写错误记录',
            lambda p: p['trace_valid'], '完整合法轨迹 / 全部正样本（%）', ylim=(0, 100),
            note='合法须完成完整搜索；仅前缀合法、提前终止不算合法。被正确撤销的错误记录不影响此指标。')
    grouped(d, '03_invalid_correct', 'CoT 不完整或不合法，但最终回答正确的数量',
            lambda p: p['correct'] - p['trace_valid'], '模型—查询评估次数（共1024次）', percent=False,
            note='含字段错误但动作可执行、非法执行、未完成轨迹；不能由此断言模型在猜答案。')

    fig, axes = plt.subplots(2, 3, figsize=(17, 10))
    for ax, task, name in zip(axes.flat, TASKS, NAMES):
        y = np.arange(len(CNAMES))
        for i, (arm, label, color) in enumerate(zip(ARMS, LABELS, COLORS)):
            p = d['tasks'][task]['groups'][arm]['pooled']; counts = causes(p)
            vals = counts / p['n'] * 100
            ax.barh(y+(i-1)*.24, vals, .22, color=color, label=label)
            for yi, v, n in zip(y+(i-1)*.24, vals, counts):
                if n: ax.text(v+.25, yi, str(n), va='center', fontsize=8)
        ax.set(title=name, yticks=y, yticklabels=CNAMES, xlim=(0, 44), xlabel='占全部正样本（%）')
        ax.invert_yaxis(); ax.grid(axis='x', alpha=.15); ax.set_axisbelow(True)
    axes.flat[5].axis('off')
    axes.flat[5].legend(*axes.flat[0].get_legend_handles_labels(), loc='center', frameon=False)
    fig.suptitle('首个错误 token 的原因：条长为占全部样本比例，标注为次数', fontsize=16)
    save(fig, '04_first_causes', '每个样本仅计首个错误；未发生错误与仅前缀合法但未完成的样本不在这些错误条中。')

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.8))
    x = np.arange(5)
    for ax, key, title in zip(axes, ['search_step', 'token'], ['首错真实搜索步', '首错 CoT token 位置']):
        for i, (arm, label, color) in enumerate(zip(ARMS, LABELS, COLORS)):
            q = [d['tasks'][t]['groups'][arm]['pooled']['positions'][key] for t in TASKS]
            mid = np.array([v['median'] for v in q]); lo = np.array([v['q25'] for v in q]); hi = np.array([v['q75'] for v in q])
            ax.errorbar(x+(i-1)*.15, mid, yerr=[mid-lo, hi-mid], fmt='o', capsize=4, color=color, label=label)
        ax.set(title=title, xticks=x, xticklabels=NAMES, ylabel='位置：中位数及25%–75%分位区间', ylim=(0, None))
        ax.grid(axis='y', alpha=.2)
    axes[0].legend(frameon=False)
    fig.suptitle('首错位置（仅在出现错误的轨迹内统计）', fontsize=16)
    save(fig, '05_first_positions', '这是条件分布；中位数后移不能单独证明整体改善。搜索步不因重试而增加；token位置从1起。')

    markers = json.loads((ROOT / 'marker_counts.json').read_text(encoding='utf-8'))
    fig, axes = plt.subplots(1, 2, figsize=(16, 6.3))
    specs = [('raw_mistake_tokens', '<MISTAKE> token总次数', '#8264a5'),
             ('true_mark_records', '可审计的正确报错次数', '#4c91b5'),
             ('repair_events', '局部修复事件次数', '#dd8844')]
    for i, (key, label, color) in enumerate(specs):
        vals = [sum(s[key] for s in markers[t]['correction']) for t in TASKS]
        bars = axes[0].bar(x+(i-1)*.25, vals, .23, color=color, label=label)
        axes[0].bar_label(bars, fontsize=8, padding=3)
    specs = [('any_raw_mistake','出现过 MISTAKE','#8264a5'), ('first_detected','首错被标记','#4c91b5'),
             ('first_repaired','首错后下一有效记录正确','#dd8844'), ('first_repaired_full_valid','首错修复后整条轨迹合法','#55a868')]
    for i, (key, label, color) in enumerate(specs):
        vals = [d['tasks'][t]['groups']['correction']['pooled'][key] for t in TASKS]
        bars = axes[1].bar(x+(i-1.5)*.2, vals, .19, color=color, label=label)
        axes[1].bar_label(bars, fontsize=8, padding=3)
    for ax, title in zip(axes, ['事件次数：同一样本可贡献多次', '样本次数：每个指标每样本最多计一次']):
        ax.set(title=title, xticks=x, xticklabels=NAMES, ylabel='次数')
        ax.grid(axis='y', alpha=.2); ax.set_axisbelow(True)
        ax.margins(y=.25); ax.legend(frameon=False, fontsize=9, loc='upper left')
    fig.suptitle('纠错模型：报错、局部重写和完整合法修复', fontsize=16)
    save(fig, '06_mistake_repairs', '两类baseline的上述计数均为0；非法动作后的状态可能无法审计，故可审计报错数不等于全部真实报错数。')

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for i, (key, label, color) in enumerate([('matched_clean','匹配干净对照',COLORS[1]),
                                            ('correction','纠错：允许 MISTAKE',COLORS[2]),
                                            ('blocked','纠错：禁用 MISTAKE','#8264a5')]):
        pools = [d['tasks'][t]['mistake_ablation']['blocked'] if key=='blocked'
                 else d['tasks'][t]['groups'][key]['pooled'] for t in TASKS]
        for ax, metric in zip(axes[:2], ['correct', 'trace_valid']):
            vals = [p[metric]/p['n']*100 for p in pools]
            bars=ax.bar(x+(i-1)*.25, vals, .23, color=color, label=label)
            ax.bar_label(bars, fmt='%.1f', padding=3, fontsize=8)
    for ax, title in zip(axes[:2], ['回答准确率（%）', '完整合法率（%）']):
        ax.set(title=title, xticks=x, xticklabels=NAMES, ylim=(0,110)); ax.grid(axis='y', alpha=.2); ax.set_axisbelow(True)
    ab = [d['tasks'][t]['mistake_ablation']['pooled'] for t in TASKS]
    helped = np.array([p['helped'] for p in ab]); harmed = np.array([p['harmed'] for p in ab])
    axes[2].bar(x-.18, helped, .35, color='#55a868', label='允许后由错变对')
    axes[2].bar(x+.18, -harmed, .35, color='#c95c5c', label='允许后由对变错')
    for j, (a,b) in enumerate(zip(helped,harmed)):
        axes[2].text(j-.18,a+2,str(a),ha='center',fontsize=9)
        axes[2].text(j+.18,-b-4,str(b),ha='center',va='top',fontsize=9)
        axes[2].text(j,0,f'净{a-b:+d}',ha='center',va='center',fontsize=9,bbox=dict(facecolor='white',edgecolor='none',alpha=.8))
    axes[2].set(title='同查询配对：允许 MISTAKE 的收益与损失', xticks=x, xticklabels=NAMES, ylabel='样本次数'); axes[2].axhline(0,color='grey',lw=.8)
    axes[2].margins(y=.2)
    axes[0].legend(frameon=False,fontsize=8,loc='upper center',bbox_to_anchor=(.5,1.0)); axes[2].legend(frameon=False,fontsize=9,loc='lower left')
    fig.suptitle('因果消融：同一个纠错checkpoint，仅将 MISTAKE 的生成logit设为 −∞', fontsize=15)
    save(fig, '07_mistake_ablation', '禁用动作没有重新训练；净收益为允许−禁用。困难图上的净收益为负，不代表每个seed都受损。')

    files = ['01_accuracy','02_legality','03_invalid_correct','04_first_causes','05_first_positions','06_mistake_repairs','07_mistake_ablation']
    text = ['# 纠错训练与 baseline：多维可视化', '', FOOT, '',
            '主比较包含原始baseline、匹配干净对照和纠错模型。匹配对照与纠错模型共享初始权重、词表和查询/轨迹采样，更适合隔离错误数据的影响。', '',
            '完整合法率允许 MISTAKE 撤销错误记录；CoT不合法但答对计数也包含未完成轨迹。所有数据来自已有冻结模型回放，无新增推理或训练。', '',
            '图中计数是模型—查询评估次数；同一256个查询在4个seed中重复评估。普通N100、窄深N100和困难DAG分别展示，后者是分布外测试。', '',
            '## 关键观察', '',
            '- 普通N100：匹配对照→纠错，答对806→856，完整合法415→506，不完整/不合法但答对391→350。',
            '- 禁用MISTAKE：N32答对918→898，N64 926→907，普通N100 856→853，窄深669→674，困难731→750。显式动作的作用随任务变化。',
            '- 普通N100首错523次中，182次被标记，36次下一有效记录修复正确，仅5次此后整条轨迹保持合法。局部修复与全程成功须分开理解。', '',
            '回放与历史保存的正样本预测有0.44%差异；图中全部使用同一批新回放的答案和轨迹，未混用历史答案。', '']
    for name in files: text += [f'![{name}](comparison_figures/{name}.png)', '']
    text += ['详细定义及逐项统计参见 [REPORT.md](REPORT.md)。', '']
    (ROOT / 'VISUAL_COMPARISON.md').write_text('\n'.join(text), encoding='utf-8')
    print(f'Wrote {len(files)} figures (PNG + SVG) and VISUAL_COMPARISON.md')


if __name__ == '__main__':
    main()
