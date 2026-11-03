"""Collect existing validation history (CPU only), or plot its compact export."""
import argparse
import json
from pathlib import Path

P = Path(__file__).resolve().parent


def collect(root, output):
    import yaml
    runs = []
    for path in sorted(root.glob('dfs_state/*/s*/metrics.jsonl')):
        cfg = yaml.safe_load((path.parent / 'resolved_config.yaml').read_text())
        tokens = 0
        history = []
        for line in path.read_text().splitlines():
            row = json.loads(line)
            tokens = row.get('tokens_seen', tokens)
            if 'free/overall/answer_accuracy' in row:
                history.append(dict(step=row['step'], tokens=tokens,
                    accuracy=row['free/overall/answer_accuracy'],
                    positive=row['free/overall/yes_recall'],
                    legal=row['free/semantic/trace_valid_rate'],
                    n=row['free/overall/n'], n_yes=row['free/overall/n_yes']))
        assert history and all(r['n'] == 64 and r['n_yes'] == 32 for r in history)
        runs.append(dict(name=str(path.parent.relative_to(root)),
            depth=cfg['model']['layers'], rate=round(100*cfg['protocol']['mistake_rate']),
            seed=cfg['train']['model_seed'], ends=cfg['train']['scale_stage_ends'],
            eval_root=cfg['data']['root'], history=history))
    assert len(runs) == 12
    assert len({r['eval_root'] for r in runs}) == 1
    output.write_text(json.dumps(runs, indent=2))


def plot(source):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    import numpy as np
    fonts = {f.name for f in font_manager.fontManager.ttflist}
    plt.rcParams.update({'font.family': 'Microsoft YaHei' if 'Microsoft YaHei' in fonts else 'DejaVu Sans',
        'axes.unicode_minus': False, 'axes.spines.top': False, 'axes.spines.right': False})
    runs = json.loads(source.read_text())
    out = P / 'figures'
    out.mkdir(exist_ok=True)
    for metric, title in [('positive', '正样本答案准确率'), ('legal', 'CoT 合法率（正负样本合并）')]:
        fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharey=True)
        for col, axis_key in enumerate(['step', 'tokens']):
            for row, depth in enumerate([6, 9]):
                ax = axes[row, col]
                for rate, color in zip([0, 30, 60], ['#487AAB', '#E19D39', '#BD5364']):
                    group = [r for r in runs if r['depth'] == depth and r['rate'] == rate]
                    grids, values = [], []
                    for run in group:
                        x = np.array([h[axis_key] for h in run['history']]) / (1000 if axis_key == 'step' else 1e9)
                        y = np.array([h[metric] for h in run['history']]) * 100
                        grids.append(x); values.append(y)
                        ax.plot(x, y, color=color, alpha=.25, lw=1)
                    assert np.allclose(grids[0], grids[1])
                    ax.plot(grids[0], np.mean(values, axis=0), color=color, marker='o', ms=3, label=f'{rate}% 错误数据')
                    # Different rates enter stages at different steps; mark each separately.
                    for end in group[0]['ends'][:-1]:
                        if axis_key == 'step':
                            ax.axvline(end/1000, color=color, alpha=.22, ls=':', lw=1)
                ax.set_title(f'{depth} 层 · ' + ('按训练步数' if axis_key == 'step' else '按累计非 padding token'))
                ax.set_xlabel('训练步数（千步）' if axis_key == 'step' else '累计 token（十亿）')
                ax.set_ylim(0, 102); ax.grid(alpha=.16)
                if col == 0: ax.set_ylabel(title + '（%）')
        axes[0, 0].legend(loc='lower right')
        fig.suptitle('不等步长、token 预算匹配训练｜' + title, fontsize=17)
        fig.text(.5, .015, '验证随课程依次切换 N16 → N36 → N64 → N100；每次 64 条（正样本 32 条）。粗线：2 种子均值；细线：各种子。\n左图彩色竖线：各比例的课程/验证规模切换；跨阶段不代表同分布表现变化。曲线无平滑，不混入最终测试。', ha='center', fontsize=10)
        fig.tight_layout(rect=[0, .065, 1, .95])
        for ext in ['png', 'svg']:
            fig.savefig(out/f'learning_{metric}.{ext}', dpi=170)
        plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path)
    parser.add_argument('--data', type=Path, default=P/'learning_history.json')
    args = parser.parse_args()
    if args.root:
        collect(args.root, args.data)
    else:
        plot(args.data)
