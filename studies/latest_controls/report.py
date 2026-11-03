"""Report the completed short-span and query-block controls; no model loading."""
import csv
import json
from pathlib import Path


def main():
    root=Path(__file__).resolve().parent
    s=json.loads((root/'summary.json').read_text())
    assert len(s['short'])==len(s['block'])==12
    assert all(r['complete'] and r['last_step']==125000 for key in ['short','block'] for r in s[key])
    rows=[]
    for suite in ['short','block']:
        for r in s[suite]:
            for tag,t in r['tests'].items():
                views=['short','long'] if suite=='short' else list(t)
                for view in views:
                    m=t[view]
                    rows.append(dict(suite=suite,protocol=r['protocol'],arm=r['arm'],checkpoint=tag,
                        step=t['checkpoint_step'] if suite=='short' else m['checkpoint_step'],view=view,
                        **{k:m.get(k) for k in ['n','n_yes','n_no','n_correct','answer_accuracy','balanced_accuracy','yes_recall','no_recall','cot_interrupted_rate','forced_stop_rate']}))
    with (root/'metrics.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    names={'direct':'Direct','path':'Path','dfs_state':'DFS state','bfs_state':'BFS state','dfs_nodes':'DFS nodes','bfs_nodes':'BFS nodes'}
    protocols=list(names)
    find=lambda suite,p,arm:next(r for r in s[suite] if r['protocol']==p and r['arm']==arm)
    lines=['# 最新训练汇总：2026-10-20','',
        '两套对照共 24 个任务模型，均训练至 125000 步并完成 best/last 测试。共享的 WALK-only 知识阶段另计。',
        '后台调度器后续承接的 20 个任务于北京时间 2026-10-20 02:36 全部完成，失败 0；另外 4 个 Direct/Path 三跳任务在此之前已完成。',
        '下表主结果使用验证集选出的 best checkpoint，没有按测试成绩挑选 checkpoint 或 seed。', '',
        '## 三跳 QUERY 训练：长度外推','',
        '固定 N100 图，short3 前向训练 QUERY 层差 1–3；all_hops 为 1–9。反向负例均占 QUERY batch 的 12.5%。',
        '同样的前向测试：2–3 跳 67 对（33 YES），4–9 跳 189 对（96 YES）。all_hops 的长跨度测试属于同跨度未见查询，不属于长度外推。','',
        '| 协议 | 三跳训练：2–3 跳准确率 | 三跳训练：4–9 跳准确率 | 三跳训练：长跨度 YES recall | 全跳训练：4–9 跳准确率 | 三跳最后 checkpoint：4–9 跳准确率 |',
        '|---|---:|---:|---:|---:|---:|']
    pct=lambda x:f'{x:.1%}'
    for p in protocols:
        a=find('short',p,'short3')['tests'];b=find('short',p,'all_hops')['tests']
        vals=[a['best']['short']['answer_accuracy'],a['best']['long']['answer_accuracy'],a['best']['long']['yes_recall'],b['best']['long']['answer_accuracy'],a['last']['long']['answer_accuracy']]
        lines.append('| '+names[p]+' | '+' | '.join(map(pct,vals))+' |')
    lines += ['',
        'DFS nodes 的外推最好（167/189=88.4%）；Path 的正例 recall 仅 14/96=14.6%。',
        'DFS state 的长跨度 STOP 率为 22.8%，其中 19.0% 是预算触发的强制 STOP；BFS state 分别为 12.7% 和 5.3%。因此状态协议的差距混有轨迹预算和提前停止问题。',
        '限制的是 QUERY 起终点层差，WALK 和 DFS/BFS 搜索展开深度没有限制。长度超过 3 的搜索可以在训练中出现。','',
        '## 跨块 QUERY 留出','',
        '同图按第 1–5 / 6–10 层分块。within_only 的 QUERY 只在块内；cross_allowed 允许跨块。',
        '两组都训练 1–4 跳前向 QUERY，按层差、标签、方向匹配独立训练查询数（各 2295 对），其余预算与 WALK 相同。',
        '验证仅使用块内查询。测试：块内 367 对；跨块 2–4 跳 266 对，其中 70 YES、196 NO。',
        '**全猜 NO 的跨块 accuracy 为 73.7%，所以主要比较 balanced accuracy 和 YES recall。**','',
        '| 协议 | 块内训练：块内 BA | 块内训练：跨块 BA | 允许跨块训练：跨块 BA | 块内训练：跨块 YES recall | 允许跨块训练：跨块 YES recall |',
        '|---|---:|---:|---:|---:|---:|']
    for p in protocols:
        a=find('block',p,'within_only')['tests']['best'];b=find('block',p,'cross_allowed')['tests']['best']
        vals=[a['within_short']['balanced_accuracy'],a['cross_short']['balanced_accuracy'],b['cross_short']['balanced_accuracy'],a['cross_short']['yes_recall'],b['cross_short']['yes_recall']]
        lines.append('| '+names[p]+' | '+' | '.join(map(pct,vals))+' |')
    lines += ['',
        'Direct 和 Path 的块内训练模型均仅识别出 1/70 个跨块正例；允许跨块监督后分别识别 38/70、53/70。',
        '块内训练的 DFS state、DFS nodes 分别识别 41/70、33/70 个跨块正例；有跨块监督时 DFS nodes 达到 66/70。','',
        '### 进一步控制','',
        '在端点角色均见过的跨块子集（61 对、21 YES）上，块内训练模型的 BA：',
        '| 协议 | BA |', '|---|---:|']
    for p in protocols:lines.append(f"| {names[p]} | {pct(find('block',p,'within_only')['tests']['best']['cross_short_role_covered']['balanced_accuracy'])} |")
    lines += ['',
        'WALK 未完整展示的跨块正例有 29 对，但它们不等于所有训练信息未见：',
        '- 在 Path 的训练轨迹中，至少 26/29 对已经以查询根到访问节点的关系出现。',
        '- 在四种 DFS/BFS 训练轨迹中，29/29 对均有这种曝光。',
        '- 因而 CoT 的跨块成功可以说明没有跨块 QUERY 标签时仍有迁移，但不能直接证明它凭完全未见关系执行了规划。',
        '- Direct 的跨块一跳 QUERY 也被禁止，边界边只通过 WALK 学习；这与论文保留全部一跳任务监督的设置不同。','',
        '## 结果边界与下一步','',
        '这里统计最终答案，不要求生成轨迹合法。所有结果来自同一图 seed 和单次训练，尚不是跨图或多训练 seed 的稳定性结论。',
        '多数模型的 last 不及验证所选 best，不能把当前问题简单归因于 125k 步仍不够；同时这也不能排除更长训练下的 grokking。',
        '最有信息量的后续对照是保留全部边的一跳 QUERY（包括边界边），仅留出跨块多跳 QUERY；另外需把跨块搜索轨迹曝光与纯知识组合能力分开。',
        '本次仅汇总结果，没有启动额外训练。','',
        '完整 best/last、各距离分桶、STOP、验证选点及 W&B 地址在 summary.json；平铺数据在 metrics.csv。']
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    fig,axes=plt.subplots(1,2,figsize=(13,4.5),constrained_layout=True)
    x=np.arange(6);w=.36
    for ax,suite,arms,view,metric,title in [
        (axes[0],'short',['short3','all_hops'],'long','answer_accuracy','4-9 hop test accuracy'),
        (axes[1],'block',['within_only','cross_allowed'],'cross_short','balanced_accuracy','Cross-block 2-4 hop balanced accuracy')]:
        for i,arm in enumerate(arms):
            vals=[find(suite,p,arm)['tests']['best'][view][metric] for p in protocols]
            bars=ax.bar(x+(i-.5)*w,vals,w,label=arm)
            ax.bar_label(bars,labels=[f'{v:.0%}' for v in vals],fontsize=8,padding=3)
        ax.set_xticks(x,['Direct','Path','DFS\nstate','BFS\nstate','DFS\nnodes','BFS\nnodes'])
        ax.set_ylim(0,1.12);ax.set_yticks([0,.25,.5,.75,1],['0%','25%','50%','75%','100%'])
        ax.set_title(title);ax.legend(fontsize=8,loc='upper center',bbox_to_anchor=(.5,-.2),ncol=2);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    fig.savefig(root/'key_results.png',dpi=180);plt.close(fig)
    print('REPORT: 24 complete runs; CSV + one key figure')


if __name__=='__main__':main()
