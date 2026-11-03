"""One key figure and report for relation-direction comparisons."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
OUT=Path.cwd()/'studies/direct_lookup_causal/relation_orientation'
TASKS=('predecessors','successors','ancestors','descendants')
NAMES=dict(predecessors='直接前驱',successors='直接后继',ancestors='所有祖先',descendants='所有后代')
def pct(x):return f'{100*x:.1f}%'
def stats(values):return f'{np.mean(values):.3f} ± {np.std(values):.3f}'

def main():
    results={s:json.loads((OUT/f'{s}.json').read_text()) for s in ('simple','hard')}
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    lines=['# 前驱、后继、祖先、后代：同一状态的方向对照','',
        '结果：Direct 节点状态中有明确的直接后继信号。原模型输出头无需新训练即可给出较高后继召回；同一状态也能被祖先探针解码，因此两种信息并不互斥。当前没有做后继信息的选择性因果干预，不能据此断定可达性答案主要由直接后继集合计算。',
        '', '## 原输出头，无新增探针','',
        '沿用 N100 简单图、困难图的原验证最佳 Direct checkpoint。冻结 native FP32 前向，把各层 source/target 状态通过原 final_norm + LM head（未校准 logit lens）。主表是目标位置状态，k 为真实直接后继数；按每个非空集合查询计算 top-k recall，再平均。候选排名排除焦点节点和查询另一个端点；样本仅原 test。另报告完整词表 softmax 后真实后继的概率质量，避免只在节点词表中重新归一化造成虚高。',
        '', '| 图 | 目标 L1 后继 recall@k | 目标 L1 完整词表后继质量 | 目标 L6 后继 recall@k | 非空后继测试 query 数 |','|---|---:|---:|---:|---:|']
    for s,r in results.items():
        a=r['roles']['target']['successors']['original_head'];lines.append(f"| {s} | {pct(a[1]['all_other_nodes']['recall_at_true_k'])} | {pct(a[1]['full_vocab_relation_mass'])} | {pct(a[6]['all_other_nodes']['recall_at_true_k'])} | {a[1]['all_other_nodes']['n']} |")
    lines+=['','起点位置也有该模式：','',
        '| 图 | 起点 L1 后继 recall@k | 起点 L6 后继 recall@k |','|---|---:|---:|']
    for s,r in results.items():
        a=r['roles']['source']['successors']['original_head'];lines.append(f"| {s} | {pct(a[1]['all_other_nodes']['recall_at_true_k'])} | {pct(a[6]['all_other_nodes']['recall_at_true_k'])} |")
    lines+=['','原节点输出头与 WALK 的下一节点训练目标天然相符，故此证据说明后继可被原模型输出头访问，但不是其用于 Direct 最终 YES/NO 的证明。',
        '', '## 相同容量探针，节点身份留出与随机向量','',
        '四个标签：直接前驱=入边邻居、直接后继=出边邻居、祖先=所有可达前驱、后代=所有可达后继，均不含自身。相同 128→100 仿射 ridge、相同查询分区。三个焦点节点身份划分 seed 17/31/47，并与原 query split 取交集。原模型见过焦点节点，probe 未见。三个固定节点随机向量 seed 201/303/407，按各层 train-only 均值/标准差调整分布，并执行相同标准化和正则选择。不存在每个 query 新抽随机向量的弱对照。',
        '', '指标固定一个候选节点，在不同 heldout 焦点节点之间比较关系是否存在：先聚合重复 query，再逐候选算 AUROC，避免固定节点出现频率获益；该频率基线为 .5。前驱/祖先只评估候选拓扑序更早的情况，后继/后代只评估更晚的情况。不是准确率，也不能把不同任务 AUROC 直接作为信息强弱排序，因为关系密度、标签支持、难度均不同。',
        '', '| 图 | 角色 | L1 关系标签 | 实际状态 AUROC（3 split） | 随机向量 AUROC（3 split×3 vector seed） | 有正负标签的候选数范围 |','|---|---|---|---|---|---|']
    for s,r in results.items():
        for role in ('source','target'):
            for task in TASKS:
                items=r['roles'][role][task]['heldout']
                actual=[q['actual'][1]['direction_valid_candidates']['fixed_candidate_across_targets']['auc'] for q in items]
                control=[q['random'][str(seed)][1]['direction_valid_candidates']['fixed_candidate_across_targets']['auc'] for q in items for seed in (201,303,407)]
                counts=[q['actual'][1]['direction_valid_candidates']['fixed_candidate_across_targets']['n_candidates'] for q in items]
                lines.append(f"| {s} | {role} | {NAMES[task]} | {stats(actual)} | {stats(control)} | {min(counts)}–{max(counts)} |")
                for q in items:
                    freq=q['frequency']['direction_valid_candidates']['fixed_candidate_across_targets']['auc']
                    assert freq is None or abs(freq-.5)<1e-9
    lines+=['','标准差混合了不同节点划分或随机种子影响，不是置信区间，也不是显著性检验。每个关系、层、角色的完整 set precision/recall/F1、exact-set、ranking 与支持数保留在两个 JSON 中；包含空集合时的集合指标与只对非平凡集合排名的指标分母不同。',
        '', '## 对机制解释的影响','',
        '不能由“祖先探针的前32维子空间不是可靠因果中介”推出“剩下的一定是后继集合”。目前确认后继信号存在，但可能与祖先、后代或身份信息纠缠。目标的直接后继 S(t) 本身不能一般性判定 s→t；更直接的判据是 s∈A(t)、t∈D(s)，或 S(s)∩(A(t)∪{t})非空。因此需要区分起点后继与目标后继，也需要区分一跳后继与多跳后代。',
        '', '例：图1有 s→a→t、t→z；图2有 s→a、t→z，但没有 a→t。两图中 S(s)={a}、S(t)={z} 都相同，s→t 的答案却不同。只读取端点的直接后继集合不足以求解一般图可达性。固定单图下，模型仍可能通过其他已学图知识快速匹配。',
        '', '下一项有区分力的实验是用同起点供体移植后继解码器子空间，与祖先子空间、同范数随机、打乱标签和身份保持控制一起比较；本次尚未执行此因果实验，不预先声称后继被实际使用。',
        '', '## 执行与复现','',
        '冻结模型前向和原头评估放 CPU、Torch 2 threads，避免占用当前8卡训练的额外前向算力；全部 ridge 拟合按要求放 GPU0。没有停止其他训练，没有写入 activations、probe 权重或模型 checkpoint。图可达性由原 loader 独立 BFS 验证。',
        '', '代码：relation_orientation.py；从 repo root 执行 `.venv/bin/python studies/direct_lookup_causal/relation_orientation.py`，再运行 summarize_relations.py。原 load(setting) 新增可选 device 参数，默认仍为 cuda:1，本实验使用 CPU，既有调用行为不变。',
        '', '每类只一张固定图、一个 checkpoint。任务差异和随机初始化、训练轨迹混在一起，不能把机制差异单独归因于图难度。']
    for col,(s,r) in enumerate(results.items()):
        ax=axes[0,col]
        for role,color in (('source','tab:blue'),('target','tab:orange')):
            scores=r['roles'][role]['successors']['original_head']
            ax.plot(range(7),[v['all_other_nodes']['recall_at_true_k'] for v in scores],'o-',color=color,label=role+' recall@degree')
            ax.plot(range(7),[v['full_vocab_relation_mass'] for v in scores],'--',color=color,label=role+' full-vocab mass')
        ax.set(title=s+': original output head / direct successors',xticks=range(7),xticklabels=r['layers'],ylim=(-.03,1.04),ylabel='Recall / probability mass')
        ax.grid(alpha=.2);ax.legend(fontsize=8)
        ax=axes[1,col];actual=[];rand=[];err=[];rerr=[]
        for task in TASKS:
            items=r['roles']['target'][task]['heldout']
            a=[q['actual'][1]['direction_valid_candidates']['fixed_candidate_across_targets']['auc'] for q in items]
            z=[q['random'][str(seed)][1]['direction_valid_candidates']['fixed_candidate_across_targets']['auc'] for q in items for seed in (201,303,407)]
            actual.append(np.mean(a));err.append(np.std(a));rand.append(np.mean(z));rerr.append(np.std(z))
        xx=np.arange(4);ax.bar(xx-.18,actual,.36,yerr=err,label='Actual target L1',color='tab:orange')
        ax.bar(xx+.18,rand,.36,yerr=rerr,label='Random node identities',color='grey')
        ax.axhline(.5,ls=':',color='black');ax.set(title=s+': held-out focal-node probes',xticks=xx,xticklabels=['predecessors','successors','ancestors','descendants'],ylim=(0,1.08),ylabel='AUROC across held-out focal nodes')
        ax.grid(axis='y',alpha=.2);ax.legend(fontsize=8)
    fig.savefig(OUT/'relation_orientation.png',dpi=180);plt.close(fig)
    (OUT/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8');print('\n'.join(lines))

if __name__=='__main__':main()
