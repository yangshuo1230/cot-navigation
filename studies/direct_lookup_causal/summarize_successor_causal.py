"""Compact final report and two key figures for successor causal tests."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
OUT=Path.cwd()/'studies/direct_lookup_causal/successor_causal'
SEEDS=(201,303,407)
def pct(x):return 'NA' if x is None else f'{100*x:.1f}%'
def mean_sd(v):return f'{100*np.mean(v):.1f}% ± {100*np.std(v):.1f}pp'
def probe_rows(r,role,method,rank,split='test'):
    return [v['result'] for v in r['roles'][role]['curves'] if v['split']==split and v['method']==method and v['rank']==rank]
def native_rows(r,role,method):return [v for v in r['roles'][role]['curves'] if v['method']==method]
def pmean(rows,group='all_four_correct'):return np.mean([v[group]['counterfactual_accuracy'] for v in rows])

def main():
    probe={s:json.loads((OUT/f'{s}.json').read_text()) for s in ('simple','hard')}
    native={s:json.loads((OUT/f'{s}_native.json').read_text()) for s in probe}
    fig,axes=plt.subplots(2,2,figsize=(13,8),layout='constrained')
    methods=[('no_op','none'),('probe_rank64','probe64'),('native_successor_union','native succ.'),
        ('random_sphere_equal_norm','random'),('unrelated_output_nodes_equal_norm','other nodes'),('full_swap','full swap')]
    aligned='different_nonempty_new_successors'
    for col,s in enumerate(probe):
        for row,role in enumerate(('target','source')):
            ax=axes[row,col];acc=[];recall=[];ae=[];re=[]
            for method,_ in methods:
                rows=native_rows(native[s],role,method)
                aa=[v[aligned]['counterfactual_accuracy'] for v in rows]
                rr=[v[aligned]['native_head_new_successors']['recall_at_true_k'] for v in rows]
                acc.append(np.mean(aa));ae.append(np.std(aa));recall.append(np.mean(rr));re.append(np.std(rr))
            xx=np.arange(len(methods));ax.bar(xx-.18,recall,.36,yerr=re,label='Original head: new-successor recall',color='tab:blue')
            ax.bar(xx+.18,acc,.36,yerr=ae,label='Counterfactual answer success',color='tab:orange')
            n=native_rows(native[s],role,'no_op')[0][aligned]['n']
            ax.set(title=f'{s}: {role} L1, changed/nonempty successors, n={n}',xticks=xx,xticklabels=[label for _,label in methods],ylim=(0,1.08),ylabel='Recall / answer rate')
            ax.grid(axis='y',alpha=.2);ax.tick_params(axis='x',labelsize=9);ax.legend(fontsize=7)
    fig.savefig(OUT/'native_successor_effects.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(12,7),layout='constrained');ranks=probe['simple']['ranks']
    for col,s in enumerate(probe):
        for row,role in enumerate(('target','source')):
            ax=axes[row,col]
            for method,label,color in [('successor','Successor probe directions','tab:blue'),('successor_complement','Complement','tab:orange'),('random_equal_norm','Equal-norm random','grey')]:
                vals=[];errors=[]
                for rank in ranks:
                    scores=[v['all_four_correct']['counterfactual_accuracy'] for v in probe_rows(probe[s],role,method,rank)]
                    vals.append(np.mean(scores));errors.append(np.std(scores))
                ax.errorbar(range(len(ranks)),vals,yerr=errors,marker='o',color=color,label=label)
            ax.set(title=f'{s}: {role} L1 diagnostic-probe subspaces',xticks=range(len(ranks)),xticklabels=ranks,ylim=(-.04,1.05),xlabel='Rank',ylabel='Counterfactual answer success')
            ax.grid(alpha=.2);ax.legend(fontsize=7)
    fig.savefig(OUT/'probe_rank_profile.png',dpi=180);plt.close(fig)
    lines=['# 后继方向因果干预：探针方向与原生输出头方向','',
        '结论：原模型输出头可见的后继相关方向，在困难图中有部分因果作用，起点位置的效果尤其明显；但节点身份等信息也被改变，尚不能证明后继集合是独立、主要或足够的因果中介。仅按新训练后继探针定义的方向干预，虽然能改变探针的读出，却经常没有改变原模型自己的后继输出，不能据此断言模型不使用后继信息。',
        '', '## 设置与正对照','',
        '沿用 N100 简单/困难固定图 Direct 验证最佳 checkpoint，第1层 block结束处。128维残差、原模型冻结、后续第2～6层重新计算。冻结推理使用 CPU FP32、2 threads，避免占用当前8卡训练的额外前向算力；所有线性拟合 GPU0。未改原训练流程，无模型或 activation checkpoint。',
        '', '目标位置：接收 a=(sa,ta)，供体 b=(sa,tb)，保持起点相同；起点位置：供体 c=(sb,ta)，保持目标相同。所有配对仍满足 R(a)=R(d) != R(b)=R(c)。a、b 原 test；c、d 不要求 test。选择只用图标签；主面板再限定四问原本均答对：简单238、困难119组。配对共享查询与节点，不能当独立样本。',
        '', '| 图 | 目标位置完整状态替换 | 起点位置完整状态替换 |','|---|---:|---:|']
    for s,r in probe.items():lines.append(f"| {s} | {pct(pmean(probe_rows(r,'target','full_swap',0)))} | {pct(pmean(probe_rows(r,'source','full_swap',0)))} |")
    lines+=['',
        '简单图仅换起点L1的正对照本身只有8.0%成功：起点信息在第1层可能已写入未修改的目标/分隔符，故单独改起点太晚。不能将该臂的低效果解释为起点或其后继信息不重要。困难图两臂正对照接近100%，可检验第1层之后的传播。',
        '', '## A. 后继诊断探针的子空间移植','',
        '原 query train 上拟合 hW+b→焦点节点的完整直接后继集合；val选正则和集合F1阈值。W换回原始hidden坐标后SVD。对rank 1/2/4/8/16/32/64，P=U_r U_r^T，分别使用 h+P(h_d−h)、h+(I−P)(h_d−h)。目标臂对比祖先解码器；起点臂对比所有后代解码器。另有逐样本等编辑范数的3组随机方向、3组打乱节点→后继标签方向。',
        '', '注意：这些诊断探针有全图后继 oracle 标签，普通 query 划分中焦点身份通常已见；它们可以学习查表。子空间是候选干预方向，不自动具有“纯后继语义”。节点身份留出与随机输入对照此前见 relation_orientation/REPORT.md。',
        '', '所有角色/图按预先固定的 val 指标“后继成功率减去随机/打乱控制均值”都选中rank1，测试效果为0%；这是没有找到特异正效应的结果，不是rank1发现了有效机制。以下rank64为预设扫描的描述性切片：','',
        '| 图 | 位置 | 供体后继集合 probe F1 | 答案按供体关系改变 | 原模型输出头对新后继 recall |','|---|---|---:|---:|---:|']
    for s,r in probe.items():
        for role in ('target','source'):
            p=probe_rows(r,role,'successor',64)[0]['all_four_correct'];h=native_rows(native[s],role,'probe_rank64')[0]['all_four_correct']['native_head_new_successors']
            lines.append(f"| {s} | {role} | {p['successor_newset_f1']:.3f} | {pct(p['counterfactual_accuracy'])} | {pct(h['recall_at_true_k'])} (n={h['n_nonempty']}) |")
    lines+=['',
        'F1按面板的全部100维标签计数，原头recall只统计新后继集合非空的查询，分母不同。重要发现是：probe F1接近1时，原模型输出头仍大多不偏好供体后继。故这个阴性结果只说明该新探针的方向不是可靠中介，不说明真正后继信息没有因果作用。',
        '', '## B. 直接干预原模型输出头，不用新探针定义方向','',
        '对每例取真实原/新后继的并集C，选择原LM head中这些节点token的输出权重。令 q=(h−mu)/sigma、LayerNorm权重为gamma；原头对这些候选的线性敏感方向为 gamma⊙E_C（去除均值方向）。在q空间把对应子空间的原/供体分量移植，再恢复接收者均值/方差。目标臂并集rank为0～7以内、起点臂2～10以内；不同样本rank不同，使用图真值选候选，不是无oracle任务求解。',
        '', '随机球面方向和同数量无关输出节点方向各3 seed，对每例保持与原生后继编辑相同的角度、L2编辑范数、均值与方差。所有编辑最终仍回到h空间，并重新计算后续层。原词表argmax不是YES/NO的计为失败，原生后继编辑本次全部输出合法YES/NO。',
        '', '下面按同一完整机制面板报告答案改变率：','',
        '| 图 | 位置 | 原生后继方向 | 等范数随机 mean±sd | 无关节点方向 mean±sd |','|---|---|---:|---|---|']
    for s,r in native.items():
        for role in ('target','source'):
            actual=native_rows(r,role,'native_successor_union')[0]['all_four_correct']['counterfactual_accuracy']
            rand=[z['all_four_correct']['counterfactual_accuracy'] for z in native_rows(r,role,'random_sphere_equal_norm')]
            other=[z['all_four_correct']['counterfactual_accuracy'] for z in native_rows(r,role,'unrelated_output_nodes_equal_norm')]
            lines.append(f"| {s} | {role} | {pct(actual)} | {mean_sd(rand)} | {mean_sd(other)} |")
    lines+=['','为对齐后继recall与答案成功率分母，下表仅保留原/供体后继集合不同且新集合非空的固定子面板；所有方法使用同一筛选：','',
        '| 图 | 位置 | n | 原新后继recall→编辑后recall | 同子面板答案改变率 | 完整替换答案改变率 |','|---|---|---:|---|---:|---:|']
    for s,r in native.items():
        for role in ('target','source'):
            orig=native_rows(r,role,'no_op')[0][aligned];edit=native_rows(r,role,'native_successor_union')[0][aligned];full=native_rows(r,role,'full_swap')[0][aligned]
            lines.append(f"| {s} | {role} | {edit['n']} | {pct(orig['native_head_new_successors']['recall_at_true_k'])} → {pct(edit['native_head_new_successors']['recall_at_true_k'])} | {pct(edit['counterfactual_accuracy'])} | {pct(full['counterfactual_accuracy'])} |")
    lines+=['',
        '困难图起点臂具有部分特异因果信号：答案46.2%按新起点关系改变，高于随机7.6%及无关节点21.3%。目标臂22.7%与无关节点19.0%较接近，不能同样声称强特异性。对照seed标准差不是置信区间，未做独立样本显著性检验。',
        '', '## 身份与混杂检查','',
        '| 图 | 位置 | 原生后继编辑后原焦点身份保留 | 供体焦点身份采用 |','|---|---|---:|---:|']
    for s,r in native.items():
        for role in ('target','source'):
            z=native_rows(r,role,'native_successor_union')[0]['all_four_correct']
            lines.append(f"| {s} | {role} | {pct(z['original_identity_retained'])} | {pct(z['donor_identity_adopted'])} |")
    lines+=['',
        '因此尚未实现“保留节点身份、只改变后继知识”。原生方向来自候选节点输出权重，也可能改变身份、其他结构或激活分布。困难图起点臂的部分因果作用不能直接归因于后继集合本身；其他方向或位置仍可能保留后继信息。',
        '', '## 目标后继集合相同，原答案却不同','',
        '测试面板中，简单图23组、困难图13组的两个目标后继集合完全相同，而同一起点对它们的可达性相反。这说明目标自己的后继集合S(t)并不足以单独决定s→t。实际例子：','']
    for s,r in probe.items():
        e=next(z for z in r['examples'] if z.get('type')=='identical_successor_sets_different_reach')
        lines.append(f"- {s}：起点 {e['source']}，目标 {e['old_target']} 与 {e['new_target']} 的后继均为 {e['successors']}；前者图标签 {e['old_label']}，后者 {e['new_label']}。因此保留原起点及其后继、仅读取目标后继也不够。不能因此否定当前起点后继与其他目标信息组合的算法。")
    lines+=['','## 有趣实例与验证','']
    for s,r in native.items():
        for role in ('target','source'):
            z=native_rows(r,role,'native_successor_union')[0]['strong_native_head_change']
            lines.append(f"- {s}/{role}：新后继recall从≤20%变为≥80%的事后质量子集n={z['n']}，答案反事实成功率 {pct(z['counterfactual_accuracy'])}。该子集是描述性筛选，不用于跨方法公平比较。")
            ex=r['roles'][role]['examples']
            neg=next((e for e in ex if e['edited_answer']==e['original_label']),None)
            if neg:
                lines.append(f"  例 query {neg['original_query_id']}→供体query {neg['donor_query_id']}：新后继 {neg['new_successors']} 的recall {pct(neg['new_successor_recall_before'])}→{pct(neg['new_successor_recall_after'])}，答案仍为 {neg['edited_answer']}，反事实标签为 {neg['desired_label']}。原始节点与少量正/负效应实例已写入*_native.json。")
    lines+=['',
        '图标签由独立BFS复核；源与目标均为单节点token；原模型冻结。原生编辑和随机对照的逐例均值、方差、L2范数检查全部通过。无反事实状态/参数写回训练模型。代码只为通用forward_patch/values按模型设备分配张量，既有cuda默认调用保持不变。',
        '', '复现：从repo root执行 `.venv/bin/python studies/direct_lookup_causal/successor_causal.py`，再执行 successor_native_control.py、summarize_successor_causal.py。数据为simple.json、hard.json（probe方向曲线）和*_native.json（原生方向、控制与实例）。只保存两张关键图，无activation/probe/模型checkpoint。',
        '', '当前结论：后继相关方向具有部分因果影响，尤其困难图起点；单纯的新探针读出变化不构成这一证据。尚未证明完整后继集合是Direct的主要因果中介，也未排除节点身份或其他关系编码解释。每类只一个图/模型，原/供体训练暴露、模型初始化与图结构仍有混杂。']
    for s,r in native.items():
        for role,z in r['roles'].items():assert z['norm_and_moment_checks_passed']
    (OUT/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8');print('\n'.join(lines))

if __name__=='__main__':main()
