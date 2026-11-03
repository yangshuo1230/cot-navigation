"""Produce only two key figures and one evidence-linked report."""
from pathlib import Path
import json
import re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
def read(name): return json.loads((HERE/f'{name}.json').read_text())
def pct(value): return f'{100*value:.1f}%'
def pp(value): return f'{100*value:.1f}'
def avg(values): return float(np.mean(values))

main = {(g, p): read(f'{g}_{p}') for g in ('simple', 'hard') for p in ('direct', 'path')}
follow = {(g, p): read(f'{g}_{p}_followup') for g in ('simple', 'hard') for p in ('direct', 'path')}
edits = {g: read(f'{g}_path_weight_edit') for g in ('simple', 'hard')}
plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})

fig, axes = plt.subplots(2, 2, figsize=(14, 8.5), layout='constrained')
for row, g in enumerate(('simple', 'hard')):
    functional = main[g, 'path']['functional']
    measures = [functional['baseline'], functional['ablations']['ffn_0_last'],
                functional['first_ffn_roles']['target'], functional['ablations']['attention_1_all']]
    x = np.arange(4)
    axes[row, 0].bar(x-.18, [100*v['node_top1_edge'] for v in measures], .36, label='Top node is a true successor')
    axes[row, 0].bar(x+.18, [100*v['within_successor'] for v in measures], .36, label='Useful choice within true successors')
    axes[row, 0].axhline(100*functional['baseline']['within_successor_random'], color='gray', ls=':', label='Uniform-neighbor useful-choice baseline')
    axes[row, 0].set(xticks=x, xticklabels=['Native', 'FFN1 off\nat current', 'FFN1 off\nat goal', 'Attention2 off\nat all positions'],
                     ylim=(0, 105), ylabel='Query-balanced rate (%)', title=f'{g.title()} PATH: functional interventions')
    records = edits[g]['interventions']; x = np.arange(len(records))
    axes[row, 1].bar(x-.18, [100*r['edited']['final_new_top1'] for r in records], .36, label='Rank-one FFN association edit')
    axes[row, 1].bar(x+.18, [100*avg([c['recipient']['final_new_top1'] for c in r['controls'].values()]) for r in records], .36, label='Same-key, equal-norm random value (3 seeds)')
    axes[row, 1].set(xticks=x, xticklabels=[r['source_node'].strip('<>')+'\n'+str(r['baseline']['n'])+' decisions' for r in records],
                     ylim=(0, 105), ylabel='Output belongs to donated successor set (%)', title=f'{g.title()} PATH: embeddings/head unchanged')
    if row == 0:
        axes[row, 0].legend(fontsize=8, loc='lower right'); axes[row, 1].legend(fontsize=8, loc='lower right')
fig.savefig(HERE/'functions_and_edits.png', dpi=170); plt.close(fig)

fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout='constrained')
for row, g in enumerate(('simple', 'hard')):
    for col, p in enumerate(('path', 'direct')):
        d = follow[g, p]; ax = axes[row, col]
        x = np.array([8, 16, 32, 64, 128]); y = np.array([d['size_scan'][str(k)]['interaction']['own_minus_other_drop'] for k in x])*100
        ci = np.array([d['size_scan'][str(k)]['interaction']['ci95'] for k in x])*100
        ax.plot(x, y, 'o-', label='Spectral regions, query-heldout')
        ax.fill_between(x, ci[:, 0], ci[:, 1], alpha=.15)
        z = d['panels']['node_heldout']['interaction']; ax.errorbar([32], [100*z['own_minus_other_drop']],
            yerr=[[100*(z['own_minus_other_drop']-z['ci95'][0])], [100*(z['ci95'][1]-z['own_minus_other_drop'])]], fmt='s', label='Discovery-node heldout')
        random = [100*v['interaction']['own_minus_other_drop'] for v in d['random_partitions'].values()]
        ax.scatter([32]*3, random, marker='x', color='gray', label='Random node partitions (3 seeds)')
        if 'large_controls' in d:
            z = d['large_controls']['node_heldout']['interaction']; ax.errorbar([128], [100*z['own_minus_other_drop']],
                yerr=[[100*(z['own_minus_other_drop']-z['ci95'][0])], [100*(z['ci95'][1]-z['own_minus_other_drop'])]], fmt='s', color='tab:orange')
            random = [100*v['interaction']['own_minus_other_drop'] for k, v in d['large_controls'].items() if k.startswith('random')]
            ax.scatter([128]*3, random, marker='x', color='gray')
        ax.axhline(0, color='black', lw=.7); ax.set(xlabel='Neurons removed per layer (of 512)', ylabel='Own-region excess drop (percentage points)',
            title=f'{g.title()} {p.upper()}', xticks=x)
        if row == 0 and col == 0: ax.legend(fontsize=8)
fig.savefig(HERE/'regional_controls.png', dpi=170); plt.close(fig)

lines = [
'# Fixed DAG：图位置、功能分工与 embedding 存储假设', '',
'结论：存在节点身份相关的 FFN 贡献差异，以及可因果区分的局部后继提取、目标条件选择和输出协议功能。尚未证实按图连通区域形成可推广到其他节点的独立模块；也不能唯一确定图知识在 embedding 与后续参数之间的存储比例。', '',
'## 范围与定义', '',
'沿用此前 N100 简单/困难 Fixed DAG，各一个 Direct、一个 PATH 验证最佳 checkpoint，共四个模型；没有重新训练主模型。架构均为 6 层、单头、hidden 128、FFN 512，checkpoint SHA256 在各 JSON 中核验。主模型训练见过所有节点；节点留出仅针对本次定位神经元或拟合探针。', '',
'冻结模型 FP32 native SDPA 使用原先空闲 GPU1；线性探针拟合使用 GPU0。未停止、修改原训练任务。没有保存模型/探针 checkpoint、activation dump 或 W&B 实验。脚本只放在 studies/fixed_graph_storage。', '',
'PATH 测量只用正例查询的原生生成轨迹：当前位置之前必须合法、尚未到达目标，每个查询最多抽三个决策点。保留当前决策可能出错的情况。训练查询用于发现神经元，原 test 用于干预评分；所有报出的 PATH 率先在每个查询内平均，再跨查询平均。它们是固定原生前缀上的下一步实验，不是干预后的完整自由生成成功率。Direct 使用正向原测试查询，包含正负例。', '',
'## 1. PATH 中可因果区分的功能', '',
'保持所有 embedding/输出头不变，单独将指定 FFN 或 attention 的 residual 分支置零，并重算后续层。为排除“只破坏输出格式”，同时看节点词表内最高分节点是否合法，以及在真实后继集合内是否选择可通往目标的节点。后者只统计同时存在有用/无用后继的分支点，不是最终答案准确率。', '',
'| 图 | 干预 | 节点词表 top1 为真实后继 | 在真实后继内选择有用分支 |',
'|---|---|---:|---:|']
for g in ('simple', 'hard'):
    d = main[g, 'path']['functional']
    for name, value in [('原模型', d['baseline']), ('只去掉当前位置 FFN1', d['ablations']['ffn_0_last']),
                        ('只去掉输入目标位置 FFN1', d['first_ffn_roles']['target']), ('去掉所有位置 attention2', d['ablations']['attention_1_all'])]:
        lines.append(f'| {g} | {name} | {pct(value["node_top1_edge"])} | {pct(value["within_successor"])} |')
for g in ('simple', 'hard'):
    d = main[g, 'path']['functional']
    lines.append(f'\n{g} 的混合分支点均匀随机后继选择基线为 {pct(d["baseline"]["within_successor_random"])}；原测试决策 {main[g,"path"]["sample_counts"]["2"]} 条。\n')
lines += ['',
'当前位置 FFN1 对本地合法后继偏好非常重要；目标位置 FFN1 和后续 attention 对方向选择也有作用。这个分工是位置/层/分支级的因果证据，不意味着一个完整功能只由单个层实现。分支删除是较强分布外干预，不能据此排除冗余或补偿电路。', '',
'## 2. Direct：结构处理与输出格式可以分开', '',
'原先若只看 FFN1 全部删除后的原生准确率，会把非法输出与关系错误混为一谈。这里补充固定 YES/NO 二选一的 balanced accuracy，并检查原生答案 token 合法率。', '',
'| 图 | 干预 | 原生准确率 | YES/NO 限定 BA | 原生输出为 YES/NO |',
'|---|---|---:|---:|---:|']
for g in ('simple', 'hard'):
    d = main[g, 'direct']['functional']
    for name, value in [('原模型', d['baseline']), ('FFN1 只在节点位置删除', d['first_ffn_roles']['nodes']),
                        ('FFN1 只在分隔符位置删除', d['first_ffn_roles']['separator']), ('全部 FFN 删除', d['ablations']['ffn_all_all'])]:
        lines.append(f'| {g} | {name} | {pct(value["exact"])} | {pct(value["constrained_answer_balanced"])} | {pct(value["answer_valid"])} |')
lines += ['', '节点位置 FFN1 的删除会在仍然输出合法 YES/NO 的情况下降低关系判断；分隔符位置在简单图上更多影响输出协议。这支持不同位置承担不同作用，不能把第一层 FFN 的全部影响解释成图查找或全部解释成格式控制。', '',
'## 3. 图区域是否对应不同 FFN 神经元', '',
'图区域由无向化邻接矩阵的归一化 Laplacian 第二特征向量划分，并在每个拓扑 rank band 内各分一半，避免区域只是浅/深层。另测深度三段与目标区域，完整结果在主 JSON；深度三段不能匹配同 rank，且 Direct 的匹配测试仅简单54/困难24条，不作强结论。', '',
'训练原生前缀上计算各层 FFN 神经元 |activation×gradient|，梯度目标为原生 top token 的 log probability。按 rank band、出度分桶、进展分桶、标签共同支持的 strata 比较区域差异，每层选固定 k 个；测试在同样 strata 内匹配。主 k=32/512，每个区域的集合分别在六层删除，仅编辑当前预测位置。两区域选出的集合不重叠是选取相反对比方向的结果，本身不是模块独立性的证据。', '',
'区域选择性定义为：(删除区域0集合时区域0比区域1多受损 + 删除区域1集合时区域1比区域0多受损)/2。Direct 用答案准确率损失，PATH 用下一节点有用率损失。CI 按查询做1000次 cluster bootstrap，只条件于当前模型与已选集合，不是跨图/跨训练置信区间。', '',
'| 图/协议 | k32 查询留出选择性 pp（95% CI） | 定位时留出节点身份 pp（95% CI） | 随机节点分组 3 seeds 范围 pp |',
'|---|---:|---:|---:|']
for g in ('simple', 'hard'):
    for p in ('direct', 'path'):
        d = follow[g, p]; a = d['size_scan']['32']['interaction']; b = d['panels']['node_heldout']['interaction']
        rand = [v['interaction']['own_minus_other_drop'] for v in d['random_partitions'].values()]
        lines.append(f'| {g}/{p} | {pp(a["own_minus_other_drop"])} [{pp(a["ci95"][0])}, {pp(a["ci95"][1])}] | {pp(b["own_minus_other_drop"])} [{pp(b["ci95"][0])}, {pp(b["ci95"][1])}] | {pp(min(rand))}–{pp(max(rand))} |')
lines += ['',
'简单 PATH 的 k32 图区域选择性较强，但留出定位节点后基本消失，随机分组也有选择性。该结果支持“已见节点身份相关的贡献分布”，尚不支持可以推广到其他节点的连通区域模块。不同划分支持的测试池不完全相同，随机分组对照只作描述比较，不当作显著性检验。', '',
'额外扫描 k=8/16/32/64/128，避免把小集合的阴性结果当成没有分工。困难 PATH 在 k128（每层25%，不是稀疏小电路）出现18.2pp查询留出选择性；节点留出为5.1pp [0.5,9.8]，但同大小随机节点分组也有7.0–26.4pp。这个弱留出信号不能确定特异的图区域模块。简单 PATH 的 k128 节点留出5.2pp，CI跨零。', '',
'还补充3种按全局贡献量分桶匹配数量的随机神经元，以及逐样本、逐层匹配被删除 residual 分支 L2 范数的随机神经元方向（3 seeds）。简单 PATH k32 的定向损害超过这些随机方向，但仍不通过节点身份泛化。范数匹配控制是缩放方向干预，可能包含放大，不能称纯置零消融；每个样本/层的范数误差有断言。', '',
'![区域选择性与关键对照](regional_controls.png)', '',
'## 4. 保持 embedding 不变，定向编辑 FFN 的节点关联', '',
'对 PATH 第一层 FFN，记 GELU 后神经元向量为 a(u)，输出矩阵为 W，输出增量 d(u)=Wa(u)。用训练查询按当前节点聚合平均 key。配对 u,v 只按图与支持数选择：同 rank band、相同非零出度、后继集合互不相交，u至少10条训练/6条测试决策，v至少10条训练决策；固定随机顺序取6组，不按干预效果筛选。', '',
'构造 key 方向 k，使 a(u)·k=1，其他观察到的节点平均 key 与 k 正交。仅在 FFN1 的 down 矩阵施加 rank-one 更新：', '',
'```text', 'ΔW = [d(v) − d(u)] kᵀ', '```', '',
'于是训练均值上的 u 关联被换成 v 的 FFN 增量，其他已观察节点的训练均值增量不变。这里转移的是原模型已有的 FFN 增量，不用测试答案训练。embedding、tied LM head、attention 和其他 FFN 全部不变。测试使用原查询的不同自然上下文。随机对照保持同一个 key，只把输出修改方向替换为同范数随机向量（3 seeds），因而整个权重更新 Frobenius 范数也一致。', '',
'| 图 | u→v | 测试决策数 | 编辑后 L1 新后继 recall@k | 最终下一 token 属于新后继 | 随机方向该比例均值 |',
'|---|---|---:|---:|---:|---:|']
for g in ('simple', 'hard'):
    for r in edits[g]['interventions']:
        control = avg([c['recipient']['final_new_top1'] for c in r['controls'].values()])
        lines.append(f'| {g} | {r["source_node"]}→{r["donor_node"]} | {r["baseline"]["n"]} | {pct(r["edited"]["l1_new_recall"])} | {pct(r["edited"]["final_new_top1"])} | {pct(control)} |')
lines += ['',
'所有配对编辑前的新后继 recall 和最终新后继 top1 均为0；新旧集合互斥，所以转向新后继会违反原图的该出边。简单6组最终转向率都是100%；困难5组100%、一组54.5%。各组样本可能来自同一图/重复查询前缀，不能把总决策数当独立训练重复。', '',
'这是“FFN 节点关联可以定向改变实际下一节点输出”的较强证据，比仅让一个新 probe 改变输出强。它并不等于模型已经变成完整、内部一致的新图，也不等于编辑只改变邻接语义：更新也可能改动身份或其他结构特征。训练均值保留不保证任意上下文保留，原模型其他区域表现仍需实际测量。', '',
'更新是低秩，但不是小范数：相对于 FFN1 down 权重 Frobenius 范数，简单为18.9%–27.0%，困难29.5%–41.4%。只在内存中修改，之后逐位恢复；恢复后的 logits 与原模型误差为0。', '',
'一个出乎意料的例子：困难图 V189→V219 的编辑在 L1 原头新后继 recall 仍为0，但最终 token 转向新后继达到100%。说明单个中间层的 logit lens 不能完全代表后续层对已修改状态的使用，不能把它当充分中介。', '',
'![功能干预与 FFN 定向编辑](functions_and_edits.png)', '',
'## 5. embedding 能否单独解释图知识', '',
'只输入一个节点，使用原输出头读取后继，比较原始 embedding 与去掉全部 attention、只保留 FFN 的逐点计算。下表 recall@真实出度，候选排除当前节点，集合非空的节点逐节点平均；完整词表后继概率质量也保存在主 JSON。单节点输入是诊断性分布外输入，不等于真实导航准确率。', '',
'| 图/模型 | 原始 E 经原头 | 仅 FFN，L1 | 仅 FFN，L6 |',
'|---|---:|---:|---:|']
for g in ('simple', 'hard'):
    for p in ('direct', 'path'):
        a = main[g, p]['pointwise']['ffn_only']
        lines.append(f'| {g}/{p} | {pct(a[0]["recall"])} | {pct(a[1]["recall"])} | {pct(a[6]["recall"])} |')
lines += ['',
'说明原模型可以在没有查询与其他节点输入时，通过节点 embedding、FFN、原头的组合提取后继；原头直接作用于 E 时做不到。不能由此断言 E 不包含完整图信息，因为 FFN 可能是必要的解码器，且 tied 输出头的 embedding 自相似会影响排名。', '',
'另用固定128维随机节点向量作对照，拟合 endpoint concat 仿射读出和 PCA16×PCA16 双线性读出（PCA 仅用 probe 训练节点）；正则/阈值仅按验证选，既有 query split 与三个双方节点身份留出 split 分别测量。后者训练端点最多60节点、验证20、测试20，因此交集测试查询很少，完整 n/各折结果见 JSON。', '',
'训练 embedding 的双线性读出在 query-heldout 的 BA：简单 Direct87.0%、PATH72.0%，困难 Direct77.7%、PATH83.3%；同容量随机身份向量约59.4%–63.3%（简单60.4%–62.8%）。说明 embedding 存在可用于关系判断的结构，但这个受限解码器没有恢复原模型表现，也不是 embedding 信息量的上界。普通 concat 高分与随机身份高分相近，仍受端点偏好影响。', '',
'最终可确定的是“embedding 有结构、后续网络对实际行为必要，并存在可编辑的节点关联”；尚不能确定“完整图信息是否已潜在包含在 embedding”。仅靠消融后续网络，逻辑上无法区分“后续网络存储图知识”和“后续网络解码 embedding 的图知识”。因此不宣称已经证明全部图知识不在 embedding，也不宣称 FFN 是唯一存储位置。', '',
'## 复现与有效性检查', '',
'从远端 repo root 执行以下脚本；large_controls 必须在 followup 完成后运行。冻结前向只用 GPU1；probe 拟合用 GPU0。', '',
'```bash', '.venv/bin/python studies/fixed_graph_storage/study.py --device cuda:1', '.venv/bin/python studies/fixed_graph_storage/followup.py --device cuda:1', '.venv/bin/python studies/fixed_graph_storage/large_controls.py', '.venv/bin/python studies/fixed_graph_storage/weight_edit.py --device cuda:1', '.venv/bin/python studies/fixed_graph_storage/report.py', '```', '',
'检查：checkpoint hash；DAG reachability DP 对照独立 BFS；原查询标签一致；gradient hook 的原生 logits 一致；按原 train/test 分离定位/评估；权重编辑只在内存发生，embedding/head 哈希不变、down 权重精确恢复、恢复 logits 完全一致；其他训练节点均值改动与指定均值转换误差均<1e-3；随机权重方向范数匹配。所有断言通过。', '',
'主结果4个JSON，神经元对照4个JSON，权重编辑2个JSON；只有2张汇总PNG和本报告，不保存中间激活或权重。每类图只有一个训练 seed，简单/困难还混有训练分布差异，不能把机制差异完全归因于图复杂度。', '']

# Add measured spillover and a complete example, rather than relying on intent.
insert = lines.index('更新是低秩，但不是小范数：相对于 FFN1 down 权重 Frobenius 范数，简单为18.9%–27.0%，困难29.5%–41.4%。只在内存中修改，之后逐位恢复；恢复后的 logits 与原模型误差为0。')
spill = []
for g in ('simple', 'hard'):
    records = edits[g]['interventions']
    drops = [r['other_baseline']['useful']-r['other_edited']['useful'] for r in records]
    unexposed = [r['unexposed_baseline']['useful']-r['unexposed_edited']['useful'] for r in records]
    spill.append(f'{g} 其他当前节点的有用率变化（下降为正）为 {pp(min(drops))}–{pp(max(drops))}pp；原前缀完全不包含被编辑节点的查询决策为 {pp(min(unexposed))}–{pp(max(unexposed))}pp。分母随配对变化，完整计数在 JSON；这不是其他功能全部保留的证明。')
lines[insert:insert] = [' '.join(spill), '']
r = edits['simple']['interventions'][0]; ex = r['examples'][0]
lines += ['## 一条具体轨迹决策', '',
    f'当前 {r["source_node"]}、目标 {ex["target"]}；原后继 {", ".join(r["old_successors"])}，供体 {r["donor_node"]} 的后继 {", ".join(r["new_successors"])}。原下一 token 为 {ex["baseline_next"]}，编辑 FFN1 后为 {ex["edited_next"]}。输入节点 token、embedding 与输出头没有改。该例是按固定配对顺序取第一条测试决策，所有12组的样例与对照均保留在两个 weight_edit JSON 中。', '']
text = re.sub(r'<(V\d+)>', r'`\1`', '\n'.join(lines))
(HERE/'REPORT.md').write_text(text, encoding='utf-8')
print('Wrote REPORT.md and two figures')
