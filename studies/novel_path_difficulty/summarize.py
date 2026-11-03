from pathlib import Path
import json
import numpy as np
HERE=Path(__file__).resolve().parent
DATA={s:json.loads((HERE/f'{s}.json').read_text()) for s in ('simple','hard')}
lines=['# 未暴露完整 witness 的 PATH 成功案例有多难？','',
    '结论：从随机游走看，这些样本不是容易的短链，随机成功概率约1%；从训练片段的组合需求看，它们很弱：全部21个成功输出都只需两个已见片段，18个只需一条边接上已见长后缀。按训练目标条件状态覆盖统计，10个没有任何“未见且有用/无用分支混合”的决策，另10个只有1处，1个有2处。不能据此把成功解释为从头进行长程规划。','',
    '## 范围和统计定义','',
    '- 旧N100简单/困难固定图PATH验证最佳checkpoint的原测试自由生成，分别15/6条既无训练PATH也无WALK完整witness的正例，全部合法到达目标且回答正确。不是新N160暴露控制实验。',
    '- 扫描原train query IDs的全部可采样PATH变体，包括正例和负例随机路径，以及全部WALK；不扫描val/test路径作训练暴露。每段逐边检查真实图合法性。',
    '- 训练暴露指缓存可供采样，不是best之前优化器实际曝光重放。片段分析给出可用训练材料的组合上界，不能证明模型实际使用了这些片段。',
    '- 有用后继等于目标或可达目标；mixed分叉同时有有用、无用后继。仅有多个后继但全通目标，不算需要辨别方向的mixed决策。',
    '- 未见目标条件决策：缓存训练任务轨迹未包含(current,target)。WALK无固定任务目标，不计入此状态键；但计入片段暴露。',
    '- 随机游走每步从真实直接后继均匀选择，目标吸收；在DAG上精确动态规划计算到达概率。最短路独立BFS，实际成功路径另外检查每条边。',
    '- 最少片段数分别测量：沿模型实际路径的连续片段拼接，以及H_train宏图中任意已暴露关系的最短拼接数。两者在这21条查询上都为2。这不是一般计算复杂度或模型算法步数。','',
    '## 两种难度','',
    '|指标|简单图15条|困难图6条|','|---|---:|---:|']
items=[('真实最短距离',lambda s:f"{min(s['shortest_hops'])}–{max(s['shortest_hops'])}跳"),
       ('实际生成路径长度',lambda s:f"{min(s['generated_edges'])}–{max(s['generated_edges'])}条边"),
       ('每条实际路径mixed分叉数',lambda s:f"{s['mixed_branch_range'][0]}–{s['mixed_branch_range'][1]}"),
       ('平均随机游走成功率',lambda s:f"{100*s['random_goal_probability_mean']:.2f}%"),
       ('同最短跳数、其他已曝光正例随机成功率',lambda s:f"{100*s['hop_matched_other_random_rate']:.2f}%"),
       ('只需两个已见片段的实际输出',lambda s:str(s['generated_fragment_count_hist']['2'])),
       ('一条边即可接上已见整段后缀',lambda s:str(s['one_new_edge_before_seen_suffix'])),
       ('起点只有一个后继',lambda s:str(s['initial_single_successor_count'])),
       ('起点全部后继都能到达目标',lambda s:str(s['initial_all_successors_useful_count'])),
       ('没有未见mixed决策',lambda s:str(s['strict_with_no_unseen_mixed_branch'])),
       ('起点入度为0',lambda s:str(s['source_root_count'])),
       ('目标出度为0',lambda s:str(s['target_sink_count']))]
for label,get in items:lines.append('|'+label+'|'+'|'.join(get(d['summary']) for d in DATA.values())+'|')
lines+=['','按最短跳数匹配的随机基线：对每个严格查询，在其余有完整训练witness的测试正例中取相同最短跳数组的均值，再对严格查询平均。仅描述这两个固定测试池，没有独立图/训练重复或显著性检验。', '',
    '## 全部逐查询记录','',
    '|图|起点→目标|最短/实际边数|全路径mixed数|未见mixed数|随机成功率|片段数|接上已见后缀前边数|',
    '|---|---|---:|---:|---:|---:|---:|---:|']
for setting,d in DATA.items():
    for r in d['queries']:
        lines.append(f"|{setting}|{r['source_token']}→{r['target_token']}|{r['shortest_hops']}/{r['generated_edges']}|{r['mixed_branch_decisions']}|{r['unseen_mixed_branch_decisions']}|{100*r['random_goal_probability']:.3f}%|{r['min_seen_fragments_for_generated_path']}|{r['novel_prefix_edges_before_seen_suffix']}|")
lines+=['','## 一个说明组合需求很低的例子','',
    '困难图查询V139→V16：最短路4跳，模型实际输出22条边；它可分成已见的V139→V50这一条边，与已见的V50→…→V16共21条边后缀。',
    '起点有4个后继，4个都可以到达V16；从模型实际轨迹的训练目标条件状态覆盖看，没有未见mixed决策。因此22-token级的长路径输出不等于22步新的定向规划。', '',
    '另一个例子V48→V46：最短7跳、输出20条边、随机成功率0.274%，有14处mixed分叉；但其中只有1处mixed目标条件状态未在训练缓存出现，整个输出可由一条边前缀和19边已见后缀拼接。因此它对“随机走路”困难，对复用训练长后缀则弱得多。','',
    '## 曝光缺失的结构原因与解释边界','',
    '简单图15个起点均为入度0节点，困难图6个中5个如此；困难图6个目标全部为出度0节点。这些端点难以作为其他正查询长路径的内部片段，随机query留出容易留下少数完整witness未覆盖的根到汇查询。负查询随机路径仍可能暴露它们，因此本次也扫描了负例缓存。',
    '本次结果进一步削弱“21条新完整路径证明通用搜索”的说法。它们排除了只逐字复述某一完整训练witness的最窄解释，仍兼容两段检索拼接、目标条件策略复用或参数关系匹配。不能据暴露片段存在断定模型确实检索/拼接了它们，也不能假定缓存中每个长后缀都在best之前真正训练过。','',
    '文件：simple.json/hard.json保留全部21条轨迹、分叉、成功路径数量、片段分解与曝光指标；run.py只读CPU复现，summarize.py生成本报告。没有新模型训练、GPU使用、checkpoint或activation中间文件。','']
(HERE/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
print(HERE/'REPORT.md')
