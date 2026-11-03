from pathlib import Path
import json

HERE = Path(__file__).resolve().parent
DATA = HERE if (HERE / 'summary.json').exists() else HERE / 'results'
R = json.loads((DATA / 'summary.json').read_text())

def pct(x):
    return f'{100*x:.2f}%'

def cell(r, condition):
    v = r['reports'][condition]
    return f"{v['correct']}/{v['n']} ({pct(v['accuracy'])})"

def table(headers, rows):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---']*len(headers)) + ' |',
                      *['| ' + ' | '.join(map(str, row)) + ' |' for row in rows]])

context = []
for key, r in R.items():
    if key.startswith('context/'):
        context.append([key.split('/')[1], *[cell(r, c) for c in ('forward_yes', 'forward_no', 'reverse', 'same_layer')],
                        pct(r['forward_reverse_both_correct'])])
fixed = []
individual = []
for model in ('direct', 'dfs_state', 'dfs_random', 'bfs_state'):
    per_world = [R[f'fixed{i}/{model}'] for i in range(3)]
    row = [model]
    for c in ('forward_yes', 'forward_no', 'reverse', 'same_layer'):
        n = sum(r['reports'][c]['n'] for r in per_world)
        correct = sum(r['reports'][c]['correct'] for r in per_world)
        row.append(f'{correct}/{n} ({pct(correct/n)})')
    fixed.append(row)
    for i, r in enumerate(per_world):
        individual.append([i, model, *[cell(r, c) for c in ('forward_yes', 'forward_no', 'reverse', 'same_layer')]])

trace = []
for key, r in R.items():
    if r['protocol'] == 'direct':
        continue
    trace.append([key, *[pct(r['reports'][c]['trajectory_valid']) for c in ('forward_yes','forward_no','reverse','same_layer')]])

start_rows = []
for model in ('dfs_state', 'dfs_random', 'bfs_state'):
    total = starts_source = starts_target = valid = 0
    for seed in range(3):
        path = HERE / 'results' / f'fixed{seed}__{model}_predictions.json'
        predictions = json.loads(path.read_text())['reverse']
        for item in predictions:
            tokens = item['tokens']
            total += 1
            starts_source += int(tokens[4] == tokens[1])
            starts_target += int(tokens[4] == tokens[2])
            valid += int(item['trace_verdict']['valid'])
    start_rows.append([model, total, starts_source, starts_target, total-starts_source-starts_target,
                       f'{valid}/{total} ({pct(valid/total)})'])

report = f'''# 同层与逆层查询：冻结模型评估

日期：2026-10-18。仅GPU0冻结推理，FP32、TF32关闭，未初始化优化器、未更新权重、未创建W&B run或checkpoint。每个checkpoint评估前后SHA256一致。共18个已有模型。扩大训练图规模的运行不纳入。

## 设计

- 新图：原N16/E24 test中选择64张图，原始正例为内部第2层到第3层的一跳查询。每图构造前向可达、前向不可达、正例起终点互换的逆层查询、同层不同节点查询，共256例。四种条件共享图、节点名字和边列表顺序。
- 起点、目标都要求有入边也有出边，因此都在边列表中出现，且不靠起点无出边/目标无入边判断。所有标签由独立BFS和bitset可达性双重校验。没有自查询。
- 固定图：三个原N32/E48世界各自评估。前向对照来自该世界既有完整test，筛选两个端点都有入/出边的内部节点，最多32例/类。逆层查询由入选前向正例反转，数量与前向正例相同。同层查询独立抽32个有序节点对。各图合格查询有限，保留全部可用样本，不复制扩充。
- 每个条件分开报告，不汇总一个受负例占比影响的准确率。成对正确率要求同一可达查询判YES、交换方向后判NO。
- Direct直接用完整词表argmax；搜索协议使用原无强制答案、无gold长度预算的greedy生成，最终YES/NO为主指标，格式失败计错，轨迹用原独立语义重放诊断。
- 没有把这些新查询加入训练。因此测量的是未训练查询方向的迁移，不是在新任务上重训后的可学习性。

## 新图结果

{table(['模型','前向可达','前向不可达','逆层','同层','正反成对均正确'],context)}

ordinary24/paired24来自刚完成的4096条一跳小实验，不是主批次数据分布的独立重复。

## 固定图结果：按样本合并三图

{table(['模型','前向可达','前向不可达','逆层','同层'],fixed)}

这些是本次筛选子集，不等同于原完整test。固定图汇总使用计数合并，不是三图等权平均。新图和固定图的查询层距分布不同，不能把两张表当作只改变知识来源的因果对照。

## 固定图逐图结果

{table(['图seed','模型','前向可达','前向不可达','逆层','同层'],individual)}

## 完整轨迹语义合法率

{table(['模型','前向可达','前向不可达','逆层','同层'],trace)}

## 逆层错误定位：第一条状态的搜索起点

{table(['固定图模型','逆层n','从真实起点开始','错误从输入目标开始','其他起点','完整轨迹合法'],start_rows)}

固定图prompt为`<QUERY> source target |`，首条状态应从source开始。随机DFS有60/74条却从target开始。一个真实例子：

```text
输入：<QUERY> <V140> <V170> |
首条输出：<V170> | <V170> | <V66> | <PUSH> <V66> ;
真实答案：NO
模型答案：YES
```

本例的source应为V140，模型却从V170开始，第一条记录即frontier_mismatch。这里是可观测的起终点角色错误，不是长轨迹后期才累积的误差。它与模型在逆层输入上复用熟悉的前向关系相容，但尚未通过内部激活干预确定原因。

## 解释边界

逆层和同层不同节点在该相邻层DAG中均为NO，但全答NO不能说明会处理图查询，必须结合前向正例与成对指标。新图Direct基线在本组全答NO，一跳Direct和paired24对查询变化反应也很弱，正反成对正确率为0。普通4096样本模型没有展示可靠迁移。新图BFS在本次全部四类输入上均答对，且多数轨迹能通过语义校验，这提供了已有模型可以处理这些分布外负例的成功对照，但仅64张图和一个模型种子，不代表普遍能力。

固定图即使已有前向高分，也不保证逆层/同层判断。尤其随机DFS对前向关系的成功不能直接外推为任意有序节点对上的方向性理解。当前结果仍不足以确定内部机制，不能把所有生成错误归于图知识缺失。

## 复现文件

- evaluate.py：独立冻结评估脚本。
- inputs.json：完整图、节点映射和查询，不修改原训练数据。
- summary.json：18个checkpoint分条件统计。
- results_bundle.tar.gz：服务器全部逐样本预测、轨迹、统计、完成记录。
- 服务器输出：`studies/query_direction/results`，本地脚本与报告：`G:/CoT/query_direction`。
'''
(HERE / 'REPORT.md').write_text(report, encoding='utf-8')
print(table(['模型','前向可达','前向不可达','逆层','同层','正反成对均正确'],context))
print(table(['固定图模型','前向可达','前向不可达','逆层','同层'],fixed))
