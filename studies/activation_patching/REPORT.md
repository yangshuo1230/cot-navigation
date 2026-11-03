# 同 token 隐藏状态干预：首轮方法验证

四个 N=100 完整状态模型（新图/固定图 × DFS/BFS），各自验证最佳 checkpoint；固定图仅 seed 0。冻结推理使用物理 GPU1，没有新训练、checkpoint、W&B run 或激活文件。

本轮完成两种读取：字段后立刻续写当前记录，以及固定当前记录后续写下一条。节点、字段分隔符、动作操作符、记录结束符分别做，另做节点+分隔符及全部组合；PUSH 参数不干预。操作符仅覆盖非终止 PUSH/POP。详细定义见 DESIGN.md。

**核心边界：token 相同而输出变化，支持这些位置的隐藏表示有因果作用；不能据此断言隐藏状态比显式文本更重要。** 隐藏状态本身就是文本进入后续计算的媒介。

## 配对覆盖

| 模型 | frontier | visited | candidates |
|---|---:|---:|---:|
| context_l10_bfs | 4 | 2 | 4 |
| context_l10_dfs | 4 | 2 | 4 |
| fixed_l10_bfs | 0 | 4 | 4 |
| fixed_l10_dfs | 4 | 4 | 4 |

每格至多四个不同 recipient 查询；目标字段逐 token 相同，另外两个字段必须语义不同。候选/visited 用集合差异，栈/队列用有序序列差异。新图 visited 只有两个可用配对；固定图 BFS frontier 在本轮有界检索中没有严格配对，不等于穷举全池或数学上不存在。不同组还会因操作符身份或空字段而缺格。

## 局部响应示例：第4层，节点+字段分隔符

“改变”指整个被生成记录与同 batch 未干预 twin 不同。“动作合法”按 recipient 原状态，允许合法候选替代；并非最终答案准确率。

| 读取 | 模型 | donor | 改变/配对 | 动作合法：对照→干预 |
|---|---|---|---:|---:|
| results | context_l10_bfs | no_history | 2/10 | 9/10 → 8/10 |
| results | context_l10_bfs | cross_example | 1/10 | 9/10 → 8/10 |
| results | context_l10_dfs | no_history | 0/10 | 8/10 → 8/10 |
| results | context_l10_dfs | cross_example | 0/10 | 8/10 → 8/10 |
| results | fixed_l10_bfs | no_history | 1/8 | 8/8 → 8/8 |
| results | fixed_l10_bfs | cross_example | 0/8 | 8/8 → 8/8 |
| results | fixed_l10_dfs | no_history | 3/12 | 12/12 → 12/12 |
| results | fixed_l10_dfs | cross_example | 2/12 | 12/12 → 12/12 |
| current_record | context_l10_bfs | no_history | 4/10 | 6/10 → 2/10 |
| current_record | context_l10_bfs | cross_example | 0/10 | 6/10 → 6/10 |
| current_record | context_l10_dfs | no_history | 5/10 | 9/10 → 4/10 |
| current_record | context_l10_dfs | cross_example | 2/10 | 9/10 → 8/10 |
| current_record | fixed_l10_bfs | no_history | 5/8 | 8/8 → 3/8 |
| current_record | fixed_l10_bfs | cross_example | 0/8 | 8/8 → 8/8 |
| current_record | fixed_l10_dfs | no_history | 7/12 | 12/12 → 6/12 |
| current_record | fixed_l10_dfs | cross_example | 1/12 | 12/12 → 12/12 |

这组预先指定的读取下，no_history 对当前动作的破坏明显多于 cross_example，但这不是语义重要性的公平排名：两种 donor 的历史长度、语法熟悉度与激活分布不同。固定其余正确文本后，下一记录的变化通常更少，是后续文本可能纠正局部扰动的线索，尚不能确定其机制。

一个候选线索：新图 DFS query 50523，用 donor 38161 在相同栈 token <V250> 上的第2层节点表示替换后，下一记录候选偏向 donor 的空候选，动作也与 donor 的 POP 一致；同时 recipient 状态正确性下降。这与 donor 影响相容，也与一般破坏相容，不能单独当作规划状态成功移植的证明。配对信息与指标在 findings.json。

## 验证与异常记录

Self-patch 共 1020 条，概率差严格为0，续写完全相同。最后一层过去位置负对照共 380 条，同样严格无影响。9 项 CPU 单元测试覆盖因果方向、cache 重算及当前记录读取等。配对前缀已经逐记录用图语义重放验证，checkpoint 在每模型运行前后校验 SHA256。

最后一层当前末尾 token 可以直接改变 next-token logits；这与最后一层过去位置无通路并不矛盾。图中必须区分节点/过去分隔符与当前末尾分隔符，不能按整层是否有效下结论。

findings.json 保存每个模型/读取最多两个变化 case 的完整匹配字段、其他字段、参考后续和配对响应指标。优先列出朝 donor 字段靠近的案例，属于异常线索，不能用筛选后的比例作总体效应。statistics.csv 保留全部统计，包括格式失败、截断、recipient/donor 相似度变化。

尚未完成的关键控制：更多查询/图 seed；历史长度精确匹配；同状态不同历史 donor；范数匹配/随机 donor；替换强度扫描；共同 cohort 上的组合效应；长程最终答案。当前 early/shallow 小样本和不同 donor 历史长度限制了外推。不能把一般输出破坏解释成语义迁移，也不能只看成功变化的样本。

![下一记录](results.png)

![当前记录](current_record.png)
