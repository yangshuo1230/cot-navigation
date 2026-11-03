# Graph scale 课程模型：CoT 非法轨迹细分测试

主实验完成 75/75 个模型×图条件，共 6,930 次查询重放；另保留 34 个已测未见固定图条件作为补充。

使用新图、固定图课程训练的 Path、DFS-state、BFS-state、DFS-nodes、BFS-nodes 共 10 个模型；全部选择各自验证集最佳 best.pt。测试集不参与 checkpoint 选择。

## 设计与统计口径

- 对每个现有 TEST 集按（层间距离，可达标签）分层，每格最多抽取 8 条、不放回；不足 8 条全部使用。种子 20261021，同一图条件的五种协议使用完全相同的查询。逐格样本数见 distributions.csv。小样本格仅用于描述，不能据此断言稳定差异。
- 新图：N=16/24/32/36/64/100；另有 N100 深窄 L20W5、浅宽 L5W20，以及 N144 L12W12。固定图主分析只用 scale_roots 实际训练的上述六种规模常规图。world_roots 注册的其余九张图作为未见世界补充对照单列，不能解释为已学图上的规模泛化。
- 正例独立 BFS 计算最短路，负例 shortest_path=null；gap 是数据集的层间距离，不能称为负例的最短路。当前分层图正例最短路等于层差，仍独立验证标签。
- 冻结模型，原生 BF16、4096 上下文及 STOP 规则；只用 GPU0，不中断其他训练。推理 batch64；与历史 batch16 的重叠查询预测对照另列。没有重新训练、写 checkpoint 或创建 W&B run。
- 完整合法性按真实标签对应的任务要求判定，答案准确率单列；不因最后答案错误而直接认定每个中间状态错误。Path 负例仅要求真实边随机游走终止于 sink，不是不可达证明。Path 正例必须到达目标；即使模型输出 NO，也不能以“合法负例随机游走”标准放过未到目标的轨迹。这比先按模型预测决定终点要求的旧口径更严格；合法边前缀率另列。
- 状态协议：从起点严格重放，visited/candidates 按集合检查，栈/队列按顺序检查，允许任意合法候选优先级；不是逐字匹配参考轨迹。动作骨架单独重放，不依赖打印状态。
- Node 协议输出首次扩展顺序，不是相邻节点路径；DFS 允许隐式回溯，BFS 允许合法的兄弟节点排列。顺序首错之后不再把推测出的队列/栈当真，只统计重复、目标后继续等可独立判断的违规。
- 首错是相对真实初始状态的第一次错误；首错后的 local 统计只检查打印状态自身及相邻记录的一致性。后续局部一致不等于恢复正确搜索，也不能把所有后续全局失配重复计成新错误。
- 若状态字段已含图外节点，该记录归入 invalid_state_nodes，不再依赖这份非法状态推断候选/PUSH 细类。后续这些局部细类是已确证的检出数，可能低估总数；候选首错则额外直接按真实 eligible 集合记录遗漏、多出和重复。
- 错误类型非互斥；“只有一处坏记录”也可能包含多个字段错误。termination_only 指无结构违规但提前结束；合法 STOP 单列，不归入完整轨迹。首错分位数以实际输出记录/节点数归一化，终止处缺失对应 1。

## 全部图条件：答案准确率 / 完整轨迹合法率

| 任务与图 | 每协议 n | Path | DFS-state | BFS-state | DFS-nodes | BFS-nodes |
|---|---:|---:|---:|---:|---:|---:|
| context N16 L4W4 | 48 | 83.3% / 83.3% | 95.8% / 75.0% | 97.9% / 89.6% | 89.6% / 81.2% | 75.0% / 37.5% |
| context N24 L4W6 | 48 | 87.5% / 87.5% | 89.6% / 66.7% | 91.7% / 77.1% | 85.4% / 77.1% | 64.6% / 20.8% |
| context N32 L4W8 | 48 | 79.2% / 77.1% | 87.5% / 50.0% | 89.6% / 62.5% | 89.6% / 75.0% | 77.1% / 18.8% |
| context N36 L6W6 | 80 | 81.2% / 81.2% | 92.5% / 66.2% | 92.5% / 65.0% | 86.2% / 57.5% | 76.2% / 30.0% |
| context N64 L8W8 | 112 | 79.5% / 78.6% | 94.6% / 56.2% | 84.8% / 52.7% | 81.2% / 61.6% | 69.6% / 27.7% |
| context N100 L5W20 | 64 | 75.0% / 73.4% | 90.6% / 50.0% | 82.8% / 51.6% | 85.9% / 60.9% | 73.4% / 23.4% |
| context N100 L10W10 | 144 | 58.3% / 58.3% | 80.6% / 42.4% | 84.7% / 36.1% | 80.6% / 47.2% | 70.8% / 26.4% |
| context N100 L20W5 | 304 | 59.9% / 57.6% | 78.6% / 37.5% | 77.3% / 37.2% | 78.3% / 47.4% | 68.8% / 25.0% |
| context N144 L12W12 | 176 | 61.4% / 58.0% | 75.6% / 19.3% | 72.2% / 32.4% | 52.8% / 23.9% | 60.8% / 25.6% |
| fixed N16 L4W4 | 19 | 89.5% / 68.4% | 73.7% / 73.7% | 78.9% / 73.7% | 89.5% / 89.5% | 73.7% / 68.4% |
| fixed N24 L4W6 | 29 | 79.3% / 75.9% | 96.6% / 89.7% | 100.0% / 96.6% | 93.1% / 69.0% | 82.8% / 62.1% |
| fixed N32 L4W8 | 32 | 81.2% / 78.1% | 96.9% / 90.6% | 100.0% / 96.9% | 96.9% / 78.1% | 90.6% / 81.2% |
| fixed N36 L6W6 | 61 | 90.2% / 82.0% | 98.4% / 86.9% | 91.8% / 83.6% | 95.1% / 86.9% | 90.2% / 72.1% |
| fixed N64 L8W8 | 93 | 84.9% / 83.9% | 95.7% / 79.6% | 98.9% / 89.2% | 93.5% / 83.9% | 90.3% / 73.1% |
| fixed N100 L10W10 | 128 | 85.2% / 82.0% | 95.3% / 71.1% | 81.2% / 60.9% | 96.1% / 78.1% | 94.5% / 57.0% |

## 错误分布与首错位置

下表合并同一任务下全部测试图条件；每条查询权重相同，因此是本次分层诊断面板的分布，不是原训练分布或自然总体的估计。首错百分位仅对非法/未完成轨迹计算。

| 任务/协议 | n | 非法/未完成 | 首错位置 Q25/中位/Q75 | 检出1处违规 | 2–3处 | ≥4处 | 合法STOP |
|---|---:|---:|---:|---:|---:|---:|---:|
| context/path | 1024 | 344 (33.6%) | 100%/100%/100% | 31 | 0 | 0 | 0 |
| context/dfs_state | 1024 | 575 (56.2%) | 3%/13%/32% | 49 | 22 | 504 | 0 |
| context/bfs_state | 1024 | 548 (53.5%) | 14%/29%/45% | 18 | 20 | 510 | 0 |
| context/dfs_nodes | 1024 | 504 (49.2%) | 41%/63%/100% | 84 | 106 | 183 | 0 |
| context/bfs_nodes | 1024 | 758 (74.0%) | 40%/50%/67% | 491 | 46 | 101 | 0 |
| fixed/path | 362 | 69 (19.1%) | 90%/100%/100% | 11 | 6 | 1 | 0 |
| fixed/dfs_state | 362 | 75 (20.7%) | 24%/59%/77% | 23 | 8 | 44 | 0 |
| fixed/bfs_state | 362 | 77 (21.3%) | 30%/65%/83% | 9 | 7 | 61 | 0 |
| fixed/dfs_nodes | 362 | 69 (19.1%) | 47%/75%/100% | 14 | 19 | 18 | 0 |
| fixed/bfs_nodes | 362 | 120 (33.1%) | 27%/48%/80% | 37 | 36 | 28 | 0 |

## N100 常规图：分跳数、分正负例

单元格为非法或未完成数 / 查询数。短、中、长分别为层差 1–3、4–6、7–9；固定图已有 TEST 不含一跳，短组实际为 2–3。

| 任务/协议 | 正例短 | 正例中 | 正例长 | 负例短 | 负例中 | 负例长 |
|---|---:|---:|---:|---:|---:|---:|
| context/path | 14/24 | 23/24 | 22/24 | 0/24 | 1/24 | 0/24 |
| context/dfs_state | 9/24 | 22/24 | 20/24 | 10/24 | 10/24 | 12/24 |
| context/bfs_state | 14/24 | 22/24 | 24/24 | 9/24 | 10/24 | 13/24 |
| context/dfs_nodes | 7/24 | 16/24 | 18/24 | 12/24 | 10/24 | 13/24 |
| context/bfs_nodes | 14/24 | 24/24 | 24/24 | 15/24 | 14/24 | 15/24 |
| fixed/path | 2/16 | 9/24 | 10/24 | 1/16 | 0/24 | 1/24 |
| fixed/dfs_state | 3/16 | 8/24 | 3/24 | 2/16 | 7/24 | 14/24 |
| fixed/bfs_state | 0/16 | 13/24 | 19/24 | 3/16 | 4/24 | 11/24 |
| fixed/dfs_nodes | 2/16 | 5/24 | 0/24 | 3/16 | 7/24 | 11/24 |
| fixed/bfs_nodes | 6/16 | 10/24 | 17/24 | 5/16 | 8/24 | 9/24 |

### context / path

答案正确但轨迹非法/未完成 17/1024；轨迹完整合法但答案错误 1/1024；STOP 0（强制 0）；动作骨架合法 680/1024；完整轨迹非法但动作骨架仍完整可执行 0/1024；结构合法前缀 993/1024。

首错类别（互斥，分母为本协议非法/未完成数）：
- `wrong_positive_endpoint`：309/344（89.8%）
- `non_edge`：16/344（4.7%）
- `non_graph_node`：12/344（3.5%）
- `unfinished_search`：4/344（1.2%）
- `continued_after_target`：3/344（0.9%）

局部错误（非互斥，分母为该协议全部查询数）：
- `wrong_positive_endpoint`：309/1024（30.2%）
- `non_edge`：28/1024（2.7%）
- `non_graph_node`：12/1024（1.2%）
- `unfinished_search`：4/1024（0.4%）
- `continued_after_target`：3/1024（0.3%）

### context / dfs_state

答案正确但轨迹非法/未完成 408/1024；轨迹完整合法但答案错误 0/1024；STOP 43（强制 29）；动作骨架合法 487/1024；完整轨迹非法但动作骨架仍完整可执行 38/1024；结构合法前缀 449/1024。

首错类别（互斥，分母为本协议非法/未完成数）：
- `candidate_mismatch`：462/575（80.3%）
- `frontier_mismatch`：85/575（14.8%）
- `wrong_action`：23/575（4.0%）
- `malformed_record`：4/575（0.7%）
- `visited_mismatch`：1/575（0.2%）

候选首错细分（非互斥）：`omitted_eligible` 433；`extra_ineligible` 453；`duplicate` 36。遗漏是缺少真实未访问后继，多出是包含不属于该 eligible 集合的 token，后者可能是已访问节点、非边节点或图外 token。

局部错误（非互斥，分母为该协议全部查询数）：
- `invalid_state_nodes`：514/1024（50.2%）
- `frontier_update_error`：270/1024（26.4%）
- `visited_update_error`：240/1024（23.4%）
- `candidate_nonedge`：144/1024（14.1%）
- `candidate_omitted`：142/1024（13.9%）
- `nonedge_push`：129/1024（12.6%）
- `frontier_duplicate`：97/1024（9.5%）
- `visited_duplicate`：91/1024（8.9%）
- `pop_with_candidates`：67/1024（6.5%）
- `wrong_action`：66/1024（6.4%）
- `frontier_missing_from_visited`：63/1024（6.2%）
- `malformed_fields`：53/1024（5.2%）
- `candidate_already_visited`：46/1024（4.5%）
- `revisited_push`：33/1024（3.2%）
- `candidate_duplicate`：31/1024（3.0%）
- `unfinished_record`：18/1024（1.8%）
- `goal_candidate_annotation`：5/1024（0.5%）
- `missed_goal`：3/1024（0.3%）
- `premature_exhaust`：2/1024（0.2%）
- `malformed_action`：1/1024（0.1%）

### context / bfs_state

答案正确但轨迹非法/未完成 364/1024；轨迹完整合法但答案错误 0/1024；STOP 82（强制 31）；动作骨架合法 484/1024；完整轨迹非法但动作骨架仍完整可执行 8/1024；结构合法前缀 476/1024。

首错类别（互斥，分母为本协议非法/未完成数）：
- `candidate_mismatch`：538/548（98.2%）
- `frontier_mismatch`：3/548（0.5%）
- `visited_mismatch`：3/548（0.5%）
- `wrong_action`：2/548（0.4%）
- `unfinished_record`：1/548（0.2%）
- `illegal_push`：1/548（0.2%）

候选首错细分（非互斥）：`extra_ineligible` 432；`omitted_eligible` 407；`duplicate` 12。遗漏是缺少真实未访问后继，多出是包含不属于该 eligible 集合的 token，后者可能是已访问节点、非边节点或图外 token。

局部错误（非互斥，分母为该协议全部查询数）：
- `invalid_state_nodes`：439/1024（42.9%）
- `candidate_omitted`：283/1024（27.6%）
- `pop_with_candidates`：276/1024（27.0%）
- `visited_update_error`：264/1024（25.8%）
- `candidate_already_visited`：240/1024（23.4%）
- `visited_duplicate`：238/1024（23.2%）
- `revisited_push`：236/1024（23.0%）
- `frontier_duplicate`：222/1024（21.7%）
- `frontier_update_error`：198/1024（19.3%）
- `wrong_action`：175/1024（17.1%）
- `frontier_missing_from_visited`：138/1024（13.5%）
- `candidate_nonedge`：128/1024（12.5%）
- `nonedge_push`：127/1024（12.4%）
- `unfinished_record`：36/1024（3.5%）
- `premature_exhaust`：24/1024（2.3%）
- `goal_candidate_annotation`：14/1024（1.4%）
- `missed_goal`：12/1024（1.2%）
- `candidate_duplicate`：11/1024（1.1%）
- `false_found`：6/1024（0.6%）
- `malformed_fields`：4/1024（0.4%）
- `push_not_displayed`：2/1024（0.2%）

### context / dfs_nodes

答案正确但轨迹非法/未完成 272/1024；轨迹完整合法但答案错误 3/1024；STOP 0（强制 0）；动作骨架合法 520/1024；完整轨迹非法但动作骨架仍完整可执行 0/1024；结构合法前缀 651/1024。

首错类别（互斥，分母为本协议非法/未完成数）：
- `non_graph_node`：182/504（36.1%）
- `not_dfs_order`：141/504（28.0%）
- `wrong_positive_endpoint`：77/504（15.3%）
- `unfinished_search`：54/504（10.7%）
- `repeated_node`：50/504（9.9%）

局部错误（非互斥，分母为该协议全部查询数）：
- `non_graph_node`：284/1024（27.7%）
- `repeated_node`：189/1024（18.5%）
- `not_dfs_order`：141/1024（13.8%）
- `wrong_positive_endpoint`：77/1024（7.5%）
- `unfinished_search`：54/1024（5.3%）
- `wrong_start`：1/1024（0.1%）

### context / bfs_nodes

答案正确但轨迹非法/未完成 442/1024；轨迹完整合法但答案错误 0/1024；STOP 0（强制 0）；动作骨架合法 266/1024；完整轨迹非法但动作骨架仍完整可执行 0/1024；结构合法前缀 386/1024。

首错类别（互斥，分母为本协议非法/未完成数）：
- `not_bfs_order`：611/758（80.6%）
- `unfinished_search`：71/758（9.4%）
- `wrong_positive_endpoint`：49/758（6.5%）
- `non_graph_node`：25/758（3.3%）
- `repeated_node`：2/758（0.3%）

局部错误（非互斥，分母为该协议全部查询数）：
- `not_bfs_order`：611/1024（59.7%）
- `repeated_node`：122/1024（11.9%）
- `unfinished_search`：71/1024（6.9%）
- `non_graph_node`：64/1024（6.2%）
- `wrong_positive_endpoint`：49/1024（4.8%）

### fixed / path

答案正确但轨迹非法/未完成 16/362；轨迹完整合法但答案错误 0/362；STOP 0（强制 0）；动作骨架合法 293/362；完整轨迹非法但动作骨架仍完整可执行 0/362；结构合法前缀 344/362。

首错类别（互斥，分母为本协议非法/未完成数）：
- `wrong_positive_endpoint`：50/69（72.5%）
- `non_edge`：9/69（13.0%）
- `non_graph_node`：6/69（8.7%）
- `repeated_node`：2/69（2.9%）
- `continued_after_target`：1/69（1.4%）
- `unfinished_search`：1/69（1.4%）

局部错误（非互斥，分母为该协议全部查询数）：
- `wrong_positive_endpoint`：50/362（13.8%）
- `non_edge`：17/362（4.7%）
- `non_graph_node`：7/362（1.9%）
- `repeated_node`：3/362（0.8%）
- `continued_after_target`：2/362（0.6%）
- `unfinished_search`：1/362（0.3%）

### fixed / dfs_state

答案正确但轨迹非法/未完成 57/362；轨迹完整合法但答案错误 0/362；STOP 6（强制 1）；动作骨架合法 302/362；完整轨迹非法但动作骨架仍完整可执行 15/362；结构合法前缀 287/362。

首错类别（互斥，分母为本协议非法/未完成数）：
- `candidate_mismatch`：61/75（81.3%）
- `wrong_action`：8/75（10.7%）
- `frontier_mismatch`：3/75（4.0%）
- `illegal_push`：1/75（1.3%）
- `malformed_record`：1/75（1.3%）
- `visited_mismatch`：1/75（1.3%）

候选首错细分（非互斥）：`omitted_eligible` 49；`extra_ineligible` 31；`duplicate` 7。遗漏是缺少真实未访问后继，多出是包含不属于该 eligible 集合的 token，后者可能是已访问节点、非边节点或图外 token。

局部错误（非互斥，分母为该协议全部查询数）：
- `candidate_omitted`：58/362（16.0%）
- `candidate_already_visited`：35/362（9.7%）
- `wrong_action`：34/362（9.4%）
- `frontier_update_error`：34/362（9.4%）
- `visited_duplicate`：33/362（9.1%）
- `candidate_nonedge`：32/362（8.8%）
- `revisited_push`：31/362（8.6%）
- `pop_with_candidates`：31/362（8.6%）
- `nonedge_push`：30/362（8.3%）
- `visited_update_error`：20/362（5.5%）
- `premature_exhaust`：18/362（5.0%）
- `frontier_missing_from_visited`：12/362（3.3%）
- `candidate_duplicate`：10/362（2.8%）
- `invalid_state_nodes`：8/362（2.2%）
- `false_found`：5/362（1.4%）
- `missed_goal`：5/362（1.4%）
- `frontier_duplicate`：3/362（0.8%）
- `push_not_displayed`：1/362（0.3%）
- `malformed_fields`：1/362（0.3%）

### fixed / bfs_state

答案正确但轨迹非法/未完成 43/362；轨迹完整合法但答案错误 0/362；STOP 54（强制 52）；动作骨架合法 286/362；完整轨迹非法但动作骨架仍完整可执行 1/362；结构合法前缀 285/362。

首错类别（互斥，分母为本协议非法/未完成数）：
- `candidate_mismatch`：70/77（90.9%）
- `wrong_action`：5/77（6.5%）
- `malformed_record`：2/77（2.6%）

候选首错细分（非互斥）：`omitted_eligible` 65；`duplicate` 33；`extra_ineligible` 12。遗漏是缺少真实未访问后继，多出是包含不属于该 eligible 集合的 token，后者可能是已访问节点、非边节点或图外 token。

局部错误（非互斥，分母为该协议全部查询数）：
- `candidate_omitted`：65/362（18.0%）
- `candidate_already_visited`：57/362（15.7%）
- `revisited_push`：56/362（15.5%）
- `candidate_duplicate`：42/362（11.6%）
- `pop_with_candidates`：36/362（9.9%）
- `malformed_fields`：20/362（5.5%）
- `frontier_update_error`：14/362（3.9%）
- `visited_duplicate`：14/362（3.9%）
- `visited_update_error`：13/362（3.6%）
- `invalid_state_nodes`：12/362（3.3%）
- `frontier_duplicate`：12/362（3.3%）
- `wrong_action`：12/362（3.3%）
- `frontier_missing_from_visited`：7/362（1.9%）
- `candidate_nonedge`：7/362（1.9%）
- `false_found`：5/362（1.4%）
- `nonedge_push`：4/362（1.1%）
- `unfinished_record`：2/362（0.6%）
- `premature_exhaust`：2/362（0.6%）
- `goal_candidate_annotation`：1/362（0.3%）
- `malformed_action`：1/362（0.3%）
- `missed_goal`：1/362（0.3%）

### fixed / dfs_nodes

答案正确但轨迹非法/未完成 50/362；轨迹完整合法但答案错误 0/362；STOP 0（强制 0）；动作骨架合法 293/362；完整轨迹非法但动作骨架仍完整可执行 0/362；结构合法前缀 311/362。

首错类别（互斥，分母为本协议非法/未完成数）：
- `repeated_node`：21/69（30.4%）
- `not_dfs_order`：20/69（29.0%）
- `wrong_positive_endpoint`：14/69（20.3%）
- `continued_after_target`：5/69（7.2%）
- `non_graph_node`：5/69（7.2%）
- `unfinished_search`：4/69（5.8%）

局部错误（非互斥，分母为该协议全部查询数）：
- `repeated_node`：37/362（10.2%）
- `not_dfs_order`：20/362（5.5%）
- `wrong_positive_endpoint`：14/362（3.9%）
- `continued_after_target`：6/362（1.7%）
- `non_graph_node`：6/362（1.7%）
- `unfinished_search`：4/362（1.1%）

### fixed / bfs_nodes

答案正确但轨迹非法/未完成 89/362；轨迹完整合法但答案错误 4/362；STOP 0（强制 0）；动作骨架合法 242/362；完整轨迹非法但动作骨架仍完整可执行 0/362；结构合法前缀 261/362。

首错类别（互斥，分母为本协议非法/未完成数）：
- `not_bfs_order`：57/120（47.5%）
- `repeated_node`：27/120（22.5%）
- `continued_after_target`：12/120（10.0%）
- `wrong_positive_endpoint`：10/120（8.3%）
- `unfinished_search`：9/120（7.5%）
- `non_graph_node`：5/120（4.2%）

局部错误（非互斥，分母为该协议全部查询数）：
- `repeated_node`：60/362（16.6%）
- `not_bfs_order`：57/362（15.7%）
- `continued_after_target`：14/362（3.9%）
- `wrong_positive_endpoint`：10/362（2.8%）
- `unfinished_search`：9/362（2.5%）
- `non_graph_node`：5/362（1.4%）

## 典型错误窗口

每个协议选取不同失败形态；下面是原始输出窗口，事件含真实候选集合及动作期望。完整逐查询事件见压缩 JSONL。

### context/path，larger，query=51

<V195> → <V19>，层差 1，标签 0，预测 0；形态 `one_bad_unit`。

首错：`{"unit": 2, "kind": "non_edge"}`

```text
<V195> <V50> <V49>
```

### context/path，larger，query=7

<V62> → <V228>，层差 1，标签 1，预测 0；形态 `termination_only`。

首错：`{"unit": 3, "kind": "wrong_positive_endpoint"}`

```text
<V62> <V124> <V117>
```

### context/dfs_state，larger，query=31

<V248> → <V90>，层差 1，标签 0，预测 0；形态 `four_or_more_bad_units`。

首错：`{"unit": 8, "kind": "candidate_mismatch", "detail": {"expected": [153, 214, 269], "shown": [148, 269, 153]}}`

```text
<V248> <V2> <V79> <V195> | <V248> <V2> <V79> <V245> <V29> <V195> | <V230> <V99> | <PUSH> <V230> ; <V248> <V2> <V79> <V195> <V230> | <V248> <V2> <V79> <V245> <V29> <V195> <V230> | <V134> <V255> <V139> | <PUSH> <V134> ; <V248> <V2> <V79> <V195> <V230> <V134> | <V248> <V2> <V79> <V245> <V29> <V195> <V230> <V134> | | <POP> ;
```

### context/dfs_state，larger，query=56

<V79> → <V30>，层差 2，标签 1，预测 1；形态 `one_bad_unit`。

首错：`{"unit": 2, "kind": "candidate_mismatch", "detail": {"expected": [53, 102, 109, 151], "shown": [151, 22, 151, 53]}}`

```text
<V79> <V140> | <V79> <V140> | <V30> <V153> | <PUSH> <V30> ; <V79> <V140> <V30> | <V79> <V140> <V30> | <V137> <V8> <V137> <V39> | <FOUND> ;
```

### context/bfs_state，larger，query=51

<V195> → <V19>，层差 1，标签 0，预测 1；形态 `four_or_more_bad_units`。

首错：`{"unit": 2, "kind": "candidate_mismatch", "detail": {"expected": [91, 92, 128, 150, 180], "shown": [92, 128, 91, 251]}}`

```text
<V195> <V50> | <V195> <V50> | | <POP> ; <V50> | <V195> <V50> | <V78> <V114> <V77> <V237> | <PUSH> <V78> ; <V50> <V78> | <V195> <V50> <V78> | <V114> <V77> <V237> | <PUSH> <V114> ;
```

### context/bfs_state，larger，query=18

<V191> → <V118>，层差 1，标签 1，预测 1；形态 `one_bad_unit`。

首错：`{"unit": 8, "kind": "candidate_mismatch", "detail": {"expected": [], "shown": [117]}}`

```text
<V135> <V118> <V103> <V88> | <V191> <V26> <V135> <V118> <V103> <V88> | | <POP> ; <V118> <V103> <V88> | <V191> <V26> <V135> <V118> <V103> <V88> | <V103> | <FOUND> ;
```

### context/dfs_nodes，larger，query=28

<V71> → <V235>，层差 1，标签 0，预测 0；形态 `one_bad_unit`。

首错：`{"unit": 3, "kind": "repeated_node"}`

```text
<V71> <V183> <V146> <V146>
```

### context/dfs_nodes，larger，query=31

<V248> → <V90>，层差 1，标签 0，预测 0；形态 `termination_only`。

首错：`{"unit": 2, "kind": "unfinished_search"}`

```text
<V248> <V199>
```

### context/bfs_nodes，larger，query=28

<V71> → <V235>，层差 1，标签 0，预测 0；形态 `one_bad_unit`。

首错：`{"unit": 2, "kind": "not_bfs_order"}`

```text
<V71> <V181> <V17>
```

### context/bfs_nodes，larger，query=41

<V116> → <V29>，层差 1，标签 0，预测 0；形态 `termination_only`。

首错：`{"unit": 4, "kind": "unfinished_search"}`

```text
<V116> <V68> <V28> <V51>
```

### fixed/path，n100_l10w10，query=184

<V154> → <V41>，层差 2，标签 1，预测 0；形态 `termination_only`。

首错：`{"unit": 4, "kind": "wrong_positive_endpoint"}`

```text
<V154> <V150> <V129> <V195>
```

### fixed/path，n100_l10w10，query=1386

<V139> → <V248>，层差 3，标签 0，预测 0；形态 `one_bad_unit`。

首错：`{"unit": 2, "kind": "non_edge"}`

```text
<V139> <V247> <V16>
```

### fixed/dfs_state，n100_l10w10，query=1667

<V60> → <V225>，层差 4，标签 1，预测 0；形态 `four_or_more_bad_units`。

首错：`{"unit": 1, "kind": "illegal_push", "detail": {}}`

```text
<V60> | <V60> | <V93> | <PUSH> <V93> ; <V60> <V93> | <V60> <V93> | <V228> <V189> <V96> | <PUSH> <V182> ; <V60> <V182> | <V60> <V93> <V182> | <V88> <V241> <V111> | <PUSH> <V88> ;
```

### fixed/dfs_state，n100_l10w10，query=172

<V236> → <V223>，层差 2，标签 1，预测 1；形态 `one_bad_unit`。

首错：`{"unit": 37, "kind": "candidate_mismatch", "detail": {"expected": [156, 237], "shown": [237]}}`

```text
<V236> <V15> <V93> | <V236> <V15> <V93> <V228> <V150> <V129> <V195> <V204> <V160> <V128> <V65> <V225> <V2> <V140> <V67> <V167> <V189> <V191> <V96> <V76> | | <POP> ; <V236> <V15> | <V236> <V15> <V93> <V228> <V150> <V129> <V195> <V204> <V160> <V128> <V65> <V225> <V2> <V140> <V67> <V167> <V189> <V191> <V96> <V76> | <V223> | <PUSH> <V223> ; <V236> <V15> <V223> | <V236> <V15> <V93> <V228> <V150> <V129> <V195> <V204> <V160> <V128> <V65> <V225> <V2> <V140> <V67> <V167> <V189> <V191> <V96> <V76> <V223> | | <FOUND> ;
```

### fixed/bfs_state，n100_l10w10，query=642

<V115> → <V182>，层差 2，标签 0，预测 1；形态 `four_or_more_bad_units`。

首错：`{"unit": 20, "kind": "candidate_mismatch", "detail": {"expected": [143, 154, 218, 239], "shown": [239, 218, 143, 239]}}`

```text
<V191> <V150> <V67> <V167> | <V115> <V238> <V103> <V27> <V224> <V218> <V57> <V228> <V191> <V150> <V67> <V167> | | <POP> ; <V150> <V67> <V167> | <V115> <V238> <V103> <V27> <V224> <V218> <V57> <V228> <V191> <V150> <V67> <V167> | <V225> <V204> <V129> <V225> | <PUSH> <V225> ; <V150> <V67> <V167> <V225> | <V115> <V238> <V103> <V27> <V224> <V218> <V57> <V228> <V191> <V150> <V67> <V167> <V225> | <V204> <V129> <V225> | <PUSH> <V204> ;
```

### fixed/bfs_state，n100_l10w10，query=2249

<V32> → <V245>，层差 5，标签 1，预测 1；形态 `one_bad_unit`。

首错：`{"unit": 30, "kind": "candidate_mismatch", "detail": {"expected": [231, 233, 242], "shown": [231, 242]}}`

```text
<V182> <V245> <V218> <V13> | <V32> <V82> <V43> <V120> <V68> <V132> <V165> <V208> <V91> <V164> <V243> <V159> <V27> <V182> <V245> <V218> <V13> | | <POP> ; <V245> <V218> <V13> | <V32> <V82> <V43> <V120> <V68> <V132> <V165> <V208> <V91> <V164> <V243> <V159> <V27> <V182> <V245> <V218> <V13> | <V217> <V228> | <FOUND> ;
```

### fixed/dfs_nodes，n100_l10w10，query=154

<V228> → <V140>，层差 2，标签 1，预测 0；形态 `four_or_more_bad_units`。

首错：`{"unit": 6, "kind": "continued_after_target"}`

```text
<V67> <V167> <V2> <V195> <V140> <V129> <V150> <V225> <V204> <V160> <V65> <V128>
```

### fixed/dfs_nodes，n100_l10w10，query=184

<V154> → <V41>，层差 2，标签 1，预测 1；形态 `termination_only`。

首错：`{"unit": 12, "kind": "wrong_positive_endpoint"}`

```text
<V129> <V195> <V225> <V2> <V157>
```

### fixed/bfs_nodes，n100_l10w10，query=656

<V206> → <V167>，层差 2，标签 0，预测 0；形态 `one_bad_unit`。

首错：`{"unit": 3, "kind": "not_bfs_order"}`

```text
<V206> <V76> <V89> <V2> <V195>
```

### fixed/bfs_nodes，n100_l10w10，query=964

<V165> → <V13>，层差 3，标签 1，预测 1；形态 `four_or_more_bad_units`。

首错：`{"unit": 7, "kind": "continued_after_target"}`

```text
<V164> <V182> <V27> <V159> <V13> <V218> <V245> <V219> <V228> <V217> <V226>
```

## 补充：未见固定图对照

下列 WORLD token 虽已注册，但对应图不在实际课程 scale_roots 中。这里的失败同时包含未学图知识，不能用来判断已学固定图的搜索能力。主分布、两张图及上面的协议汇总均排除这些条件。

| 未见图 | Path | DFS-state | BFS-state | DFS-nodes | BFS-nodes |
|---|---:|---:|---:|---:|---:|
| N16 L2W8 | 无测试查询 | 无测试查询 | 无测试查询 | 无测试查询 | 无测试查询 |
| N16 L8W2 | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| N36 L3W12 | 0.0% | 0.0% | 本轮未测 | 0.0% | 0.0% |
| N36 L12W3 | 1.8% | 5.4% | 9.9% | 0.0% | 10.8% |
| N64 L4W16 | 0.0% | 本轮未测 | 本轮未测 | 0.0% | 0.0% |
| N64 L16W4 | 0.0% | 本轮未测 | 本轮未测 | 1.9% | 3.8% |
| N100 L5W20 | 2.1% | 本轮未测 | 本轮未测 | 2.1% | 2.1% |
| N100 L20W5 | 0.0% | 本轮未测 | 本轮未测 | 3.9% | 0.0% |
| N144 L12W12 | 1.2% | 本轮未测 | 本轮未测 | 0.0% | 0.0% |

## 验证与局限

- 所有已完成条件的 checkpoint SHA256 与原验证集最佳选择清单相同；查询选择跨协议一致。原历史预测可比 2690 条，batch64 与历史 batch16 的最终预测差异 77 条（详见 verification.json）。
- 检查器通过 7680 项参考断言、24576 项任意节点序列的旧/新判定等价检查，以及 FOUND→POP 目标漏检变异测试；参考状态完整性额外在全部四节点 DAG、端点和两种搜索下检查。
- 仅现有 curriculum 的 graph seed0，不是多训练种子或多固定图种子的显著性结论；不同图规模也同时改变形状/拓扑，不能把差异完全归因于节点数。
- 没有该系列课程训练的 Filler checkpoint；Direct 无图轨迹。此次结构合法性覆盖现有五种有轨迹协议。
- Node 协议的“检出1处违规”仅表示检出了一个违规位置。首次顺序错误之后不能唯一重建隐式搜索状态，后续顺序不再判定；不能据此认定它确实只错一次。状态协议的打印字段允许持续进行局部检查，两者的坏单元计数不能直接当作同等完整的误差率比较。
- 本实验是轨迹观察分析，不足以证明模型内部是否在执行搜索；局部违规的扩散形态也不能单凭观察认定因果。

## 文件

- `distributions.csv`：每模型×图条件的总体及精确 gap×label 统计。
- `per_query_errors.jsonl.gz`：所有查询首错、坏单元、局部事件、终止/STOP 和答案信息；不保存大批激活或完整 rollout。
- `summary.json`：条件元信息、checkpoint、样本格及少量原始错误窗口。
- `invalid_by_scale_gap.png` / `failure_shapes.png`：仅两张关键结果图。
- `queries/`：固定 TEST 查询 ID 和每格可用数，后续可直接复用。

## BF16 分批敏感性复核

在预先列出的五个模型×条件中，各取一个含较多差异查询的原始 batch16，共 80 条；历史最终预测复现 80/80。这些是有意选择的诊断样本，不能用来估计总体差异率。主表保留统一 batch64 的全部结果，没有择优替换。分批差异和轨迹合法性分别记录，不能把数值敏感性误报为 checkpoint 改变。详细逐查询比较见 batch_sensitivity.json。
