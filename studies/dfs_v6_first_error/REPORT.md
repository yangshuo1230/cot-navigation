# v6 DFS nodes：不同查询划分的首错分析

冻结4个N80 graph0 final50k checkpoint，全部10848条原生自由生成。推理GPU1，无训练、权重/activation保存或W&B。
按照生成事件的时间顺序检查，首次违规之后停止归因；包含最终答案/EOS事件。合法隐式回退之后才检查下一节点是否为当前真实未访问后继，绝不将所有相邻输出非边都判错。
完整协议成功要求节点展开合法、完整终止并回答正确；另记录仅节点轨迹语义合法性。此处“首错因”是行为分类，不能直接等同于内部机制原因。
已有replay_nodes会先全局检查非节点token；本次流式判定避免未来语法错误遮蔽更早的搜索错误。349184个小DAG测试及全部10848条实际轨迹的最终语义合法性与原验证器一致。

## 主要发现

- 导航训练query完整协议合法率：短程约99.0%、长AA约99.3%；未见AT/TT约61%。当前差距主要体现在查询泛化，并非训练query普遍仍未拟合。
- 合法到达目标之后，AA/TA未正确YES停止的条件率约1.2%/2.1%，AT/TT约24.2%/23.0%；满足穷尽条件后的NO停止接近正确。V目标也有类似差别：AV约21.5%、VA约1.3%。这些均为观察性分组，未控制图节点属性等因素。
- 短程测试也有36/213的目标停止失败，而短程训练为0/1980，因此问题不只发生在长程留出目标角色；未见query组合也出现迁移缺口。
- long_AT_test：到目标后继续的85次中，78次下一节点符合忽略目标的合法DFS；搜索展开类首错共142/261，占非法轨迹54.4%。目标停止问题不是全部失败的解释。
- long_TT_test：到目标后继续的34次中，32次下一节点符合忽略目标的合法DFS；搜索展开类首错共76/130，占非法轨迹58.5%。目标停止问题不是全部失败的解释。
- 高visited/过滤探针成绩与这些行为错误可以同时成立；平均排序可读性不是原生选择正确率，也没有在本次首错位置检验模型是否使用该方向。

## 导航模型原始查询划分

|原始面板|每seed查询数|完整协议合法率|答案BA|非边跳转|跳过未探索分支|重复访问|目标停止错误|过早结束|其他|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|short_train|1234|99.0%|99.8%|0.3%|0.2%|0.4%|0.0%|0.1%|0.0%|
|short_val|151|82.0%|90.6%|3.6%|1.0%|5.3%|5.5%|1.3%|1.3%|
|short_test|151|79.6%|89.7%|3.5%|3.0%|5.6%|6.0%|1.7%|0.7%|
|long_train|274|99.3%|100.0%|0.3%|0.1%|0.3%|0.1%|0.0%|0.0%|
|long_AA_val|31|79.8%|93.6%|10.5%|4.0%|4.8%|0.0%|0.0%|0.8%|
|long_AA_test|31|83.1%|94.9%|6.5%|3.2%|5.6%|0.8%|0.8%|0.0%|
|long_VA_val|168|80.8%|96.1%|6.7%|3.7%|4.9%|0.9%|1.3%|1.6%|
|long_AV_val|168|61.8%|86.4%|8.0%|4.9%|6.1%|12.5%|5.5%|1.2%|
|long_VV_val|84|59.5%|89.1%|9.5%|6.5%|6.8%|9.8%|5.1%|2.7%|
|long_TA_test|168|85.6%|96.4%|5.1%|2.4%|4.0%|1.3%|1.0%|0.6%|
|long_AT_test|168|61.2%|84.5%|7.9%|6.8%|6.4%|12.8%|4.2%|0.7%|
|long_TT_test|84|61.3%|85.7%|10.4%|6.8%|5.4%|10.4%|4.2%|1.5%|

## 已合法到达终止条件的机会

分母是终止条件出现前始终合法的机会数，不是全部正例/负例；计数为4seed合计，百分比为合计计数之比。图中为逐seed条件比例均值，可能略有不同。

|面板|目标后错误/机会|穷尽后错误/机会|继续节点时仍符合忽略目标的DFS|
|---|---|---|---|
|short_train|0/1980 (0.0%)|0/2906 (0.0%)|0/0|
|short_val|33/216 (15.3%)|3/315 (1.0%)|32/32|
|short_test|36/213 (16.9%)|1/305 (0.3%)|31/31|
|long_train|1/874 (0.1%)|0/215 (0.0%)|1/1|
|long_AA_val|0/80 (0.0%)|0/19 (0.0%)|0/0|
|long_AA_test|1/83 (1.2%)|0/21 (0.0%)|1/1|
|long_VA_val|6/465 (1.3%)|0/84 (0.0%)|5/5|
|long_AV_val|84/391 (21.5%)|0/108 (0.0%)|73/83|
|long_VV_val|33/182 (18.1%)|0/51 (0.0%)|30/31|
|long_TA_test|9/436 (2.1%)|1/149 (0.7%)|8/9|
|long_AT_test|86/355 (24.2%)|0/142 (0.0%)|78/85|
|long_TT_test|35/152 (23.0%)|0/89 (0.0%)|32/34|

## 非边、回溯与重复访问的进一步分解

- non_edge_jump：按应有隐式回退后的栈顶判断，下一未访问节点不是真实后继，且不是当前栈祖先的未访问后继。
- skipped_live_frontier：当前栈顶仍有合法候选，却输出严格祖先的某个未访问后继。这是跳过未探索分支的行为，不证明模型内部确实弹出了栈。
- revisited_node：nodes协议输出已发现节点。区分重输当前点、重输栈祖先、重访已关闭分支，以及把隐式返回栈顶显式打印出来；不能全部叫visited编码失败。
- reachable_descendant_skip：非边跳转的落点在应有栈顶的可达后代中，至少缺少一个中间节点；这是多跳跳转签名，不等于证明模型把可达性当成直接边。

|测试面板|非边跳转：跳可达后代/总数|非边直接跳目标|跳过分支直接跳目标|重复节点是真实后继|重复访问细分|
|---|---|---:|---:|---|---|
|short_test|9/21|1|2|12/34|{'revisited_closed_node': 25, 'repeated_stack_ancestor': 6, 'repeated_current': 2, 'printed_implicit_return_top': 1}|
|long_AA_test|4/8|1|0|3/7|{'repeated_current': 1, 'repeated_stack_ancestor': 4, 'printed_implicit_return_top': 1, 'revisited_closed_node': 1}|
|long_TA_test|13/34|0|1|10/27|{'revisited_closed_node': 19, 'repeated_stack_ancestor': 7, 'printed_implicit_return_top': 1}|
|long_AT_test|23/53|8|1|16/43|{'revisited_closed_node': 31, 'printed_implicit_return_top': 1, 'repeated_stack_ancestor': 8, 'repeated_current': 3}|
|long_TT_test|18/35|6|0|9/18|{'revisited_closed_node': 10, 'repeated_stack_ancestor': 5, 'repeated_current': 3}|

## 正例/负例与层差

以下“gap”为层差；正例已由独立BFS核验等于最短路，负例没有有限最短路。计数合并4seed，主导错误取首次违规最多的一类。

|测试面板|标签|gap|n（含4seed）|完整协议合法率|最多的首错及次数|
|---|---|---:|---:|---:|---|
|short_test|1|1|64|53.1%|[('continued_after_goal', 24)]|
|short_test|1|2|84|73.8%|[('continued_after_goal', 7)]|
|short_test|1|3|96|84.4%|[('skipped_live_frontier', 7)]|
|short_test|0|1|164|84.8%|[('revisited_node', 12)]|
|short_test|0|2|120|88.3%|[('revisited_node', 4)]|
|short_test|0|3|76|77.6%|[('revisited_node', 9)]|
|long_AA_test|1|4|24|91.7%|[('continued_after_goal', 1)]|
|long_AA_test|1|5|24|87.5%|[('non_edge_jump', 1)]|
|long_AA_test|1|6|20|80.0%|[('revisited_node', 2)]|
|long_AA_test|1|7|16|87.5%|[('revisited_node', 1)]|
|long_AA_test|1|8|8|75.0%|[('skipped_live_frontier', 1)]|
|long_AA_test|1|9|4|75.0%|[('revisited_node', 1)]|
|long_AA_test|0|4|12|83.3%|[('non_edge_jump', 1)]|
|long_AA_test|0|5|4|75.0%|[('revisited_node', 1)]|
|long_AA_test|0|6|4|75.0%|[('revisited_node', 1)]|
|long_AA_test|0|7|4|75.0%|[('non_edge_jump', 1)]|
|long_AA_test|0|8|4|50.0%|[('non_edge_jump', 2)]|
|long_TA_test|1|4|132|84.8%|[('non_edge_jump', 7)]|
|long_TA_test|1|5|112|86.6%|[('revisited_node', 7)]|
|long_TA_test|1|6|100|88.0%|[('non_edge_jump', 4)]|
|long_TA_test|1|7|72|79.2%|[('revisited_node', 5)]|
|long_TA_test|1|8|56|83.9%|[('non_edge_jump', 5)]|
|long_TA_test|1|9|32|81.2%|[('non_edge_jump', 4)]|
|long_TA_test|0|4|60|91.7%|[('revisited_node', 2)]|
|long_TA_test|0|5|48|89.6%|[('non_edge_jump', 3)]|
|long_TA_test|0|6|28|89.3%|[('revisited_node', 2)]|
|long_TA_test|0|7|24|91.7%|[('skipped_live_frontier', 1)]|
|long_TA_test|0|8|8|37.5%|[('revisited_node', 2)]|
|long_AT_test|1|4|100|72.0%|[('continued_after_goal', 11)]|
|long_AT_test|1|5|108|57.4%|[('continued_after_goal', 20)]|
|long_AT_test|1|6|104|52.9%|[('continued_after_goal', 12)]|
|long_AT_test|1|7|80|55.0%|[('continued_after_goal', 11)]|
|long_AT_test|1|8|64|42.2%|[('continued_after_goal', 22)]|
|long_AT_test|1|9|32|28.1%|[('continued_after_goal', 9)]|
|long_AT_test|0|4|92|81.5%|[('revisited_node', 8)]|
|long_AT_test|0|5|52|82.7%|[('revisited_node', 5)]|
|long_AT_test|0|6|24|70.8%|[('revisited_node', 3)]|
|long_AT_test|0|7|16|43.8%|[('revisited_node', 5)]|
|long_TT_test|1|4|48|68.8%|[('revisited_node', 5)]|
|long_TT_test|1|5|52|65.4%|[('non_edge_jump', 5)]|
|long_TT_test|1|6|48|56.2%|[('non_edge_jump', 8)]|
|long_TT_test|1|7|32|37.5%|[('continued_after_goal', 6)]|
|long_TT_test|1|8|32|25.0%|[('continued_after_goal', 9)]|
|long_TT_test|1|9|16|18.8%|[('continued_after_goal', 6)]|
|long_TT_test|0|4|48|79.2%|[('non_edge_jump', 3)]|
|long_TT_test|0|5|28|82.1%|[('non_edge_jump', 3)]|
|long_TT_test|0|6|16|81.2%|[('non_edge_jump', 1)]|
|long_TT_test|0|7|16|93.8%|[('non_edge_jump', 1)]|

## 原模型划分与后续probe重划分不能混用

|划分来源|split|每seed查询数|完整协议合法率|答案BA|
|---|---|---:|---:|---:|
|main_split|train|1508|99.0%|99.8%|
|main_split|val|602|72.8%|91.2%|
|main_split|test|602|73.8%|90.1%|
|probe_split|train|1678|87.3%|95.0%|
|probe_split|val|515|88.2%|95.9%|
|probe_split|test|519|88.1%|95.9%|

## 目标端点角色的完整性与错误位置

|原测试面板|正例协议合法率|负例协议合法率|答案对但协议错（4seed计数）|首错时已访问节点占全图 Q25/中位/Q75|
|---|---:|---:|---:|---|
|short_test|72.5%|84.4%|69|3.8%/15.0%/29.4%|
|long_AA_test|85.4%|75.0%|16|10.0%/13.8%/41.2%|
|long_TA_test|84.7%|88.1%|69|7.5%/16.2%/30.0%|
|long_AT_test|55.1%|77.2%|131|11.2%/21.2%/27.5%|
|long_TT_test|51.3%|82.4%|67|8.8%/18.8%/25.0%|

## 可复核实例

### long_AT_test，query36，seed0：continued_after_goal

查询：<V207> → <V88>；label=1；首错token=<V73>。

首错前栈：`<V207> <V106> <V77> <V34> <V14> <V88>`

执行必要回退后的栈：`<V207> <V106> <V77> <V34> <V14> <V88>`

当时合法候选：``

局部输出：`<V105> <V254> <V88> <V73> <V82> <V178> <V186>`

进一步分类：{'event_kind': 'node', 'follows_DFS_if_goal_ignored': True, 'ignored_goal_required_pops': 2, 'ignored_goal_has_remaining_candidates': True, 'actual_is_target': False}

### long_TT_test，query99，seed0：non_edge_jump

查询：<V49> → <V1>；label=1；首错token=<V201>。

首错前栈：`<V49> <V123> <V63>`

执行必要回退后的栈：`<V49>`

当时合法候选：`<V50> <V2>`

局部输出：`<V39> <V199> <V63> <V201> <V115> <V250> <V130>`

进一步分类：{'jump_subtype': 'reachable_descendant_skip', 'graph_distance_from_top': 2, 'has_edge_from_closed_visited_node': False, 'actual_is_target': False}

### long_AT_test，query58，seed0：skipped_live_frontier

查询：<V207> → <V109>；label=1；首错token=<V164>。

首错前栈：`<V207> <V106> <V77> <V34> <V14> <V88> <V40> <V69>`

执行必要回退后的栈：`<V207> <V106> <V77> <V34> <V14> <V88> <V40> <V69>`

当时合法候选：`<V109> <V179>`

局部输出：`<V40> <V76> <V69> <V164> <V174> <V55> <V29>`

进一步分类：{'alternative_ancestor': 54, 'actual_is_target': False}

### long_TT_test，query104，seed0：revisited_node

查询：<V49> → <V56>；label=0；首错token=<V93>。

首错前栈：`<V49> <V50> <V201> <V115> <V73> <V88> <V40> <V164> <V174> <V169>`

执行必要回退后的栈：`<V49> <V50> <V201> <V115> <V73> <V88> <V40> <V164>`

当时合法候选：`<V179>`

局部输出：`<V174> <V29> <V169> <V93> <V179> <V69> <V109>`

进一步分类：{'repeat_subtype': 'revisited_closed_node', 'edge_from_decision_top': True, 'actual_is_target': False}

## 解释边界

- 条件停止错误和搜索展开错误都发生在已验证合法的历史之后，但本实验未在首错位置测activation或做干预，不能据此定位某层FFN、attention或visited子空间失效。
- 查询角色、长度、分支和可达性分布不同；条件率仍是观察性比较，不是只改变目标角色的受控因果实验。
- 非边跳转、跳过活跃分支、重复输出可能来自局部记忆、栈/visited状态使用、协议序列化或解码等多种机制，不强行单因解释。
- 首错后不继续猜测内部状态或累计次生错误。零次某类首错不意味着后续永不发生。
- 新BF16生成与历史final.json有少量差异，逐seed逐面板记录于results.json；所有结论用本次实际轨迹。
- 只有一张N80图，4初始化重复同一组query；不以四倍query数当独立图样本。
