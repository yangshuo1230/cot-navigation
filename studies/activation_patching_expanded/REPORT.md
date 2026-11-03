# 扩大合法配对的 activation patch 实验

保持原四个 N=100 完整状态 checkpoint，使用所有数据 split 与合法人工构造。这里只研究局部因果响应，结果不是独立测试集泛化准确率，也没有重新训练模型。

## 配对规模

| 模型 | 精确字段配对：栈/visited/候选 | 共同历史配对：栈/visited/候选 |
|---|---|---|
| context_l10_bfs | 128 / 128 / 128 | 128 / 128 / 128 |
| context_l10_dfs | 128 / 128 / 128 | 128 / 128 / 128 |
| fixed_l10_bfs | 0 / 128 / 128 | 128 / 128 / 128 |
| fixed_l10_dfs | 128 / 128 / 128 | 128 / 128 / 128 |

共 2944 对；两种读取重复使用这些配对。L2/L4全量，L6与self-patch稀疏校验。每格以 visited 大小1–3、4–7、8–15、16+分层，具体覆盖写入 datasets。

同一基础单位在不同字段/配对类中可能重复。每格分层覆盖并不总能完全均衡，具体以数据清单为准。

exact_field 要求一个字段逐 token 一致，其他两个字段语义不同。新图通过完整、单射的节点重命名实现；固定图保留原图，用新查询与合法邻居优先级匹配。shared_history 保证当前整条 CoT 前缀相同且等长：新图只改后续未出现节点的名称，固定图只改合法查询目标。两类不能混为一种实验。

固定图BFS缺少exact_field/frontier是结构性限制：相邻层DAG上的合法BFS满足 CAND = successors(queue[0]) - set(queue)，同队列不可能有不同候选集合。这里保留为空，不伪造配对。固定图recipe中的base_split=train只是读取原图的锚点，实际source/target已另行生成，不意味着这些查询全属于原训练集。

## 未干预控制的支持度

| 模型 | 配对类 | recipient完整参考正确 | donor完整参考正确 | 两者都正确 |
|---|---|---:|---:|---:|
| context_l10_bfs | exact_field | 224/384 | 295/384 | 197/384 |
| context_l10_bfs | shared_history | 0/384 | 0/384 | 0/384 |
| context_l10_dfs | exact_field | 293/384 | 270/384 | 215/384 |
| context_l10_dfs | shared_history | 153/384 | 160/384 | 130/384 |
| fixed_l10_bfs | exact_field | 230/256 | 225/256 | 199/256 |
| fixed_l10_bfs | shared_history | 326/384 | 324/384 | 319/384 |
| fixed_l10_dfs | exact_field | 343/384 | 354/384 | 329/384 |
| fixed_l10_dfs | shared_history | 326/384 | 325/384 | 316/384 |

context_l10_bfs / shared_history 下一记录的recipient完整参考正确数为0，故该格没有合格方向性迁移分母；不能用低响应推断不存在相关能力。

## 第4层跨 donor 的下一记录响应

| 模型 | 配对类 | 干预 | 续写改变 | 动作改变 | 未解析 | 动作合法：对照→干预 |
|---|---|---|---:|---:|---:|---:|
| context_l10_bfs | exact_field | operator | 0/266 | 0/266 | 0/266 | 227/266 → 227/266 |
| context_l10_bfs | exact_field | semicolon | 120/384 | 0/384 | 0/384 | 258/384 → 258/384 |
| context_l10_bfs | exact_field | nodes_bar | 49/384 | 4/384 | 1/384 | 258/384 → 256/384 |
| context_l10_bfs | exact_field | all | 27/266 | 3/266 | 1/266 | 227/266 → 225/266 |
| context_l10_bfs | shared_history | operator | 0/384 | 0/384 | 0/384 | 0/384 → 0/384 |
| context_l10_bfs | shared_history | semicolon | 0/384 | 0/384 | 0/384 | 0/384 → 0/384 |
| context_l10_bfs | shared_history | nodes_bar | 6/384 | 1/384 | 0/384 | 0/384 → 0/384 |
| context_l10_bfs | shared_history | all | 6/384 | 1/384 | 0/384 | 0/384 → 0/384 |
| context_l10_dfs | exact_field | operator | 146/291 | 113/291 | 6/291 | 242/291 → 167/291 |
| context_l10_dfs | exact_field | semicolon | 174/384 | 26/384 | 0/384 | 335/384 → 324/384 |
| context_l10_dfs | exact_field | nodes_bar | 10/384 | 5/384 | 0/384 | 335/384 → 333/384 |
| context_l10_dfs | exact_field | all | 218/291 | 131/291 | 7/291 | 242/291 → 148/291 |
| context_l10_dfs | shared_history | operator | 286/384 | 244/384 | 10/384 | 270/384 → 92/384 |
| context_l10_dfs | shared_history | semicolon | 3/384 | 3/384 | 0/384 | 270/384 → 272/384 |
| context_l10_dfs | shared_history | nodes_bar | 5/384 | 3/384 | 0/384 | 270/384 → 272/384 |
| context_l10_dfs | shared_history | all | 286/384 | 244/384 | 10/384 | 270/384 → 93/384 |
| fixed_l10_bfs | exact_field | operator | 0/128 | 0/128 | 0/128 | 128/128 → 128/128 |
| fixed_l10_bfs | exact_field | semicolon | 130/256 | 86/256 | 9/256 | 256/256 → 164/256 |
| fixed_l10_bfs | exact_field | nodes_bar | 0/256 | 0/256 | 0/256 | 256/256 → 256/256 |
| fixed_l10_bfs | exact_field | all | 84/128 | 45/128 | 10/128 | 128/128 → 73/128 |
| fixed_l10_bfs | shared_history | operator | 0/384 | 0/384 | 0/384 | 384/384 → 384/384 |
| fixed_l10_bfs | shared_history | semicolon | 1/384 | 0/384 | 0/384 | 384/384 → 384/384 |
| fixed_l10_bfs | shared_history | nodes_bar | 0/384 | 0/384 | 0/384 | 384/384 → 384/384 |
| fixed_l10_bfs | shared_history | all | 1/384 | 0/384 | 0/384 | 384/384 → 384/384 |
| fixed_l10_dfs | exact_field | operator | 44/196 | 41/196 | 0/196 | 196/196 → 161/196 |
| fixed_l10_dfs | exact_field | semicolon | 24/384 | 8/384 | 4/384 | 384/384 → 374/384 |
| fixed_l10_dfs | exact_field | nodes_bar | 5/384 | 3/384 | 0/384 | 384/384 → 384/384 |
| fixed_l10_dfs | exact_field | all | 56/196 | 44/196 | 2/196 | 196/196 → 155/196 |
| fixed_l10_dfs | shared_history | operator | 7/384 | 5/384 | 0/384 | 384/384 → 384/384 |
| fixed_l10_dfs | shared_history | semicolon | 1/384 | 0/384 | 0/384 | 384/384 → 384/384 |
| fixed_l10_dfs | shared_history | nodes_bar | 1/384 | 0/384 | 0/384 | 384/384 → 384/384 |
| fixed_l10_dfs | shared_history | all | 6/384 | 4/384 | 0/384 | 384/384 → 384/384 |

动作改变要求对照与干预都能解析；未解析不计动作改变，单独列出。动作合法是相对于 recipient 参考状态判断，不能代替整条生成记录合法率。不同组支持的case不同，不按不同分母直接排名组合效应。

## 有方向的迁移

directional_transfer.csv 只保留参考后续确实不同、该字段实际生成、recipient配对控制和donor控制均完整复现参考记录的样本。字段迁移要求完全匹配donor对应集合/序列；动作要求精确匹配donor动作及参数。条件较严格，合法但不同显示优先级可能被排除，分母小应如实解释。

| 模型 | 配对类 | 干预 | 目标 | 迁移 / 合格对照 |
|---|---|---|---|---:|
| context_l10_dfs | exact_field | operator | candidates | 18/101 |
| context_l10_dfs | exact_field | operator | action | 18/106 |
| context_l10_dfs | exact_field | semicolon | frontier | 40/215 |
| context_l10_dfs | exact_field | semicolon | visited | 21/214 |
| context_l10_dfs | exact_field | semicolon | candidates | 1/143 |
| context_l10_dfs | exact_field | semicolon | action | 5/149 |
| context_l10_dfs | exact_field | all | frontier | 40/153 |
| context_l10_dfs | exact_field | all | visited | 17/152 |
| context_l10_dfs | exact_field | all | candidates | 25/101 |
| context_l10_dfs | exact_field | all | action | 30/106 |
| context_l10_dfs | shared_history | operator | candidates | 24/130 |
| context_l10_dfs | shared_history | operator | action | 32/121 |
| context_l10_dfs | shared_history | all | candidates | 24/130 |
| context_l10_dfs | shared_history | all | action | 32/121 |
| fixed_l10_bfs | exact_field | semicolon | frontier | 4/199 |
| fixed_l10_bfs | exact_field | semicolon | candidates | 39/120 |
| fixed_l10_bfs | exact_field | semicolon | action | 35/112 |
| fixed_l10_bfs | exact_field | all | frontier | 4/110 |
| fixed_l10_bfs | exact_field | all | candidates | 22/36 |
| fixed_l10_bfs | exact_field | all | action | 16/37 |
| fixed_l10_dfs | exact_field | operator | candidates | 28/60 |
| fixed_l10_dfs | exact_field | operator | action | 28/65 |
| fixed_l10_dfs | exact_field | all | candidates | 26/60 |
| fixed_l10_dfs | exact_field | all | action | 26/65 |

较干净的同历史新图DFS实验：L4操作符替换后，指定donor动作迁移 32/121，候选集合迁移 24/130；无历史替换的对应计数为 0/121 和 0/130。前者的输入图不同，但查询名称、整段CoT文本及位置一致，支持操作符表示携带特定后续信息，而不只是一般生成破坏。

key_examples.json保存少量两边控制均完整正确的实际续写、参考状态及可复现数据索引。案例是定向挑出的说明例子，不用于估计迁移比例。

这些计数仍需和同token随机donor、范数控制等比较，不能单凭匹配donor就排除偶然错误。summary.json中的置信区间按recipient基础图/查询unit聚类bootstrap，donor复用的相关性未完整建模；固定图依然只有seed0，区间不表示跨图seed的不确定性。

## 当前记录控制

shared_history 两边当前完整记录相同，字段后立即续写当前记录的正确输出应相同。如果干预改变这条记录，这是局部扰动或不必要的信息干扰，不能算向不同正确后续迁移。exact_field当前记录可有不同正确后续。两种读取及所有层、字段、失败统计保留在压缩统计中。

## 校验与边界

每条生成轨迹经独立状态机重放，配对加载时校验前缀哈希；checkpoint运行前后校验。self-patch 6165条、末层过去位置负对照 2226条均严格无变化。

同一基础图的命名变体共享unit，不被当作独立图。新图基础来源包含train/val/test；固定图为同一已知世界上的额外合法查询。新图BFS沿用验证最佳完整状态checkpoint，训练step仅5000，控制表现不足时不能把缺少迁移归结为该架构没有相关信息。

没有导航模型训练、checkpoint、W&B或激活/KV文件。这里只续写一条记录，未测最终YES/NO、长程恢复或完整规划。

![精确字段配对](exact_field.png)

![共同历史配对](shared_history.png)
