# 最新scale实验：CoT轨迹语义合法性检查

主分布完成 51/51；主分布及扩展条件完成 84/84；所有条件完成的模型 51/51。主分布使用与原评估相同的全部查询（最多256）和batch16；N100静态、课程及固定迁移探针另测deep/wide/larger，每条件64条分层子集。

冻结latest checkpoint、BF16、原4096上下文及原STOP策略。原模型文件检查前后SHA256一致。不创建W&B或训练checkpoint。语义验证器接受任意合法邻居优先级；已通过7680项4节点DAG穷举断言、STOP检查，以及49152个候选节点序列与全部参考邻居优先级的穷举等价检查。

完整轨迹合法要求所有状态/动作正确并合法终止；STOP后的未完成搜索不算完整合法。前缀合法允许未完成但已输出部分必须全部正确，空STOP单列。动作骨架忽略显示状态，仍要求动作可执行且完整终止。Path负例随机walk只检查起点、真实边及sink终点，不构成NO证明，所以合法路径和正确答案必须分别报告。non_node_token表示序列包含非本图节点或语法token，并不只表示特殊token。

## 新图：答案准确率 / 完整轨迹合法率

| 训练 | path | dfs_state | bfs_state | dfs_nodes | bfs_nodes |
| --- | --- | --- | --- | --- | --- |
| l4 | 72.27% / 54.30% | 75.00% / 42.97% | 51.56% / 26.95% | 90.62% / 67.97% | 46.88% / 0.00% |
| l6 | 54.30% / 3.12% | 67.58% / 25.39% | 66.41% / 31.64% | 67.58% / 35.16% | 55.08% / 14.06% |
| l8 | 58.98% / 22.66% | 91.80% / 55.86% | 51.95% / 8.20% | 50.00% / 10.55% | 52.73% / 9.77% |
| l10 | 55.47% / 6.64% | 70.70% / 31.25% | 50.00% / 12.50% | 50.00% / 13.28% | 58.20% / 24.61% |
| curriculum | 65.23% / 93.36% | 83.98% / 43.75% | 84.77% / 39.84% | 76.56% / 45.70% | 62.11% / 24.61% |

## 固定图：答案准确率 / 完整轨迹合法率

| 训练 | path | dfs_state | bfs_state | dfs_nodes | bfs_nodes |
| --- | --- | --- | --- | --- | --- |
| l4 | 94.74% / 94.74% | 73.68% / 47.37% | 73.68% / 57.89% | 78.95% / 47.37% | 68.42% / 47.37% |
| l6 | 90.35% / 93.86% | 78.07% / 64.04% | 74.56% / 59.65% | 93.86% / 92.11% | 85.09% / 73.68% |
| l8 | 94.92% / 96.48% | 95.31% / 84.38% | 94.53% / 89.06% | 96.88% / 90.23% | 94.92% / 87.11% |
| l10 | 98.44% / 98.83% | 96.48% / 85.55% | 97.66% / 83.98% | 97.27% / 88.28% | 98.05% / 88.67% |
| curriculum | 88.67% / 94.53% | 95.31% / 78.52% | 96.88% / 87.89% | 97.66% / 83.20% | 92.19% / 78.52% |

## 全部分项

| 模型 | 条件 | n | 答案 | 完整合法 | 答案且完整合法 | 动作骨架 | 前缀合法 | STOP | STOP中前缀合法 | 原答案一致 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| context/curriculum/bfs_nodes | deep | 64 | 59.38% | 17.19% | 17.19% | 17.19% | 25.00% | 0.00% | — | 64/64 |
| context/curriculum/bfs_nodes | larger | 64 | 56.25% | 28.12% | 28.12% | 28.12% | 39.06% | 0.00% | — | 64/64 |
| context/curriculum/bfs_nodes | main | 256 | 62.11% | 24.61% | 24.61% | 24.61% | 32.81% | 0.00% | — | 256/256 |
| context/curriculum/bfs_nodes | wide | 64 | 64.06% | 20.31% | 20.31% | 20.31% | 29.69% | 0.00% | — | 64/64 |
| context/curriculum/bfs_state | deep | 64 | 82.81% | 45.31% | 45.31% | 45.31% | 45.31% | 17.19% | 0.00% | 64/64 |
| context/curriculum/bfs_state | larger | 64 | 81.25% | 40.62% | 40.62% | 40.62% | 40.62% | 17.19% | 0.00% | 64/64 |
| context/curriculum/bfs_state | main | 256 | 84.77% | 39.84% | 39.84% | 41.02% | 39.84% | 8.98% | 0.00% | 256/256 |
| context/curriculum/bfs_state | wide | 64 | 84.38% | 46.88% | 46.88% | 51.56% | 46.88% | 1.56% | 0.00% | 64/64 |
| context/curriculum/dfs_nodes | deep | 64 | 78.12% | 40.62% | 40.62% | 40.62% | 40.62% | 0.00% | — | 64/64 |
| context/curriculum/dfs_nodes | larger | 64 | 51.56% | 23.44% | 23.44% | 23.44% | 90.62% | 0.00% | — | 64/64 |
| context/curriculum/dfs_nodes | main | 256 | 76.56% | 45.70% | 45.70% | 45.70% | 48.05% | 0.00% | — | 256/256 |
| context/curriculum/dfs_nodes | wide | 64 | 76.56% | 56.25% | 56.25% | 56.25% | 57.81% | 0.00% | — | 64/64 |
| context/curriculum/dfs_state | deep | 64 | 78.12% | 37.50% | 37.50% | 39.06% | 37.50% | 6.25% | 0.00% | 64/64 |
| context/curriculum/dfs_state | larger | 64 | 81.25% | 39.06% | 39.06% | 46.88% | 39.06% | 9.38% | 0.00% | 64/64 |
| context/curriculum/dfs_state | main | 256 | 83.98% | 43.75% | 43.75% | 47.66% | 43.75% | 3.91% | 0.00% | 256/256 |
| context/curriculum/dfs_state | wide | 64 | 81.25% | 54.69% | 54.69% | 57.81% | 54.69% | 0.00% | — | 64/64 |
| context/curriculum/path | deep | 64 | 64.06% | 92.19% | 57.81% | 92.19% | 93.75% | 0.00% | — | 64/64 |
| context/curriculum/path | larger | 64 | 59.38% | 90.62% | 53.12% | 90.62% | 93.75% | 0.00% | — | 64/64 |
| context/curriculum/path | main | 256 | 65.23% | 93.36% | 62.50% | 93.36% | 93.36% | 0.00% | — | 256/256 |
| context/curriculum/path | wide | 64 | 73.44% | 95.31% | 70.31% | 95.31% | 95.31% | 0.00% | — | 64/64 |
| context/l10/bfs_nodes | deep | 64 | 53.12% | 20.31% | 20.31% | 20.31% | 25.00% | 0.00% | — | 64/64 |
| context/l10/bfs_nodes | larger | 64 | 56.25% | 21.88% | 21.88% | 21.88% | 37.50% | 0.00% | — | 64/64 |
| context/l10/bfs_nodes | main | 256 | 58.20% | 24.61% | 24.61% | 24.61% | 28.12% | 0.00% | — | 256/256 |
| context/l10/bfs_nodes | wide | 64 | 56.25% | 14.06% | 14.06% | 14.06% | 21.88% | 0.00% | — | 64/64 |
| context/l10/bfs_state | deep | 64 | 50.00% | 7.81% | 7.81% | 7.81% | 7.81% | 0.00% | — | 64/64 |
| context/l10/bfs_state | larger | 64 | 50.00% | 14.06% | 14.06% | 14.06% | 14.06% | 0.00% | — | 64/64 |
| context/l10/bfs_state | main | 256 | 50.00% | 12.50% | 12.50% | 12.50% | 12.50% | 0.00% | — | 256/256 |
| context/l10/bfs_state | wide | 64 | 50.00% | 6.25% | 6.25% | 6.25% | 6.25% | 0.00% | — | 64/64 |
| context/l10/dfs_nodes | deep | 64 | 50.00% | 7.81% | 7.81% | 7.81% | 100.00% | 0.00% | — | 64/64 |
| context/l10/dfs_nodes | larger | 64 | 51.56% | 9.38% | 9.38% | 9.38% | 59.38% | 0.00% | — | 64/64 |
| context/l10/dfs_nodes | main | 256 | 50.00% | 13.28% | 13.28% | 13.28% | 100.00% | 0.00% | — | 256/256 |
| context/l10/dfs_nodes | wide | 64 | 50.00% | 7.81% | 7.81% | 7.81% | 98.44% | 0.00% | — | 64/64 |
| context/l10/dfs_state | deep | 64 | 67.19% | 26.56% | 26.56% | 26.56% | 26.56% | 0.00% | — | 64/64 |
| context/l10/dfs_state | larger | 64 | 62.50% | 25.00% | 25.00% | 29.69% | 25.00% | 4.69% | 0.00% | 64/64 |
| context/l10/dfs_state | main | 256 | 70.70% | 31.25% | 31.25% | 34.77% | 31.25% | 0.00% | — | 256/256 |
| context/l10/dfs_state | wide | 64 | 79.69% | 26.56% | 26.56% | 32.81% | 26.56% | 0.00% | — | 64/64 |
| context/l10/path | deep | 64 | 54.69% | 3.12% | 3.12% | 3.12% | 17.19% | 0.00% | — | 64/64 |
| context/l10/path | larger | 64 | 43.75% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | — | 64/64 |
| context/l10/path | main | 256 | 55.47% | 6.64% | 6.64% | 6.64% | 14.45% | 0.00% | — | 256/256 |
| context/l10/path | wide | 64 | 48.44% | 6.25% | 6.25% | 6.25% | 21.88% | 0.00% | — | 64/64 |
| context/l4/bfs_nodes | main | 256 | 46.88% | 0.00% | 0.00% | 0.00% | 1.95% | 0.00% | — | 256/256 |
| context/l4/bfs_state | main | 256 | 51.56% | 26.95% | 26.95% | 26.95% | 26.95% | 0.00% | — | 256/256 |
| context/l4/dfs_nodes | main | 256 | 90.62% | 67.97% | 67.97% | 67.97% | 77.73% | 0.00% | — | 256/256 |
| context/l4/dfs_state | main | 256 | 75.00% | 42.97% | 42.97% | 48.83% | 42.97% | 0.00% | — | 256/256 |
| context/l4/path | main | 256 | 72.27% | 54.30% | 48.05% | 54.30% | 67.97% | 0.00% | — | 256/256 |
| context/l6/bfs_nodes | main | 256 | 55.08% | 14.06% | 14.06% | 14.06% | 71.09% | 0.00% | — | 256/256 |
| context/l6/bfs_state | main | 256 | 66.41% | 31.64% | 31.64% | 33.59% | 31.64% | 0.00% | — | 256/256 |
| context/l6/dfs_nodes | main | 256 | 67.58% | 35.16% | 35.16% | 35.16% | 36.33% | 0.00% | — | 256/256 |
| context/l6/dfs_state | main | 256 | 67.58% | 25.39% | 25.39% | 34.38% | 25.39% | 0.00% | — | 256/256 |
| context/l6/path | main | 256 | 54.30% | 3.12% | 3.12% | 3.12% | 14.84% | 0.00% | — | 256/256 |
| context/l8/bfs_nodes | main | 256 | 52.73% | 9.77% | 9.77% | 9.77% | 41.80% | 0.00% | — | 256/256 |
| context/l8/bfs_state | main | 256 | 51.95% | 8.20% | 8.20% | 8.98% | 8.20% | 0.00% | — | 256/256 |
| context/l8/dfs_nodes | main | 256 | 50.00% | 10.55% | 10.55% | 10.55% | 100.00% | 0.00% | — | 256/256 |
| context/l8/dfs_state | main | 256 | 91.80% | 55.86% | 55.86% | 59.77% | 55.86% | 0.00% | — | 256/256 |
| context/l8/path | main | 256 | 58.98% | 22.66% | 21.09% | 22.66% | 35.55% | 0.00% | — | 256/256 |
| fixed/curriculum/stage_b/bfs_nodes | main | 256 | 92.19% | 78.52% | 78.52% | 78.52% | 82.03% | 0.00% | — | 256/256 |
| fixed/curriculum/stage_b/bfs_state | main | 256 | 96.88% | 87.89% | 87.89% | 87.89% | 89.45% | 11.33% | 13.79% | 256/256 |
| fixed/curriculum/stage_b/dfs_nodes | main | 256 | 97.66% | 83.20% | 83.20% | 83.20% | 90.23% | 0.00% | — | 256/256 |
| fixed/curriculum/stage_b/dfs_state | main | 256 | 95.31% | 78.52% | 78.52% | 82.03% | 78.91% | 3.12% | 12.50% | 256/256 |
| fixed/curriculum/stage_b/path | main | 256 | 88.67% | 94.53% | 85.55% | 94.53% | 96.88% | 0.00% | — | 256/256 |
| fixed/knowledge_transfer/stage_b/bfs_state | deep | 64 | 57.81% | 0.00% | 0.00% | 0.00% | 0.00% | 10.94% | 0.00% | 64/64 |
| fixed/knowledge_transfer/stage_b/bfs_state | larger | 64 | 48.44% | 9.38% | 9.38% | 10.94% | 10.94% | 1.56% | 0.00% | 64/64 |
| fixed/knowledge_transfer/stage_b/bfs_state | main | 256 | 96.88% | 87.89% | 87.89% | 87.89% | 96.48% | 8.98% | 95.65% | 256/256 |
| fixed/knowledge_transfer/stage_b/bfs_state | wide | 64 | 43.75% | 1.56% | 1.56% | 1.56% | 1.56% | 12.50% | 0.00% | 64/64 |
| fixed/l10/stage_b/bfs_nodes | main | 256 | 98.05% | 88.67% | 88.67% | 88.67% | 89.45% | 0.00% | — | 256/256 |
| fixed/l10/stage_b/bfs_state | main | 256 | 97.66% | 83.98% | 83.98% | 83.98% | 91.41% | 10.94% | 67.86% | 256/256 |
| fixed/l10/stage_b/dfs_nodes | main | 256 | 97.27% | 88.28% | 88.28% | 88.28% | 89.84% | 0.00% | — | 256/256 |
| fixed/l10/stage_b/dfs_state | main | 256 | 96.48% | 85.55% | 85.55% | 87.50% | 88.28% | 4.69% | 58.33% | 256/256 |
| fixed/l10/stage_b/path | main | 256 | 98.44% | 98.83% | 98.05% | 98.83% | 98.83% | 0.00% | — | 256/256 |
| fixed/l4/stage_b/bfs_nodes | main | 19 | 68.42% | 47.37% | 47.37% | 47.37% | 78.95% | 0.00% | — | 19/19 |
| fixed/l4/stage_b/bfs_state | main | 19 | 73.68% | 57.89% | 57.89% | 57.89% | 57.89% | 0.00% | — | 19/19 |
| fixed/l4/stage_b/dfs_nodes | main | 19 | 78.95% | 47.37% | 47.37% | 47.37% | 52.63% | 0.00% | — | 19/19 |
| fixed/l4/stage_b/dfs_state | main | 19 | 73.68% | 47.37% | 47.37% | 47.37% | 47.37% | 0.00% | — | 19/19 |
| fixed/l4/stage_b/path | main | 19 | 94.74% | 94.74% | 89.47% | 94.74% | 94.74% | 0.00% | — | 19/19 |
| fixed/l6/stage_b/bfs_nodes | main | 114 | 85.09% | 73.68% | 73.68% | 73.68% | 74.56% | 0.00% | — | 114/114 |
| fixed/l6/stage_b/bfs_state | main | 114 | 74.56% | 59.65% | 59.65% | 61.40% | 59.65% | 0.00% | — | 114/114 |
| fixed/l6/stage_b/dfs_nodes | main | 114 | 93.86% | 92.11% | 92.11% | 92.11% | 93.86% | 0.00% | — | 114/114 |
| fixed/l6/stage_b/dfs_state | main | 114 | 78.07% | 64.04% | 64.04% | 64.04% | 64.04% | 0.00% | — | 114/114 |
| fixed/l6/stage_b/path | main | 114 | 90.35% | 93.86% | 85.09% | 93.86% | 94.74% | 0.00% | — | 114/114 |
| fixed/l8/stage_b/bfs_nodes | main | 256 | 94.92% | 87.11% | 87.11% | 87.11% | 87.89% | 0.00% | — | 256/256 |
| fixed/l8/stage_b/bfs_state | main | 256 | 94.53% | 89.06% | 89.06% | 89.06% | 89.06% | 0.39% | 0.00% | 256/256 |
| fixed/l8/stage_b/dfs_nodes | main | 256 | 96.88% | 90.23% | 90.23% | 90.23% | 91.41% | 0.00% | — | 256/256 |
| fixed/l8/stage_b/dfs_state | main | 256 | 95.31% | 84.38% | 84.38% | 89.06% | 84.38% | 0.00% | — | 256/256 |
| fixed/l8/stage_b/path | main | 256 | 94.92% | 96.48% | 92.19% | 96.48% | 96.88% | 0.00% | — | 256/256 |

## 首错类别

| 模型 | 条件 | 首次导致不合法的原因 |
| --- | --- | --- |
| fixed/l6/stage_b/dfs_nodes | main | {"continued_after_target": 4, "missing_or_inconsistent_terminal": 2, "repeated_node": 3} |
| fixed/l6/stage_b/dfs_state | main | {"candidate_mismatch": 4, "wrong_action": 35, "frontier_mismatch": 2} |
| fixed/l6/stage_b/path | main | {"non_edge": 4, "non_node_token": 2, "wrong_positive_endpoint": 1} |
| fixed/l6/stage_b/bfs_state | main | {"wrong_action": 40, "malformed_fields": 4, "candidate_mismatch": 2} |
| fixed/l6/stage_b/bfs_nodes | main | {"not_bfs_order": 8, "continued_after_target": 14, "non_node_token": 2, "repeated_node": 4, "wrong_start": 1, "missing_or_inconsistent_terminal": 1} |
| fixed/knowledge_transfer/stage_b/bfs_state | deep | {"candidate_mismatch": 56, "frontier_mismatch": 4, "malformed_fields": 4} |
| fixed/knowledge_transfer/stage_b/bfs_state | larger | {"frontier_mismatch": 20, "candidate_mismatch": 37, "missing_or_inconsistent_terminal": 1} |
| fixed/knowledge_transfer/stage_b/bfs_state | wide | {"candidate_mismatch": 57, "frontier_mismatch": 5, "malformed_fields": 1} |
| fixed/knowledge_transfer/stage_b/bfs_state | main | {"missing_or_inconsistent_terminal": 22, "candidate_mismatch": 2, "wrong_action": 7} |
| fixed/l8/stage_b/dfs_nodes | main | {"non_node_token": 6, "repeated_node": 13, "missing_or_inconsistent_terminal": 3, "continued_after_target": 2, "not_dfs_order": 1} |
| fixed/l8/stage_b/dfs_state | main | {"malformed_fields": 4, "candidate_mismatch": 27, "wrong_action": 8, "visited_mismatch": 1} |
| fixed/l8/stage_b/path | main | {"non_node_token": 3, "non_edge": 5, "wrong_positive_endpoint": 1} |
| fixed/l8/stage_b/bfs_state | main | {"wrong_action": 15, "candidate_mismatch": 12, "frontier_mismatch": 1} |
| fixed/l8/stage_b/bfs_nodes | main | {"not_bfs_order": 17, "continued_after_target": 6, "repeated_node": 3, "non_node_token": 5, "missing_or_inconsistent_terminal": 2} |
| fixed/l10/stage_b/dfs_nodes | main | {"not_dfs_order": 6, "repeated_node": 17, "non_node_token": 2, "missing_or_inconsistent_terminal": 4, "continued_after_target": 1} |
| fixed/l10/stage_b/dfs_state | main | {"candidate_mismatch": 14, "visited_mismatch": 2, "malformed_fields": 4, "wrong_action": 10, "missing_or_inconsistent_terminal": 7} |
| fixed/l10/stage_b/path | main | {"non_edge": 3} |
| fixed/l10/stage_b/bfs_state | main | {"candidate_mismatch": 20, "wrong_action": 2, "missing_or_inconsistent_terminal": 19} |
| fixed/l10/stage_b/bfs_nodes | main | {"non_node_token": 3, "not_bfs_order": 9, "repeated_node": 11, "missing_or_inconsistent_terminal": 2, "continued_after_target": 4} |
| fixed/curriculum/stage_b/dfs_nodes | main | {"missing_or_inconsistent_terminal": 18, "repeated_node": 21, "not_dfs_order": 3, "continued_after_target": 1} |
| fixed/curriculum/stage_b/dfs_state | main | {"wrong_action": 6, "candidate_mismatch": 41, "visited_mismatch": 4, "push_not_first_candidate": 1, "missing_or_inconsistent_terminal": 1, "frontier_mismatch": 2} |
| fixed/curriculum/stage_b/path | main | {"non_edge": 7, "wrong_positive_endpoint": 4, "negative_path_not_at_sink": 2, "non_node_token": 1} |
| fixed/curriculum/stage_b/bfs_state | main | {"candidate_mismatch": 22, "missing_or_inconsistent_terminal": 4, "malformed_fields": 2, "frontier_mismatch": 2, "wrong_action": 1} |
| fixed/curriculum/stage_b/bfs_nodes | main | {"continued_after_target": 15, "repeated_node": 19, "missing_or_inconsistent_terminal": 9, "not_bfs_order": 11, "non_node_token": 1} |
| fixed/l4/stage_b/dfs_nodes | main | {"not_dfs_order": 3, "missing_or_inconsistent_terminal": 1, "continued_after_target": 3, "wrong_start": 3} |
| fixed/l4/stage_b/dfs_state | main | {"candidate_mismatch": 3, "malformed_fields": 2, "wrong_action": 5} |
| fixed/l4/stage_b/path | main | {"wrong_start": 1} |
| fixed/l4/stage_b/bfs_state | main | {"candidate_mismatch": 2, "wrong_action": 6} |
| fixed/l4/stage_b/bfs_nodes | main | {"continued_after_target": 1, "missing_or_inconsistent_terminal": 6, "not_bfs_order": 3} |
| context/l6/dfs_nodes | main | {"not_dfs_order": 16, "non_node_token": 138, "missing_or_inconsistent_terminal": 3, "repeated_node": 9} |
| context/l6/dfs_state | main | {"candidate_mismatch": 191} |
| context/l6/path | main | {"non_node_token": 149, "non_edge": 61, "negative_path_not_at_sink": 29, "repeated_node": 8, "wrong_positive_endpoint": 1} |
| context/l6/bfs_state | main | {"candidate_mismatch": 174, "malformed_fields": 1} |
| context/l6/bfs_nodes | main | {"missing_or_inconsistent_terminal": 146, "non_node_token": 16, "not_bfs_order": 54, "repeated_node": 4} |
| context/l8/dfs_nodes | main | {"missing_or_inconsistent_terminal": 229} |
| context/l8/dfs_state | main | {"candidate_mismatch": 113} |
| context/l8/path | main | {"non_edge": 71, "wrong_positive_endpoint": 4, "negative_path_not_at_sink": 29, "non_node_token": 80, "repeated_node": 14} |
| context/l8/bfs_state | main | {"candidate_mismatch": 235} |
| context/l8/bfs_nodes | main | {"not_bfs_order": 92, "non_node_token": 50, "missing_or_inconsistent_terminal": 82, "repeated_node": 7} |
| context/l10/dfs_nodes | deep | {"missing_or_inconsistent_terminal": 59} |
| context/l10/dfs_nodes | larger | {"missing_or_inconsistent_terminal": 32, "wrong_start": 13, "non_node_token": 12, "empty_nodes": 1} |
| context/l10/dfs_nodes | wide | {"missing_or_inconsistent_terminal": 58, "not_dfs_order": 1} |
| context/l10/dfs_nodes | main | {"missing_or_inconsistent_terminal": 222} |
| context/l10/dfs_state | deep | {"candidate_mismatch": 43, "frontier_mismatch": 3, "visited_mismatch": 1} |
| context/l10/dfs_state | larger | {"candidate_mismatch": 46, "visited_mismatch": 2} |
| context/l10/dfs_state | wide | {"candidate_mismatch": 47} |
| context/l10/dfs_state | main | {"candidate_mismatch": 176} |
| context/l10/path | deep | {"non_node_token": 34, "non_edge": 18, "negative_path_not_at_sink": 9, "repeated_node": 1} |
| context/l10/path | larger | {"non_node_token": 27, "wrong_start": 19, "empty_path": 18} |
| context/l10/path | wide | {"non_node_token": 31, "non_edge": 19, "negative_path_not_at_sink": 10} |
| context/l10/path | main | {"negative_path_not_at_sink": 20, "non_edge": 93, "non_node_token": 119, "repeated_node": 7} |
| context/l10/bfs_state | deep | {"candidate_mismatch": 59} |
| context/l10/bfs_state | larger | {"candidate_mismatch": 55} |
| context/l10/bfs_state | wide | {"candidate_mismatch": 60} |
| context/l10/bfs_state | main | {"candidate_mismatch": 224} |
| context/l10/bfs_nodes | deep | {"not_bfs_order": 19, "non_node_token": 28, "missing_or_inconsistent_terminal": 3, "repeated_node": 1} |
| context/l10/bfs_nodes | larger | {"not_bfs_order": 24, "non_node_token": 16, "missing_or_inconsistent_terminal": 10} |
| context/l10/bfs_nodes | wide | {"non_node_token": 33, "not_bfs_order": 17, "missing_or_inconsistent_terminal": 5} |
| context/l10/bfs_nodes | main | {"not_bfs_order": 65, "non_node_token": 115, "repeated_node": 4, "missing_or_inconsistent_terminal": 9} |
| context/curriculum/dfs_nodes | deep | {"non_node_token": 31, "repeated_node": 4, "not_dfs_order": 3} |
| context/curriculum/dfs_nodes | larger | {"missing_or_inconsistent_terminal": 43, "not_dfs_order": 4, "non_node_token": 1, "repeated_node": 1} |
| context/curriculum/dfs_nodes | wide | {"not_dfs_order": 4, "non_node_token": 19, "repeated_node": 4, "missing_or_inconsistent_terminal": 1} |
| context/curriculum/dfs_nodes | main | {"non_node_token": 113, "repeated_node": 13, "missing_or_inconsistent_terminal": 6, "not_dfs_order": 7} |
| context/curriculum/dfs_state | deep | {"candidate_mismatch": 29, "frontier_mismatch": 11} |
| context/curriculum/dfs_state | larger | {"candidate_mismatch": 31, "malformed_fields": 2, "frontier_mismatch": 4, "visited_mismatch": 1, "wrong_action": 1} |
| context/curriculum/dfs_state | wide | {"candidate_mismatch": 29} |
| context/curriculum/dfs_state | main | {"candidate_mismatch": 144} |
| context/curriculum/path | deep | {"non_edge": 4, "negative_path_not_at_sink": 1} |
| context/curriculum/path | larger | {"negative_path_not_at_sink": 2, "non_node_token": 1, "non_edge": 3} |
| context/curriculum/path | wide | {"non_node_token": 2, "non_edge": 1} |
| context/curriculum/path | main | {"non_edge": 6, "non_node_token": 11} |
| context/curriculum/bfs_state | deep | {"candidate_mismatch": 35} |
| context/curriculum/bfs_state | larger | {"candidate_mismatch": 34, "visited_mismatch": 3, "wrong_action": 1} |
| context/curriculum/bfs_state | wide | {"candidate_mismatch": 33, "frontier_mismatch": 1} |
| context/curriculum/bfs_state | main | {"candidate_mismatch": 153, "malformed_fields": 1} |
| context/curriculum/bfs_nodes | deep | {"non_node_token": 15, "not_bfs_order": 26, "missing_or_inconsistent_terminal": 5, "repeated_node": 7} |
| context/curriculum/bfs_nodes | larger | {"not_bfs_order": 29, "non_node_token": 3, "repeated_node": 7, "missing_or_inconsistent_terminal": 7} |
| context/curriculum/bfs_nodes | wide | {"not_bfs_order": 36, "repeated_node": 5, "missing_or_inconsistent_terminal": 6, "non_node_token": 4} |
| context/curriculum/bfs_nodes | main | {"non_node_token": 20, "not_bfs_order": 126, "repeated_node": 26, "missing_or_inconsistent_terminal": 21} |
| context/l4/dfs_nodes | main | {"missing_or_inconsistent_terminal": 25, "not_dfs_order": 28, "non_node_token": 22, "repeated_node": 7} |
| context/l4/dfs_state | main | {"candidate_mismatch": 146} |
| context/l4/path | main | {"non_edge": 29, "non_node_token": 45, "negative_path_not_at_sink": 29, "repeated_node": 8, "wrong_positive_endpoint": 6} |
| context/l4/bfs_state | main | {"wrong_action": 22, "candidate_mismatch": 165} |
| context/l4/bfs_nodes | main | {"wrong_start": 148, "repeated_node": 11, "non_node_token": 87, "empty_nodes": 1, "not_bfs_order": 4, "missing_or_inconsistent_terminal": 5} |

每个错误类别计该轨迹的首个失败原因；这不表示后续动作、状态一定正确。完整合法率低于答案准确率时，不能认为都是无关状态注释错误，还要看动作骨架。OOD为64例诊断子集，其答案不与此前256例总体直接作差；BF16重组batch也可能影响临界生成。
