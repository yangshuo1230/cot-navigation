# N32纯任务训练：当前结果

固定图N32/E48，4层×8节点；随机初始化，无Stage A，无Stage B WALK。任务50k步、batch64、单seed0。训练288条，验证37条，测试59条（YES22、NO37），与匹配WALK对照使用完全相同划分及任务采样设置。

以下只列已有50k末次测试的纯任务分支，不代表整个13作业批次已完成；其余协议及WALK对照仍在运行。没有对照结果前，不判断删除WALK的因果收益。始终NO的原始准确率为62.71%，因此同时报告平衡准确率。

| 协议 | last步数 | 正确/测试 | 准确率 | 平衡准确率 |
| --- | --- | --- | --- | --- |
| bfs_nodes | 50000 | 54/59 | 91.53% | 89.56% |
| dfs_nodes | 50000 | 54/59 | 91.53% | 89.56% |
| direct | 50000 | 47/59 | 79.66% | 75.49% |
| path | 50000 | 51/59 | 86.44% | 82.74% |

已检查实际checkpoint：task_only=true、walk_sampler=None、walks_seen=0。W&B effective=online，group=task_only_n32_s0。GPU1–7，GPU0保留。当前数据全部目标及四轨迹变体通过语义检查；没有创建smoke run或smoke checkpoint。

匹配对照：同一N32单图，先1k步WALK知识预训练，再50k步QUERY+WALK；各协议与纯任务组共享数据划分、词表、查询/轨迹采样种子和任务优化日程。共6纯任务+1共享知识阶段+6对照任务。

训练结束后启动器会自动补测新模型的完整轨迹合法率，输出在该批次semantic_audit/，与原模型的冻结审计结果分开。

脚本：scripts/task_only_n32.sh；运行目录：outputs/task_only_n32_v1/launches/task_only_n32_s0；进度日志：outputs/task_only_n32_v1/launches/task_only_n32_s0.launch.log。启动与续跑说明见README.md。
