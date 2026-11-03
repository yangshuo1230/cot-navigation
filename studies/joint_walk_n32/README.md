# N32补充组：无知识预训练，任务期QUERY+WALK

本组仅补充`A=无知识预训练，B=有任务期WALK`，不重跑已经完成的两组，也不额外启动历史已探索的`A有/B无`。

- 图：与task_only_n32_s0完全相同的N32/E48、4层×8节点单世界；train288、val37、test59。
- 模型：同骨干、286-token词表、WORLD标识和seed0；从原始随机权重开始，不加载任何知识阶段checkpoint。
- 任务：Direct、带负例随机路径的Path、随机DFS/BFS全状态、无重复DFS/BFS节点六种协议。
- 训练：每个分支50k步，从第1步开始每步64 QUERY+16 WALK；WALK loss权重1；查询一跳25%、多跳75%，各自平衡标签。
- 复用现有图、WALK及四变体轨迹缓存，保持任务学习率、query/WALK采样流和轨迹增强流与原WALK对照一致。
- 预算：没有1k步知识阶段；与原对照的任务阶段预算相同，总预算少了知识阶段。
- 评估：相同完整val/test、best/last口径；结束后自动补做独立轨迹语义重放，保留STOP前缀与完整轨迹的区别。
- GPU：限定1–7，保留GPU0。W&B online，group=`joint_walk_n32_s0`。

已验证：六个作业均无Stage A依赖；CPU构造出的初始权重与同seed原始模型逐参数相等；QUERY和WALK采样器均开启，与原WALK对照前10批和增强随机流一致；老启动计划和checkpoint仍可兼容恢复。该检查没有优化器更新、W&B或smoke checkpoint。

配置显式设置`train.skip_knowledge_stage=true`、`train.task_only=false`及`walk_loss_mode=fixed`。skip_knowledge_stage默认false，禁止同时加载预训练初始化；断点记录并校验此标志，避免将旧训练误续为新消融。

启动入口（本次已经启动）：

```bash
cd /mnt/ningxuefei/yangshuo/cot-navigation
bash scripts/task_only_n32.sh \
  --arm joint_from_scratch \
  --run-name joint_walk_n32_s0 \
  --gpus 1 2 3 4 5 6 7 \
  --wait --background
```

中断后使用同一命令并追加`--resume`。

运行目录：`outputs/task_only_n32_v1/launches/joint_walk_n32_s0`。

进度日志：`outputs/task_only_n32_v1/launches/joint_walk_n32_s0.launch.log`。

原对照：`outputs/task_only_n32_v1/launches/task_only_n32_s0`，包含`task_only`（无A无B WALK）和`walk_control`（有A有B WALK）。
