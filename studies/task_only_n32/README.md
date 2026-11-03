# N32固定图：纯任务训练与WALK对照

沿用最新scale数据的N32/E48、4层×8节点固定世界及原划分。训练288条（192单跳+96多跳），验证37条、测试59条，无跨split查询重叠。训练25%一跳、75%多跳，分别平衡正负标签。

- `task_only`：随机初始化，不进行Stage A，Stage B无WALK采样和loss。
- `walk_control`：相同随机初始架构，先共享1k步Stage A，再每步64 QUERY+16 WALK。
- 两组各Direct、负例随机路径Path、随机DFS/BFS全状态、无重复DFS/BFS节点六种协议；任务均50k步，batch64，单seed0，不混其他规模。
- 相同286-token词表、原WORLD标识、原四变体缓存、query/轨迹采样种子、优化器与任务学习率。沿用原骨干，不更改模型结构。
- W&B online，group=`task_only_n32_s0`；GPU1–7，保留GPU0。数据只读复用，不重新生成。
- `train.task_only`默认false，不改变旧训练；仅允许固定图Stage B、walk_loss_mode=none，拒绝预训练初始化。断点记录并校验此标志，旧checkpoint按false兼容。

启动：

```bash
cd /mnt/ningxuefei/yangshuo/cot-navigation
bash scripts/task_only_n32.sh --run-name task_only_n32_s0 --gpus 1 2 3 4 5 6 7 --wait --background
```

续跑时追加`--resume`。只跑六个纯任务分支可追加`--no-with-walk-control`（不能用于修改已有13项计划）。默认13作业=6纯任务+1共享知识阶段+6对照任务。

已进行CPU配置/构造检查：随机权重与同seed原始模型逐参数相同、无WALK sampler、零优化器更新；检查所有缓存、split与词表。没有创建smoke run或smoke checkpoint。
