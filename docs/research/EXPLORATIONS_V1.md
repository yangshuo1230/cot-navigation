# 探索实验入口

2026-10-16。已实现用户确认的探索；Filler不进入本入口。代码和入口已部署，交付时没有启动正式训练。

## 运行

```bash
cd /mnt/ningxuefei/yangshuo/cot-navigation

# 先查看实验计划；不准备数据、不创建run、不启动训练。
bash scripts/explore.sh --run-name explore_v1_s012 --dry-run

# 默认完整探索：标准图，新图50k独立训练图，固定图seed0/1/2。
bash scripts/explore.sh --run-name explore_v1_s012 --gpus 0 1 2 3 4 5 6 7 --wait --background

# 分批运行：Direct的基线、单跳和课程。
bash scripts/explore.sh --setting context --experiments baseline direct_onehop direct_curriculum \
  --run-name explore_direct --gpus 0 1 2 3 4 5 6 7 --wait --background

# Path负例路径；同时训练新图和三个独立固定图。
bash scripts/explore.sh --experiments baseline path_negative --run-name explore_path \
  --gpus 0 1 2 3 4 5 6 7 --wait --background

# 搜索协议比较，包含固定图随机DFS。
bash scripts/explore.sh --experiments baseline dfs_random bfs_state dfs_nodes bfs_nodes \
  --run-name explore_search --gpus 0 1 2 3 4 5 6 7 --wait --background

# 固定图WALK消融。
bash scripts/explore.sh --setting fixed --experiments baseline walk_balanced walk_none \
  --run-name explore_walk --gpus 0 1 2 3 4 5 6 7 --wait --background

# 相同N32/M48下的宽浅与窄深；可按需缩小experiments列表。
bash scripts/explore.sh --shapes wide deep --experiments baseline bfs_state dfs_nodes bfs_nodes \
  --run-name explore_shapes --gpus 0 1 2 3 4 5 6 7 --wait --background

# 仅准备相同选项对应的数据/token缓存，不训练、不占GPU、不创建训练run。
bash scripts/explore.sh --prepare-only

# 恢复时保留原run-name及全部实验/形状/数据/种子/步数选项。
bash scripts/explore.sh --run-name explore_v1_s012 --gpus 0 1 2 3 4 5 6 7 --wait --background --resume
```

调度器等待空闲卡，不打断其他进程。新入口沿用单卡独立训练、BF16、SDPA、torch.compile、可用的fused AdamW；8卡调度不同实验，不是DDP。不自动启动smoke或计时训练。

默认任务阶段50k updates，知识阶段1k；有效查询batch64，固定图任务阶段通常另加16条WALK。模型仍为6层、单头、d128、FFN512、RoPE、约1.22M参数。`--context-steps`、`--task-steps`、`--knowledge-steps`控制预算，`--model-seed`控制初始化。

默认完整批次60项：新图9个分支；固定图每个seed含1个Stage A与16个Stage B，三图共51项。每张固定图的所有分支共用同一Stage A；不同图各自学习，绝不把三图混在一个模型里。只选择固定图专属实验却指定`--setting context`会报空计划，不会悄悄启动别的组。

## 分支与比较关系

| experiment | 协议 | 设置 | 相对共同基线的变化 |
| --- | --- | --- | --- |
| baseline | Direct / Path / DFS完整状态 | 两种 | 新图每次曝光重排边；固定图沿用combined查询池 |
| direct_onehop | Direct | 两种 | 仅训练单跳，单跳内正负平衡 |
| direct_curriculum | Direct | 两种 | 逐跳扩展，阶段和已纳入跳数均分 |
| walk_balanced | Direct / Path / DFS完整状态 | 固定图 | 自动平衡查询与WALK损失贡献 |
| walk_none | Direct / Path / DFS完整状态 | 固定图 | Stage B不采样WALK、无WALK损失；Stage A保留 |
| path_negative | Path | 两种 | 负例每次曝光重新采样起点出发的随机路径 |
| dfs_random | DFS完整状态 | 固定图 | 每次曝光独立重排各节点邻居次序 |
| bfs_state | BFS完整状态 | 两种 | 队列搜索，沿用已有行为token |
| dfs_nodes | DFS节点序列 | 两种 | 仅首次展开节点序列，无行为/状态字段 |
| bfs_nodes | BFS节点序列 | 两种 | 仅首次展开节点序列，无行为/状态字段 |

这些是隔离变量的分支，不是所有开关的笛卡尔积。没有默认把随机DFS、负例路径和WALK平衡全叠到同一个分支。协议数不同，训练token/FLOPs也不同；本轮首先保持查询曝光一致，不宣称等计算量。

## 课程与单跳

阶段k仅取训练划分中的1..k跳，按跳数均分批次，每个跳数内部正负均分。整数余数轮换，长期每桶相差至多1条；不会永久丢弃桶内查询。各阶段平分总训练步数，除不超过1步的整数余数。

标准新图最多3跳，50k步分为16667/16667/16666；第二阶段单跳/两跳各1/2，第三阶段各1/3。标准固定图最多7跳，前6阶段各7143步，末阶段7142步。宽浅图最多3跳，窄深图最多7跳。课程切换不重置optimizer或学习率，连续使用50k步cosine；不是每阶段各训50k。

新图单跳分支仅在独立val/test图的单跳查询上验证/测试，建立基础任务成功锚点；课程分支仍在完整held-out跳数上评估，另报分跳数结果。固定图的现有单跳节点对全部属于train，故固定图单跳分支用训练拟合检查基础关系，并在原多跳val/test上检查迁移；不声称拥有固定图未见单跳测试，也不改变原划分来制造它。

## WALK损失平衡

查询损失保持原定义：Direct为答案CE；有轨迹协议为答案CE加每例轨迹CE均值。查询均值记作Q，WALK转移CE均值记作W。

- 基线：`Q + W`。
- 平衡：`Q + lambda * W`，lambda使用上一update的`EMA(Q)/EMA(W)`，默认EMA decay=0.98、首步lambda=1、安全上限100、分母下限1e-12。系数detach，整批所有microbatch共用，不对lambda求导。
- 无WALK：只有Q。

这里平衡的是损失数值贡献，不是梯度范数。权重在训练早期也可能大于1；查询CE降下来后通常会降低WALK权重。滞后的EMA不能保证每个随机batch严格50%，因此同时记录`task_loss`、原始`walk_loss`、`walk_weight`、`weighted_walk_loss`、`walk_loss_share`；以实际占比核查是否达到目标。没有额外模型forward。

## Path和搜索格式

负例随机路径从查询起点开始，每步均匀选择真实出边，直到无出边，最后NO；无出边时只输出起点。不是在所有完整路径之间均匀采样，也不使用目标/参考成功路径挑路。每次曝光重采样，不要求每次结果必然不同。正例沿用原成功路径规则：固定图在该查询所有合法成功路径间随机选路，新图保持原成功路径。评估参考负例路径用独立固定seed，不消耗训练RNG；自由生成只输入原题目，由模型自行作答。

完整BFS记录为`queue | visited | candidates | action [node] ;`，沿用PUSH/POP/FOUND/EXHAUST，无新增token。PUSH将一个未访问邻居加入队尾，同时标记visited；POP移除队首。队首等于目标时先FOUND，否则扩展候选；队列只余当前节点且无候选时EXHAUST。代码复用Step.stack字段存FIFO队列，BFS当前节点是首元素，DFS仍为末元素。

DFS/BFS节点序列均按当前节点首次展开顺序输出，每个节点至多一次，直到找到目标或搜索耗尽，再输出YES/NO。DFS回溯不输出父节点；BFS按展开而非发现顺序输出。因此序列中相邻节点未必相连，它是搜索顺序，不是成功路径。去掉栈/队列、visited、候选、行为及记录分隔符；保留原最终答案和受监督EOS。推理不额外用oracle或节点去重mask修正模型，主指标仍仅看最终答案。

## 图形状与数据身份

| shape | 新图 | 固定图 |
| --- | --- | --- |
| standard | N16/M24，4层×4宽 | N32/M48，8层×4宽 |
| wide | N32/M48，4层×8宽 | N32/M48，4层×8宽 |
| deep | N32/M48，8层×4宽 | N32/M48，8层×4宽 |

`wide/deep`控制相同节点数、边数，但候选边密度、距离分布仍会随形状变化，不将差异单独归因于深度。固定图standard与deep几何相同，比较宽深时选择`--shapes wide deep`即可，不必重复standard。

新图默认train50k、val/test各10k；可用`--context-sizes 10000 50000 150000`扩展。相同形状/seed下不同训练规模共享先生成的held-out图，各协议共享底层图和查询。新数据位于`data/explore_v1/`，不覆盖旧数据与结果。原始标签/划分继续由独立BFS和bitset审计；BFS与节点序列另有独立FIFO搜索及不重复性测试。

静态token缓存仍离线准备、训练时读内存。新图边重排后的完整搜索轨迹、固定图随机DFS轨迹和负例随机路径必须随曝光在线生成；只处理已在内存中的小DAG，不在update中读取文件或采样新图，不把随机轨迹当成无限缓存键。每组有独立edge/search/negative RNG，checkpoint保存其状态，验证不推进训练RNG。

注意：固定图训练路径（包括新负例路径、BFS/DFS序列与WALK）仍可能展示其他保留查询的完整见证。本轮保持既定query划分；高精度本身不能排除完整见证曝光，后续分析需按实际轨迹单独审计。随机化格式也不自动消除这一问题。

## 输出、W&B和恢复

输出为`outputs/explore_v1/launches/<run-name>/`，含固定plan、status、各分支配置/日志/checkpoint，以及完整test的best/last报告与逐例预测。日志位于相邻的`<run-name>.launch.log`。W&B沿用现有账号、项目、按run-name分组，`tracking.variant`明确标记分支；同时上传课程阶段、采样比例、分跳数准确率和加权损失贡献。现有W&B视图整理脚本的分组已加入variant与图层数，避免将不同课程或形状混成同一条曲线；交付没有写入外部W&B工作区。

若沿用旧的项目汇总视图，可先预览布局，再自行应用更新，保留Group选择并按variant/形状分组：

```bash
.venv/bin/python scripts/organize_wandb.py
.venv/bin/python scripts/organize_wandb.py --apply
```

所有分支主指标仅为自由生成最终YES/NO；错误路径不否决正确答案。搜索参考轨迹完整匹配只作辅助诊断，随机访问顺序训练尤其不能把与固定参考不一致直接当成算法错误。没有按oracle长度分配测试预算。

恢复检查覆盖课程阶段、增强开关、负例格式与WALK策略；所有新RNG和EMA状态随checkpoint恢复。课程阶段预算不能在恢复时改变，增加总步数请另建实验，而不是静默重划阶段。BFS与节点序列仍使用原词表，可共用本次同图Stage A；不复用历史不匹配的run配置。

smoke现在覆盖六协议的两个设置及知识阶段，共13项，不含Filler。临时数据清理、W&B禁用、不保存日志/checkpoint的约束保留。独立测试产生的临时恢复checkpoint在测试退出时删除，不属于正式训练产物。

## 验证

专项测试覆盖独立BFS oracle、唯一节点序列、边重排与候选次序一致性、随机合法负例路径、课程每阶段桶配额、静态新协议缓存、线上增强/EMA/随机选路的断点恢复，以及训练入口的父子依赖和形状配置。完整回归检查在CPU运行；随后GPU空闲时单独完成编译后的BF16/SDPA前向与梯度数值检查。未启动正式探索训练。

CPU完整检查118项通过；另行运行的CUDA编译/BF16/SDPA数值检查1项通过，共119项。补充的宽浅/窄深临时数据检查核对16320条缓存样本（含两个设置、六协议、三个固定图种子），全部符合长度上限和节点唯一性，退出后临时数据清理。真实CLI dry-run确认60个作业、未创建run目录。历史launch plan省略新增默认字段时的兼容检查也已通过。

N32图上仅在线变换的CPU测量约为Direct每64例1ms、完整DFS/BFS每64例7—9ms；这不包含图字段取出、模型计算、编译和验证，不能当作端到端训练速度。
