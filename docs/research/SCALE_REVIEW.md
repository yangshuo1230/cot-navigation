# 规模实验代码 review 与启动说明

服务器目录：`/mnt/ningxuefei/yangshuo/cot-navigation`。本次已修复并部署；未启动训练、未生成正式规模数据、未运行训练 smoke。

## 修复的主要问题

1. **旧实验兼容性**：Cursor 把 STOP 插入节点前，导致所有节点 ID 位移，旧 checkpoint 和手写数据脚本不再兼容。默认恢复 v2 的 270-token 词表和原 hash；规模实验在全部节点之后追加 STOP 和 WORLD token，单独使用 `v3_append`。真实旧 checkpoint 已严格加载验证。新规模实验须从新 run 开始，不与旧 checkpoint 混续。
2. **固定图混合的可辨识性**：多个世界复用节点名时，单靠起终点无法区分世界。新增 WORLD 前缀，知识阶段 WALK、任务阶段查询及测试使用一致世界标识，正确平移 loss mask。世界名是身份，不提供边列表。
3. **固定图泛化定义**：主设置只测试训练图上的未见查询及已训练的小规模世界；不把从未学过的固定世界作为普通长度泛化。另设 Direct/BFS-state 两个探针：知识阶段 WALK 涵盖全部测试世界，任务阶段查询只来自六个训练规模，WALK 持续覆盖全部世界。报告应称“知识已见、查询未见”。
4. **数据覆盖与采样**：补入 N24、N32；固定图恢复完整查询池、25% 一跳监督和标签平衡；混合采样器实际执行这些规则，而非只在配置里声明。训练期间新图每次出现重新排列边列表。
5. **缓存与数据准备**：搜索轨迹提前生成有限随机变体并按 context 截断，训练只采样缓存，不再反复执行 DFS/BFS。准备入口尊重 setting、数据 seed、用户参数；避免再由通用 launcher 重复生成完整搜索缓存；共享数据集的缓存写入加锁。
6. **训练/推理截断一致性**：训练长轨迹监督 STOP→原 YES/NO；推理到预算也注入 STOP，全状态轨迹回退至最后完整 `;` 记录。STOP 后只在模型的 YES/NO logits 中选择，不使用正确标签。保存中断率、强制中断率、中断/未中断准确率和逐查询中断标识。
7. **续跑和评估**：checkpoint provenance 覆盖整个训练混合集；课程验证随当前主规模切换；最终主测试切回主世界；完成判断必须包含额外测试报告；GPU 参数仅允许 1–7。

## 最终任务表

两种知识设置均跑六种协议：Direct、正例路径不变/负例起点随机游走 Path、随机 DFS-state、随机 BFS-state、无重复节点 DFS-nodes、无重复节点 BFS-nodes。全状态中的栈/队列/VIS 本来就会重复展示节点；“不重复”约束指搜索不重复发现节点及 nodes 输出不重复节点。

| 主规模 | N / E | 每协议任务更新数 | 混合比例 |
|---|---:|---:|---|
| 4×4 | 16 / 24 | 50k | 仅 N16 |
| 6×6 | 36 / 54 | 75k | N36 70%，N16/24/32 各 10% |
| 8×8 | 64 / 96 | 100k | N64 70%，N16/24/32/36 各 7.5% |
| 10×10 | 100 / 150 | 125k | N100 70%，N16/24/32/36/64 各 6% |
| 课程 | 16→36→64→100 | 200k | 每 50k 切换主规模，使用该阶段上述混合比例 |

主任务 2×5×6=60 项；固定图五组各共享 1k 知识更新；两个额外迁移探针各 125k，并共享另一组 1k 知识更新。合计 **68 个作业、6,856,000 次更新**，单模型 seed=0，数据 seed=0。每个作业独占一张卡，1–7 卡排队，0 卡留空。

固定图课程在最初 1k 知识阶段只学习 N16，后续新规模通过对应阶段 WALK 回放接触。它同时改变图知识覆盖及查询难度，不能解释为“所有图事先已知”的纯查询课程。

## 数据、模型与验证

- 新图每个训练规模默认 50k 图，val/test 各 512 图；深/宽/额外更大规模只准备极小 train/val 占位和 512 test。原始划分与可达性检查保留；新图抽检比例 1%，固定图 100%，不是宣称所有新图都做了完整独立审计。
- 同节点数的深图将层数加倍、宽度减半；宽图反之。新图每个主规模测主规模、已混合小规模、深、宽、下一档大图，最大测 N144/E216。
- 默认模型保持 6 层、1 head、d=128、FFN=512、RoPE、context=4096。bf16、SDPA、fused optimizer（可用时）、torch.compile；规模变长使用 dynamic compile。未做实际吞吐/显存测试，不承诺加速倍数。
- 默认 query batch=64；固定图任务阶段 WALK batch=16，知识阶段 WALK batch=64。lr=1e-3、min_lr=1e-4、warmup=100。继承固定图 WALK 权重模式 `fixed`、weight=1；可用 `--set train.walk_loss_mode=balanced` 改为历史探索中的 EMA 近似平衡模式，或 `none` 做无 WALK 消融。
- 每 5k 更新保存/验证；teacher-forced 128，free validation 16；最终每测试集最多 256 查询，分别评估 best/last。best 使用 free validation balanced accuracy，课程仅从最后阶段开始选择。16 条验证较噪，最终比较应以测试结果为主。
- 为减慢验证成本，规模批次关闭逐字段轨迹诊断，但继续保存最终准确率、格式失败和中断统计。轨迹因果/语义分析应另做离线评估，不能用“与某一条随机参考串完全一致”判断随机搜索正确性。

随机轨迹采用**每查询默认 4 个离线随机变体**，每次曝光随机选一个；不同 seed 不保证得到不同遍历（链图通常只有一种）。这是有限变体库，并非每次重新生成无限新轨迹。Path 正例从原准备的路径中抽样，负例从起点随机沿出边走到 sink，允许只有起点；不严格匹配正例长度。可调 `--variants`，会增加准备、磁盘和内存开销。

context 末尾预留 8 个位置；长全状态 CoT 在完整记录边界截断。中断后的答案属于预算限制下的推理结果，不能称为完整搜索结果。训练随机变体的截断正确性已用小型 DAG 检查，真实规模截断比例需准备数据后统计。

## 启动与续跑

```bash
cd /mnt/ningxuefei/yangshuo/cot-navigation

# 只预览，不准备数据或训练
bash scripts/scale.sh --run-name scale_v1_s0 --dry-run

# 可选：提前准备。正式入口也会先准备完再启动 GPU 作业。
bash scripts/scale.sh --prepare-only

# 正式启动（本次没有执行）
bash scripts/scale.sh --run-name scale_v1_s0 --gpus 1 2 3 4 5 6 7 --wait --background

# 相同参数续跑；若启动时改了 variants/context-train/set，续跑要保持相同。
bash scripts/scale.sh --run-name scale_v1_s0 --gpus 1 2 3 4 5 6 7 --wait --background --resume
```

W&B 默认 online，group=`scale_v1_s0`，每项单独 name；沿用现有认证配置。可设 `--tracking offline`，没有验证远端账号写入。输出位于 `outputs/scale_v1/launches/scale_v1_s0`，launcher 日志位于其上级 `scale_v1_s0.launch.log`。

可选参数：`--setting context|fixed|all`、`--context-train 50000`、`--variants 4`、`--no-transfer-probes`、`--no-compile` 以及非 runner 专属的 `--set KEY=VALUE`。保留原 `retrain.sh`、`explore.sh` 和 GPU0 的 `research/direct_minimal_v2_20261018/launch.sh`；默认旧词表/旧协议语义不变，不把这批大规模预算强加给旧研究。

## 验证记录与限制

服务器执行同一组 CPU pytest（首次发现旧测试仍断言默认 v3；修正断言后复跑通过）：

```bash
CUDA_VISIBLE_DEVICES="" PYTHONPATH=src .venv/bin/python -m pytest -q \
  tests/test_scale_review.py tests/test_scale.py tests/test_protocols.py
# 21 passed
```

覆盖真实旧 checkpoint、68 项配置/世界知识边界、prefix/mask、六协议离线变体与截断、CPU 假模型的预算中断和答案选择，以及 extra-set 完成检查。没有执行 optimizer 更新、W&B 上传或 GPU 训练 smoke。只做这些定向检查，未运行完整测试集或准备全部规模数据；因此大规模数据生成耗时、RAM/磁盘峰值、编译和真实 GPU 速度仍待正式准备/运行确认。
