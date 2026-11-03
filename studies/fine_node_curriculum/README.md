# 更细、更充分的 DFS/BFS-node 课程训练

运行项目：`/mnt/ningxuefei/yangshuo/cot-navigation-unified`。
组名：`fine_node_curriculum_20261028`。

12 个独立模型：DFS-node / BFS-node × 简单 / 困难图 × 初始化 seed 0/1/2。
三个初始化共用 data seed 0，不等于三个独立生成的数据集。

| 节点数 | 形状（层/带数 × 宽度） | 本级更新数 | 累计更新数 |
|---|---|---:|---:|
| 12 | 3 × 4 | 100,000 | 100,000 |
| 16 | 4 × 4 | 100,000 | 200,000 |
| 24 | 4 × 6 | 100,000 | 300,000 |
| 32 | 4 × 8 | 100,000 | 400,000 |
| 36 | 6 × 6 | 100,000 | 500,000 |
| 48 | 6 × 8 | 100,000 | 600,000 |
| 64 | 8 × 8 | 100,000 | 700,000 |
| 80 | 8 × 10 | 100,000 | 800,000 |
| 100 | 10 × 10 | 200,000 | 1,000,000 |

最小规模从 N12 开始，保持每个规模 50k 训练图与 512/512 验证/测试图。N8 简单图在现有密度和形状下独立结构过少，不能继续沿用这组不重复图数量。每个阶段独立于验证结果固定推进，不以测试成绩决定课程或停止。

第一阶段只训练 N12，之后沿用 native ScaleMixtureSampler：70% 当前规模，30% 等分回放此前各规模；其累计配额算法、采样 RNG 与 checkpoint 保持原框架语义。简单图保持 M=1.5N，困难图保持 M=2N，沿用原来的 layered / multiscale_dag_v1 分布。两个家族各自扩规模，没有新增跨家族迁移阶段。

主干、目标和优化配方保持上一轮：单路严格因果 Transformer，6L/1H/d128/FFN512、RoPE/GELU、271 tokens、4096 context、dropout=0；batch64、AdamW、weight decay 0.01、clip1、答案 CE + 每例平均轨迹 CE。完整 1M 步的单次 warmup100 + cosine，LR 1e-3→1e-4，不在阶段切换时重置 optimizer/LR。每个查询四条离线随机合法搜索轨迹；每次暴露重排边列表，独立 RNG。

这轮同时增加课程细度和训练预算，与旧 200k 实验的比较不能单独分离二者的贡献。

## 复用与效率

- 直接导入统一框架的 Trainer、数据生成/校验、PreparedRows、轨迹准备器、语义校验器、checkpoint 与多卡 scheduler。
- 仅增加 `src/cot_navigation/experiments/fine_node_curriculum.py` 实验适配器；没有修改模型、loss、generator、通用训练器、通用调度器。
- 18 个图数据集全组共享：12 个旧规模数据集复用；仅新建两个家族的 N12/N48/N80 共 6 个数据集。
- 数据按三个独立 CPU 进程准备，每个生成任务最多四个 native generation workers，避免 Python GIL 将轨迹准备串行化。
- 两种协议各准备一次 flat 四变体 token 数组。兼容较大 context 的旧缓存时必须身份完全匹配、校验 checksum、所有真实行完整适配 4096，无截断。
- 原生全数据结构/label/split/checksum 校验；各数据集每协议另抽取 train/val/test 各12查询的全部4变体，共144条语义审计。
- 每数据集和整个批次保存文件状态凭据；重启复用完成数据与缓存，不重新生成已完成部分。
- 训练读取内存 token 数组，保留32,768条有界 Python 行缓存，避免九个规模全部 eagerly materialize 导致启动慢、内存膨胀。optimizer update 中无图生成/搜索、无数据文件读写。
- 使用原生 BF16、SDPA、torch.compile、编译 loss、fused AdamW、65,536 token 微批预算、长度分组、30% padding 约束以及编译 KV-cache 自回归评估。
- 每模型独占一张空闲卡，12任务在8卡上排队；不共享已有进程正在使用的卡。

## 指标与留档

- 每20k：当前阶段固定256题验证，YES/NO各128题；同时记录自由生成答案、teacher-forced诊断、全体/正例完整CoT合法率、重复节点/顺序/终止错误数。允许任意合法邻居顺序，不要求匹配参考字符串。
- 每阶段末：追加以前各规模的固定验证查询，观察小图能力保留；留存该阶段独立checkpoint。
- 每10k保存可恢复checkpoint，每100步训练日志与W&B；仅保留最近周期checkpoint，阶段checkpoint另存。
- 仅最终last在本家族N100测试512题，不根据test挑选模型或决定课程。
- 正例答案准确率与正例完整合法率分开记录。teacher-forced答案高分不能作为已学会搜索的证据。

## 入口

配置在 `configs/generated/fine_node_curriculum_20261028/`，实验辅助文件在 `studies/fine_node_curriculum/`。

```bash
cd /mnt/ningxuefei/yangshuo/cot-navigation-unified
# 查看计划
.venv/bin/python -m cot_navigation.experiments.fine_node_curriculum
# 只准备数据
.venv/bin/python -m cot_navigation.experiments.fine_node_curriculum --prepare-only
# 正式后台启动（数据凭据完成后）
.venv/bin/python -m cot_navigation.experiments.fine_node_curriculum --execute --wait --background
# 恢复同一批次
.venv/bin/python -m cot_navigation.experiments.fine_node_curriculum --execute --wait --background --resume
```

输出：`outputs/fine_node_curriculum_20261028/`；原生调度器 `status.json` 提供 pending/running/completed。

验证凭据：`studies/fine_node_curriculum/verification.json`（108个边界、CPU真实数据更新和完全一致的跨阶段恢复）；`verification_cuda_{dfs,bfs}_nodes.json`（真实batch64、N16/N100编译前后向、自由生成、完整forward/缓存logits一致、无编译fallback）。所有短检查均未创建正式训练run或W&B run；临时恢复checkpoint已自动清理。
