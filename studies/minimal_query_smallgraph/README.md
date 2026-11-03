# 同图多 query / 小图混入：最小测试

只添加独立研究脚本，复用原项目 Trainer、loss、模型、SDPA、torch.compile、
fused AdamW、checkpoint 和 W&B。不会修改原始数据、主框架或已有实验输出。

## 默认四组

| 组 | 初始化 | 数据 | 默认预算 |
|---|---|---|---|
| direct_single | 从零，model seed 0 | 4096 张原训练图，每图固定 1 个一跳 query | 10k steps |
| direct_multi | 与 single 相同 | 同一批图，每图 8 个不同一跳 query，4 YES / 4 NO | 10k steps |
| bfs_control | 原 BFS 50k latest | 原 N16/M24 数据 | 额外 5k steps |
| bfs_small_mix | 与 control 相同 | 每批 N16/M24 50%，N12/M12 25%，N8/M6 25% | 额外 5k steps |

架构始终为 6 层、1 头、d128、FFN512；batch 64。Direct 使用原学习率
1e-3 → 1e-4 cosine，100 步 warmup；BFS 续训两组均重置优化器，
1e-4 恒定学习率，100 步 warmup。固定单个模型种子，是用于筛查的最小测试。

### Direct 的控制条件

- 两组独立从相同权重初始化；不加载失败的 Direct 旧 checkpoint。
- 训练图来自原 train 的一跳样本，anchor query 正负各半；保留原节点名字。
- 每个优化步两组抽到的 **graph ID 顺序相同**，每张图曝光次数一致。
- single 反复监督 anchor；multi 每次曝光选一个 query，每图每八次曝光覆盖全部八个。
- 每条样本仍只含完整边列表、一个 query、一个答案；不是多问答拼接。
- 边列表每次曝光重排；对同一图两组使用相同的边重排 RNG 流。
- 最终保存初始化 hash、每图曝光计数和图采样 RNG hash，供成对核验。

评估：自己的训练 query、两组共享的 anchor query、训练图中两组都未训练的
query、独立新图 val/test。未见 query 诊断对前 256 张训练图各取 1 正 1 负，
排除 multi 全部八个 query。val 512 / test 1024，正负平衡，严格限定一跳。
train/val/test 检查拓扑边集合不重叠；这不是一般图同构去重。

**限制：**同图多 query 必然增加不同标签覆盖数，无法同时固定图数量、不同
query 数和总训练曝光。本测试固定图数量与总曝光，测的是同图监督覆盖的效果。
4096 图的小池与原 16668 条一跳训练集不同，应该比较本测试中的成对结果。
10k 阴性结果不能推出 50k 仍学不会；可用新 run-name 将预算改为 50k。

### 小图混入的控制条件

- 仅训练更小的图，最大 N16、M24、层数 4、query hop 3 都不增加。
- prompt 长度为 76 / 40 / 22，来自真实边列表，没有 filler 或额外独立子图。
- 小图也是四层相邻层 DAG；各 size × hop × label 单独生成 256 个样本，
  总共 3072 个。若某图不存在所需正/负查询，重采图，不改 hop 或 label。
- 混合组替换一个原训练样本时，严格保留其 hop 和 label；边顺序在线重排，
  BFS 轨迹同步按新邻接顺序重新生成，不截断。
- 两组原始 query 抽样流、原始边重排流相同，新增小图采样流独立并随 checkpoint 保存。
- 小图样本会重复使用；这是有限数据池，不是每次曝光生成新图。

最终采用先前已保存的 held-out 输入，每个 hop/label 取前 16 个：
ID、原查询加不可达分量、宽图、同输入长度深图、N24 深图、N32 深图。
同时报告最终答案、严格状态轨迹、可执行搜索骨架和逐 hop 指标，保存完整生成轨迹。
评估使用完整上下文预算，没有 gold 长度截断，也不会强制输出答案。

**限制：**该对照改变长度、图规模和稀疏度，不能单独证明位置依赖。
总 query 曝光相同，但混合组原 N16 样本曝光减半；计算量也不完全相同。
若有改善，后续再拆分变边数、变节点数和大图曝光控制。本测试不包含额外训练大图。

## 服务器使用

脚本目录：`/mnt/ningxuefei/yangshuo/cot-navigation/research/minimal_query_smallgraph_20261017`

只检查数据，不训练、不初始化模型/W&B、不保存 smoke 记录或 checkpoint：

```bash
cd /mnt/ningxuefei/yangshuo/cot-navigation
bash research/minimal_query_smallgraph_20261017/launch.sh --check-only
```

手动启动默认四组，在物理 0 卡依次执行，前台运行：

```bash
cd /mnt/ningxuefei/yangshuo/cot-navigation
GPU=0 bash research/minimal_query_smallgraph_20261017/launch.sh
```

后台运行并保留正式启动日志：

```bash
cd /mnt/ningxuefei/yangshuo/cot-navigation
GPU=0 nohup bash research/minimal_query_smallgraph_20261017/launch.sh \
  > research/minimal_query_smallgraph_20261017/launch.log 2>&1 &
```

可调参数示例（创建独立输出，不覆盖默认运行）：

```bash
GPU=0 bash research/minimal_query_smallgraph_20261017/launch.sh \
  --direct-steps 50000 --cot-steps 5000 --graphs 4096 \
  --run-name minimal_query_smallgraph_50k
```

只运行 Direct：

```bash
JOBS="direct_single direct_multi" GPU=0 \
  bash research/minimal_query_smallgraph_20261017/launch.sh
```

如需 DFS 的相同对照，脚本已支持（不是默认任务）：

```bash
JOBS="dfs_control dfs_small_mix" GPU=0 \
  bash research/minimal_query_smallgraph_20261017/launch.sh
```

使用完全相同参数再次执行：跳过已完成任务，未完成任务从自己的 latest.pt 恢复
优化器、采样器、RNG 和小图 batch quota。尚无 checkpoint 的任务会重新开始。
GPU 显存占用超过 1000 MiB 或同脚本已有 launcher 占卡时拒绝启动；不会杀其他进程。

默认 W&B online，沿用原项目凭证，上传失败直接报错。group 为 run-name，
默认 `minimal_query_smallgraph_v1`。训练指标、Direct 诊断和长度评估都上传，
checkpoint 仅存服务器。可用 `--tracking offline` 或 `--tracking disabled` 显式更改。

输出在脚本目录 `runs/<run-name>/<job>/`；checkpoint 的 step 对于 CoT 指**额外步数**。
`completed.json` 表示训练和最终评估均已完成。没有该文件时不要把已有 checkpoint
误判为完整结果。
