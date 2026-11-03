# Direct 一跳学习诊断与课程：80 GPU 小时后台队列

服务器：`/mnt/ningxuefei/yangshuo/cot-navigation`。
研究目录：`research/direct_learning_20261020`。
W&B group：`direct_learning_20261020_v1`；沿用项目 `cot-graph-reasoning`。

## 范围与预算

40 个分支，各最多 7200 秒，累计约 80 GPU 小时。只允许物理 GPU 1–7，GPU0 保留。最多同时运行 7 个模型，理想队列约 12 小时，已有其他进程占卡会延后。编译、周期评估计入每个分支的时间；末尾评估、保存、上传有少量额外开销。失败重试计入原分支预算，不另开 2 小时。

这个预算比固定 50k/100k 长得多，但并非数学上的不收敛证明。报告最终更新次数、查询曝光、完整训练准确率、末期 loss，以及在当前预算内是否仍停滞。

## 40 个分支

| 类别 | 数量 | 内容 |
|---|---:|---|
| seed0 基础/扩展 | 18 | A 固定图、B 固定 query、C 固定 source、D 固定 target；E 完整 query 固定边序、F 完整 query 随机边序；完整 query 的 64/256 名字×固定/随机边序；ordinary M2/M24；256 名字的 A/C/D；完整 M2 成组采样；16/256 名字 lr=1e-3 |
| seed1 匹配重复 | 10 | A/B/C/D/E/F、64/256 名字随机边序、ordinary M2/M24 |
| 端点课程 | 6 | source/target 门控课程各 seed0/1，固定时间课程各 seed0 |
| 混合教学课程 | 2 | source/target 受限查询占 80%→50%→20%→0%，然后扩名字与边数 |
| 其他门控课程 | 2 | 固定边序→随机边序，及直接按名字集合扩展 |
| 最终分布等时间对照 | 2 | 从头直接训练完整 paired M24，seed0/1，各 2 小时 |

模型 seed 同时设置模型初始化和采样 RNG；所有种子共享同一数据池，数据种子固定。不是每个实验都两个初始化，不把没有重复的结果当稳定结论。

## 数据与训练

- 保持原 6L/1H/d128/FFN512、RoPE/GELU、270 词表，无结构改动、无 CoT、无辅助监督、无 WALK。
- 所有输入为边三元组后接 `QUERY s t |`，只在最后 prompt 位置计算完整词表的 YES/NO CE，batch64。
- M2 只有四个可见端点；M4/8/16/24 在 4×4 相邻层 DAG 中加入干扰边，查询始终一跳。
- 16/64/256 指数据使用的节点名字范围，词表始终 256 个节点 token，不通过换模型缩词表。
- 非 ordinary 分支使用 G/G' 图对及受限/完整 query；每批正负各半。默认跨图/query 混排；packed 对照按 8 个完整图对组、每组 8 条采样。
- full 每池 train512、val128、test128 图对，完整 query 每图对8条。fixed-query16 为96/32/32图对。ordinary 每规模从4096/512/1024个图对中各取一个随机 query，再平衡标签，实际数量以 manifest 为准。
- A 固定图是已知世界拟合控制：train/val/test 采用同一组 query，明确不解释为未见图泛化。
- 其他池在生成时全局限制 M2 核心图的 split 所有权，避免课程不同池之间图泄露；独立遍历和直接边查找核对每条标签。查询始终邻层，因此二者必须一致。
- 全部数据先缓存 NPZ；训练时 GPU 张量采样、GPU 边序重排，不在线生成图。
- 复用原模型和原 fused AdamW 参数分组；BF16、SDPA、torch.compile。主 LR3e-4，两项 LR1e-3 对照；1000 更新 warmup；按占用时间前70%余弦下降、后30%保持峰值的1/10。
- 优化后的 prompt-only loss 与原项目 full-sequence shifted/masked CE 及参数梯度已经 CPU 对齐验证；没有训练 smoke。

## 课程规则

典型课程：端点受限 M2 → 完整 query/固定边序 M2 → 随机边序 M2 →64名字→256名字→M4→M8→M16→M24。

门控：当前阶段至少20k更新，连续3次评估满足完整训练准确率≥99%、CE≤0.02、验证≥95%、图翻转两边均正确≥90%；随机边序阶段还要求反转边序训练/验证满足相同准确率门槛。未达标不自动扩难，继续训练至预算结束。

固定时间课程不等待掌握，用于检验门控和按时扩难的差异。阶段切换不重置模型/优化器，LR沿共同的时间日程。混合课程在各阶段准确实现受限/完整 query 配额，切到下一阶段前按完整 query 评估。

注意：当前实现的普通阶段课程**没有额外回放旧阶段池**；只有明确标记 mix 的课程在该阶段混合受限与完整查询。门控课程优先隔离变化因素。若后续增加常规回放，应另开版本，不能改正在运行的定义。

## 评估与保存

100/500/1k/5k及之后每10k更新评估当前完整 train/val，并测试反转边序。最终另测test、训练图上未见query（受限/ordinary池适用）、统一full M2 v16/64/256及M24 v256测试。

成对指标包括 graph/source/target 的两边均正确率及标签有向 P(YES) 差；报告 P(YES) 均值/标准差、完整词表格式失败。M2只有两种边序，canonical+reverse覆盖全部边序；M24反转只是一种顺序诊断，不声称穷举24!种置换。

少量固定验证图跟踪各层末prompt位置的公共/输入变化RMS及source/target/graph变化，不保存activation数组、不训练探针。主比较使用时间预算耗尽后的 latest；best只按当前阶段验证选择，不按test选模型。不同阶段best不可视为同分布全局排名。

每分支保存 latest、验证 best、optimizer/RNG/课程/曝光状态、周期评估、最终结果和W&B状态。不上传checkpoint artifact。只有正式实验产物，没有smoke记录或checkpoint。

## 查看与恢复

```bash
cd /mnt/ningxuefei/yangshuo/cot-navigation
bash research/direct_learning_20261020/status.sh
tail -f research/direct_learning_20261020/runs/direct_learning_20261020_v1/launcher.log
```

若队列退出，确认没有活跃的旧 launcher 后，使用相同入口恢复；启动器加锁，拒绝重复队列；已完成跳过，未完成从latest恢复并扣除已使用预算。

```bash
bash research/direct_learning_20261020/launch.sh
```

原始数据契约：`research/direct_learning_20261020/data_v1/manifest.json`。
完整逐项计划：`research/direct_learning_20261020/runs/direct_learning_20261020_v1/launch_plan.json`。
各分支目录含 `experiment.json`、`resolved_config.yaml`、`initialization.json`、`metrics.jsonl`、`eval/`、`checkpoints/`、`wandb_status.json`、`progress.json` 和最终完成凭据。
