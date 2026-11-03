# Direct 学习验证：六组，物理0卡顺序运行

当前 `launch.sh` 默认运行下面六组，W&B group 和输出分组为 `direct_minimal_v3`。
保留原两组实现与数据；`suite.py` 复用 `train.py`，没有修改主项目或旧实验输出。

| 任务 | 与对照的差别 | 默认步数 | 学习率 |
|---|---|---:|---:|
| ordinary24 | 普通随机新图、一图一query；同样4096不同查询 | 50k | 3e-4 |
| paired24 | 成对约束、24条边、同图对八样本集中 | 50k | 3e-4 |
| paired24_interleaved | 与paired24完全相同的样本，随机交错到batch | 50k | 3e-4 |
| curriculum | 与paired24相同图对，2→4→8→16→24条边 | 50k | 3e-4 |
| paired24_low_lr | 与paired24相同数据/抽样，学习率降低三倍 | 50k | 1e-4 |
| paired24_long | 与paired24相同数据，第一段调度相同，额外继续训练 | 100k | 3e-4 |

六组总计350k个优化步，均从相同seed0初始化，无结构改动。不是6个随机种子。
`paired24_long`从零运行100k：前50k沿用paired24的50k cosine调度，之后在3e-5下继续50k，
不重置优化器、不重新warmup。不是将cosine直接拉长到100k；保存prefix_summary.json核对共同前缀。
长组会重新计算前50k，未通过复制训练run的方式省略这段计算，避免改变恢复或W&B语义。

普通组取原train的4096个独立图/一跳查询，正负各半。其余组使用512个图对、4096个查询，
共1024张M24图。普通组和成对组的不同查询数量匹配，独立图数不匹配，应如实区分。
集中与交错组的底层样本完全一致、总曝光相同；区别包括batch内相关性和标签波动。
交错组与普通组按IndexSampler遍历所有训练行，不会每batch强制正负各半；整轮严格平衡。

推荐比较顺序：ordinary24↔paired24（数据构造）；paired24↔interleaved（batch组织）；
paired24↔curriculum（递进难度）；paired24↔low_lr（优化）；paired24↔long（额外训练）。
评估输入、普通新图测试、成对干预定义保持一致；普通组的训练拟合probe使用它自己的普通训练数据，
不会把未训练的成对图误标为训练probe。

启动命令仍然是：

```bash
cd /mnt/ningxuefei/yangshuo/cot-navigation
nohup bash research/direct_minimal_v2_20261018/launch.sh \
  > research/direct_minimal_v2_20261018/launch.log 2>&1 &
```

只检查六组，不训练：

```bash
bash research/direct_minimal_v2_20261018/launch.sh --check-only
```

只选择部分实验（仍固定0卡）：

```bash
JOBS="paired24 curriculum" bash research/direct_minimal_v2_20261018/launch.sh
```

修改基础步数：`--steps 25000 --run-name direct_minimal_v3_25k`；五组25k，long组50k。
学习率由各实验预设，不接受原两组入口的`--lr`参数。旧`train.py --mode`仍保留原两组入口。

当前六组输出：`research/direct_minimal_v2_20261018/runs/direct_minimal_v3/<job>/`。
同参数重启自动跳过完成任务，恢复未完成任务。长组保留10个周期checkpoint，其余保留5个。

下文说明沿用的成对数据、课程和评估设计，以及原两组入口的参数。

---

不改主项目、模型、词表或输出形式，不训练 CoT，不引入辅助 loss。
每条样本仍为边列表 + 一个 query + YES/NO，只在答案位置计算原始 CE。
从零训练，使用原 6L/1H/d128/FFN512 模型，约 1.22M 参数。

| 任务 | 训练数据 | 默认预算 |
|---|---|---|
| paired24 | 成对约束，始终24条边 | 50k步 |
| curriculum | 同一组约束，2→4→8→16→24条边，每阶段10k步 | 50k步 |

每组 batch64、单种子0、AdamW、BF16、SDPA、torch.compile，学习率3e-4，
warmup500步后cosine到3e-5；其余优化设置沿用原配置。两组初始化和图对抽样流相同。
原两组共100k个优化步；六组默认总预算见上表。没有自动启动训练。

## 数据

512个基础图对，每个图对由G/G'及8条独立查询样本组成。
在同一相邻层选择A/C和B/D：G有A→B/C→D，无A→D/C→B；G'交换这四角边。
每张图都问四个角查询，形成每图2正2负；相同query在G/G'上标签相反。

因此单独依赖图、source、target或节点对名字，都不能正确完成一个完整图对。
两图节点名相同，逐节点入度/出度相同；关键差异仅是两条边的终点交换。
输入顺序每个样本每次曝光重排，不把同图问答拼进上下文。
batch64=8图对×8样本，只在同一次更新聚合这些样本的梯度，样本之间不互相注意。

256个独立held-out图对，val/test各128。另沿用原数据的普通随机新图：
val512/test1024，均为一跳、正负各半，单独检查跨分布迁移。
图对训练集不包含这些普通held-out图。数据文件和hash随正式运行保存。

课程只增加干扰边，不改四角标签；保持4层、相邻层边、16个节点名字的基础映射。
少边时有孤立节点，不会额外输入节点列表；模型可见的节点可能少于16。
M=2阶段四个查询端点均已出现在边列表中，负例不靠缺失节点识别。
阶段2起20%旧难度回放，按完整图对分配，累计配额准确实现比例。
固定阶段预算不会根据测试分数调整；不能假定进入下一阶段前已经学会。

注意：paired24的实际训练图是512对/1024张，不是原4096图池；它和旧实验还同时改变了
样本组织、学习率和训练预算。应先比较本次两组，不能将对旧run的改善只归因于某一个改动。
本次不再增加第三个普通数据baseline，以保持验证任务集最小。

## 评估

每5k步在独立val上检查全部五种边数及普通N16/M24新图；同时看训练子集能否拟合。
最终再测试一次held-out test。评估为FP32，不用完整标签序列作输入。

除准确率、答案CE，还报告：

- 关键边翻转后，两边答案同时正确的比例。
- 固定图改target、改source后，两边同时正确的比例。
- 标签翻转方向上的P(YES)变化和输出概率标准差，检测常量输出。

所有结果保存本地并上传W&B。测试结果不选checkpoint；使用最终权重。
若只在特殊成对图上成功、普通随机新图仍失败，不能称为一般一跳算法已学会。

## 手动运行：固定物理0卡

```bash
cd /mnt/ningxuefei/yangshuo/cot-navigation
bash research/direct_minimal_v2_20261018/launch.sh
```

后台入口：

```bash
cd /mnt/ningxuefei/yangshuo/cot-navigation
nohup bash research/direct_minimal_v2_20261018/launch.sh \
  > research/direct_minimal_v2_20261018/launch.log 2>&1 &
```

脚本强制CUDA_VISIBLE_DEVICES=0。发现0卡已有计算进程/超过1000MiB占用，或旧launcher占锁，
直接退出，不切换GPU、不停止他人进程。两任务顺序执行，每任务前都会再检查0卡。

只做CPU数据与采样检查，不训练、不创建模型/W&B/checkpoint或smoke记录：

```bash
bash research/direct_minimal_v2_20261018/launch.sh --check-only
```

默认W&B在线，六组group=`direct_minimal_v3`，沿用服务器已有凭证，上传失败报错。
训练与诊断指标上传，checkpoint不上传为artifact。

六组输出：`research/direct_minimal_v2_20261018/runs/direct_minimal_v3/<job>/`。
同参数再次启动：跳过completed任务，从未完成任务的latest.pt恢复模型、优化器、
调度器、采样器、重排RNG、回放配额和曝光计数。每10k步保留checkpoint，最多5个。
completed.json只在最终测试也完成后写入。

调参数需新run-name，避免改变已有课程边界或学习率调度：

```bash
bash research/direct_minimal_v2_20261018/launch.sh \
  --steps 25000 --run-name direct_minimal_v3_25k
```

这会创建六组新实验，基础预算25k，课程每阶段5k，long组50k；不等同于从50k方案中途截取。
