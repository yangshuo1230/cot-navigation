# Direct：成功一跳数据设计下，从零混合多跳

服务器目录：`/mnt/ningxuefei/yangshuo/cot-navigation/research/direct_mixed_curriculum_20261021`。

W&B project：`cot-graph-reasoning`；group：`direct_mixed_curriculum_20261021_v1`。

三个分支均从随机初始化开始，无一跳权重、CoT、WALK、辅助损失或模型结构变化。复用单跳研究的模型、BF16、torch.compile、融合AdamW、batch64、学习率3e-4和1000步warmup。每组4小时，合计12 GPU小时；GPU1/6/7，GPU0不使用。

|分支|seed|课程|GPU|
|---|---:|---|---:|
|target_gated|1|固定终点→自由查询固定边序→随机边序→扩展名字范围和边数，验证门控晋级|1|
|source_timed|0|固定起点→自由查询固定边序→随机边序→扩展名字范围和边数，八阶段按时间均分|6|
|mix_source_gated|0|固定起点比例80%→50%→20%→全自由查询→扩展名字范围和边数，验证门控晋级|7|

每阶段均从开始混合一/二/三跳，期望比例各1/3；采样每对保持同跳YES/NO平衡。每次曝光独立重排边列表，但为复用原课程，指定的固定边序过渡阶段例外。无阶段之间的数据回放；mix_source是同阶段固定起点与自由查询混合。

名字范围16→64→256；边数6→8→16→24。每图16节点，4层×4节点，相邻层有向边。不能机械使用原来2条边的起始阶段：这里用两条长度3的链作为6边核心，并构造图边翻转与端点变化的正负查询。干扰边必须保持三个跳数全部配对标签。固定终点版本为不同起始层到第4层，其余版本为第1层到不同目标层。

这是原成功数据设计的多跳改编，图分布不等同普通随机DAG；其配对测试成绩需与此前普通图测试区别解释。full训练池每阶段512个组，每组2图×4查询×3跳=24条，共12,288条；验证和测试各128组，各3,072条。锚定池每组12条，样本数减半。每图不跨split；不同课程池与边数规模用全局图身份注册避免跨split泄漏。全部标签通过遍历与独立邻接矩阵传播检查。

课程门控复用原训练/验证/反序/配对条件，并额外要求每跳训练准确率≥99%、验证≥95%。不达标不强制晋级，定时分支作为能够走完整个课程的对照。最终测试包含当前阶段、各名字范围以及最终图难度。保存latest及best，best仅按固定full-v256-m24验证集选择，不用test选checkpoint。保留分跳准确率与YES/NO召回。

启动入口（同一队列未运行时使用，避免重复启动）：

```bash
cd /mnt/ningxuefei/yangshuo/cot-navigation
bash research/direct_mixed_curriculum_20261021/launch.sh
```

日志：`runs/direct_mixed_curriculum_20261021_v1/launcher.log`及各分支`train.log`。配置为各分支`experiment.json`，运行状态在`progress.json`，结果在`eval/`，结束凭据在`completed.json`。

仅进行了语法检查、离线数据审计和正式训练启动后的状态检查，没有额外smoke训练。
