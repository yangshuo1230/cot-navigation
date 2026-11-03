# Direct 从一跳继续学习二跳、三跳

服务器：`/mnt/ningxuefei/yangshuo/cot-navigation`。
研究目录：`research/direct_multihop_20261021`。
W&B group：`direct_multihop_20261021_v1`，项目沿用 `cot-graph-reasoning`。

## 实验目标与预算

保持原 N16/E24、4层×4宽、256节点名字、76-token边列表输入和6L/1H/d128/FFN512模型。只改变query跨度及采样比例，仍然直接输出YES/NO，无CoT、无辅助目标、无WALK、无RL。

15个分支，每个最多2小时，合计约30 GPU小时。后台只使用空闲物理GPU1–7，GPU0保留；最多并行7个，约6小时排完，不含占卡等待。周期评估计入分支时间；最终全量评估、保存、W&B关闭有少量额外开销。

## 一跳父模型

|简称|实际父checkpoint|原始一跳1024条测试（此前冻结评估）|
|---|---|---:|
|target_best|`direct_learning_20261020_v1/curr_target_gate_s1/checkpoints/best.pt`，610k|99.51%|
|mix_source_last|`direct_learning_20261020_v1/curr_mix_source_gate_s0/checkpoints/latest.pt`|98.05%|
|timed_source_last|`direct_learning_20261020_v1/curr_source_timed_s0/checkpoints/latest.pt`|94.73%|

路径前缀均为 `research/direct_learning_20261020/runs/`。父文件只读，SHA256写入每分支experiment.json，加载时复核。加载全部模型权重，不冻结层。因为best没有可用的对应optimizer，**所有父模型与scratch都统一从新AdamW开始**；这是模型权重续训，不声称延续原Adam动量。

## 15项矩阵

三个父模型各4项，共12项：

|后缀|采样与课程|峰值LR|
|---|---|---:|
|joint|hop1/2/3各1/3，直接混合训练|1e-4|
|timed|前1小时hop1=25%、hop2=75%；后1小时hop1=25%、hop2=25%、hop3=50%|1e-4|
|gated|相同两阶段；达到二跳门槛才开放三跳，未达标则持续二跳阶段|1e-4|
|joint_lr3e4|hop1/2/3各1/3，高学习率适应对照|3e-4|

三个从零对照：`scratch_joint`、`scratch_timed`、`scratch_joint_lr3e4`。它们与相应warm-start使用相同数据、采样和时间预算，模型seed0。父模型有不同预训练历史，不能把三个父模型当成同方案三次独立重复。

每batch64 query，每个hop内YES/NO各半，按32个正负对的累积配额实现hop比例。每次曝光随机重排边列表。没有“停止一跳监督”的分支。

门控要求：当前阶段至少20k更新、连续3次评估，二跳固定训练probe准确率≥99%且CE≤0.02，二跳验证balanced accuracy≥95%、YES/NO recall均≥90%，一跳验证balanced accuracy≥95%。门槛不使用test。它是入门检查，不保证后续一定稳定或学到一般算法。

LR在最初1000更新warmup，前84分钟余弦降至峰值1/10，后36分钟恒定。阶段切换不重置LR和optimizer。

## 数据与独立检查

使用已有主数据 `data/explore_v1/context_standard_n16_m24_train50000_val10000_test10000_s0`，不重新划分，不从test借训练样本。

|split|hop1|hop2|hop3|
|---|---:|---:|---:|
|train|16668|16666|16666|
|val|3334|3334|3332|
|test|3334|3334|3332|

各桶YES/NO各半。原70,000条全部由独立BFS、拓扑bitset closure以及实际输入token重解析检查，标签一致；图身份train/val/test重叠为零。图只含相邻层边，故正例hop为路径长，负例hop仅是层差。

数据预先缓存 `data_v1/arrays.npz`，训练张量一次上传GPU，不在线构图或生成轨迹。另在test图上构造每hop256对端点反事实：同图、同层差，只换source或target翻转标签；只有真实存在反标签端点时纳入，不由模型结果选样。

## 评估与checkpoint

- 训练前保存step0的一/二/三跳基线，确认父模型恢复及原子能力。
- 100/500/1k/5k和每10k更新，评估各hop固定2048条训练probe、512条验证query，均按标签平衡。这不是全训练集收敛证明。
- 最终latest对全部50k train、10k val、10k test评估，另测test反序及端点反事实两边同时正确率。
- `best.pt`按固定验证集合三个hop的balanced accuracy宏平均选择；训练完成后独立评估best，避免把曾学会但后期退化误认为从未学会。final全量val不会反过来更改best选择。
- 每次保存latest optimizer/RNG/课程/曝光状态，保留best；不上传模型artifact。不存在训练smoke run或smoke checkpoint。
- 记录训练总CE、一跳保持、二/三跳能力、YES/NO recall和阶段曝光，不将多跳全答NO的50%解释成成功。

## 查看与恢复

```bash
cd /mnt/ningxuefei/yangshuo/cot-navigation
bash research/direct_multihop_20261021/status.sh
tail -f research/direct_multihop_20261021/runs/direct_multihop_20261021_v1/launcher.log
```

队列已在后台运行。仅在原launcher已经退出、且没有存活旧子任务时用同一入口恢复；重复launcher会被锁拒绝，完成项跳过，未完成项从latest恢复并扣除已占用预算。

```bash
bash research/direct_multihop_20261021/launch.sh
```

配置：每分支 `experiment.json`、`resolved_config.yaml`。
评估：`eval/initial.json`、`step_*.json`、`final.json`、`final_best.json`。
队列凭据：`launch_plan.json`、`data_adapter_verification.json`、`preflight.json`、`status.json`、`ledger.json`。

本阶段首先检验同图规模、同输入长度下的二/三跳组合学习，不同时扩大图规模。是否进一步训练更深图应根据本批结果另定。
