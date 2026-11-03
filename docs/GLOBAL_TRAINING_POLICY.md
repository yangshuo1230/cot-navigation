# 全局训练与评估策略更新

已应用到服务器 `/mnt/ningxuefei/yangshuo/cot-navigation`。

| 项目 | 新建正式实验的默认值 |
| --- | --- |
| Teacher-forced 验证 | 每 20,000 个 optimizer updates |
| 自回归自由生成验证 | 每 20,000 个 optimizer updates |
| Checkpoint | 每 10,000 步，并在训练结束保存 |
| W&B 训练日志 | 每 100 步 |
| 最终测试 | 仅 last checkpoint、主测试集 256 个查询 |
| 额外图尺度、OOD、专用查询视图 | 默认关闭，按需开启 |
| 逐字段轨迹诊断 | 默认关闭，按需开启 |

共享训练器保留 65,536 token 微批预算、30% padding 约束、长度分组、编译 loss、减少 H2D 传输与日志同步等优化。评估使用独立自由生成批量、缓存/prefill 预算、有效前缀 KV 移行和独立编译入口。普通 Transformer 的历史缓存读取对齐也已与双路 attention 同步。CPU、smoke 和 record-SWA 按支持情况保留参考路径。

默认不改变训练数据、逻辑 batch、训练总步数、学习率、loss 权重或搜索长度预算。评估频率降低会减少 best checkpoint 的候选点。显式同时测试 best/last 时，仅在两者所有权重完全相同时复用评估报告，保留两个标签及逐样本预测。

已同步共享配置、正式预设，以及 first-round、retrain、explore、scale、hard-DAG、task-only、short-hop、block-query、自纠错、自然纠错、课程/SWA、dual-attention 的相关训练入口。从历史配置创建新实验时会应用当前策略；历史 generated/resolved 配置和运行结果未改写。续跑仍检查模型、数据和目标函数的一致性，允许运行效率及评估策略调整。

通用 CLI 可以在实验配置之后叠加 `--config configs/train/throughput.yaml`。支持 `--set` 的入口可以显式覆盖策略，例如：

```bash
--set train.eval_interval=20000 \
--set train.free_eval_interval=20000 \
--set evaluation.final_checkpoints='[last]' \
--set evaluation.test_samples=256
```

需要额外分析时，可显式设置 `evaluation.final_checkpoints=[best,last]`、`evaluation.extra_sets=[wide=数据目录]`、`evaluation.trace_diagnostics=true`。short-hop/block-query 的专用视图使用 `evaluation.extra_query_views=true`；task-only 的训练后语义回放使用 `--semantic-audit`。

验证：35 个正式实验预设通过全局审计；CPU 回归 243 passed、13 skipped；CUDA 定向检查 23 passed，覆盖训练分批/加权、loss、缓存移行、STOP 回退、普通模型长历史编译解码及双路 attention。检查未创建新的正式训练或 smoke 记录、checkpoint、W&B run。

修改不会热更新已经加载代码和配置的进程。本轮双路模型已完成 140k 训练，尚在进行的旧版最终评估没有被中断或重启。

本目录保存修改补丁、文件哈希、配置审计及测试日志；服务器 `research/global_training_policy_20261023/before/` 保存修改前文件。
