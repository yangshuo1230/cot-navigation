# DFS纠错训练：层数与错误数据比例

12个从头训练的任务：6/9层 × 错误样本30%/60%/90% × 模型种子0/1。
同种子、同层数的三个比例共享相同初始化。模型、曝光与错误采样种子同步。

复用 correction_reproduction/worker.py、trace_weight.py 和公共 scheduler，
不修改公共训练器。模板为上一批 correction/s0 的 resolved_config.yaml。
准备时逐字段检查，只允许深度、错误比例、种子及输出/日志标识变化。

其余沿用200k步、N16→N36→N64→N100四阶段混合课程、batch64、BF16、
SDPA、torch.compile、缓存轨迹和RAM错误池。每20k步验证、每10k步保存。
错误候选token、MISTAKE与修复均计算loss，整条CoT的相对权重为
`clip(actual_cot_tokens / 128, 1, 16)`；保持原样本级归一化。

错误内容来自原冻结自然候选列表错误池，不重新采集、不扩充错误类型。
比例按样本采样概率定义，不是错误token比例。按当前课程规模和标签抽样；
50%概率优先抽复杂子池，另50%抽全池。高掺入率会增加有限错误池的重复曝光。
训练包含可达和不可达查询。评估设置也与上一批一致。

准备命令只进行CPU配置、缓存哈希和初始化检查，不做smoke训练：

```bash
.venv/bin/python studies/correction_depth_rate/run.py --wait
.venv/bin/python studies/correction_depth_rate/run.py --launch --background --wait
```

默认run名为 `dfs_depth_rate_20261031`，按用户要求禁用W&B，
只记录本地metrics、评估结果和checkpoint。8张卡每卡一个任务，剩余4个排队；
大模型优先。输出、队列状态与启动日志位于：

```text
outputs/correction_depth_rate/launches/dfs_depth_rate_20261031/
outputs/correction_depth_rate/launches/dfs_depth_rate_20261031.prepared.json
outputs/correction_depth_rate/launches/dfs_depth_rate_20261031.launch.log
```

恢复使用原参数加 `--launch --background --wait --resume`。
不要在已有任务目录上直接重新开始。配置和源码变化会触发准备一致性检查。
