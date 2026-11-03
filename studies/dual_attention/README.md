# 双路 attention 实验启动记录

服务器项目：`/mnt/ningxuefei/yangshuo/cot-navigation`。

已后台启动，launcher PID `1474627`，训练 PID `1474648`–`1474655`。
启动检查时8组均有正常loss记录、W&B online，进度300–4500步。
每组1,217,664个参数。GPU0–3为DFS，GPU4–7为BFS，顺序均为
parallel/causal、parallel/bidirectional、serial/causal、serial/bidirectional。
状态快照见 `launch_verification.json`，它是检查时刻的记录而非实时进度。

本轮为 hard_dag_v1 的 DFS/BFS × 并行/串行 × 图前缀因果/双向，共 8 组。
使用已有审计数据与四种离线轨迹，不重新生成。每次训练曝光继续打乱边列表。
课程统一为 N16 40k、N36 30k、N64 30k、N100 40k，总计 140k。
其他尺度重放权重、逻辑查询 batch64、优化器、答案/轨迹损失与历史 hard DAG 设置一致。
历史普通模型为 200k、单头，本轮为 140k、两条半宽分支；历史比较并非严格单变量实验。

每层保留 width128、FFN512，两路 attention 各64维，总注意力投影参数量与普通128维 attention一致。
图分支只读取 EDGE 列表与原始 QUERY；图表示不接收任何 CoT 信息。
图 token 自身只经过图分支及共享 FFN；CoT token 经过图分支和历史分支。
历史分支只读取当前已消费 token 及之前 CoT，不读取图前缀，不读取未来 CoT。
并行：两路基于同一归一化残差，投影输出相加后进入 FFN。
串行：历史更新残差，再归一化并查询图分支，最后进入 FFN。
两种顺序使用相同参数集合、参数量和初始化规则；绝对 RoPE 位置不重置。
图前缀双向只改变图 token 之间的可见性。
生成使用每层固定图 KV 与因果历史 KV，兼容行移除、不同前缀长度、STOP 回退。

训练采用 SDPA、BF16、torch.compile、fused AdamW 与已有吞吐 overlay。
图注意力将前缀查询与长 CoT 尾部分开，不构造 B×L×L 图 mask。
图缓存界限868覆盖最终 N144/M288 外推集，超出会明确报错。
teacher forcing 每5k、自由生成每10k（64条）；主测试与扩展集保留 best/last、每集256条。
自由生成保持完整32768上下文和原STOP规则；没有增加512等短生成限制。
BF16 编译及批量变化可改变接近并列的生成分支，历史 eager 生成不保证逐 token 一致。

启动：

```bash
cd /mnt/ningxuefei/yangshuo/cot-navigation
.venv/bin/python scripts/train_dual_attention.py --run-name dual_attention_v1_s0_20261023 --background
```

同一入口加 `--resume --background` 可按各自 latest checkpoint 续跑；已有 run 不会被覆盖。
W&B group 为 run-name，项目 `cot-graph-reasoning`，online 认证失败即报错。
正式日志位于 `outputs/dual_attention_v1/launches/dual_attention_v1_s0_20261023`。
smoke 使用 `runtime.smoke=true`，禁止 run 记录、W&B 和 checkpoint。

所有自由验证及最终测试保存每条 prompt/生成轨迹，支持事后语义合法率审计。
注意：沿用历史 trace_diagnostics=false。自由生成准确率与中断指标用于训练跟踪；
若要比较任意合法随机搜索顺序，应使用语义搜索检查，不能把与单条参考轨迹完全相同当作合法率。
