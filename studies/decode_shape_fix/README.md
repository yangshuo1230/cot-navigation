# decode 动态形状修复

项目：`/mnt/ningxuefei/yangshuo/cot-navigation`。
本轮继续使用 group `dual_attention_v1_s0_20261023_r1`，不重置预算/种子/数据/优化器。

此前8组均在 `_decode` 累积8个版本后永久切到eager，日志涉及：
历史KV长度对8的余数、是否越过图KV上限868、batch是否为1。
已有900步测试只覆盖一次prefill与少数batch变化，没有覆盖跨尺度重复prefill累积的版本。

修复仅改变 decode 的读取视图：

- 历史KV读取上界向上对齐到8，受max_seq_len限制。
- 图KV读取长度固定为min(graph_max_tokens,max_seq_len)。
- 原绝对位置mask继续排除未来/未写入历史，图mask继续只允许图/QUERY前缀。
- 不改变KV写入位置、生成预算、STOP回退、训练前向或参数格式。
- 保留独立full/prefill/decode编译入口与明确的异常回退，不放大默认编译缓存上限。

回归测试跨100/148/220/604/868长度重复使用同一wrapper，batch64/32/16/8/1，
逐步压缩到7/2/1，含STOP回退与上下文末端。要求全过程无编译回退。
既有缓存/信息隔离CPU测试与900步CUDA测试同样通过。

真实checkpoint基准使用同一批8条N100验证查询，完整32768上下文，max_new_tokens=0。
分别记录旧eager、新compiled冷启动、新compiled预热耗时，以及生成token数和答案差异。
训练仍并发运行，因此结果不是独占GPU测量，且BF16输出长度变化会影响耗时。
同一强制token工作负载下FP32 logits完全一致；BF16 greedy可能因mask形状与kernel改变而分支不同。
不能将少量查询上的答案变化解释成能力提升或退化。

当前进程已在内存中持有旧代码，需从生产checkpoint续跑才能应用修复；
沿用已有run ID。具体切换位置和验证结果记录在本目录JSON中。

已部署：新启动器PID1521431，8组全部online续跑，run ID不变。
恢复点为120k/130k/140k；部分未保存更新需要重算，最大约9.8k步。
续跑会先补齐checkpoint落盘后被中断的验证，避免跳过该轮best选择。
CPU全套与CUDA跨尺度、长解码检查通过；正式N100的256条best测试完成，
generation_compile_fallbacks=0，其后N16/N24/N32各256条测试也为0。

| 完整预算8条验证查询 | 旧eager | 新compiled冷启动 | 新compiled预热 | 旧/新生成token |
|---|---:|---:|---:|---:|
| DFS并行因果 | 75.38s | 78.33s | 47.53s | 19683 / 17424 |
| BFS串行因果 | 224.23s | 83.93s | 73.95s | 34379 / 23328 |

两种协议各8条的最终预测全部与旧版相同；逐轨迹相同分别4/8、2/8。
强制token检查FP32最大logit差均为0。BF16 greedy轨迹不是逐token等价。
DFS两种模式测量时均有同卡训练；BFS测速跨过任务切换，同卡训练负载变化，
加上生成长度不同，上表不能解释成严格等工作量的kernel加速倍数。
