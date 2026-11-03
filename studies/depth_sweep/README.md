# 4096上下文层数对比：重新启动

框架：`/mnt/ningxuefei/yangshuo/cot-navigation-unified`。
新组名：`depth_sweep_4096_20261101`，后台调度PID3406250。

保持32个模型：simple/hard × dfs_state/dfs_nodes × 3/6/9/12层 × seed0/1。两种协议统一4096上下文；其余沿用九级N12→16→24→32→36→48→64→80→100课程，每模型1M步、batch64、70/30规模回放、干净四变体轨迹、严格因果单路attention、d128/FFN512/1head、dropout0、原AdamW与全程cosine设置。

旧32768批次的输出、生成配置和启动日志已删除。本轮新增的7份32768轨迹缓存转换为4096后，连同其metadata/complete标记一并删除；其他实验复用的原始图数据和既有缓存保留。清理清单见新输出目录的superseded_cache_cleanup_done.json。

## 4096语义

DFS Node的参考序列完整保留。完整DFS状态轨迹若超过4088 token（4096减8 margin），直接复用原生truncate_example：在完整`;`记录边界截断，添加STOP，再保留真实YES/NO与EOS。不会删掉超长查询或打断状态记录。派生缓存按root/protocol共享，记录truncated_rows；每种协议/每个规模只准备一次，转换无需重新运行DFS搜索。

因此这组是4096预算下的能力实验，不能声称所有DFS训练轨迹都完整执行到终止。最终评估继续独立报告答案和完整CoT合法率，以及原生forced_stop/cot_interrupted统计。截断前缀通过状态转移语义检查，但不作为完整合法遍历计分。

所有参考缓存完成checksum检查。对train/val/test各8题的四个变体检查语义；完整轨迹校验完整状态或节点顺序，截断轨迹检查记录边界、STOP位置及合法状态前缀。verification_4096.json另检查了最长龙12557→4079 token时答案保留、记录边界和前缀合法性；32份配置均确认4096及offline。

之前的32768启动因W&B 403退出。新批次W&B使用offline，本地metrics/checkpoint和离线日志正常保存，避免远端日志服务阻断训练。

输出：`outputs/depth_sweep_4096_20261101/`；缓存：该输出目录下`derived/`；来源图数据不变。

```bash
cd /mnt/ningxuefei/yangshuo/cot-navigation-unified
.venv/bin/python studies/depth_sweep/experiment.py --execute --wait --background --resume
```

当前启动状态与初始训练进度见launch_receipt.json。共享模型、loss、训练器和调度器未改动，新增的预算派生和语义统计都在study适配层实现。
