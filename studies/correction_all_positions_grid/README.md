# DFS：全位置错误池的层数／混合比例实验

12个从头训练任务：6/9层 × 0%/30%/60%错误轨迹 × 初始化种子0/1。
仅DFS state。所有任务使用相同数据曝光种子0与错误采样种子0，只改变初始化。
同层、同初始化的三种比例初始参数哈希一致。0%组为纯干净轨迹，保留相同MISTAKE词表。

使用data/natural_correction/all_positions_v2_20261102中已完成的六个DFS池，
拒绝旧4位置池。准备时核验v2身份、全位置采样标记、文件hash、train隔离、全部训练
事件的原样错误token与正确修复、4K长度；独立验证/测试缓存也在启动前读取检查。
错误池按当前图规模和原查询label采样；natural_complex_share保持0.5。

采用逐阶段token预算匹配，60%组200k步为基准；0%组262370步、30%组226929步。
阶段步数分别为0%：[62641,65127,67722,66880]；30%：[55611,56570,57527,57221]；
60%：[50000,50000,50000,50000]。各阶段预计非padding完整序列token相近，总计约101.61亿。
计入图/查询/CoT/答案/EOS，不计padding和验证。实际采样会有少量波动；不是等FLOPs设计。
四阶段数据权重不变：N16；70%N36+前三档各10%；70%N64+前四档各7.5%；
70%N100+前五档各6%。学习率按每组自己的总步数衰减，阶段边界采用各自累计步数。

其余继承原复现实验resolved_config：d128、1头、FFN512、RoPE4K、有效batch64、
BF16、SDPA、torch.compile、缓存数据、长度分桶与token预算microbatch。
所有CoT错误与修复token参与loss；每样本lambda=clip(CoT长度/128,1,16)，
总任务loss为2*(answerCE+lambda*meanCoTCE)/(1+lambda)。每20k步验证、每10k步保存。
不使用RL。禁用W&B，保留本地metrics、checkpoint和最终评估。主终点各组最终checkpoint。

复用原correction_reproduction/worker.py和公共scheduler，不另建训练器。
公共调度器只使用空闲GPU，繁忙卡不抢占，余下任务自动排队。

```bash
.venv/bin/python studies/correction_all_positions_grid/run.py --wait
.venv/bin/python studies/correction_all_positions_grid/run.py --launch --background --wait
```

恢复用相同参数加--resume。新计划源码、配置或池hash变更会拒绝直接launch。
输出outputs/correction_all_positions_v2/launches/dfs_all_positions_tokenmatched_20261102。
之前的等步数批次已由用户停止并加保护，不恢复、不纳入正式比较。
prepared.json记录CPU审计和初始化哈希，status.json记录当前GPU分配和待运行任务。

## 2026-11-03：统一400k步的新批次

使用独立run名dfs_all_positions_equal400k_20261103，从头训练同一12条件矩阵。
传入--uniform-steps 400000覆盖token匹配预算，四阶段边界为100k/200k/300k/400k；
图规模混合权重、损失、学习率上限/下限、batch、曝光/错误种子和W&B禁用等保持不变。
学习率按照400k总步数衰减；不从上一批checkpoint继续，也不覆盖历史结果。

```bash
.venv/bin/python studies/correction_all_positions_grid/run.py --run-name dfs_all_positions_equal400k_20261103 --uniform-steps 400000 --wait
.venv/bin/python studies/correction_all_positions_grid/run.py --run-name dfs_all_positions_equal400k_20261103 --uniform-steps 400000 --launch --background --wait
```
