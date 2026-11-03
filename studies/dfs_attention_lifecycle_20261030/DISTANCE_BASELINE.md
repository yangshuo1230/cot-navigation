> 此文件仅记录已撤回的距离校正实验。当前HTML已恢复原始attention分数。

# 距离校正展示

2026-10-31 更新：`behavior_atlas.html` 的 attention 表格和全局热力图显示 `A[l,q,k] − μ[l,q−k]`。

每条完整参考序列、每层/head分别对所有相同 q−k 距离的因果可见格子求算术均值，包含图、查询、CoT和最终答案位置。未来遮罩格不参与；距离d有L−d个样本。本模型只有一个head。

先逐格减去基线，再按原语义路线/类别求和，之后按query内事件平均、跨query平均；置信区间仍为600次query bootstrap。统计对象为原有18条N100 test查询，事件位置与原分析一致，没有重新选择展示层。

正值高于距离均值、负值低于；pp=百分点，不是概率，不重新归一化。对角线均值为0，但每行之和不要求为0。各距离的样本数量不同，可在distance_baselines.json查看。

- `distance_baseline.py`：在服务器CPU读取同一checkpoint，重算18条序列的距离校正统计。
- `statistics_distance_adjusted.json`：校正后的分组、路线、bootstrap区间及子条件。
- `distance_baselines.json`：每条序列的逐层/逐距离均值和样本数量。
- `build_report.py`：生成校正后的HTML/图，同时保留原始研究报告的历史统计。

干预成功次数、自由生成正确率等不是attention分数，保持原值。原始statistics.json与原始模型数据均保留。
