# 隐去终止动作的正确CoT诊断

输入为正确完整轨迹直到最后状态的第三个竖线，去掉最后FOUND/EXHAUST及其分号，并且不输入YES/NO/EOS。模型全词表greedy续写，不强制动作、分号或答案。每例最多续写32 token；主要联合指标为前三个token恰好等于“正确动作 ; 正确答案”。完整参考过长的查询沿用上一轮排除，报告保留样本数。

九个完整状态checkpoint：固定图DFS/BFS、新图DFS/BFS验证best，及新图DFS loss/SWA五组最终checkpoint；同一批TEST查询。

|模型|条件|标签|n|首动作正确|动作+分号+答案全对|32token内答案对|动作正确后答案对|直接给正确终止动作后答案对|
|---|---|---|---:|---:|---:|---:|---:|---:|
|context_curriculum_bfs_state|deep|all|113|92.04%|92.04%|92.04%|100.00%|100.00%|
|context_curriculum_bfs_state|deep|1|55|83.64%|83.64%|83.64%|100.00%|100.00%|
|context_curriculum_bfs_state|deep|0|58|100.00%|100.00%|100.00%|100.00%|100.00%|
|context_curriculum_bfs_state|larger|all|60|90.00%|90.00%|90.00%|100.00%|100.00%|
|context_curriculum_bfs_state|larger|1|31|80.65%|80.65%|80.65%|100.00%|100.00%|
|context_curriculum_bfs_state|larger|0|29|100.00%|100.00%|100.00%|100.00%|100.00%|
|context_curriculum_bfs_state|n100|all|66|96.97%|96.97%|96.97%|100.00%|100.00%|
|context_curriculum_bfs_state|n100|1|33|93.94%|93.94%|93.94%|100.00%|100.00%|
|context_curriculum_bfs_state|n100|0|33|100.00%|100.00%|100.00%|100.00%|100.00%|
|context_curriculum_dfs_state|deep|all|126|100.00%|100.00%|100.00%|100.00%|100.00%|
|context_curriculum_dfs_state|deep|1|69|100.00%|100.00%|100.00%|100.00%|100.00%|
|context_curriculum_dfs_state|deep|0|57|100.00%|100.00%|100.00%|100.00%|100.00%|
|context_curriculum_dfs_state|larger|all|69|76.81%|76.81%|94.20%|100.00%|100.00%|
|context_curriculum_dfs_state|larger|1|39|89.74%|89.74%|89.74%|100.00%|100.00%|
|context_curriculum_dfs_state|larger|0|30|60.00%|60.00%|100.00%|100.00%|100.00%|
|context_curriculum_dfs_state|n100|all|69|100.00%|100.00%|100.00%|100.00%|100.00%|
|context_curriculum_dfs_state|n100|1|35|100.00%|100.00%|100.00%|100.00%|100.00%|
|context_curriculum_dfs_state|n100|0|34|100.00%|100.00%|100.00%|100.00%|100.00%|
|context_loss_full_l1|deep|all|126|99.21%|99.21%|99.21%|100.00%|100.00%|
|context_loss_full_l1|deep|1|69|98.55%|98.55%|98.55%|100.00%|100.00%|
|context_loss_full_l1|deep|0|57|100.00%|100.00%|100.00%|100.00%|100.00%|
|context_loss_full_l1|larger|all|69|97.10%|97.10%|97.10%|100.00%|100.00%|
|context_loss_full_l1|larger|1|39|94.87%|94.87%|94.87%|100.00%|100.00%|
|context_loss_full_l1|larger|0|30|100.00%|100.00%|100.00%|100.00%|100.00%|
|context_loss_full_l1|n100|all|69|100.00%|100.00%|100.00%|100.00%|100.00%|
|context_loss_full_l1|n100|1|35|100.00%|100.00%|100.00%|100.00%|100.00%|
|context_loss_full_l1|n100|0|34|100.00%|100.00%|100.00%|100.00%|100.00%|
|context_loss_full_l16|deep|all|126|72.22%|72.22%|73.81%|100.00%|100.00%|
|context_loss_full_l16|deep|1|69|49.28%|49.28%|52.17%|100.00%|100.00%|
|context_loss_full_l16|deep|0|57|100.00%|100.00%|100.00%|100.00%|100.00%|
|context_loss_full_l16|larger|all|69|91.30%|91.30%|91.30%|100.00%|100.00%|
|context_loss_full_l16|larger|1|39|84.62%|84.62%|84.62%|100.00%|100.00%|
|context_loss_full_l16|larger|0|30|100.00%|100.00%|100.00%|100.00%|100.00%|
|context_loss_full_l16|n100|all|69|100.00%|100.00%|100.00%|100.00%|100.00%|
|context_loss_full_l16|n100|1|35|100.00%|100.00%|100.00%|100.00%|100.00%|
|context_loss_full_l16|n100|0|34|100.00%|100.00%|100.00%|100.00%|100.00%|
|context_loss_full_l4|deep|all|126|98.41%|98.41%|99.21%|100.00%|100.00%|
|context_loss_full_l4|deep|1|69|97.10%|97.10%|98.55%|100.00%|100.00%|
|context_loss_full_l4|deep|0|57|100.00%|100.00%|100.00%|100.00%|100.00%|
|context_loss_full_l4|larger|all|69|98.55%|98.55%|98.55%|100.00%|100.00%|
|context_loss_full_l4|larger|1|39|97.44%|97.44%|97.44%|100.00%|100.00%|
|context_loss_full_l4|larger|0|30|100.00%|100.00%|100.00%|100.00%|100.00%|
|context_loss_full_l4|n100|all|69|98.55%|98.55%|98.55%|100.00%|100.00%|
|context_loss_full_l4|n100|1|35|97.14%|97.14%|97.14%|100.00%|100.00%|
|context_loss_full_l4|n100|0|34|100.00%|100.00%|100.00%|100.00%|100.00%|
|context_loss_swa1_l1|deep|all|126|95.24%|95.24%|98.41%|100.00%|100.00%|
|context_loss_swa1_l1|deep|1|69|91.30%|91.30%|97.10%|100.00%|100.00%|
|context_loss_swa1_l1|deep|0|57|100.00%|100.00%|100.00%|100.00%|100.00%|
|context_loss_swa1_l1|larger|all|69|100.00%|100.00%|100.00%|100.00%|100.00%|
|context_loss_swa1_l1|larger|1|39|100.00%|100.00%|100.00%|100.00%|100.00%|
|context_loss_swa1_l1|larger|0|30|100.00%|100.00%|100.00%|100.00%|100.00%|
|context_loss_swa1_l1|n100|all|69|100.00%|100.00%|100.00%|100.00%|100.00%|
|context_loss_swa1_l1|n100|1|35|100.00%|100.00%|100.00%|100.00%|100.00%|
|context_loss_swa1_l1|n100|0|34|100.00%|100.00%|100.00%|100.00%|100.00%|
|context_loss_swa2_l1|deep|all|126|76.98%|76.98%|76.98%|100.00%|100.00%|
|context_loss_swa2_l1|deep|1|69|57.97%|57.97%|57.97%|100.00%|100.00%|
|context_loss_swa2_l1|deep|0|57|100.00%|100.00%|100.00%|100.00%|100.00%|
|context_loss_swa2_l1|larger|all|69|97.10%|97.10%|98.55%|100.00%|100.00%|
|context_loss_swa2_l1|larger|1|39|97.44%|97.44%|97.44%|100.00%|100.00%|
|context_loss_swa2_l1|larger|0|30|96.67%|96.67%|100.00%|100.00%|100.00%|
|context_loss_swa2_l1|n100|all|69|100.00%|100.00%|100.00%|100.00%|100.00%|
|context_loss_swa2_l1|n100|1|35|100.00%|100.00%|100.00%|100.00%|100.00%|
|context_loss_swa2_l1|n100|0|34|100.00%|100.00%|100.00%|100.00%|100.00%|
|fixed_curriculum_bfs_state|n100|all|60|98.33%|98.33%|98.33%|100.00%|100.00%|
|fixed_curriculum_bfs_state|n100|1|30|96.67%|96.67%|96.67%|100.00%|100.00%|
|fixed_curriculum_bfs_state|n100|0|30|100.00%|100.00%|100.00%|100.00%|100.00%|
|fixed_curriculum_dfs_state|n100|all|62|98.39%|98.39%|98.39%|100.00%|100.00%|
|fixed_curriculum_dfs_state|n100|1|32|96.88%|96.88%|96.88%|100.00%|100.00%|
|fixed_curriculum_dfs_state|n100|0|30|100.00%|100.00%|100.00%|100.00%|100.00%|

动作错后的32token内答案仅是有上限的续写结果，不能说明更长续写一定不会答对；三token联合指标没有这个歧义。前缀包含完整正确最终状态，是搜索终点处的局部判断诊断，不检验模型是否会独立到达这个状态。PATH、节点版和Direct没有训练过这些动作token，不纳入此项。

[统计JSON](summary.json)；[CSV](summary.csv)；[错误例子](examples.json)。
