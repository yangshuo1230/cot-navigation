# Fixed DAG PATH：按跳数的行为与逐层有用后继概率

## 范围和定义

- 原 N100 简单/困难固定图 PATH 验证最佳 checkpoint，分别 step20000/10000，图边数150/200。每类只有一个图/模型，训练设置不同，不能把差异只归因于图复杂度。未分析最新限制暴露模型。
- 使用原测试集自由贪心生成；跳数用独立BFS重新核对，为真实最短有向距离。跳数图仅含可达正例，因此答案准确率等于该桶YES recall。负例不可定义有限最短跳数，单独列出。
- 路径成功要求从正确起点出发、整条轨迹每一步都是真实边、最后一个节点恰好等于目标；不要求随后答案正确。此前“合法前缀曾经到达目标”是更宽松的指标，本图不使用该定义。
- 图中均匀随机游走从真实后继等概率选择，目标吸收，DAG上精确动态规划得出成功概率；不加额外训练。图上贪心成功比例不是温度1采样到达概率。
- 有用后继为真实直接后继中仍能到达目标者，包含目标自身。概率图只取正例、自由生成的合法前缀、到达目标之前，且同时含有用与无用后继的mixed分叉；当前模型已走错的轨迹不纠正。每条查询先平均，再跨查询等权平均，避免长路径占权重。
- 所有13个位置使用原final_norm和LM head计算softmax，不训练新probe。A为attention残差相加后，F为FFN残差相加后；前12个是未校准logit lens，只有L6F是模型实际下一token分布。
- 完整词表概率质量：P(Succ(u))及P(Succ(u)∩ReachTo(t))；限定后继概率为两者之比。采用logsumexp避免中间层概率极小时的下溢。限定后继概率高而合法后继总质量极低时，不代表模型实际会生成有用节点。

## 按查询跳数

|图|最短跳数|n|答案准确率|整条路径合法且结束于目标|随机游走到达率|
|---|---:|---:|---:|---:|---:|
|simple|2|60|96.7%|96.7%|33.0%|
|simple|3|66|95.5%|93.9%|19.8%|
|simple|4|64|95.3%|95.3%|17.8%|
|simple|5|59|94.9%|94.9%|12.6%|
|simple|6|48|100.0%|100.0%|10.7%|
|simple|7|38|100.0%|100.0%|9.5%|
|simple|8|32|100.0%|100.0%|4.0%|
|simple|9|19|100.0%|100.0%|1.9%|

simple：可达正例n=386，答案准确率97.2%，严格路径到达率96.9%；不可达负例n=705，答案准确率99.4%。

|hard|1|9|88.9%|88.9%|44.2%|
|hard|2|27|92.6%|92.6%|26.0%|
|hard|3|42|100.0%|100.0%|13.4%|
|hard|4|71|98.6%|98.6%|9.1%|
|hard|5|68|100.0%|100.0%|8.8%|
|hard|6|41|100.0%|100.0%|6.0%|
|hard|7|29|100.0%|100.0%|2.1%|
|hard|8|6|100.0%|100.0%|2.4%|
|hard|9|3|100.0%|100.0%|0.5%|
|hard|10|3|100.0%|100.0%|0.5%|
|hard|11|1|100.0%|100.0%|0.1%|

hard：可达正例n=300，答案准确率98.7%，严格路径到达率98.7%；不可达负例n=255，答案准确率100.0%。


高跳数桶有些只有1–3条，100%不能解释为稳定泛化；不同桶不是同难度随机重复，且原训练可能披露过完整witness。不能从曲线推出跳数越多推理越容易。

## 逐层概率：全部mixed前缀、查询等权

|图|位置|合法后继完整词表质量|有用后继完整词表质量|后继内有用概率|
|---|---|---:|---:|---:|
|simple|E|0.0%|0.0%|50.0%|
|simple|L1_attention|0.0%|0.0%|49.2%|
|simple|L1_FFN|97.5%|46.3%|47.6%|
|simple|L2_attention|91.2%|59.7%|65.2%|
|simple|L2_FFN|95.0%|69.0%|72.4%|
|simple|L3_attention|92.1%|73.0%|79.0%|
|simple|L3_FFN|94.2%|80.7%|85.7%|
|simple|L4_attention|91.0%|78.3%|85.7%|
|simple|L4_FFN|94.4%|86.4%|91.5%|
|simple|L5_attention|93.9%|86.6%|92.1%|
|simple|L5_FFN|98.0%|92.7%|94.6%|
|simple|L6_attention|98.1%|93.2%|95.0%|
|simple|L6_FFN|99.8%|95.5%|95.7%|
|hard|E|0.0%|0.0%|61.3%|
|hard|L1_attention|31.0%|20.7%|65.6%|
|hard|L1_FFN|96.3%|50.2%|52.1%|
|hard|L2_attention|91.4%|44.8%|48.9%|
|hard|L2_FFN|94.5%|45.8%|48.5%|
|hard|L3_attention|94.8%|59.9%|63.2%|
|hard|L3_FFN|96.4%|60.2%|62.5%|
|hard|L4_attention|96.4%|76.3%|79.0%|
|hard|L4_FFN|98.0%|78.9%|80.4%|
|hard|L5_attention|98.2%|79.5%|80.9%|
|hard|L5_FFN|98.9%|80.8%|81.6%|
|hard|L6_attention|98.5%|91.0%|92.3%|
|hard|L6_FFN|99.4%|97.0%|97.5%|

首步mixed查询的对照曲线单独画出，避免全部结果只来自模型已经走对的中间前缀。E位置对后继的完整词表质量通常极低，因此只看条件概率容易误判。

这与之前报告的“后继内最高分节点是否有用”的top-1选择率不同；例如最后层接近99%的top-1选择率，不等于给有用节点分配99%的概率。也与按每次决策直接平均不同，JSON同时保留决策等权作为辅助口径。

## 验证与复现

每个checkpoint校验SHA256；可达闭包逐起点与BFS核对，跳数再独立核对；最后层head结果与原生logits逐决策对齐，最后层各概率与上一轮PATH行为分析对齐误差小于1e-5。仅冻结CPU FP32前向，无新训练、探针、W&B或checkpoint/activation存盘。

- simple：{"final_stage_head_matches_native_max_error": 0, "final_stage_vs_previous_max_errors": {"mixed_all_useful_given_successor": 3.507231717270187e-07, "mixed_all_successor_full_vocab": 1.6176210679841319e-06, "mixed_all_useful_full_vocab": 1.3958938276381616e-06, "mixed_first_step_useful_given_successor": 2.3879271937587987e-07, "mixed_first_step_successor_full_vocab": 8.143088667811327e-07, "mixed_first_step_useful_full_vocab": 1.197995770918503e-06}}
- hard：{"final_stage_head_matches_native_max_error": 0, "final_stage_vs_previous_max_errors": {"mixed_all_useful_given_successor": 2.4496092598802477e-07, "mixed_all_successor_full_vocab": 8.093698317113152e-07, "mixed_all_useful_full_vocab": 9.757900385487517e-07, "mixed_first_step_useful_given_successor": 3.201789178230996e-08, "mixed_first_step_successor_full_vocab": 5.980917051617851e-07, "mixed_first_step_useful_full_vocab": 1.7034072496979746e-07}}

服务器复现：repo根目录执行 `.venv/bin/python research/path_behavior_figures_20261024/run.py` 和 `plot.py`。
