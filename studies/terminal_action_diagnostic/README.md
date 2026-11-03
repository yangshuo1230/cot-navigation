# 从正确搜索终点预测FOUND/EXHAUST和YES/NO

服务器目录：`/mnt/ningxuefei/yangshuo/cot-navigation/research/terminal_action_diagnostic_20261022`。

冻结上一轮九个完整状态checkpoint：固定图DFS/BFS、in-context DFS/BFS、full λ1/4/16与SWA1/2。使用完全相同的TEST查询，排除上一轮参考轨迹因上下文限制被截断的例子。不训练，不保存checkpoint，不创建W&B run，GPU0保留。

输入保留图、查询、全部正确非终止搜索记录以及最后一条记录的完整frontier、visited和候选。截在动作前的竖线，例如 `... frontier | visited | candidates |`。无FOUND/EXHAUST、终止分号、答案或EOS。让模型自由生成 `FOUND/EXHAUST ; YES/NO`，不将正确动作反馈给续写。

三个主要指标：首token是否为正确终止动作；前三token是否完整匹配“动作 ; 答案”；首动作正确时答案是否正确。另保留最多32token内的答案、正确终止动作已给定的配对控制、与上一轮自由搜索结果的对照。动作错而没有在32token内答对，不能解释为任意长续写均失败。

终点状态本身是正确的强条件输入，这测局部终止判断，不是独立搜索能力。Path、节点版和Direct没有对应动作协议，不强行要求其输出未训练的终止动作。

入口为`launch.sh`，worker日志在本目录，结果在`results/`。推理完成后运行`summarize.py`生成报告、CSV、JSON和错误例子。
