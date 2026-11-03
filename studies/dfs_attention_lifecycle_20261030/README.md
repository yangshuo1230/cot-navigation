# 长CoT attention研究

从 [行为图谱](behavior_atlas.html) 或 [完整报告](REPORT.md) 开始。

同一185k In Context DFS-state模型，52条完整参考轨迹/7255个行为事件，12条原生自由生成及42个局部反事实接收位置案例。没有重新训练、没有占用GPU、没有生成MP4。

研究区分稳定attention位置模式、局部value干预证据和未解决的完整计算。主要发现包括prompt-bar的source初始化接口、历史重复节点作为上下文寄存器、fresh/revisit候选的不同路线，以及L1终止动作到最终答案的功能性通路。字段结束判据和深栈action比较尚未完整定位。

分析程序在服务器项目根目录通过`.venv/bin/python studies/dfs_attention_lifecycle_20261030/<程序名>.py`执行：

1. `analyze.py` 固定分层panel、统计参考轨迹及最初4条原生轨迹。
2. `extend_free.py` 对panel中剩余全部middle/long test查询完成原生生成，最终共12条；不按表现筛选。
3. `register_swap.py`、`visited_register_swap.py`、`terminal_test.py` 运行全部局部反事实及sham/低权重对照。
4. `writer_audit.py`、`carrier_audit.py` 核查低层写入候选和高层历史root读取。
5. 结果同步至本目录后，在Windows运行`summarize.py`、`build_report.py`生成统计、报告和图谱。后者使用Microsoft YaHei字体。
6. `verify_results.py`验证统计、位置、query分离、对照及结果一致性；`verify_atlas.mjs`用本地headless Edge验证7个行为页、全部图和结果表。验证凭据为`validation.json`、`browser_validation.json`。

`refresh.py`仅用于在不重做自由生成的情况下，对原panel补算新增的语义路线。当前`analyze.py`已包含这些路线，完整重跑无需再额外运行refresh。

原始模型、图与既有研究均保留。新记录仅写入本研究目录。主统计teacher-forced，不将参考轨迹的稳定性冒充整条自由生成成功率；全部12条原生结果保留，答案正确9/12，完整严格合法2/12。

字段终止专项复查见 [FIELD_TERMINATION.md](FIELD_TERMINATION.md)。新增18条已有test查询的1,331个精确路线事件，以及其中6条long查询上的432个局部输出交换条件。程序与全部阴性结果保留；`build_termination_review.py`汇总并检查，`build_report.py`更新图谱。52条查询和12条自由生成的原分母不变。


2026-10-31：已按要求恢复原始attention分数展示；距离校正实验文件仅作历史记录。
