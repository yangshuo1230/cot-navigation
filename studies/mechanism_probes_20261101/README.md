# 2026-11-01 机制假设复查

完整结论与限制见 [REPORT.md](REPORT.md)，机器可读汇总见 `summary.json`，检查结果为 `validation.json`，probe曲线为 `probe_curves.png`。

模型与数据不变：原185k冻结模型，复用18条N100 val与18条test参考查询。probe按query划分，12条val训练/6条val开发选择正则强度，再18条val refit，18条test评估；不是新增自由生成成功率。

先在服务器运行 `collect.py`，同步activation/metadata后本地运行 `probe.py`。把probe权重同步服务器，运行 `length_intervention.py`、`candidate_tests.py`、`push_carrier.py`、`layer_bar_tests.py`。同步结果后运行 `readout_probe.py`、`summarize.py`。服务器运行命令统一为项目根目录 `.venv/bin/python studies/mechanism_probes_20261101/<脚本>`，本地用 `D:\anaconda\python.exe`。CPU线程数4，未训练原模型，未占用GPU。

共4334个stack probe样本、4491个bar probe样本、193个候选生成条件、144个历史PUSH载体条件、48个长度方向条件、72个bar干预条件。条件之间并非独立query，详见报告分母。
