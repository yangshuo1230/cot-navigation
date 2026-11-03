# 2026-11-02 因果复查

详见 [REPORT.md](REPORT.md)。同一冻结模型，复用已有N100 test查询；未训练模型，CPU FP32，4线程。

服务器项目根目录运行 `.venv/bin/python studies/causal_followup_20261102/<脚本>`：

- `fresh_candidates.py`：13个首次展开跨分支过滤状态，合法图/历史配对、历史角色定位，221条件。
- `verify_pairs.py`：独立验证配对前缀及当前正确候选/PUSH记录合法。
- `length_layers.py`：L2–L6长度方向、三档剂量及等范数对照，420条件。
- `length_donor.py`：同token、同已输出数量=1的自然表示交换及子空间对照，300条件。
- `bar_count.py`：数量/距离/语法分开的L1单行虚拟计数与真实输入控制，324条件。

`count_probe.py`和`summarize.py`在本地用`D:\anaconda\python.exe`运行，读取上一轮query-split probe数据和本轮JSON；输出计数probe、汇总、验证与PNG。不覆盖上一轮结果。
