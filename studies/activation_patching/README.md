# Activation patching

实验设计见 DESIGN.md，本轮结果见 REPORT.md。

服务器：`~/cot-navigation/research/activation_patching_20261021/`。可移除的 PyTorch block hook 不修改训练源码；只需冻结推理，不训练。

在服务器项目根目录运行。更改参数时用新的 `--out`，已有结果不会覆盖。

```bash
PYTHONPATH=src .venv/bin/python research/activation_patching_20261021/test_patch.py
PYTHONPATH=src .venv/bin/python research/activation_patching_20261021/run.py --gpu 1 --readout next_record --out research/activation_patching_next_run
PYTHONPATH=src .venv/bin/python research/activation_patching_20261021/run.py --gpu 1 --readout current_record --out research/activation_patching_current_run
```

参数支持 models、recipient-queries、donor-queries、cases（每字段查询数）、max-prefix、max-new、batch、layers（从1开始）、groups。audit-only 仅使用CPU审计配对覆盖，需指定单独 out 目录。

current_record 自动跳过尚未输入的操作符/分号等组。next_record 的统一记录结束切点支持全部组。操作符 token 不同则跳过对应组及包含它的组合；PUSH 参数始终不替换。

本地汇总重画：`python G:\CoT\activation_patching_20261021\report.py`。只保存两张图、统计、配对信息和少量异常；不保存激活、KV 或 checkpoint。

selection.json 固定四个完整状态变体的 validation-best checkpoint 路径和 SHA256。可扩展清单添加其他规模，但当前数据适配器针对 scale 离线格式，不能直接假定兼容任意旧格式。
