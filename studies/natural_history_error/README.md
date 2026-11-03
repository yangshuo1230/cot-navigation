# 自然 CoT 的历史错误读出

服务器项目根目录：`/mnt/ningxuefei/yangshuo/cot-navigation`。
分析目录：`research/natural_history_error_20261022`。

使用上一轮相同的四个冻结 N100 state 模型。先完成自然生成和严格单 token 修复对照，再做未经编辑的自然前缀分类。正式轮次沿用完整上下文预算；1024 token 试跑只作诊断，不进入正式图表。

确认指定 GPU 空闲后，从服务器项目根目录执行，例如：

```bash
.venv/bin/python research/natural_history_error_20261022/run.py \
  --gpu 7 --models context_l10_dfs_state --max-new 0
.venv/bin/python research/natural_history_error_20261022/stream.py \
  --gpu 7 --models context_l10_dfs_state
```

其余三个模型 ID 为 `context_l10_bfs_state`、`fixed_l10_dfs_state`、`fixed_l10_bfs_state`。同一模型不要同时启动两个实例。已完成的模型会跳过，生成缓存会核对查询 ID、checkpoint hash 与预算；更改预算或查询数量请使用全新的分析目录，避免复用旧缓存。

四个模型均完成后，执行 CPU 验证：

```bash
OMP_NUM_THREADS=2 .venv/bin/python research/natural_history_error_20261022/verify.py
OMP_NUM_THREADS=2 .venv/bin/python research/natural_history_error_20261022/verify_stream.py
```

`verify.py` 使用独立状态机校验严格配对的全部自然首错和每个合法修复读出位置。`verify_stream.py` 校验纯自然分类每条前缀的标签、位置和查询隔离。两个验证均 PASS 后才能生成正式报告。

将小型结果 JSON 和验证结果同步到本地后，在 Windows 执行：

```powershell
python G:\CoT\natural_history_error_20261022\report.py
```

模型保持冻结；没有训练导航模型，没有 activation 缓存、探针权重、导航 checkpoint 或 W&B run。服务器保留压缩自然 rollout，便于 CPU 重建验证。图中 n 是每类测试前缀数；不同 anchor 独立训练探针，误差棒是初始化标准差，不是抽样置信区间。
