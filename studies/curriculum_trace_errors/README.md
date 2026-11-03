# Curriculum CoT trace errors

服务器目录：`~/cot-navigation/research/curriculum_trace_errors_20261021`。

在项目根目录执行。仅冻结推理，默认复用已完成条件；不训练、不写 checkpoint、不创建 W&B run。

```bash
.venv/bin/python research/curriculum_trace_errors_20261021/audit.py --test-only
CUDA_VISIBLE_DEVICES=0 .venv/bin/python research/curriculum_trace_errors_20261021/audit.py --per-cell 8 --batch 64
CUDA_VISIBLE_DEVICES=0 .venv/bin/python research/curriculum_trace_errors_20261021/check_batch.py
.venv/bin/python research/curriculum_trace_errors_20261021/summarize.py
```

`--per-cell` 控制每个 TEST 层差×标签格的样本上限；`--worker i --workers k` 按模型分工，同一模型不可并发运行。更改抽样规模、精度、batch 或 checkpoint 时应换输出目录，避免误用已完成结果。该版本的模型选择清单来自此前验证集最佳 checkpoint 记录，图/查询内容由原数据集加载；默认固定图只测实际训练的六张图。

`--include-unseen` 可额外测注册但没参加课程训练的固定图，必须在报告中按 `unseen_world` 单列。注册 WORLD token 不代表相应图被训练过。

错误检查函数 `sequence_audit`、`full_state_audit` 不依赖训练器，可接任意图、起终点和生成序列。状态首错严格接续真实历史；首错后的局部检查仅判断打印记录自身是否一致。Node-only 输出首次扩展节点，不能当普通相邻路径判定；其顺序首错后的隐式状态不再强行重建。

主结果保留统一 BF16 batch64；`batch_sensitivity.json` 有历史 batch16 的配对复现。分批导致的差异不能归因于 checkpoint 改变。

最终交付 `REPORT.md`、`distributions.csv`、`summary.json`、`per_query_errors.jsonl.gz`、`verification.json`、两个 PNG 和固定查询清单。JSONL 只保留错误事件和统计，不保存大批激活/完整 rollout；少量原始输出窗口位于 summary/report。
