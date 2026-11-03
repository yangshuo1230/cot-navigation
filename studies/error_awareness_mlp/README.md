# 最后输入 token 的错误探针

服务器项目：`/mnt/ningxuefei/yangshuo/cot-navigation`。

输入停在 `x[t-1]`，读取 L4–L6 最后输入位置的 residual，预测这个已输入 token 是否违反导航协议；此时 `x[t]` 尚未生成，也没有输入占位 token。标准 decoder 没有一个可以在不提供输入的情况下单独读取的第 t 个位置 activation。

例如合法前缀的 candidates 应该是节点 B：输入分别截止于 `… A | A | B` 和 `… A | A | C`（C 是图内错误节点），探针分别学习标签 0 和 1。后面的分隔符或动作完全不提供给模型。合法候选可有多个，标签允许所有合法选择。

新增模块 `src/cot_navigation/analysis/error_awareness.py`；报告模块 `report_error_awareness.py`。源文件副本及测试位于本地 `source/`。不修改已有导航模型、旧实验和训练管线。

复现完整实验，在服务器项目根目录运行：

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m cot_navigation.analysis.error_awareness \
  --output research/error_awareness_mlp_20261019/results \
  --train-pairs 8192 --val-pairs 2048 --test-pairs 4096 \
  --train-queries 4096 --val-queries 1024 --test-queries 2048 \
  --natural-queries 256 --hidden 64 --steps 1000 --seeds 0 1 2 --resume

.venv/bin/python -m cot_navigation.analysis.report_error_awareness \
  research/error_awareness_mlp_20261019/results

.venv/bin/python -m pytest tests/test_error_awareness.py \
  tests/test_mechanistic_followup.py tests/test_mechanistic_analysis.py -q
```

更改设置或代码后使用新的 output 目录；当前设置有指纹检查，不允许悄悄混用旧结果。`--models context_dfs` 可限制模型，`--forward-batch`、`--probe-batch` 可调整批大小。所有冻结模型读出与探针训练限定物理 GPU0。Direct/Filler 不存在相应自由轨迹错误任务；两种图任务的 Path/DFS/BFS 加三个固定图 BFS-state 对照共 15 个 checkpoint。

训练完全使用人工错误，真实生成错误只用于外部分布测试，保证没有拿自然测试数据训练或选择探针。训练正负样本成对共享同一个合法历史，不监督错误以后的连锁 token。数据量是“错误/合法前缀对”，不是独立图的数量；固定图查询数量仍受既有划分限制。只检查局部协议违规，合法但最终未能解决问题的分支不自动视作错误。

只保存设置、紧凑指标、训练曲线、少量异常片段和两张结果图；不保存 activation、rollout 大缓存、probe checkpoint 或 W&B run。模型保持冻结，eval 模式。

`results/REPORT.md` 是完整实验结果，`summary.csv` 可用于后续对照，`models/*.json` 包含类型/字段拆分。测试中的标准差只来自三次探针初始化，不是测试样本置信区间。
