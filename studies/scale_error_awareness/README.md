# scale checkpoint 上的 token 错误探针

部署目录：服务器 `/mnt/ningxuefei/yangshuo/cot-navigation`。新增分析模块，不修改旧模型、旧分析模块或旧结果。

覆盖新图/固定图 N36、N64、N100、课程至 N100，共 24 个设置。在每个任务/规模内，Path 使用其 best checkpoint，DFS/BFS 从 state/nodes 两变体中按存储的验证 best_score 选胜者；同分优先 state。所有候选及选择均记录在 selection.json。没有按测试表现选择模型。

scale 批次只有固定图 seed0；三个 probe 初始化 seed 不等于三个图 seed。图规模、主模型训练步数、世界混合和随机轨迹设置均与上一轮小图不同，不将跨轮差异单独归因于图大小。

探针输入严格截止最后已消费 token x[t-1]，读取该位置 L4–L6 residual，预测 x[t-1] 是否违反局部协议；x[t] 尚未生成。没有 dummy 或额外查询 token。

数据读取实际 ScalePreparedRows，按 index%4 选离线随机轨迹变体；固定图加训练时的 WORLD 标识。保持 STOP 策略，STOP/最终答案/EOS 不构成探针标签。每个 checkpoint 的 probe 只在主规模/主世界取原始查询；全体人工改写按原查询分组隔离。

目标 train/val/test 为8192/2048/4096个错误-合法前缀对；每对共享同一个合法历史，只改变最后一个 token。训练只使用人工数据，自然首次错误用于外部测试。新图val/test只有512个原始查询，增加配对样本不增加独立查询数。固定图同样报告原始查询数。

训练 Linear 与 64 hidden GELU MLP，三个初始化；另训练匹配架构的 token embedding/位置/字段基线。L4–L6分别读出，验证BCE选择最优状态。节点版oracle检查真正的合法搜索序列，不把BFS距离相同视为任意可交换；已穷举验证4节点DAG的4950个前缀。全状态沿用旧探针较宽松的集合顺序/PUSH语义口径。

服务器项目根目录启动正式分析：

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u -m cot_navigation.analysis.scale_error_awareness \
  --output research/scale_error_awareness_20261019/results \
  --train-pairs 8192 --val-pairs 2048 --test-pairs 4096 \
  --train-queries 4096 --val-queries 1024 --test-queries 2048 \
  --natural-queries 256 --hidden 64 --steps 1000 --seeds 0 1 2 --resume

.venv/bin/python -m cot_navigation.analysis.report_scale_error_awareness \
  research/scale_error_awareness_20261019/results

.venv/bin/python -m cot_navigation.analysis.verify_scale_error_awareness
```

支持 `--models context_l10_dfs` 选择设置；`--forward-tokens` 限制特征提取每批 token 数，长轨迹自动减小 batch。默认只用物理GPU0。更改实验设置/源码后使用新 output，避免不同实验结果混用。probe不保存权重，activation不落盘，不创建W&B run。

错误替换使用索引抽样，仅构造实际抽中的tuple，不枚举全部替换对象。与原展开列表shuffle的随机数序列及抽样结果一致；14项真实数据对照见results/sampler_equivalence.json。

本次实际运行在7个设置完成后，使用研究目录中的run_parallel.py将剩余设置分成三个不重叠进程，均限定GPU0；启动时顺序写入共同设置，之后各自只写自己的模型结果。parallel_plan.json保留分配，最终核验与两张图由finalize.py自动生成。上面的顺序命令可复现相同实验设置。

回归测试：

```bash
.venv/bin/python -m pytest tests/test_scale_error_awareness.py \
  tests/test_error_awareness.py tests/test_mechanistic_followup.py \
  tests/test_mechanistic_analysis.py -q
```

结果见 results/REPORT.md；只生成两张关键图，紧凑JSON保存训练曲线、分层/分类型指标、少量低错误分数的真实错误前缀。自然评估是“真实首次错误 vs 单token合法修复”的配对，不是连续自然token流中的发生率或校准评估。很少自然错误的设置只作诊断；低泛化分数不证明activation完全不含错误信息。
