# 历史 CoT 错误的延迟激活读出

服务器目录：`/mnt/ningxuefei/yangshuo/cot-navigation/research/history_error_20261022`。

首轮使用旧 scale_v1 的 N100 新图/固定图 DFS-state、BFS-state 四个 validation-best checkpoint，均冻结。没有混入正在训练的 hard-DAG 模型。

每模型 train/val/test 为768/192/256个不同查询，每查询一对轨迹。将未被选择的最后一个 CAND 节点替换为同层非邻居，保留 PUSH 首项及相同合法后续。读取错误处、固定 token 偏移和后续记录 VIS 结束位置的六层 residual。

当前步骤之前的非法候选注释是唯一新增错误；后续实际状态相同。本轮没有让模型自由生成后续，也没有测试自然错误或自动纠错。

主要对照包括：错误前的相同激活；当前 token 身份；历史编辑节点+当时当前节点身份；新图同一 CoT 的图关系对照；BF16/FP32 同配对重跑。每位置独立拟合 Linear/64 hidden MLP，各3个初始化；数据与指标见 REPORT.md。

运行前先检查 GPU7 空闲。直接命令：

```bash
.venv/bin/python research/history_error_20261022/run.py --gpu 7
.venv/bin/python research/history_error_20261022/run.py --gpu 7 --precision fp32 --output research/history_error_20261022/results_fp32
OMP_NUM_THREADS=2 .venv/bin/python research/history_error_20261022/verify.py
```

`launch.py` 检查GPU7空闲后启动BF16正式评估，`launch.py --fp32` 启动FP32对照；已完成模型会跳过。不要同时启动两轮。

本地重画：

```powershell
python G:\CoT\history_error_20261022\report.py
```

不保存 activation 缓存或探针权重、不创建导航 checkpoint 或 W&B run。`run_bf16_source.py` 是正式 BF16 轮次执行的精确源码，便于核对其 source_sha256；当前 run.py 增加了精度和输出目录参数。

初次提取在训练探针前被因果性检查拦截，未产生结果；其原因是配对跨批的BF16数值差异。正式版本固定同批配对，错误前激活差严格为0。AUROC验证允许1e-12的浮点求和误差，例如数学上的0.5可能存储为0.4999999999999999；不修改原始指标。
