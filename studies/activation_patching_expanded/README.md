# 扩大合法配对的 activation patch

服务器目录：`~/cot-navigation/research/activation_patching_expanded_20261021/`。

设计见 DESIGN.md，完整结果与两张图见 REPORT.md。共2944对合法配对，目标字段多数为每格128对；固定图BFS的“同队列、不同候选”在当前图族中不可实现，保留缺格。

在服务器项目根目录执行。所有命令均为单行，已有同配置数据/结果会校验并复用。

```bash
PYTHONPATH=src .venv/bin/python -m unittest discover -s research/activation_patching_expanded_20261021 -p 'test_*.py'
PYTHONPATH=src .venv/bin/python research/activation_patching_expanded_20261021/worker.py --mode prepare --models context_l10_dfs context_l10_bfs fixed_l10_dfs fixed_l10_bfs
PYTHONPATH=src .venv/bin/python research/activation_patching_expanded_20261021/worker.py --mode run --gpu 0 --models context_l10_dfs context_l10_bfs fixed_l10_dfs fixed_l10_bfs
PYTHONPATH=src .venv/bin/python research/activation_patching_expanded_20261021/verify.py
```

最后一个worker命令按模型串行运行；多个worker可分配互不重叠的模型和GPU。同一模型/读取不能并发写。以上是复用示例，不代表GPU0当前空闲；启动时选实际可用GPU。

`worker.py` 支持 cases（每类/字段对数）、queries、variants、seed、max-prefix、max-new、batch、chunk-cases。更改数据设置时，请在新的研究目录中复制代码及selection.json运行，避免混用旧数据。生成器/引擎哈希和参数都记录，旧结果不会静默覆盖。`engine.py` 供worker导入，用户入口为worker。

数据文件是 `datasets/*.json.gz`，只含基础图索引、查询、邻居种子、节点映射、记录序号、前缀哈希。无需保留完整轨迹或激活，就能确定性重建。固定图的base_index/base_split只是读取原WORLD的锚点，新查询由source/target覆盖，不应将base_split误读为新查询所属split。

结果为 `results/*.json.gz`，每模型/读取一个；summary.json和statistics.csv.gz包含全部条件。directional_transfer.csv只统计两边控制完整复现参考、且有不同正确后续的定向迁移。key_examples.json是少量定向挑出的实际续写，不能用它估计总体比例。

本地重画汇总：`python G:\CoT\activation_patching_expanded_20261021\report.py`。16项单元测试仅在内存中运行，不生成smoke记录或checkpoint；全流程不训练导航模型、不创建W&B run、不保存activation或KV。
