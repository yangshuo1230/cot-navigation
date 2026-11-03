# DFS baseline与纠错模型的图规模泛化

使用上一批已完成的6层matched_clean/correction模型，各4个seed，固定latest200k。
不使用尚在训练的depth/rate实验。N=32/64/100/144/169/196，每个规模两图族：
相邻层简单DAG，M=floor(1.5N)；原有multiscale_dag_v1复杂DAG，M=2N。
形状依次4×8、8×8、10×10、12×12、13×13、14×14。

每设置512个独立新图查询，可达/不可达各256；复用原生成器的rank-band距离配额、
查询采样、节点重命名和边呈现。额外用独立BFS检查全部标签，并排除原训练/验证/
测试图的结构hash和新测试集内部重复。固定数据只生成一次，全部模型复用。
复杂图与简单图同时改变图族和边密度，结论不能单独归于拓扑变化。

原生4K、BF16、贪心自由生成、margin8、compiled KV-cache，固定batch32。
不加oracle约束、不过滤难例、不延长RoPE或上下文、不禁用MISTAKE。
大图可能受预算限制：报告截断、强制停止、参考DFS可完成子集和参考长度。
参考DFS顺序不是模型唯一合法顺序，其长度只能作诊断。

复用correction_replay.replay：接受visited/candidate合法重排和MISTAKE取消修复。
报告答案准确率、有效CoT合法率、完全无错合法率、不合法但答对、首错原因/字段/
动作位置、无首错但未完成、MISTAKE与修复次数。首错分布避免首错后的连锁计数。
指标按positive（主分析）/negative/all和每个模型seed分别记录，不混淆分母。

```bash
.venv/bin/python studies/dfs_generalization/run.py --prepare
.venv/bin/python studies/dfs_generalization/run.py --launch --background
```

调度器先等待正在运行的训练队列分配完所有待训练任务，之后只使用真正空闲、
且训练status中没有占用的GPU，不中断或挤占训练。断点重启跳过完整评估单元。
禁用W&B，不保存activation和整条生成轨迹；保留紧凑逐查询审计和少量错误片段。

数据位于data/evaluation/dfs_generalization_20261031；结果、状态、日志和自动
更新的summary.json/REPORT.md/dashboard.html位于
outputs/analysis/dfs_generalization_20261031。新模型或更多规模可按同样底座扩展。
