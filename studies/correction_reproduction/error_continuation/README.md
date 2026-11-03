# 自然错误历史与后续生成

问题：baseline与纠错训练模型自然生成的CoT，在先前已有不同类型、不同数量错误时，
后续记录是否更容易错误、直接恢复，或沿用错误节点？这是观察性分析，不识别因果。

复用最终200K的matched_clean/correction各4seed；N32、N64、普通N100、窄深N100、
困难N100各256个相同可达测试查询。只使用GPU0，BF16原生贪心生成至答案或4K预算；
不修改任何生成token，不屏蔽MISTAKE。按正样本batch32重新生成，数值batch与旧的
含负样本batch16不同，单独记录与历史预测的差异，不宣称逐token重现历史结果。

复用ContextStore、ScalePreparedRows、CompiledInference、generate、共享语义重放器。
数据/模型不重新训练，不写W&B或checkpoint。这里只保存紧凑逐记录标注及轨迹哈希，
不持久化完整token或激活。每个模型-图任务完成后原子提交，可断点续算。

## 语义口径

- 一条记录为 `stack | visited | candidates | action ;`，只评分完整记录。
- 从起点重建真实状态，只执行真实图上合法动作；打印字段不修改真状态。
- visited/candidates允许任意合法顺序，候选不要求固定路径。PUSH遵循首候选优先协议；
  即使仅违反候选优先而图动作合法，仍可继续核验真实状态。
- 真实图上非法动作/格式无法执行后停止严格语义评分；不把其后全部token自动判错。
- MISTAKE取消本条记录，不推进搜索状态；错误取消和无错误报分别计数。
- 主历史计数是当前记录之前错误完整记录数，分0、1、2、3–4、5+；同记录错误再多
  也只增加1。敏感性另外统计错误事件数及历史未被MISTAKE取消的错误记录数。
- 不将当前错误计入自己的历史。首错类型来自此前最早错误记录，并按实际字段扫描
  顺序指定主类型；本条的完整错误类型保留为多标签事件计数。
- 后续正确记录不等于整段CoT合法。无标记恢复与MISTAKE后恢复不能混为一谈。
- 主风险按记录加权，另看首错后+1至+5（每轨迹每lag最多1条）、各seed、排除已达
  目标位置、进度/状态复杂度共同支持分层。共同支持差值仍不是因果干预结果。
- 95%区间使用共享query和model seed交叉重采样；仅4个seed，区间应谨慎解释。
- 高错误次数组有幸存偏差。动作非法、早停、预算截断会让后续无法观测，必须一并看
  样本量和删失；自然首错发生在未完成记录时也单独计数，不填充后续结果。

## 运行

```bash
.venv/bin/python -u studies/correction_reproduction/error_continuation/run.py
.venv/bin/python studies/correction_reproduction/error_continuation/report.py
.venv/bin/python -m pytest tests/test_error_continuation.py -q -o addopts=''
```

报告脚本只用CPU，可读取原子提交的部分结果，明确显示未完成状态。默认完整结果为
40个gzip JSON文件，共10,240条模型-查询轨迹（1,280个不同图查询，各8模型）。
`summary.json`、`REPORT.md`和单文件离线HTML是主要产物；主框架原有研究不覆盖。
