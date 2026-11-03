# 自然错误的长度控制与结构依赖

复用上一轮GPU0自然生成代码和公共模型/数据/语义底座。旧记录没有节点身份，无法
回溯追踪同一错误候选；新增纯CPU `error_dependencies` 审计，在自然生成后只保存
紧凑结构标注，不改token、不训练模型、不保存激活或完整轨迹。

## 三类数据，不混合

1. `../length_analysis.json`：原4seed自然完整轨迹中已完成且两组配对的数据。
   token位置首错风险与记录位置条件错误率分开。首错风险只计算首错前风险token，
   除去末尾STOP；同长度共同支持比较只看此前0与1条错误记录，再控制状态/进度。
   这些调整仍不是因果识别，且状态可能是错误之后的中介；显示覆盖量，不外推。
2. `native_window_summary.json`：已有普通N100自然探针语料，各4seed×256可达查询。
   最多到首错后两条记录，立即提供非法动作后局部执行与短窗口回访检查；不代表
   长期回访率、不与完整轨迹合并。保存少量候选字段片段用于核查有趣现象。
3. `results/*.json.gz`：新补充完整自然轨迹，初期seed0、1，两组，5图×256查询。
   在原任务释放GPU0后串行运行，支持以后添加seed。与旧轨迹数值batch条件一致，
   但这次为独立重放，不声称所有token逐字相同。

## 结构检验

- 字段写错但动作合法：同时构造真实状态执行后的预测、模型打印错误状态执行后的
  预测，比较下一条栈/visited。分别记录只符合错误状态、只符合真状态、两者都符
  合、两者都不符合。未检出所考察关系不等于统计独立。
- 首个非法动作后：另建局部执行分支与拒绝动作分支，对后1–2条记录逐字段检查。
  图外PUSH仍能建立局部栈与visited；空栈POP等无法解释的情况保留NA，不强行打分。
  全字段局部一致不代表全局图导航正确，也不能排除内部共同错误原因。
- 回溯：只把合法PUSH离开后返回同一祖先栈的记录当回访；MISTAKE原地重写不算。
  检查此前未使用的额外候选token是否再次出现；遗漏只看返回时仍应存在的后继。
  无回访不算纠正，并保留原干净候选回访变错的描述性对照。

## 入口

```bash
.venv/bin/python studies/correction_reproduction/error_continuation/length_analysis.py
.venv/bin/python studies/correction_reproduction/error_continuation/dependencies/native_window_check.py
.venv/bin/python studies/correction_reproduction/error_continuation/dependencies/run.py --seeds 0 1
.venv/bin/python studies/correction_reproduction/error_continuation/dependencies/report.py
.venv/bin/python -m pytest tests/test_error_dependencies.py tests/test_error_continuation.py -q -o addopts=''
```

`launch_after.py OLD_PID`为这次顺序调度入口，先等旧GPU0任务结束，再运行上述补充并按
完成任务更新同一份结果。`sync_results.ps1`仅将紧凑结果下载到本地对应目录。
交互页为`dependency_dashboard.html`，始终明确各数据源的seed覆盖与完成状态。
