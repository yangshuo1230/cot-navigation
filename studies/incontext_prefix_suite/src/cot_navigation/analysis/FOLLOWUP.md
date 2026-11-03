# CoT 机理补充分析

服务器项目：`~/cot-navigation`。本地结果：`results/REPORT.md`、`summary.csv`、三个 PNG 和 `models/*.json`。
沿用上一轮按验证集选择的 20 个主模型，并保留三个固定图 BFS-state 对照。模型冻结，所有线性头在物理 GPU0 训练。

## 运行

在服务器项目根目录依次执行：

```bash
.venv/bin/python -m cot_navigation.analysis.followup --resume
.venv/bin/python -m cot_navigation.analysis.error_training_controls
.venv/bin/python -m cot_navigation.analysis.followup_report
.venv/bin/python -m cot_navigation.analysis.followup_verify
```

主入口支持 `--previous`、`--output`、`--models`、`--stages sets errors blocking`。
其余入口使用 `--input` 指定本轮结果目录；错误训练来源对照另支持 `--models`。
更改方法或数据设置时使用新的输出目录，避免把不同实验静默混在一起。
全量核验要求模型清单全部完成以及两个新图错误训练来源对照齐全。

## 三类实验

1. **集合探针的概率与排序**：同配置重新拟合原有后继／visited 探针，核对旧 F1。
   同时报正确／错误节点的 sigmoid 分数、错误节点误报率、状态内正确—错误配对 AUC、
   严格分离比例、已知真实集合大小时的 top-k 诊断。空集和全集不进入排序分母。
   sigmoid 经正类加权训练，未做概率校准，不是下一 token 的 softmax 概率。
2. **第一处 CoT 错误探针**：图规则允许任意合法 Path、集合排列、PUSH 选择；
   BFS 节点版仅检查可达节点的距离层级与重复，是较弱的语义检查。
   当前支持 Path、DFS-state、BFS-state、BFS-nodes；其他协议先扩展语义 oracle 和测试。
   只取第一处错误避免连锁标注。YES/NO/EOS 不进入读出；提前 FOUND/EXHAUST 会被检查。
   错误前缀与只修复该 token 的前缀配对，分别读错误处及下一处 residual。
   +1 两侧追加同一个实际后继；额外筛选修复后仍合法的严格子集。
   自然／人工错误、图内语义／其他错误分开评分；新图另比较自然、人工、混合训练头。
   对照包含当前 token，以及记住前一个 token 身份的 embedding 控制。
3. **全部路径 visited 阻断**：从完整 stack／queue 的所有节点考虑替代路径，
   原状态必须仍可搜索成功。把所有剩余通往目标的节点写入 visited，另做不写目标的版本；
   图可达性和标准状态机都验证阻断成立。另测所有未访问节点、仅目标、等长重复 padding、
   有足够节点时的等长不同死分支控制。重生成 candidates 及后缀，不保留原候选列表。

## 解释边界

- 修改 visited 不改变原图：原查询仍是 YES，服从被篡改状态的搜索机应输出 NO。
  这两个标签必须分开；保持 YES 可能来自状态修复，也可能来自另外的答案通路。
- 不写目标与写目标的两个版本不一定使用同一批查询，不直接比较其原始比例。
  匹配统计只在同一查询内计算；置信区间使用 2,000 次配对 bootstrap。
- visited 编辑可能与历史矛盾。重复 padding 是等长压力对照，不是合法搜索状态。
  所有节点入 visited 的长度变化更大，也不能单独排除序列长度影响。
- 高分探针是外部监督可读性证据，不能直接称为模型原生错误觉察或自主纠错。
  自然错误较少、错误类型分布不同和不合法修复后继都会影响结论。
- 不同协议的原结构读出位置不同：Direct/Filler 在 QUERY 起点，Path 在路径节点，
  state 后继在 candidates 前、visited 在 visited 前。跨协议差异还包含读出位置差异。
- 不保存模型／探针 checkpoint 或 activation 缓存；每模型只保留少量解码案例。
  原研究结果和训练代码不改动，SHA256、旧 F1 复现、查询划分及重放情况均核验。

## 实现位置

- `analysis/set_metrics.py`：有效节点 mask 下的概率与集合排序。
- `analysis/error_detection.py`：语义 oracle、第一错误、修复配对及 activation 读取。
- `analysis/route_blocking.py`：完整 frontier 的阻断与反事实搜索机。
- `analysis/followup.py`：分阶段执行与恢复。
- `analysis/error_training_controls.py`：探针训练分布与前一 token 身份控制。
- `analysis/followup_report.py`、`followup_verify.py`：三个关键图、报告和 provenance 检查。
- `tests/test_mechanistic_followup.py`：替代路径、已入队目标、语义合法变体与集合评分测试。
