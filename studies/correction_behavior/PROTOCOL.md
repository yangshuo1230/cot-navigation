# 纠错模型行为分析固定流程 v1

新纠错模型必须走此流程。只改变cohort manifest中的模型路径、分组和输出目录；
任何统计定义、查询、生成条件变化都需要新协议版本，不覆盖旧结果。

## 固定输入和评估口径

- 每个模型使用冻结的最终200K checkpoint并记录SHA256，不按测试成绩挑选。
- DFS state，新图任务；N32/M48、N64/M96、普通N100/M150、窄深N100/M150、
  困难N100/M200，分别统计，不混合。每图256个官方可达测试查询，固定随机顺序。
- 准备阶段验证全部模型使用同一图fingerprint、query ID、距离及顺序。
- GPU0，BF16，batch32，原生4096上下文，贪心生成至答案/预算。不得人为编辑token、
  压制MISTAKE或人为延长上下文。显式记录中断与删失。
- 同次生成同时完成下述模块，保留紧凑记录级/节点关系标注和轨迹哈希，不保存
  激活、完整token副本、模型checkpoint或W&B。按模型—图原子提交、可断点恢复。

## 必须覆盖的模块

1. 最终答案、CoT语义合法性、首错类型/位置、MISTAKE真/误报与修复次数。
2. 历史错误记录0/1/2/3–4/5+，首错类型、各类型累计次数、未取消历史、事件数、
   排除到达目标位置的敏感性结果，首错后+1至+5及相应删失和样本量。
3. 首错前token位置风险、按CoT位置的记录/字段/动作错误率、长度及进度/状态
   复杂度共同支持分层，显示覆盖，不做因果归因。
4. 错误候选到非法PUSH、错误节点后续复现，以及字段错但动作合法时，下一字段
   符合真状态更新/错误打印状态更新/两者/两者都不符合的分类。
5. 首个非法动作后+1/+2局部执行状态、打印状态和拒绝动作状态的一致性；不把局部
   一致叫全局正确，不强行解释无法建立局部状态的动作。
6. 合法离开再回溯同一祖先节点的未用错误候选复现，仍需后继的重复遗漏，原正确
   候选回访变错；未回访单列，MISTAKE原地重试不算回溯。
7. 保留每seed、每图结果和样本量；模型/查询交叉重采样置信区间，明确只有2seed
   等支持限制。缺失对照不能用不同深度或未匹配历史模型冒充。

## 标准入口

```bash
.venv/bin/python -m cot_navigation study correction_behavior/run --manifest studies/correction_behavior/cohorts/NAME.json --prepare-only
.venv/bin/python -m cot_navigation study correction_behavior/run --manifest studies/correction_behavior/cohorts/NAME.json
.venv/bin/python -m cot_navigation study correction_behavior/report --output outputs/analysis/correction_behavior/NAME
```

准备只审计配置/数据/哈希，不训练或生成smoke数据。计算完成前页面和状态必须显示
部分结果。失败保存failure.json且不能标记完成。训练/探针/activation干预/不可达
查询/BFS/固定图和更长错误分支属于独立扩展，不能宣称本v1已经覆盖。
