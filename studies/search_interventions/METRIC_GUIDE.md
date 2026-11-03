# 按干预目的选择指标

沿用已有冻结评估结果，未重新推理。每张图：行是 DFS/BFS，列是新图/固定图；横轴为 DAG setup。

n 为该面板各曲线的指标分母范围；不同实验样本组不同，具体分母见 plot_data.csv。无适用样本保留空缺。

Frontier/终止动作等反事实修改可能与历史冲突；仅改 PUSH 参数也偏离训练的首候选惯例。

| 图 | 纵轴指标 | 文件 |
|---|---|---|
| 目标位置干预：是否在当前步骤终止 | 本步 FOUND 率 | [PNG](G:/CoT/search_interventions_20261020/metric_figures/01_target_found.png) · [SVG](G:/CoT/search_interventions_20261020/metric_figures/01_target_found.svg) |
| 目标位置干预：最终答案是否服从修改后的状态 | 状态答案正确率 | [PNG](G:/CoT/search_interventions_20261020/metric_figures/02_target_state_answer.png) · [SVG](G:/CoT/search_interventions_20261020/metric_figures/02_target_state_answer.svg) |
| 当前节点替换：能否区分可达分支与死分支 | 状态答案正确率 | [PNG](G:/CoT/search_interventions_20261020/metric_figures/03_branch_state_answer.png) · [SVG](G:/CoT/search_interventions_20261020/metric_figures/03_branch_state_answer.svg) |
| Frontier 顺序干预：下一条状态是否正确更新 | 下一状态正确率 | [PNG](G:/CoT/search_interventions_20261020/metric_figures/04_frontier_next_state.png) · [SVG](G:/CoT/search_interventions_20261020/metric_figures/04_frontier_next_state.svg) |
| PUSH 分支干预：动作是否传递到下一条状态 | 下一状态正确率 | [PNG](G:/CoT/search_interventions_20261020/metric_figures/05_push_next_state.png) · [SVG](G:/CoT/search_interventions_20261020/metric_figures/05_push_next_state.svg) |
| PUSH 分支干预：后续搜索是否完整且合法 | 完整后缀合法率 | [PNG](G:/CoT/search_interventions_20261020/metric_figures/06_push_complete_search.png) · [SVG](G:/CoT/search_interventions_20261020/metric_figures/06_push_complete_search.svg) |
| 候选列表干预：动作是否跟随首个展示候选 | PUSH 首个展示候选的比例 | [PNG](G:/CoT/search_interventions_20261020/metric_figures/07_candidate_following.png) · [SVG](G:/CoT/search_interventions_20261020/metric_figures/07_candidate_following.svg) |
| 候选列表干预：下一动作是否符合图搜索规则 | 首动作合法率 | [PNG](G:/CoT/search_interventions_20261020/metric_figures/08_candidate_action_legal.png) · [SVG](G:/CoT/search_interventions_20261020/metric_figures/08_candidate_action_legal.svg) |
| 终止动作干预：最终答案是否跟随指定的终止含义 | 指定答案输出率 | [PNG](G:/CoT/search_interventions_20261020/metric_figures/09_terminal_answer_following.png) · [SVG](G:/CoT/search_interventions_20261020/metric_figures/09_terminal_answer_following.svg) |

## 目标位置干预：是否在当前步骤终止

纵轴：**本步 FOUND 率**。预期：目标在当前位置时应为 100%；另外三种条件本步均不应 FOUND。该指标只检查终止动作。

![目标位置干预：是否在当前步骤终止](G:/CoT/search_interventions_20261020/metric_figures/01_target_found.png)

## 目标位置干预：最终答案是否服从修改后的状态

纵轴：**状态答案正确率**。预期答案由修改后的状态机计算，区别于原始图标签；各曲线都以高值表示服从状态。

![目标位置干预：最终答案是否服从修改后的状态](G:/CoT/search_interventions_20261020/metric_figures/02_target_state_answer.png)

## 当前节点替换：能否区分可达分支与死分支

纵轴：**状态答案正确率**。可达/死分支由状态模拟器及独立可达性检查认证；生成失败记为不正确。

![当前节点替换：能否区分可达分支与死分支](G:/CoT/search_interventions_20261020/metric_figures/03_branch_state_answer.png)

## Frontier 顺序干预：下一条状态是否正确更新

纵轴：**下一状态正确率**。同时核对下一条 frontier 顺序和 VIS 集合；所有应有非终止转移的样本进入分母，缺失/非法转移记失败。

![Frontier 顺序干预：下一条状态是否正确更新](G:/CoT/search_interventions_20261020/metric_figures/04_frontier_next_state.png)

## PUSH 分支干预：动作是否传递到下一条状态

纵轴：**下一状态正确率**。分母为全部分支样本。仅改 PUSH 参数偏离训练中选首个展示候选的惯例；同步重排分支保持该惯例。

![PUSH 分支干预：动作是否传递到下一条状态](G:/CoT/search_interventions_20261020/metric_figures/05_push_next_state.png)

## PUSH 分支干预：后续搜索是否完整且合法

纵轴：**完整后缀合法率**。要求每步状态/候选/动作合法并完成搜索、回答正确；STOP 中断记为未完成，不等同于算法错误。

![PUSH 分支干预：后续搜索是否完整且合法](G:/CoT/search_interventions_20261020/metric_figures/06_push_complete_search.png)

## 候选列表干预：动作是否跟随首个展示候选

纵轴：**PUSH 首个展示候选的比例**。此图测行为跟随，不是正确率：非出邻居曲线越高，表示越常 PUSH 注入的假节点。

![候选列表干预：动作是否跟随首个展示候选](G:/CoT/search_interventions_20261020/metric_figures/07_candidate_following.png)

## 候选列表干预：下一动作是否符合图搜索规则

纵轴：**首动作合法率**。按图的真实出邻居、VIS 和当前节点验证动作；强制给定的错误 CAND 字段本身不计为模型动作错误。

![候选列表干预：下一动作是否符合图搜索规则](G:/CoT/search_interventions_20261020/metric_figures/08_candidate_action_legal.png)

## 终止动作干预：最终答案是否跟随指定的终止含义

纵轴：**指定答案输出率**。强制动作曲线测终止提示控制；两条未改动曲线提供各自样本组的原答案正确率对照。

![终止动作干预：最终答案是否跟随指定的终止含义](G:/CoT/search_interventions_20261020/metric_figures/09_terminal_answer_following.png)
