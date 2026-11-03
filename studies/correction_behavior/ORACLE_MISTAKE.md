# 完整记录oracle-MISTAKE干预

同12模型、5图、官方256可达查询、BF16 batch32、4K上下文；只用GPU0。
每批分别重跑自然条件与干预条件，同模型/图/查询，不从旧汇总直接借基线。
自然输出另与原始轨迹哈希逐条比较；各模型/提示长度首次遇到时验证no-op回调
与自然输出完全一致。不同活跃批形状仍可能有数值差异，未触发查询的漂移单列。

每条记录的分号准备输出时，外部真实图执行机检查已生成的栈、visited、候选和
动作；字段允许合法的任意集合顺序，PUSH遵循首候选规则。若错误，保留前三字段，
撤销尚未提交的错误动作，将KV位置回退到动作开头，输出MISTAKE并强制其后分号。
下条记录完全由模型生成，不提供正确节点、状态、动作或答案。取消记录不推进
oracle搜索状态；合法记录按真实动作推进。自然MISTAKE不重复干预。

这是有lookahead与回退的外部oracle实验，不是模型自身的早期错误检测。原上下文
4096不变，但被回退的动作带来额外计算。无法定位动作开头的畸形记录，以及已经
终止后多余记录，使检查器停止并单列unsupported。提前YES/NO或没有闭合分号的
记录不强制重写。预算仍按原生成器处理，反复MISTAKE耗尽上下文属于真实代价。

`rollout.generate`新增可选`action_override`，默认None完全保持原行为。只有这项
独立研究启用；原固定流程v1和旧结果不覆盖，新计划明确记录改动后的源码SHA。

```bash
.venv/bin/python studies/correction_behavior/oracle_mistake.py --output outputs/analysis/correction_behavior/depth_rate_20261101
.venv/bin/python studies/correction_behavior/oracle_mistake.py --output outputs/analysis/correction_behavior/depth_rate_20261101 --report
```

输出在原cohort的`oracle_mistake/`子目录：逐查询配对标注、comparison.json、REPORT.md。
统计回答正确/合法轨迹、helped/harmed、触发次数、修复、预算中断；不生成网页。
不重新训练导航模型、不保存激活或checkpoint，不改原自然轨迹结果。
