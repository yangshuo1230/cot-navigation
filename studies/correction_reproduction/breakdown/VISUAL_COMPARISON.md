# 纠错训练与 baseline：多维可视化

仅 DFS 正样本；每任务相同256个查询 × 4个seed = 1024次模型评估（非1024张独立图）。

主比较包含原始baseline、匹配干净对照和纠错模型。匹配对照与纠错模型共享初始权重、词表和查询/轨迹采样，更适合隔离错误数据的影响。

完整合法率允许 MISTAKE 撤销错误记录；CoT不合法但答对计数也包含未完成轨迹。所有数据来自已有冻结模型回放，无新增推理或训练。

图中计数是模型—查询评估次数；同一256个查询在4个seed中重复评估。普通N100、窄深N100和困难DAG分别展示，后者是分布外测试。

## 关键观察

- 普通N100：匹配对照→纠错，答对806→856，完整合法415→506，不完整/不合法但答对391→350。
- 禁用MISTAKE：N32答对918→898，N64 926→907，普通N100 856→853，窄深669→674，困难731→750。显式动作的作用随任务变化。
- 普通N100首错523次中，182次被标记，36次下一有效记录修复正确，仅5次此后整条轨迹保持合法。局部修复与全程成功须分开理解。

回放与历史保存的正样本预测有0.44%差异；图中全部使用同一批新回放的答案和轨迹，未混用历史答案。

![01_accuracy](comparison_figures/01_accuracy.png)

![02_legality](comparison_figures/02_legality.png)

![03_invalid_correct](comparison_figures/03_invalid_correct.png)

![04_first_causes](comparison_figures/04_first_causes.png)

![05_first_positions](comparison_figures/05_first_positions.png)

![06_mistake_repairs](comparison_figures/06_mistake_repairs.png)

![07_mistake_ablation](comparison_figures/07_mistake_ablation.png)

详细定义及逐项统计参见 [REPORT.md](REPORT.md)。
