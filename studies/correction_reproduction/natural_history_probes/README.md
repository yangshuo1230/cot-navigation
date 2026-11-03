# 自然 CoT 训练和测试探针

当前可视化入口已更新为 [当前节点输出前的层间修正](../within_token_correction/README.md)。下文保留历史“已消费token后的错误历史”实验说明；不应将其混同输出前探针。

**8/8已完成。结果见 [REPORT.md](REPORT.md)，逐位置汇总见summary.json；40项相关测试通过，5632条原始自然前缀和全部匹配读出均已审计。**

交互结果：[natural_probe_dashboard.html](natural_probe_dashboard.html)。单文件离线打开，无服务器、CDN或联网依赖；支持自然共享/自身测试、线性/MLP、逐层逐token热图、类型与对照分析、CSV及SVG导出。用`build_dashboard.py`从summary.json重新构建；图表口径及支持阈值在页面内说明。浏览器验证见dashboard_verification.json。

当前页面默认汇报**预测准确率，固定概率阈值 0.5**：概率 ≥0.5 判为截至当前位置已出错。主测试、严格同 token 子集和各类对照均使用相同阈值，不根据测试成绩调节。每个有效读出 cell 的正负配对平衡，因此准确率可由已保存的 TPR／FPR 精确恢复为 `(TPR + 1 − FPR) / 2`；总体仍按 token 位置、探针 seed、模型 seed 宏平均。这不等于自然自由生成错误比例下的总体分类准确率。页面及 CSV 不再汇报旧排序指标，旧指标链接自动迁移到准确率；原始历史 JSON 保留，构建时生成准确率，不重训导航模型或探针。

### 偏移 0–5 增量分析

原热图按**动作记录起点**计数：首错记录偏移 0–3 没有可匹配的自然已错正例，无法训练当前目标的二分类探针；偏移 4、5，以及后两条记录的偏移 0–5 已有结果。页面补充对应训练／验证／测试的逐位置正负查询覆盖表。

新增**首个错误 token 对齐 +0…+5**：+0 是已消费错误 token、尚未生成下一个 token 的残差；可以跨记录，遇首个 MISTAKE 截断。两种对齐已整合到同一个“每层 × 每个 token 位置”模块，通过“位置对齐方式”切换，共用热图、曲线、差值模式、支持筛选及 CSV／SVG 导出；分别保留选中的偏移，URL 保存对齐方式。目标均为“截至当前是否已经语义出错”，不将未来错误回标到合法前缀。负例仍匹配实际的记录偏移、字段和生成 checkpoint，优先同 token；两端均是原始自然生成，无编辑。

`early_offsets.py` 复用公共 `match_readouts(error_offsets=range(6))`、原 `run.py` 的冻结模型与特征提取和公共 `fit_positions`。每层×相对偏移独立训练线性／MLP64，5000步、2个探针 seed，逐位置验证 BCE 选步。保留相同8个导航 checkpoint、原数据划分、共享／自身测试、严格同 token 子集及 token／历史 embedding／随机标签对照；没有重新训练导航模型或生成新主数据。

```bash
.venv/bin/python studies/correction_reproduction/natural_history_probes/early_offsets.py
.venv/bin/python studies/correction_reproduction/natural_history_probes/build_dashboard.py
```

只增加 `early_offsets_results/` 紧凑指标、`early_offsets_summary.json`、来源指纹计划及日志；不保存激活或探针权重。构建器存在该汇总时会自动将增量结果嵌入同一个离线 HTML。原 `summary.json` 和原实验指标不变。

替代人工错误作为主训练分布；旧人工训练结果保留作历史记录。

## 主数据：原样自由生成，两个分支都不修补

- 图任务、模型容量、latest200k checkpoint与之前相同：正样本、普通N100、新图DFS；matched_clean/correction各4个模型seed。
- 训练采集：matched_clean_s0 / correction_s0分别在同一1536个官方train正查询上自由生成，共3072条轨迹。
- 验证采集：两个来源分别在同一256个官方val正查询上生成，共512条轨迹。
- 测试采集：全部8个checkpoint分别在同一256个官方test正查询上生成，共2048条轨迹。这仍是256个独立测试查询，不能计作2048张独立图。
- 只在首错记录结束后再完整生成两条记录时停止分析续写，使用公共generate可选record_stop；不更改任何采样token、不拼接参考续写、不要求修复错误。未出错的生成沿用原生4K/STOP预算。分析停止标志独立于预算耗尽。
- 主读出严格早于第一个显式MISTAKE，避免探针只识别报错标记；记录因此损失的后续位置覆盖率。该子集不能代表已经显式报错后的全部行为。

## 自然正负读出及划分

正例：未经编辑的自然前缀，语义重放证明此前或当前token已错。负例：另一个未经编辑的、截至读出仍合法的自然前缀，可以来自最后失败的轨迹；不要求最终答案正确。

每对匹配生成checkpoint、语法字段、记录内token偏移；优先匹配当前token，否则匹配token家族，并尽量匹配推理位置和记录编号。不会将模型生成的错误改成正确答案作为主负例。每个原始查询在共享数据中只选一条自然错误轨迹；每个cell的负查询不重复。

官方train/val/test先按原始查询及带节点名图哈希隔离，再做所有读出选择；同一个图的所有checkpoint续写和所有token位置留在同一划分。所有被分析checkpoint使用完全相同的主训练、验证和共享测试前缀。另单列各checkpoint自己生成的错误测试，以区分共同输入上的可读出性与自身错误分布差异。

## 探针和对照

- 每层×首错/后1/后2记录×token偏移独立线性、MLP64；5000更新、2个探针seed，逐cell验证BCE选步，标准化仅使用该cell训练数据；复用history_probe / ParallelProbe。
- 复用token+位置/字段、CoT embedding均值对照；测试单列exact-token匹配子集；查询/事件级随机翻转标签保留样本结构。
- 自然测试按首错类型、字段、层、token位置汇报，不按测试成绩挑最优层/模型。
- **人工错误只在最后作为受控测试**：错误类型须在自然训练中实际观察到；各类型使用同一批测试图、查询及同一条参考记录，仅改变指定字段。考察非邻接/图外/已访问/重复/遗漏候选，以及实际观察到的visited/栈错误。它们不参与探针训练、验证或选择。
- 主结论依据自然共享测试与自身自然测试；人工控制用于解释错误类型敏感性。不能将前缀分类等同主动纠错，也不把平衡采样下的sigmoid当作自由生成的已校准真实后验。

## 运行和产物

全部推理和探针训练只使用GPU0、CPU线程2，单进程显存上限32%。导航模型冻结；不产生导航训练、W&B运行或探针checkpoint。只保存12个紧凑压缩自然前缀文件、每模型结果和关键汇总图；激活仅驻留内存。

公共study入口：

```
.venv/bin/python -m cot_navigation study correction_reproduction/natural_history_probes
```

pipeline.py顺序完成准备、探针训练及汇总。已有准备任务在运行时会等待其完成，避免重复采集；无周期消息。完成的缓存和模型结果会校验并复用。

中文图已在Windows用可用中文字体渲染并同步到服务器。服务器无中文字体时，直接复用已验证的PNG/SVG；重新绘制可在具有中文字体的环境中运行report.py。
