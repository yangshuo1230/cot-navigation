# 新 cohort 的统一机制实验

当前 manifest：`studies/correction_behavior/cohorts/all_positions_tokenmatched_20261103.json`。
6/9 层 × 0/30/60% × 初始化 seed0/1，共 12 个有效新模型。0/30/60% 分别使用
最终 262370/226929/200000 步；使用同深度、同初始化对照，不复活废弃模型。

```bash
.venv/bin/python studies/correction_mechanisms/pipeline.py --launch
```

2026-11-03 最新用户指令：**完成探针/lens → 补齐自然分析 → 画图；暂不做强制MISTAKE**。
选择保存在 `execution_choices.json`：`natural_after_probes=true`、`skip_natural=false`、
`skip_forced=true`。启动/恢复可用 `pipeline.py --launch --natural-after-probes --defer-forced`。
此前的 skip-natural 指令已被最新指令覆盖，自然阶段将从39/60项继续，不重做已完成项。
当前正在运行的历史探针进程由新协调器接管，不中断、不重复拟合；退出后逐项核验
全部24个结果文件，不能把提前失败当成完成。强制MISTAKE保持暂缓，不会自动启动。

已有8个普通N100测试轨迹直接复用，缺少的4个模型只生成探针所需N100窗口。
训练/验证自然错误样本仍是探针必要输入，不做五种图的行为统计。
若以后重新启用强制MISTAKE，仍使用同期配对原生生成作对照；历史自然结果只对已有
查询核对哈希，未覆盖查询记为无历史参考，不能计成哈希不一致或零表现。

流水线在 GPU0 串行执行；重复启动活跃 PID 会被拒绝，已有完成项跳过。日志和阶段状态：
`outputs/analysis/correction_mechanisms/all_positions_tokenmatched_20261103/` 下
`pipeline.log`、`pipeline_status.json`。失败会保留错误；重启命令可恢复完成阶段。
导航模型冻结，不重训、不修改 checkpoint，不上传 W&B。

以下是实验模块说明；实际执行顺序以以上最新指令为准。

1. 公共 behavior v2：5 种图，每模型每图 256 个官方可达查询，4K，贪心，BF16。
   同次自然生成完成首错、记录错误、历史计数、长度/状态共同支持、删失、非法动作后
   局部一致性、合法回溯后的错候选重现、原正确候选变错等完整固定流程。
   只有普通 N100 的原始 token 保留一份压缩主数据，供后续探针复用。
2. 同深度 0/30/60% 的 seed0 模型提供自然训练/验证轨迹：每个来源 1536/256 个
   查询，首错完整记录后再生成两条记录停止。测试使用本批次自己的原始自然生成，
   做同样的事后窗口裁剪，不从旧模型借 donor。图哈希/查询严格跨 split 隔离。
3. 字段身份、栈/visited/候选集合、真实后继、合法候选等读出：独立线性/MLP64，
   逐层独立拟合，1500 步，每来源前512/128训练/验证查询，256查询自身自然测试。
   保留 token 和 token+位置对照；集合阈值0.5，检查非成员误报、F1、完整集合恢复。
4. 历史错误探针：自然正负匹配，首错 token +0…+5 和首错动作后0/1/2条记录两种
   对齐。每层/位置独立线性、MLP64；5000步、两个探针seed；按验证集选参数。
   共同前缀、自身错误、严格同token、标签打乱对照；阈值0.5，不汇报AUROC。
   不够查询支持的 cell 记为缺失，不能解释成不能感知错误。
5. 基础 raw logit lens 不训练。额外校准 lens 用前4条动作、完整记录在首错前、
   receiver 下一token预测正确的训练/验证位置拟合，1500步；最多4096/2048位置。
   全测试、早期、首错前正确、早期正确和较晚正确子集分别统计；测试不选参数。
   第5层及倒数第二层到最终层的语义合法转变，复用真实图/协议 replay；首错预测位置
   保留，前序已错排除。接受不同的合法 token 顺序，不使用自然 token 一致性代替合法。
6. 公共 oracle MISTAKE：外部识别错误完整记录后回退替换动作，不提供正确修复，
   后续自由生成。与同批本模型自然生成逐查询比较，检查 sham 与自然轨迹哈希。
7. 自动汇总7张关键图，PNG/SVG，按层数/图设置分开。每组仅两个模型种子，阴影
   是种子差异，不把不同模型对同查询的重复测量当成独立图。

不保存 activation/probe 权重，不重复保存强制干预的 token 副本。完整记录级行为
结果在 `outputs/analysis/correction_behavior/all_positions_tokenmatched_20261103`，
探针/lens、阶段日志与最终图在上述 mechanisms 目录。代码复用 common checkpoint
adapter、公共 rollout/replay、natural_history 匹配、history_probe 和 state_readout
拟合器，以及已验证的字段抽样和 lens 校准函数；不复制旧 runner 的硬编码模型列表。

2026-11-03 图表增补：原七张图的标题、图例、坐标说明已中文化。新增第8张自然
MISTAKE 标注/本条修复/整条修复计数，第9张普通N100原始token全量与可核验计数。
`marker_summary.py --manifest ...` 仅在CPU聚合已有冻结文件，不运行模型；然后运行
`report.py --manifest ...` 复用同一底座绘图。Windows自动使用微软雅黑；Linux可通过
`COT_PLOT_FONT` 指定中文字体。其他图未保留原始token，原始全量标记次数为空，
不能用可评分记录中的标记数量替代。所有新图附有统计口径，原始结果文件不变。
