# 简单图 CoT 自纠错实验（2026-10-22）

本轮按用户最终选择启动 8 个从头预训练实验，不新增干净对照：

|GPU（按空闲卡调度）|协议|错误样本曝光比例|错误字段 loss|
|---:|---|---:|---|
|0|DFS state|15%|屏蔽|
|1|DFS state|15%|计算|
|2|DFS state|30%|屏蔽|
|3|DFS state|30%|计算|
|4|BFS state|15%|屏蔽|
|5|BFS state|15%|计算|
|6|BFS state|30%|屏蔽|
|7|BFS state|30%|计算|

实际 GPU/PID 以服务器 `outputs/self_correction_v1/launches/self_correction_s0/status.json` 为准。启动器会在 12 组数据方案全部准备和验证完成后启动训练。

## 沿用的实验条件

基于 `outputs/scale_v1/launches/scale_v1_s0/context/curriculum/{dfs_state,bfs_state}/resolved_config.yaml` 克隆。使用之前的简单、严格分层 DAG，未使用 hard_dag_v1 数据：

- 6 层、1 头、d_model=128、d_ff=512、标准 FFN、RoPE、BF16、SDPA、torch.compile；上下文上限维持 4096。
- 图规模依次包含 N16/M24、N24/M36、N32/M48、N36/M54、N64/M96、N100/M150，每种 50k 训练图；沿用原始训练、验证、测试划分及四个离线随机搜索轨迹变体。
- 200k optimizer updates、每步 64 查询；四阶段各 50k，主要图规模为 N16→N36→N64→N100，后期保留原配置的小图混合比例。
- 其余优化器、学习率、微批次 token 上限、5k 验证/保存间隔、验证选 best 的限制、最终 best/last 及跨图规模测试设置均沿用原配置。训练 model seed=0。

## 错误与修正语义

记录格式沿用：`stack/queue | visited | candidates | action [node] ;`。

例如，正确记录为：

```text
V0 | V0 | V1 V2 | PUSH V1 ;
```

删除候选 V2 后，训练序列插入：

```text
V0 | V0 | V1 | MISTAKE ;
V0 | V0 | V1 V2 | PUSH V1 ;
```

第二行是第一行原本应该输出的完整记录，不是执行错误动作之后再从错误状态继续搜索。MISTAKE 不推进搜索，不携带节点参数。后续正确轨迹和标签保持原语义。真实训练 token 为 `<Vn>`、`<PUSH>`、`<MISTAKE>` 等；上述省略尖括号仅用于阅读。

每次查询曝光独立以 15% 或 30% 的概率注入一个事件；其中候选与 visited 各占 50%。每个事件只改一个字段，通过单节点增加、删除或替换制造真正的集合错误，而非仅重排。节点取自输入图；候选增加/替换避免选已 visited 的节点；visited 错误保留起点及栈/队列端点，以减少过于明显的缺失当前节点线索。由于部分状态不能使用某种编辑，增加/删除/替换的最终频率不强制等量，实际分布保存在数据审计报告中。

两种 loss 设置的输入 token 完全相同：

- masked：出错字段的全部节点及该字段结束的 `|` 不计算 trace loss。屏蔽分隔符是为了避免删除节点时仍监督模型过早结束错误字段。
- full_loss：上述位置也计算 trace loss，观察监督模型生成错误会产生什么差异。
- 两者都监督 MISTAKE、后续正确状态和操作、其他正常轨迹以及最终答案，保留原本按样本归一化的 answer/trace loss。

两种比例与两种 loss 的实验使用独立但可配对的错误 RNG；错误采样不改变查询、原始轨迹变体或边顺序 RNG 的曝光流。

## 正确性与效率

只追加一个 MISTAKE token，原有 token ID 全部保持不变，直接复用原正确轨迹缓存。错误目录单独写在 `data/mistake_v1`，不修改原图数据或 clean cache。

准备阶段对每条训练轨迹变体独立重放图搜索，检查 stack/queue、visited、候选集合、PUSH/POP/FOUND/EXHAUST 和最终标签一致；再逐个检查错误确实发生、节点不重复、修正记录与正确记录一致且能完整放入上下文。原正确尾部若因新增记录超过上限，继续使用原来的按记录边界 STOP 规则；保证错误后的完整修正记录仍在截断点之前。

每个错误方案仅存 8 个 int32，两个字段共 64 字节/正确轨迹变体（未压缩）；不复制数百万条完整轨迹。12 个协议/图规模组合共 240 万条正确变体、480 万个错误方案。训练启动时载入数组，训练循环只做 RNG、数组查找和 token 列表拼接，没有图搜索、记录解析或磁盘 IO。

已完成 61 项相关检查，另对两个图规模、两个协议的 2048 个实际事件检查两种 loss 的 token 一致性和 mask 差异。CPU 微基准中，该采样与拼接路径约 0.2–0.8ms / 64 查询；微基准只代表所取样本及数据处理开销，不含额外 token 的 GPU 计算。详细数据见 data_benchmark.json。

## 评估和 W&B

正常自由生成验证仍从干净的边列表和查询出发，模型必须自己决定是否输出 MISTAKE。保持原答案准确率选 checkpoint；中间错误不直接否定正确最终答案。

另增加 16 条 held-out 错误前缀诊断（候选、visited 各 8 条）：输入停止在错误记录的操作符之前，检查模型是否自主预测 MISTAKE，及是否输出下一条完整正确记录；匹配的干净状态前缀用于检查 MISTAKE 误报。不会在诊断输入中提前提供 MISTAKE，也不会强制模型修正。

该诊断的 rollout 上限取所需局部修正长度，只衡量局部修复，不作为完整搜索成功率。报告 exact repair 和允许候选顺序变化的合法 state repair。人工错误识别与自然生成错误的识别存在分布差异，后续分析需分别看待。

W&B group 为 `self_correction_s0`，沿用原任务图；新增默认图仅为识错率、完整记录修复率、干净前缀误报率。分字段指标、实际注入比例和自然生成中的 MISTAKE 使用频率保留在本地记录；使用 MISTAKE 本身不被当作已经成功纠错。

周期错误诊断仅覆盖写 `correction_latest.json`，最终 best/last 分别保留报告，避免增加大量中间文件。没有保存 smoke run、临时训练 checkpoint 或 activation dumps。

## 启动与恢复

本轮启动命令（已执行，不要重复提交）：

```bash
.venv/bin/python scripts/train_self_correction.py --run-name self_correction_s0 --gpus 0 1 2 3 4 5 6 7 --tracking online --wait --background
```

日志：`outputs/self_correction_v1/launches/self_correction_s0.launch.log`。中断后需使用同名配置及 `--resume` 恢复；禁止以同名实验无 checkpoint 覆盖已训练记录。

启动核验：8 个进程均进入实际优化，loss 为有限值，W&B 均为 online；随后 8 个任务均完成首轮 5000 步的正常验证与错误前缀诊断，并保存正式 latest checkpoint。未发现训练异常。启动器 PID 为 1315281，具体 GPU、PID、W&B URL 和首轮运行速度见 launch_status.json；该文件是采样时刻快照，不是实时状态。
