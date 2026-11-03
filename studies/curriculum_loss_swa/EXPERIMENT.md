# 首轮：in-context DAG / DFS-state 的 loss 与操作窗口对照

这轮仅准备代码和配置，未启动正式训练。BFS 与固定图均不在本轮。

## 五个条件

| 配置名 | λ | 可直接关注的历史 CoT |
|---|---:|---|
| full_l1 | 1 | 全部历史 |
| full_l4 | 4 | 全部历史 |
| full_l16 | 16 | 全部历史 |
| swa1_l1 | 1 | 最近 1 条完整状态记录 + 当前未完成记录 |
| swa2_l1 | 1 | 最近 2 条完整状态记录 + 当前未完成记录 |

任务目标为 `2 * (answer_CE + λ * mean(trace_CE)) / (1 + λ)`，查询之间平均。
λ=1 与原任务目标一致；λ=4/16 比较相对权重，不额外把总系数放大为 5/17。
日志同时保留原始 answer_loss、trace_loss，以及 weighted_answer_loss、weighted_trace_loss、task_loss。

一个操作序列是 `stack | visited | candidates | action [node] ;`。
EDGE/QUERY 输入始终按因果顺序可见；当前正在生成的记录不计入最近 K 条配额。
已输入 `;` 的位置预测下一条记录，所以刚完成的记录从此占历史配额。
窗口只读取实际输入/生成的分隔符，不使用真实搜索状态或答案构造掩码。

训练与生成使用相同记录窗口；绝对 RoPE 位置保留，不删除或重排 token。
原 4096 总上下文与 STOP 策略保留，SWA 不会自动延长可生成总长度。
最近记录的隐藏状态仍可能间接携带更早历史；这是局部直接注意力实验，不是严格遗忘实验。

## 课程、数据与评估

原配置来源：`outputs/scale_v1/launches/scale_v1_s0/context/curriculum/dfs_state/resolved_config.yaml`。
除实验旋钮、输出目录、tracking 标识和 SWA 的编译包装外，逐项继承原配置。

- 6 层、1 个注意力头、d_model=128、d_ff=512、RoPE、4096 上下文、BF16。
- 六个课程数据源 N=16/24/32/36/64/100，每个已有训练集 50k；复用原验证/测试划分。
- 总计 200k 步，四个 50k 阶段；主规模依次 N16→N36→N64→N100，沿用原小图混合权重，包含 N24/N32。
- query batch=64，lr=1e-3→1e-4，原 warmup、随机边顺序、4 个离线随机搜索变体、种子 0 均保留。
- 原课程每 5000 步验证/存档，原验证选 best 规则、最早 150k 参与 best 选择、best/last 两种最终测试均保留。
- 测试：六个已训练规模；N100 深窄 L20W5、浅宽 L5W20；N144 L12W12 泛化。每条件原 test_samples=256，batch16，seed1729。
- 只读核对 9 个数据源的现有 DFS 缓存，训练前不重新生成数据。

## 文件与命令

服务器项目根目录：`~/cot-navigation`。

```bash
# 只检查缓存、生成五组配置，不启动训练
.venv/bin/python scripts/train_curriculum_loss_swa.py \
  --prepare-only --run-name curriculum_dfs_loss_swa_s0

# 查看五个训练任务及实际命令
.venv/bin/python scripts/train_curriculum_loss_swa.py \
  --dry-run --run-name curriculum_dfs_loss_swa_s0

# 正式启动时使用；后台调度独立实验，等待空闲卡，不中断已有任务
.venv/bin/python scripts/train_curriculum_loss_swa.py \
  --run-name curriculum_dfs_loss_swa_s0 \
  --gpus 0 1 2 3 4 5 6 7 --tracking online --background --wait

# 训练完成后，分别汇总 best/last 的正负例准确率及精确层差
.venv/bin/python scripts/evaluate_curriculum_loss_swa.py --checkpoint best
.venv/bin/python scripts/evaluate_curriculum_loss_swa.py --checkpoint last

# 可选：冻结 best，使用上一轮相同的逐跳分层面板检查轨迹语义
CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/evaluate_curriculum_loss_swa.py \
  --checkpoint best --audit-traces --per-cell 8 --batch 16
```

配置：`configs/generated/curriculum_loss_swa/curriculum_dfs_loss_swa_s0/`。
训练输出：`outputs/curriculum_loss_swa_v1/launches/curriculum_dfs_loss_swa_s0/`。
W&B 项目沿用 `cot-graph-reasoning`，group 为 run-name，五个 arm 分别命名，不保存密钥到配置。
`--resume` 恢复同一批任务；更改配置使用新 run-name，防止覆盖已生成实验定义。

## 实现与验证口径

默认 SWA 在 AMP 且长度为 128 倍数的训练块上使用 FlexAttention 编译内核；非对齐长度、FP32 和缓存 decode 使用相同掩码语义的 SDPA/参考路径，以避开当前 PyTorch 2.6 的不兼容形状。不会在训练中静默吞掉内核错误。
`--swa-backend sdpa` 可完全使用 SDPA 参考实现，需使用不同 run-name。FlexAttention 接口参考 [PyTorch 官方文档](https://docs.pytorch.org/docs/stable/nn.attention.flex_attention.html)，实际兼容性以服务器 PyTorch 2.6 的检查结果为准。
SWA 关闭模型外层 compile，单独编译注意力；完整注意力沿用原 compile。性能差异应按实测吞吐量记录，不能直接宣称 SWA 一定更快。

测试不创建训练 checkpoint 或 W&B run。覆盖 λ 公式及梯度、微批累积、λ=1 兼容、因果与记录边界、当前记录不占额度、输入始终可见、padding、KV-cache、STOP 回退，以及真实 head_dim=128 / BF16 / 4096 长度的 GPU 前反向一致性。
