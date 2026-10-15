# cot-navigation

从零训练 causal Transformer，在 DAG 上判断起点到目标是否可达。支持 `direct`、`path`、`dfs_state`、`filler`；主指标只看最终 YES/NO，错误轨迹不否决正确答案。最新研究规格为 `docs/research/FIRST_ROUND.md`，优先于历史方案。

首轮 v2：新图 N=16/M=24/4×4；固定图 N=32/M=48/8×4。四协议独立训练，Filler 默认 K=N（16/32 DOT）。270-token 紧凑词表；完整 DFS，没有 ANS、模式、预算或 CUT token。旧 checkpoint 不兼容，数据与输出使用独立的 first_round_v2 路径。

## 安装与环境

```bash
cd ~/cot-navigation
uv sync --group dev
.venv/bin/cot-nav env --probe-kernels
```

锁定 Python 3.11、PyTorch 2.6.0+cu124、Triton 3.2.0。每个节点是独立 token。模型为 6 层、1 head、d=128、FFN=512、RoPE、GELU，约 1.22M 参数。

## Smoke：不保存记录、checkpoint 或 W&B run

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 .venv/bin/cot-nav smoke --steps 3
# 没有空闲 GPU 时可用 CPU；只输出终端摘要
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 .venv/bin/cot-nav smoke --steps 1 \
  --set runtime.device=cpu --set train.precision=fp32
```

该命令覆盖新图四协议、固定图 stage A 和四个 stage B。临时合成数据在退出时清理，A→B 权重直接在内存中传递，不落 checkpoint。`runtime.smoke=true` 强制 `train.persist_outputs=false`、W&B disabled、checkpoint disabled，CLI 覆盖 online 或 persist_outputs 也不能解除这一限制。所有 shipped smoke preset 都设置了该开关。

直接使用 smoke preset 调用 train/evaluate/benchmark 也不保存运行文件，但需要自行提供数据。建议使用上面的专用命令，不再依赖旧 `outputs/smoke` checkpoint。smoke 不是能力实验，不要求几步训练就达到高准确率。

## 正式数据

先使用单一中间形状和小图，仅采样相邻层边。新图 N=16/M=24/4×4，按 distance/label 配额生成 10k/2k/2k，节点重命名、边重排，每图一个查询；2k updates、batch 64，共128k次查询曝光。固定图 N=32/M=48/8×4，三个 graph seeds，每个世界独立训练；5k 条 1–4 边游走，不足时补充一跳记录，保证全部基础边至少监督 100 次，因此最终 walk 数可能略高于5k。一跳不作为主要泛化测试。扩图和宽浅/窄深比较留作后续显式配置。

```bash
.venv/bin/cot-nav data generate --config configs/experiments/main_context.yaml --dry-run
.venv/bin/cot-nav data generate --config configs/experiments/main_context.yaml
.venv/bin/cot-nav data validate --config configs/experiments/main_context.yaml

.venv/bin/cot-nav data generate --config configs/experiments/main_stage_a.yaml
.venv/bin/cot-nav data validate --config configs/experiments/main_stage_a.yaml
```

`data.root` 绑定不可变 generation fingerprint：seed、N/M、形状、生成器/schema 版本等必须匹配。worker/audit 设置不改变语义；chunk_size 改变布局，恢复时也需匹配。修改参数请用新 root。旧版无 fingerprint 的数据需在新 root 重新生成，不能给旧数据贴上新 seed。

```bash
.venv/bin/cot-nav data generate --config configs/experiments/main_context.yaml \
  --set data.workers=8 --resume
```

完成标记包含真实 checksum，恢复会重新生成损坏/未完成 shard。训练 manifest 来自列出的 shard，不扫描任意遗留文件。sorted-edge hash 不是通用去同构算法。

固定图查询按 20%/10%/70% 划分，按距离平衡正负。游走可能展示测试查询的某条见证；该事实须单独审计，d≥5 不可能被最多 4 边的 walk 完整展示。

## 正式训练

正式配置默认允许持久化并启用 W&B online。先配置凭据或显式选择 offline/disabled。

```bash
export WANDB_API_KEY='YOUR_KEY'             # 已有登录时可不设置
export WANDB_ENTITY='your_entity'           # 可选

CUDA_VISIBLE_DEVICES=0 .venv/bin/cot-nav train \
  --config configs/experiments/main_context_direct.yaml --set model.compile=true
```

另有 `main_context_path.yaml`、`main_context_dfs_state.yaml`、`main_context_filler.yaml`，每组默认独立 output_dir。四组使用同一底层数据和 exposure seed。有效 task batch=64，microbatch 与 token 上限不改变每例损失权重。

固定图：

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/cot-nav train --config configs/experiments/main_stage_a.yaml
CUDA_VISIBLE_DEVICES=0 .venv/bin/cot-nav train --config configs/experiments/main_stage_b_direct.yaml \
  --init-from outputs/first_round_v2/parametric/balanced_seed0/stage_a/checkpoints/latest.pt
```

每张固定图独立训练。Stage A 仅 walk/一跳边，1k updates、batch 64，不加载查询数组、不训练或验证查询。Stage B 从同一 Stage A 权重复制为四组，2k updates，每步 64 查询+16 walk；两部分分别归一化后以 1:1 相加。各阶段 warmup 为100步。Stage B 必须提供 `--init-from`（或 resume），不能意外从零跳过知识阶段。另有 `main_stage_b_path/dfs_state/filler.yaml`。

覆盖 `data.graph_shape`、`data.graph_seed` 时也使用独立输出目录。初始化校验图文件、节点映射、语料、划分及模型语义；不接受另一张图的同形状权重。Stage B 创建新 optimizer 和 W&B run；普通 resume 才续原 run。

```bash
.venv/bin/cot-nav train --config configs/experiments/main_context_optimized.yaml \
  --resume outputs/first_round_v2/context_direct/checkpoints/latest.pt
```

断点保存模型、optimizer、scheduler、sampler/RNG、数据/词表身份、W&B id。关键参数不匹配时拒绝恢复；max_steps 仅可增加，增加后保留原 scheduler 的 horizon 和已衰减学习率，不重启高学习率。legacy checkpoint 不允许静默严格恢复。

CLI 可覆盖有类型的默认字段，例如 `--set evaluation.batch_size=4`。未知键报错。图大小变化时同时保证 layers×width=N。`shared_dense` 真正共享 FFN，attention/norm 保持独立；MoE、RL、DDP 尚未实现，不伪称已完成。

## 损失与生成

- 新图输入：`<EDGE> u v ... <QUERY> s t |`；固定图：`<QUERY> s t |`。每个节点是独立 token，无层号、距离或图结构 embedding；四组共享词表定义，不共享训练后的权重。
- Direct 直接预测 YES/NO，只学习答案；首个 YES/NO 即结束响应，不要求额外预测未受监督的分隔符。
- Path/DFS：每例答案 CE + 每例有效轨迹 CE 均值，再按逻辑 batch 平均。
- Filler：题目 + 固定 DOT×K + YES/NO，默认 K=N，只计算答案 CE。`protocol.filler_tokens` 可显式覆盖；固定 prefill 不当成未知 CoT 的等时解码，不要求各组计算量相同。
- DFS 每条记录：`stack | visited | candidates | action [node] ;`。CUR 是栈顶；动作前快照，POP 不清空 visited。候选和状态均由模型生成。Path 负例直接 NO，无 NONE。
- DFS 训练全部完整，不按预算截断、不根据长度删样本。配置检查保守完整轨迹界，渲染再检查实际含 padding 长度。测试遇 EOS 前无答案、context 或 `evaluation.max_new_tokens` 安全上限均失败，不插入答案或将失败变成 NO。默认安全上限使用剩余 context，不能按测试样本的参考长度设限。
- teacher-forced 指标有 gold trace，仅作诊断。主 best.pt 由分层验证集上自由生成最终答案准确率选择。

## 分批评估

```bash
.venv/bin/cot-nav evaluate --config configs/experiments/main_context_optimized.yaml \
  --checkpoint outputs/first_round_v2/context_direct/checkpoints/best.pt --split test \
  --set output_dir=outputs/eval/context_direct \
  --set evaluation.batch_size=8 --set evaluation.test_samples=0
```

`evaluation.test_samples=0` 是全量测试，不继承训练期 `free_gen_samples` 的短验证限制。全量评估流式分批，不将所有测试查询同时放进 KV cache。

训练期验证按 shape/label/distance 的固定分层 sample-id 清单选取，独立于训练 RNG 和协议；小子集先覆盖 shape/label，足够大时遍历所有非空距离桶。分组指标保留分母，不能取有序数据开头冒充代表性验证。

主分数与 DFS 过程诊断独立。DFS 输出候选、栈、visited、动作的参考匹配率、正确前缀、已完成与 malformed 记录数和安全上限失败率。错误过程不否决正确答案。数据 validate 另报完整轨迹长度、回退、访问节点、最大栈和分支记录分布。

## 加速与性能

`main_context_optimized.yaml` 开启 torch.compile/Inductor/Triton。主路径使用 BF16、SDPA、可用时 fused AdamW；保留 math/eager 参考。每个逻辑 batch 内按长度和 token 上限分 microbatch，减少 padding。训练端当前为同步渲染/collate + 缓存，未实现异步 DataLoader，因此移除了空转的 train.num_workers 配置；离线数据生成仍支持多进程。

```bash
.venv/bin/cot-nav benchmark --config configs/experiments/main_context_optimized.yaml \
  --set output_dir=outputs/benchmark/main --batch 8 --length 256 --warmup 5 --steps 20
```

基准单列冷启动，组合优化对比包含精度/backend/optimizer 差异，不将它全部归因于 compile。真实长序列收益需用实际样本 profile。smoke preset 下 benchmark 不写文件。

## W&B

支持 online/offline/disabled，密钥不进入 YAML 或 artifact。同一 optimizer update 可有多次日志事件，使用自定义 `optimizer_step` 横轴，SDK history step 单调推进，因此 train/teacher-forced/free 不互相丢弃。

```bash
# 正式训练离线记录；不是 smoke
.venv/bin/cot-nav train --config configs/experiments/main_context.yaml \
  --config configs/tracking/offline.yaml --set output_dir=outputs/first_round_v2/offline_direct
.venv/bin/wandb sync outputs/first_round_v2/offline_direct/wandb/offline-run-*
```

online 失败按 online_failure 明确报错或明确降级，不声称已上传。`tracking.resume` 用于恢复原 run，新 run 使用 never；`tracking.upload_artifacts=true` 时结束上传配置及 best/latest，而非所有数据/周期 checkpoint。日志频率统一使用 `train.log_interval`，不再保留第二个空转开关。

没有 git commit 时仍记录源代码 SHA256 fingerprint。source/dirty、数据、词表和参数信息用于复现，但不上传 .env、认证文件或环境全集。

## 多卡独立运行

```bash
.venv/bin/cot-nav launch \
  --command '.venv/bin/cot-nav train --config configs/experiments/main_context_optimized.yaml --set output_dir=outputs/first_round_v2/direct' \
  --command '.venv/bin/cot-nav train --config configs/experiments/main_context_optimized.yaml --set protocol.name=path --set output_dir=outputs/first_round_v2/path' \
  --output-root outputs/launch
```

默认 dry-run；检查后加 `--execute --gpus 0,1`。每卡一独立实验，不占用已有任务的卡。识别出的 smoke 命令不写 launcher log/PID 文件。

## 测试与扩展

`pytest` 覆盖数据、mask/cache、梯度累积、断点，以及 Direct 接口、分层验证、数据 fingerprint、W&B 多事件、free best、分批全量评估、DFS 字段和 smoke 零持久化回归。测试涉及 checkpoint 的写入仅在测试临时目录，用完清理；smoke 本身不写 checkpoint。

新图生成器在 data/graph.py 与 data/generate.py，协议在 protocols/render.py，FFN 在 models/ffn.py。新增生成语义应升级 generator/schema 版本；新增 token 需升级词表版本。普通训练不缓存 attention matrix；机理分析可显式选择 capture / math attention 路径。
