# cot-navigation

从零训练 causal Transformer，在 DAG 上判断起点到目标是否可达。支持 `direct`、`path`、`dfs_state`、`bfs_state`、`dfs_nodes`、`bfs_nodes`；主指标只看最终 YES/NO，错误轨迹不否决正确答案。当前探索入口见 `docs/research/EXPLORATIONS_V1.md`；此前数据覆盖与一跳监督重训见 `docs/research/RETRAIN_V3.md`。

新图 N=16/M=24/4×4；固定图 N=32/M=48/8×4。三协议独立训练。270-token 紧凑词表；完整 DFS，没有 ANS、模式、预算或 CUT token。旧 checkpoint 不兼容，首轮数据与输出使用独立的 first_round_v2 路径。

2026-10-16起暂时撤下Filler。旧训练入口及默认数据准备保留Direct、Path、DFS；新增探索入口提供课程、WALK消融、负例随机路径、随机DFS、BFS及无重复节点序列。历史Filler配置、token兼容和训练结果保留；恢复历史四协议批次时显式传入原`--protocols direct path dfs_state filler`。

当前探索训练入口（交付未自动启动）：

```bash
bash scripts/explore.sh --run-name explore_v1_s012 --dry-run
bash scripts/explore.sh --run-name explore_v1_s012 --gpus 0 1 2 3 4 5 6 7 --wait --background
```

标准图默认60项，任务阶段50k步、知识阶段1k步；可用`--experiments`分批、`--shapes wide deep`比较图形状。完整参数和实验表见`docs/research/EXPLORATIONS_V1.md`。

## 安装与环境

```bash
cd ~/cot-navigation
uv sync --group dev
.venv/bin/cot-nav env --probe-kernels
```

锁定 Python 3.11、PyTorch 2.6.0+cu124、Triton 3.2.0。每个节点是独立 token。模型为 6 层、1 head、d=128、FFN=512、RoPE、GELU，约 1.22M 参数。

固定图数据使用新目录 `data/first_round_v2/parametric_n32_m48_allpaths_v1`；旧划分的数据与 checkpoint 请使用独立 run-name，不能直接恢复到新数据上。

## 自动创建目录并在后台启动

```bash
bash scripts/train_first_round.sh --background --setting all --graph-seeds 0 1 2 --gpus 0 1 2 3 4 5 6 7 --run-name first_round_s012_10k --context-steps 10000 --tracking online --wait
```

`--background` 先校验配置/GPU编号，自动创建日志父目录，再脱离终端启动调度进程。无需额外 `nohup`、`>` 或 `&`。默认日志为 `outputs/first_round_v2/launches/<run-name>.launch.log`，命令会打印 PID、日志位置和 tail 命令；实际运行目录仍为 `outputs/first_round_v2/launches/<run-name>/`。`--output-root` 会同时改变这两处位置；`--dry-run` 不创建目录或后台进程。运行名重复时仍需 `--resume` 或新名称。

外部 Bash 的 `> path/log` 在脚本启动前执行；若仍使用外部重定向，需先 `mkdir -p` 对应目录，脚本内部无法补救这一步失败。

## 一次准备，后续复用

```bash
bash scripts/prepare_first_round.sh
```

此脚本只做 CPU 数据准备：新图固定 train/val/test=10000/2000/2000；固定图生成 seeds 0/1/2 的图、全部 walk、查询划分和完整路径表，并对所有查询做 BFS 审计。三种当前协议的 token 序列和 walk token 也提前写入各数据目录的 `prepared/`；Path 保留每个查询的所有路径变体，训练时独立随机选路。不会启动模型训练或 W&B。

`prepared_data.json` 保存生成身份、验证报告与文件状态。再次执行或正式训练启动时，若身份/文件状态与所需协议匹配，直接复用，不重生成、不重跑全部 DFS/BFS 校验；文件变化会使凭据失效。需要重新完整审计时执行 `bash scripts/prepare_first_round.sh --force-validate`。手工审计也可继续使用 `cot-nav data validate`。

训练启动时，当前协议的 token 数组一次读入内存，训练查询（包含全部路径变体）和 walk 提前转换并缓存。取消旧的8192条缓存上限；optimizer update 中只做内存索引、选路和 batch 拼装，无文件读取、图生成、路径枚举或 DFS。token 缓存绑定源数据 fingerprint、词表、协议、Filler 长度与 renderer 版本；修改这些设置需重新准备。原始数据仍可单独使用，缺少 token 缓存时会在训练启动阶段一次构造训练样本。

正式训练每 **1000 optimizer updates** 做一次 teacher-forced 与自由生成验证；默认2000步对应第1000、2000步。知识阶段没有查询验证。验证集仍固定，最终测试集单独评估。

## Smoke：不保存记录、checkpoint 或 W&B run

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 .venv/bin/cot-nav smoke --steps 3
# 没有空闲 GPU 时可用 CPU；只输出终端摘要
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 .venv/bin/cot-nav smoke --steps 1 \
  --set runtime.device=cpu --set train.precision=fp32
```

该命令覆盖六协议的新图与固定图任务及共享知识阶段，共13项，不含Filler。临时合成数据在退出时清理，A→B 权重直接在内存中传递，不落 checkpoint。`runtime.smoke=true` 强制 `train.persist_outputs=false`、W&B disabled、checkpoint disabled，CLI 覆盖 online 或 persist_outputs 也不能解除这一限制。所有 shipped smoke preset 都设置了该开关。

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

固定图查询按 50%/20%/30% 划分，按距离平衡正负；比例在每个距离/标签桶内取整，随后各距离保留相同数量的正负例，因此实际训练查询数小于原始 split。游走可能展示测试查询的某条见证；该事实须单独审计，d≥5 不可能被最多 4 边的 walk 完整展示。

## 正式训练

正式配置默认允许持久化并启用 W&B online。先配置凭据或显式选择 offline/disabled。

### 正式启动脚本（不运行 smoke 或性能测试）

```bash
# 查看完整计划，不创建数据、输出目录或 W&B run
bash scripts/train_first_round.sh --setting all --run-name first_s0 --dry-run

# 两种知识设置各三协议；固定图先训练共同知识模型，再自动衔接三个分支
bash scripts/train_first_round.sh --setting all --gpus 0 1 2 3 4 5 6 7 \
  --run-name first_s0 --wait

# 只启动一个正式 DFS 实验
bash scripts/train_first_round.sh --setting context --protocols dfs_state \
  --gpus 0 --run-name context_dfs_s0 --wait

# 首次就采用更长训练，而不是在 cosine 降到最低学习率后才续训
bash scripts/train_first_round.sh --setting all --context-steps 5000 --task-steps 5000 \
  --gpus 0 1 2 3 4 5 6 7 --run-name longer_s0 --wait
```

脚本默认知识阶段1k、任务阶段及新图2k步，启用 torch.compile、BF16、SDPA 和可用的 fused AdamW；`--no-compile` 可关闭编译。2k步是首轮检查点，不保证收敛。默认只训练固定图seed0；`--graph-seeds 0 1 2` 扩到三张固定图，总共15个训练作业（默认两种设置+单图为7个）。各阶段仍按配置独立调度学习率，知识阶段不加载查询。

启动时自动生成/校验数据，数据文件指纹匹配时复用；数据生成有文件锁，训练只使用无计算进程且有足够显存的GPU。无空闲卡时，`--wait`等待，否则直接退出；不终止其他训练。每个作业独占一张卡，无DDP。

输出在 `outputs/first_round_v2/launches/<run-name>/`，包含计划、状态、数据校验日志和各作业 `train.log`、metrics、checkpoint、W&B记录。建议在tmux中运行；也可 `nohup bash scripts/train_first_round.sh ... > first_s0.launch.log 2>&1 &`。

```bash
# 使用原来的参数和名称恢复；已完成的作业会跳过
bash scripts/train_first_round.sh --setting all --gpus 0 1 2 3 4 5 6 7 \
  --run-name first_s0 --wait --resume
```

恢复依赖已写入的latest checkpoint。首次checkpoint前中断的作业不能恢复权重，应使用新run-name。可在恢复时增加context/task步数，但保留checkpoint的原cosine学习率horizon；不能改变既有分支的知识阶段训练长度。若打算训练5k步，优先首次就指定5k。

运行100个新update后，每30秒打印已运行时间和粗略ETA（包含已经发生的编译、验证等开销，后续验证成本可能变化）。脚本不做计时试跑、不自动执行最终test；正式训练中的验证按现有配置执行。`--tracking offline`可离线记录W&B，`--set train.lr=0.0005`等可覆盖普通超参。dry-run不会触发训练或写文件。

```bash
export WANDB_API_KEY='YOUR_KEY'             # 已有登录时可不设置
export WANDB_ENTITY='your_entity'           # 可选

CUDA_VISIBLE_DEVICES=0 .venv/bin/cot-nav train \
  --config configs/experiments/main_context_direct.yaml --set model.compile=true
```

另有 `main_context_path.yaml`、`main_context_dfs_state.yaml`，每组默认独立 output_dir。三组使用同一底层数据和 exposure seed。有效 task batch=64，microbatch 与 token 上限不改变每例损失权重。

固定图：

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/cot-nav train --config configs/experiments/main_stage_a.yaml
CUDA_VISIBLE_DEVICES=0 .venv/bin/cot-nav train --config configs/experiments/main_stage_b_direct.yaml \
  --init-from outputs/first_round_v2/parametric/balanced_seed0/stage_a/checkpoints/latest.pt
```

每张固定图独立训练。Stage A 仅 walk/一跳边，1k updates、batch 64，不加载查询数组、不训练或验证查询。Stage B 从同一 Stage A 权重复制为三组，2k updates，每步 64 查询+16 walk；两部分分别归一化后以 1:1 相加。各阶段 warmup 为100步。Stage B 必须提供 `--init-from`（或 resume），不能意外从零跳过知识阶段。另有 `main_stage_b_path/dfs_state.yaml`。上述为首轮入口步数，当前重训入口步数见RETRAIN_V3。

覆盖 `data.graph_shape`、`data.graph_seed` 时也使用独立输出目录。初始化校验图文件、节点映射、语料、划分及模型语义；不接受另一张图的同形状权重。Stage B 创建新 optimizer 和 W&B run；普通 resume 才续原 run。

```bash
.venv/bin/cot-nav train --config configs/experiments/main_context_optimized.yaml \
  --resume outputs/first_round_v2/context_direct/checkpoints/latest.pt
```

断点保存模型、optimizer、scheduler、sampler/RNG、数据/词表身份、W&B id。关键参数不匹配时拒绝恢复；max_steps 仅可增加，增加后保留原 scheduler 的 horizon 和已衰减学习率，不重启高学习率。legacy checkpoint 不允许静默严格恢复。

CLI 可覆盖有类型的默认字段，例如 `--set evaluation.batch_size=4`。未知键报错。图大小变化时同时保证 layers×width=N。`shared_dense` 真正共享 FFN，attention/norm 保持独立；MoE、RL、DDP 尚未实现，不伪称已完成。

## 损失与生成

- 新图输入：`<EDGE> u v ... <QUERY> s t |`；固定图：`<QUERY> s t |`。每个节点是独立 token，无层号、距离或图结构 embedding；三组共享词表定义，不共享训练后的权重。
- Direct 直接预测 YES/NO，只学习答案；首个 YES/NO 即结束响应，不要求额外预测未受监督的分隔符。
- Path/DFS：每例答案 CE + 每例有效轨迹 CE 均值，再按逻辑 batch 平均。
- DFS 每条记录：`stack | visited | candidates | action [node] ;`。CUR 是栈顶；动作前快照，POP 不清空 visited。候选和状态均由模型生成。Path 负例直接 NO，无 NONE。
- 固定图 Path：准备数据时将每个正例查询的全部合法路径写入 `paths.npz`，负例路径集合为空。训练先按原查询 sampler 选择查询，再在该查询路径中均匀随机选一条；不把所有路径展平成样本，避免路径较多的查询权重变大。独立选路 RNG 写入 checkpoint，缓存区分路径，三组查询/walk 顺序仍对齐。
- 验证不消耗训练选路 RNG。teacher-forced 诊断固定使用路径表第 0 条；free-generation 只输入查询，主指标仍只判断最终 YES/NO。保存的验证/测试参考路径不进入训练。
- 全路径表用扁平数组与两级 offsets 存储；先以 DAG 动态规划计算容量，再剪枝枚举。校验独立计数、路径唯一性及边合法性，保证没有漏路。`data.max_prepared_paths=1000000`、`data.max_prepared_path_nodes=20000000` 为单图资源上限，超限明确报错，不能截断路径集合。
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

默认图按任务精简：新图导航 4 张（答案 loss、CoT loss、自由生成答案准确率、DFS 完整参考轨迹率），固定图导航再加 walk loss，共 5 张；知识阶段只留 walk loss 和学习率 2 张。Direct/Filler 的 run 不生成 CoT 图。导航曲线按协议汇总图 seed，阴影为标准差；准确率始终只看最终答案，不要求轨迹正确。DFS 参考轨迹率只是独立的辅助诊断。

完整细分指标仍写入每个 run 的 `metrics.jsonl` 和评估文件；W&B 只上传核心曲线和少量隐藏诊断，不自动创建分距离、分图形状、计数及 DFS 字段面板。旧 run 的历史不会删除。三个手工视图关闭自动面板生成，可用下面脚本切换要显示的训练批次（依赖 dev 组的 `wandb-workspaces`）：

```bash
# 不加 --apply 只预览，不联网写入；保存前自动备份已有视图至 outputs/wandb_layout/
.venv/bin/python scripts/organize_wandb.py --group first_round_s012_50k --apply
```

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
# 数据覆盖与一跳监督重训（v3）

新入口：`bash scripts/retrain.sh --run-name retrain_v3_s012_50k --wait --background`。
默认新图10k/50k/150k训练图、固定图完整训练查询加一跳监督，保留旧数据和run。
先查看计划：`bash scripts/retrain.sh --dry-run`；只准备数据：`bash scripts/retrain.sh --prepare-only`。
完整参数、消融、W&B及best/last独立测试说明见[RETRAIN_V3.md](docs/research/RETRAIN_V3.md)。

