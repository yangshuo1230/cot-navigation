> 本文是第一阶段记录。当前独立版本及最新测试结果见 [STANDALONE_MIGRATION.md](STANDALONE_MIGRATION.md)。

# cot-navigation 统一框架与迁移说明

服务器新目录：`/mnt/ningxuefei/yangshuo/cot-navigation-unified`。
原目录：`/mnt/ningxuefei/yangshuo/cot-navigation`，本次整理不向该目录写入。

## 使用

```bash
cd /mnt/ningxuefei/yangshuo/cot-navigation-unified
# 查看跨协议配置矩阵；只计划，不生成数据、训练或启动 W&B
.venv/bin/python -m cot_navigation workflow configs/workflows/context_protocols.yaml
# 可选择一个新 run_name，然后显式执行
.venv/bin/python -m cot_navigation workflow configs/workflows/context_protocols.yaml --run-name my_experiment --execute
# 固定图 v6/v7 的计划；示例没有 --execute，子步骤也只显示配置
.venv/bin/python -m cot_navigation workflow configs/workflows/fixed_packed.yaml
# 检索保留下来的研究程序
.venv/bin/python -m cot_navigation catalog attention
# 原有训练/评估/数据命令仍可直接使用
.venv/bin/python -m cot_navigation train --config configs/experiments/main_context.yaml --dry-run
```

工作流执行器顺序执行依赖图，适合串联数据准备、训练、评估、分析。GPU 空闲检测、多卡任务调度、Stage A/B 依赖及断点恢复仍由 `experiments.scheduler` 负责；保留的实验 launcher 直接使用它。没有另写第二套多 GPU 调度器。工作流的 `--execute` 只执行它列出的 argv，不会替 packed 子命令自动补 `--execute`。

新工作流运行保存在 `outputs/workflows/<run_name>`，同名目录已存在时拒绝执行，防止误覆写。失败后的训练续跑使用已有训练入口的 `--resume`；工作流自身不自动跳过或重跑步骤。历史 launcher 的 dry-run 行为保持原样；严格无写入预览请使用新工作流的默认计划模式。

## 框架职责

| 层 | 位置 | 职责 |
|---|---|---|
| 声明式实验 | `configs/workflows/*.yaml` | 参数矩阵、步骤依赖、训练/评估/研究组合 |
| 编排 | `experiments/workflow.py` | 只读展开、配置校验、argv 执行、状态和日志 |
| 调度 | `experiments/scheduler.py` | GPU 调度、checkpoint、完成判定、数据准备回调、后台运行 |
| 传统训练 | `training/trainer.py` | 原有 context/fixed、课程、多尺度、纠错、SWA、双路 attention |
| 冻结语料训练 | `training/packed.py` | v6/v7 共用训练循环，实验差异由 backend 提供 |
| 语料与目标函数 | `data/packed_bank.py`、`training/packed_loss.py` | 独立的 token 存储、抽样、长度分桶和全局分母损失 |
| 评估与语义 | `evaluation/packed.py`、`protocols/search_replay.py` | 共用自由生成和严格 DFS/PATH 重放 |
| 研究 | `analysis/`、`migration/research_catalog.json` | 现有共享分析模块和完整历史脚本索引 |
| 路径与记录 | `runtime/workspace.py`、`experiments/artifacts.py` | 历史绝对路径内存重定位、哈希、原子 JSON |

两个训练引擎保留不同的目标函数与数据契约：传统训练的归一化 trace/answer 目标，和冻结语料 QUERY+ADJ 的全局逻辑 batch 目标。将它们强行改成同一损失会改变实验，因此统一配置/编排层和可复用组件，保留两个明确的执行后端。

## 新旧接口对应

以下路径均相对新项目根目录；旧命令在新目录中继续可用。

| 旧入口 / 符号 | 新实现 / 推荐入口 | 兼容性 |
|---|---|---|
| `scripts/train_first_round.py:Job, execute, run_jobs, prepare_data, job_complete, check_resume_plan` | `cot_navigation.experiments.scheduler` 中同名接口 | 旧脚本继续导出 |
| `scripts/train_first_round.py:parse_args, build_jobs, main` | `cot_navigation.experiments.first_round` | 原有参数保留 |
| retrain、explore、scale、task_only_n32、train_hard_dag、train_dual_attention、train_curriculum_loss_swa、train_self_correction、train_natural_correction | 同名脚本直接引用共享 scheduler；新实验可用 workflow + 现有配置 | 原有特殊数据准备逻辑保留 |
| `research/fixed_navigation_protocols_v6_20261026/train_protocol.py` | `training.packed.main(argv, backend=...)` | 旧入口选择 v6 后端 |
| `research/fixed_navigation_hard_v7_20261026/train_protocol.py` | 同一 `training.packed.main` | 旧入口选择 v7 后端，额外支持 PATH |
| 上述两目录的 `packed_bank.PackedBank` | `data.packed_bank.PackedBank(..., bank_root=..., world_root=...)` | 旧构造参数由适配器补目录 |
| 上述两目录的 `packed_bank.logical_loss` | `training.packed_loss.logical_loss` | 参数和数值保持一致 |
| 上述两目录的 `evaluate.evaluate/actual_exposure` | `evaluation.packed` 同名函数 | 旧 import 保留；采用 v7 的 PATH 支持 |
| 上述两目录的 `semantics` | `protocols.search_replay` | 公共函数继续导出 |
| `python -m cot_navigation.cli ...` / `cot-nav ...` | 原接口或 `python -m cot_navigation ...` | 命令参数保留 |
| 手工执行多个训练命令 | `workflow CONFIG [--run-name NAME] [--execute]` | 一个 YAML 可展开多个协议/种子 |
| 手工查找研究脚本 | `catalog [关键词]` | 完整路径、是否入口、docstring 可查 |
| 其他日期目录研究代码，包括 v5 和早期实验 | 原路径保留，workflow 的 `research/legacy` 步骤调用 | 未将不同研究假设强行合并 |

`migration/research_catalog.json` 是全部历史 Python 文件索引；并非每个文件都可单独执行，`entrypoint` 标记可执行入口。未来研究优先从 `analysis`、共享数据/评估模块 import；仅把实验特有统计量放在新模块。

## 配置扩展

```yaml
version: 1
run_name: protocol_ablation
steps:
  - name: train/${protocol}/seed${seed}
    kind: train
    configs: [configs/experiments/main_context.yaml, configs/train/throughput.yaml]
    matrix:
      protocol: [direct, path, dfs_state]
      seed: [0, 1]
    set:
      protocol.name: ${protocol}
      train.model_seed: ${seed}
      tracking.mode: disabled
```

支持 `train/evaluate/data/benchmark/research/legacy/packed`。标准步骤通过 `configs` 顺序叠加配置，`set` 使用已有 dotted-key 类型校验。`data` 用 `action: prepare/generate/validate`。研究步骤指定 `script` 或包内 `module`，`args` 必须是字符串列表，禁止 shell 拼接。`depends_on` 是展开后的步骤名列表，可使用相同的 `${变量}`；未知依赖、循环依赖、重复名称、越界脚本路径会在执行前报错。`${root}` 是新目录绝对路径，`${run}` 是运行名。

新增协议若需要改变模型/语料语义，仍需实现并测试相应 tokenizer/render/replay 契约；配置矩阵不意味着任意协议天然兼容全部模型。当前保留原校验规则，不能兼容的组合会提前报错。

## 实现选择与性能

采用服务器当前工作树（包括未提交修改），没有回退到旧 commit。保留现有 65,536-token 微批预算、padding 约束、编译 loss、独立自由生成批量与 KV/prefill 预算，以及 CUDA/CPU/SWA 各自的支持路径。未为整理目录修改训练数据、总步数、学习率或默认目标函数。

v6/v7 的 packed bank 和语义重放在迁移前逐字比对一致，训练循环仅协议选择不同；因此使用一套共享实现。评估采用 v7 的超集实现，保留 PATH 合法性检查。没有声称所有历史实验中某个模型精度最高，也没有进行大规模重新训练；这里选择的是已有优化实现及经等价性验证的公共路径。

## 结果保留与隔离

全目录独立复制，包括 `.venv`、数据、输出、figures、research 和备份。无结果删除，无数据/结果 hardlink 复用。复制环境的 editable-install 路径和启动脚本已指向新目录，内部绝对符号链接也重定向到副本。历史 resolved/generated 配置原文保留，标准配置加载器在内存中重定位旧绝对路径；38 个历史脚本中的旧路径字面量已迁移（详见清单）。

复制的 Python 环境通过 `sitecustomize` 阻止常见文件写操作进入旧项目，避免遗漏的旧路径误写。这是 Python 进程的误操作保护，不是 OS 沙箱；手工 shell 命令应始终确认当前目录，外部程序不受 Python audit hook 全面约束。

原目录当时仍在运行 correction reproduction。`migration/preservation_audit.json` 记录逐文件 SHA-256 对比及原源码基线检查。复制后新生成的 30 个结果文件补入副本；发生变化的 17 个文件保留初始副本，并将新版本存入 `migration/live_updates/`。`migration/live_updates.json` 给出补充文件哈希。新目录是复制期间的快照，不是持续同步镜像，原任务后续产出仍留在旧目录，不会因本次整理丢失。

## 续跑

传统训练器保留原 checkpoint/配置检查。packed v6/v7 的原始续跑规则将源码哈希纳入身份，重构后默认仍拒绝不匹配的 checkpoint。需要迁移复制过来的旧 checkpoint 时显式加 `--migrate-checkpoint --resume`：只接受 `migration/legacy_source_hashes.json` 中记录的迁移前哈希，其他 job 字段必须一致；不接受未知代码版本。第一次保存后写入新源码哈希。历史 checkpoint 的代码版本若早于迁移基线，应使用其对应的冻结源码，不绕过校验。

## 验证

参见 `migration/VALIDATION.md` 与同目录测试日志。验证包括全量原测试、新旧抽样/微批/loss/梯度对照、工作流只读计划/错误检测/实际研究入口执行，以及空闲 GPU 上真实 packed 数据的临时训练检查。没有启动新的正式长训练或 W&B run，也没有中断原任务。

## 已迁移的现有实验配置

除三个基础示例外，`configs/workflows/` 还提供 self_correction、natural_correction、curriculum_loss_swa、两批 dual_attention 和 correction_reproduction 共 6 个工作流、49 个历史配置任务。它们使用同一个执行器，不再要求针对每批实验编写训练 launcher；保留原数据和超参数，将输出放入新的 workflow 目录，默认关闭 tracking。详见 `migration/preset_workflows.json`。依赖额外数据预处理时可在前面增加显式步骤；这些预设复用本次复制的已有 prepared 数据，不会隐式重建语料。
