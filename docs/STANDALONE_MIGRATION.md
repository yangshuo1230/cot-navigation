# 独立运行版本：迁移与验证

项目：`/mnt/ningxuefei/yangshuo/cot-navigation-unified`

本版取代上一轮“主要公共代码已提取、历史研究脚本兼容保留”的运行方式。执行所需代码、配置、数据和研究资源全部位于新项目内。旧项目、当前项目的 `research/` 档案和 `.code_backups/` 不再是运行依赖；这些目录中的历史内容保留，不删除结果。

## 使用入口

```bash
cd /mnt/ningxuefei/yangshuo/cot-navigation-unified

# 普通训练、评估、数据准备保持原 CLI
.venv/bin/python -m cot_navigation train --config configs/experiments/main_context.yaml --dry-run

# 枚举/检索研究入口
.venv/bin/python -m cot_navigation study --list
.venv/bin/python -m cot_navigation catalog attention
.venv/bin/python -m cot_navigation study fixed_navigation_hard_v7/train_protocol -- --help

# packed 后端直接从正式包运行，不加载旧研究脚本
.venv/bin/python -m cot_navigation.training.packed --backend v7 --protocol path --tracking disabled
# 上一条只打印计划。正式运行需显式 --execute --output 新目录。
# 临时真实数据检查，不保存训练结果：
.venv/bin/python -m cot_navigation.training.packed --backend v7 --protocol path --gpu 2 --eager --check

# 配置矩阵和步骤依赖
.venv/bin/python -m cot_navigation workflow configs/workflows/context_protocols.yaml
```

`study` 使用 `configs/studies.json` 的注册表在独立进程中执行，避免不同研究程序的 `common/data/prepare` 同名模块互相污染。嵌入式研究运行时按包模块启动，正确处理相对 import。255 个注册入口包含验证、报告、维护程序和历史对照程序，不应一次性全部执行；`study --list` 给出对应说明。

工作流仍采用顺序执行，训练 launcher 仍使用共享多 GPU 调度器；本次专注完成代码独立迁移，没有改变已有并行调度与实验语义。工作流的 packed 示例本身仍是计划示例，正式执行时需要在对应子步骤声明训练的 `--execute` 和新输出路径。

## 目录与接口对应

| 原位置/调用 | 本版位置/调用 |
|---|---|
| `research/<研究名称>_<日期>/...` | `studies/<研究名称>/...`；108 个目录的精确对应见 `configs/study_paths.json` |
| v6/v7 `train_protocol.py` 中加载 v5 脚本 | `training.packed` + `experiments.packed_backend`，不再存在 `load_v5` 依赖 |
| 固定图脚本中的 `Samples` | `cot_navigation.data.adjacency.Samples` |
| `evaluate_adjacency` | `cot_navigation.evaluation.adjacency.evaluate_adjacency` |
| `panel_subset` | `cot_navigation.evaluation.panels.panel_subset` |
| `save_atomic` | `cot_navigation.training.atomic_checkpoint.save_atomic` |
| 分析中的 `legacy_trace_audit` 导入 | `cot_navigation.protocols.trace_audit` |
| 各脚本导入 `train_first_round` 作为引擎 | `cot_navigation.experiments.scheduler` |
| 查找日期目录中的程序 | `study <稳定标识> -- <原参数>` |
| 测试读取旧项目的参考实现 | `tests/fixtures/packed_reference.py.txt`，独立的本地数值对照夹具 |
| 旧 checkpoint 源码版本映射 | `configs/checkpoint_migration.json`，运行时不再读取迁移日志 |

完整的逐文件对应和核查结果在 `migration/standalone/final_audit.json`。研究程序、嵌入式运行时、辅助函数、配置和所需数据均迁入 `studies/`，不是通过指向 `research/` 的符号链接访问。保留独立研究方法和不同损失契约；没有为了合并文件而替换实验算法。

旧 `research/` 路径现为档案，不作为本版执行入口。需要执行对应功能时按表使用 `studies/` 或 `study`。原项目目录依然保持其原来的使用方式。

## 路径和结果保护

- 活跃代码移除旧项目绝对路径，研究目录使用不带日期的稳定名称。
- 配置及研究元数据中的路径已重定位；共 2,742 个元数据文件的修改前副本保存在 `migration/standalone/metadata_originals/`。
- 数据、checkpoint 和既有研究结果没有被删除。档案目录保留原内容；验证生成的报告写入 `migration/standalone/`，没有覆盖已有研究验证报告。
- 历史 run 标识和结果目录名保持不变，只迁移研究目录名，避免找不到已有结果或改变实验分组。
- `runtime.workspace` 对历史配置中的路径提供纯字符串映射，不搜索、读取或回退到旧项目。

## checkpoint 与续跑

`training.packed` 新增 `--stop-after N`：在绝对第 N 个 update 保存 checkpoint 并退出，未完成训练不会写 `COMPLETE.json`；随后用相同参数加 `--resume` 继续。

普通恢复要求 job 身份一致。标准训练器和启动计划只归一化已知项目根路径，目录迁移不再被误判为数据或目标变化；数据 manifest 哈希及其他科学参数仍严格比较。显式 `--migrate-checkpoint` 只允许 `configs/checkpoint_migration.json` 中登记的源码版本转换，数据、模型、损失及其他 job 字段必须相同；未知源码哈希或目标变化会被拒绝。不同历史算法的 checkpoint 不做无条件互换。

## 如何验证“独立于历史目录”

本服务器不允许 mount namespace，因此采用 Linux Landlock 对测试进程施加内核级规则：禁止读取和执行旧项目、`research/` 档案、`.code_backups/` 内的文件。规则由所有子进程继承，不能在进程内撤销；不修改任何目录的权限、名称或内容。目录元数据仍可查看，但文件内容访问被拒绝。日志开头列出 `LANDLOCK ABI 1 DENIED ...`，运行前实际尝试读文件并确认得到 `PermissionError`。

复跑：

```bash
CUDA_VISIBLE_DEVICES='' .venv/bin/python migration/standalone/isolate.py
CUDA_VISIBLE_DEVICES=2 .venv/bin/python migration/standalone/isolate.py \
  .venv/bin/python -m pytest tests/test_cuda.py tests/test_training_throughput.py \
  tests/test_inference_compilation.py tests/test_dual_attention.py tests/test_record_swa.py -o addopts='' -q
.venv/bin/python migration/standalone/isolate.py \
  .venv/bin/python migration/standalone/audit_delivery.py
```

`isolate.py` 是这台服务器的隔离验证工具，需要启用 Landlock 的 Linux；普通训练/研究运行不依赖 Landlock。

## 验证结果

| 验证 | 结果 |
|---|---|
| 隔离后的主测试集，含完整训练/恢复/最终报告 | **265 passed，13 skipped**（CPU 环境下跳过 GPU 条件项） |
| 隔离后的 CUDA 回归 | **39 passed** |
| 研究侧 record-SWA、因果干预与数据测试 | **41 passed** |
| 探针与因果捕获 | 仿射恢复、二分类、AUROC、图属性、禁止未来信息泄漏检查通过 |
| 搜索语义重放 | **1,536** 个小图查询/搜索案例，错误答案、非边 PUSH、错误字段识别通过 |
| 真实 checkpoint 解码 | simple/hard × DFS/BFS 的生成前缀一致性通过 |
| v5、v6、v7 真实数据短训练 | PATH、完整 DFS-state、packed PATH 前后向与自由生成通过，无正式长训练 |
| 前缀注意力分支 | 七种模式的可见性、缓存、编译检查通过；bootstrap、main boundary、PATH main 的恢复权重最大差异 **0.0** |
| 课程 bootstrap | 十个阶段边界检查通过 |
| packed 持久化恢复 | 连续训练与中断恢复后的权重、RNG、曝光计数及最终报告完全一致；目标变化被拒绝 |
| 完整配置/工作流 | **106** 份完整配置、**9** 份工作流通过 |
| 研究入口参数解析 | **107/107** 通过，测试期间禁止项目写入和任务启动 |
| 文件/依赖审计 | 遗漏文件 **0**、历史路径运行依赖 **0**、错误指向旧目录的链接 **0**、Python 语法错误 **0** |

证据在 `migration/standalone/`：`isolated_cpu.log`、`isolated_cuda.log`、`help_audit.json`、`research_tests/`、`backend_checks/`、`final_audit.json`。

这些检查验证代码独立运行、数值行为、恢复和代表性研究流程。没有重新执行所有历史长训练或全部研究实验，不把 CLI 检查等同于完整实验复现。不同实验的算法、数据划分及损失定义保持各自契约。
