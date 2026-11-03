> 本文是第一阶段记录。当前独立版本及最新测试结果见 [STANDALONE_MIGRATION.md](../docs/STANDALONE_MIGRATION.md)。

# 验证结果

- CPU 原有回归与新增等价性测试：**257 passed, 13 skipped**。跳过项为 CUDA/编译相关环境条件。7 条 warning 来自依赖的弃用提示。
- CUDA 定向回归：**39 passed**，覆盖训练吞吐、推理编译、双路 attention、record-SWA 等。10 条 warning 保留在日志。
- packed v7 direct 真实数据检查：PASS，3 个临时 optimizer updates、最长训练参考前后向、4 个自由生成查询；没有持久化训练结果。
- packed v6 full DFS-state 真实数据检查：PASS，最长参考 5,080 tokens、13 个微批、4 个自由生成查询；没有持久化训练结果。
- 新旧 v6/v7：direct/dfs_state/dfs_nodes 的抽样序列、分桶、batch 张量、loss、梯度一致；全 batch 与微批梯度一致。
- 工作流：所有预设可只读展开；重复名称、未知依赖、空矩阵、路径逃逸检查通过；研究 `--help` 工作流实际执行成功。
- 10 个历史 launcher 的 `--help` 入口验证成功。
- 原目录源码/配置/研究文本基线：未发现变化。结果/数据/figures/备份共 43,894 个文件、34,151,130,227 bytes 完成逐文件比对，无共享 inode。
- 复制期间旧任务新增/变化文件：30 个新增文件补齐；17 个变化文件另存新版本；补充复制期间没有观察到文件变化。初始快照未覆写。

## 证据

| 文件 | 内容 |
|---|---|
| `regression_tests.log` | 全量 CPU 回归 |
| `cuda_tests.log` | CUDA 回归 |
| `packed_cuda_check.log` | direct 真实训练检查 |
| `packed_dfs_cuda_check.log` | 完整 DFS 真实训练检查 |
| `workflow_validation.json` | 每份工作流的任务数 |
| `legacy_cli_validation.json` | 旧入口兼容检查 |
| `source_baseline.json`、`delivery_summary.json` | 原始源码基线与最终检查 |
| `preservation_audit.json`、`live_updates.json` | 数据/结果保留和活动任务增量 |
| `changes.json`、`reorganization.patch` | 相对原工作树的修改清单和补丁 |
| `interface_map.json` | 1,082 个旧路径到新路径对应 |

没有执行全部历史研究脚本或完整长训练；数值等价性和回归验证不等同于重新复现每个历史实验。旧任务仍可能继续产生结果，新副本不是持续同步镜像。packed 的旧 checkpoint 迁移需匹配记录的源码哈希；迁移选项未用于真实长训练续跑。
