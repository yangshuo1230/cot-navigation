# 训练吞吐优化

2026-10-23 已应用到服务器 `/mnt/ningxuefei/yangshuo/cot-navigation`。
本次优化不实现双路 attention，不启动8组结构实验，不改变课程步数、逻辑batch、样本分布或loss权重。

## 实测

使用现有 hard DAG N100、6层/d128/FFN512 的 DFS/BFS 模型，以及相同64条真实训练查询。
临时模型和AdamW只在内存中更新；没有导航checkpoint、W&B run或activation缓存。
每项先预热5步，再测30步。原/新训练器使用同一批缓存样本。

| 协议 | 原更新ms | 优化后ms | 有效吞吐提升 | 原/新微批次数 |
|---|---:|---:|---:|---:|
| DFS | 118.95 | 96.23 | 23.6% | 14 / 5 |
| BFS | 146.64 | 114.49 | 28.1% | 17 / 6 |

上表为不收集日志的完整 `_update` 路径中位数，包含collation、H2D、forward/backward、梯度裁剪、optimizer；不包含样本重抽取/边重排和验证。
收集日志时，DFS为123.33→97.89ms，BFS为150.68→116.34ms。
这些是固定真实批次的短基准，不代表所有图大小、所有架构或整轮训练均有同样加速。

## 修改

- 默认微批次token预算从16k提高到64k，逻辑query batch仍为64。hard DAG启动器和optimized context预设同步更新。
- 长度排序后限制混合不同长度产生的额外padding，默认上限30%；单条样本自身的bucket取整不算入该上限，不丢样本、不截断CoT。
- loss的FP32 NLL与per-example masked reduction使用`torch.compile`融合；CPU和smoke走eager参考路径，可设置`train.compile_loss=false`关闭。
- H2D仅传tokens和三个loss mask，不再搬运未参与训练计算的标签、长度等CPU元数据。
- 日志loss汇总合并为一次小向量CPU传输，减少逐微批次的GPU同步。总loss继续记录实际反向传播目标，保留lambda和WALK加权。
- 新增`update_time`、`end_to_end_tokens_per_s`、`gpu_reserved_mb`和验证`wall_time`，区分计算、渲染/汇总及评估时间。
- 新增独立`train.free_eval_interval`；默认0保持原评估行为。显式使用throughput配置时，teacher-forced每5k、自由生成每10k、W&B每100步、保存每10k。

16k→64k并不意味着padding更低。朴素64k会把DFS padding从8.6%推到29.3%；长度约束将它压到20.4%，以少量padding换取更少微批次。BFS朴素64k为26.4%，约束后16.7%。
实测继续扩大到128k/256k并没有更快，最严格5%长度约束也会产生过多小微批次。因此当前选择64k/30%，没有通过盲目占满80GB显存追求表面利用率。

## 使用

新配置默认获得64k、长度分组、loss融合和同步优化。旧resolved/generated配置若显式写了16k，会继续遵守该值；不要修改历史配置文件。可在实验配置之后叠加：

```bash
--config configs/train/throughput.yaml
```

完整的只读配置检查示例：

```bash
.venv/bin/cot-nav train \
  --config outputs/hard_dag_v1/launches/hard_dag_v1_s0/context/curriculum/dfs_state/resolved_config.yaml \
  --config configs/train/throughput.yaml \
  --set output_dir=outputs/throughput_new_run \
  --dry-run
```

throughput overlay不改变max_steps、学习率计划、数据、轨迹变体、逻辑batch、loss权重或最终测试覆盖；自由验证频率改变会改变best checkpoint候选集合。8组结构比较应统一使用同一评估配置。
CUDA首次编译有额外开销，基准需预热；不同微批次和融合kernel不保证BF16位级复现。

## 验证

- 全套CPU回归：221 passed、6 skipped。
- CUDA定向检查：6 passed，包含融合前后loss值与梯度对照、不同序列长度。
- CPU覆盖训练续跑与不中断训练一致性；新增测试覆盖按长度分批不丢样本、全局query/WALK加权梯度、日志归约和独立验证调度。
- smoke继续禁止保存训练记录、导航checkpoint和W&B run。

可以复测：

```bash
.venv/bin/python research/training_speed_20261023/benchmark.py \
  --gpu 7 --protocol bfs_state --compile-loss \
  --budgets 16384 65536 131072 --padding-fractions 0.15 0.30 --tag recheck
```

请先确认目标GPU空闲。基准不保存临时训练权重，但会保存小型性能JSON。
原实现对照保存在本次研究目录中，未创建`.code_backups`。
