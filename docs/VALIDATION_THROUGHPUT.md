# 验证与最终评估吞吐优化

2026-10-23，已应用到服务器 `/mnt/ningxuefei/yangshuo/cot-navigation`。
未启动正式训练，也未启动双路 attention 的8组实验。

## 完整预算实测

冻结现有 hard DAG N100 的 DFS/BFS 模型，同一32个验证查询。原方式batch16、eager生成；快速方式请求batch64（本批实际32）、编译生成、优化KV移行。两者都使用原上下文上限32768及原STOP规则，`max_new_tokens=0`，没有额外截断。

| 协议 | 原耗时 | 快速方式 | 查询吞吐提升 | 原/新生成token数 |
|---|---:|---:|---:|---:|
| DFS | 35.79秒 | 16.81秒 | 2.13倍 | 25,635 / 21,999 |
| BFS | 523.66秒 | 195.73秒 | 2.68倍 | 34,393 / 38,404 |

是单次小批实测，不是整轮训练加速承诺。计时不包含模型载入和最初4个查询×8token预热；首次遇到新形状的编译成本仍可能计入测量。BF16分支不同导致实际输出长度不同，耗时比较同时包含这部分工作量变化，不能解释成严格等量token工作负载的纯kernel加速。

另测同一64查询、最多生成512token的短程基准：DFS 12.56→0.98秒，BFS 13.27→1.00秒，约12.8/13.3倍。该结果主要说明小批自回归调用开销，不能代替上面的完整预算结果；这项512token限制仅用于基准，未加入正式评估配置。

## 优化内容

1. **KV cache移行只复制有效前缀**。过去有样本结束时，会对每层整个32k分配区做`index_select`，复制未使用的未来区域。现在复用已有分配，只移动有效前缀。未来区域仍由因果mask排除，并在消费前被新K/V覆盖；保留STOP回退和记录历史。未提供有效边界的外部调用仍使用原全量路径。
2. **自由生成独立批量**。新增`evaluation.free_batch_size`，与teacher-forced的`batch_size`独立；不改变查询选择、顺序、数量或标签。
3. **缓存与prefill资源保护**。新增`free_max_cache_mb`和`free_max_prefill_tokens`。前者依据标准Transformer配置估算K/V容量，后者只根据输入prompt长度分批，不使用gold CoT长度；不丢查询、不缩短搜索。单个超长prompt仍单独执行。缓存预算不含编译workspace等额外内存，也不是通用模型的显存硬上限。
4. **编译生成前向**。`compile_generation=true`时GPU使用动态形状`torch.compile`，复用同一个wrapper；best/last权重载入后读取新参数。CPU保持eager。默认配置关闭该选项，throughput overlay开启；目前不支持record-SWA与此选项组合。
5. **分项计时**。每次自由评估记录`generation_wall_time`、`diagnostics_wall_time`、`generation_batches`、`generation_batch_limit`、`compiled_generation`，以及已有总wall_time，便于后续判断真正的验证占比。

同batch64、同编译方式的DFS短程对照中，仅改变KV移行策略：1.453→0.984秒，64条输出完全一致，峰值allocated约8.75→6.18GiB。大批量的内存代价由显式预算控制。

## 数值与质量检查

| 协议 | 原/新答案正确数 | 原/新合法生成前缀数 | 最终答案发生变化 | 完全相同token序列 |
|---|---:|---:|---:|---:|
| DFS | 27 / 28 | 14 / 14 | 3 / 32 | 14 / 32 |
| BFS | 25 / 25 | 10 / 10 | 4 / 32 | 15 / 32 |

合法前缀使用语义状态机，接受合法集合顺序及PUSH选择；它不等于搜索完成率。以上没有空轨迹。自然/强制STOP合计，DFS原/新均0，BFS为3/2；因此这些计数不是全部完成搜索后的准确率。

**不能将优化宣称为BF16逐token无损**。批量变化和编译融合会改变舍入，随机搜索多分支的近似等分logits可能选择不同路径，最终答案也可能改变。32条上未观察到整体答案或前缀合法率降低，但不能据此证明统计等价。历史旧配置结果和新配置结果不应混为严格对照；8组新结构应统一评估方式与精度。

需要与旧评估方式对照时：

```yaml
evaluation:
  free_batch_size: 16
  compile_generation: false
```

KV移行优化本身另有相同batch的BF16一致性检查，以及CPU/CUDA前缀、移行、未来写入、STOP回退和FP32编译生成对照。

## 使用

服务器 `configs/train/throughput.yaml` 已补充：

```yaml
evaluation:
  free_batch_size: 64
  free_max_cache_mb: 8192
  free_max_prefill_tokens: 65536
  compile_generation: true
```

在实验配置后叠加此文件即可。原有每5k teacher-forced、每10k自由验证的节奏保持；本轮没有进一步减少验证频率、样本数量、最终best/last测试或额外测试集。训练batch、loss、数据、步数均未改变。

只检查配置、不启动训练：

```bash
.venv/bin/cot-nav train \
  --config outputs/hard_dag_v1/launches/hard_dag_v1_s0/context/curriculum/dfs_state/resolved_config.yaml \
  --config configs/train/throughput.yaml \
  --set output_dir=outputs/validation_speed_new_run \
  --dry-run
```

## 验证记录

全套CPU：224 passed、7 skipped。CUDA定向：16 passed，覆盖缓存裁剪、移行、EOS、STOP回退、不同prompt长度、FP32编译生成和复用wrapper后的参数更新。

临时基准不训练导航模型，不保存导航checkpoint或W&B run；性能JSON包含冻结模型的生成轨迹，便于核对。原数据、checkpoint和历史评估文件未改动。
