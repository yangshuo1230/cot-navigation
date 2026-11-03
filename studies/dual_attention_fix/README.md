# 首次自由验证失败修复与重跑

原 group：`dual_attention_v1_s0_20261023`。
8 组都到达10k，7组明确出现 `RecompileLimitExceeded`，其余被启动器终止。
原保存顺序为训练→验证→checkpoint，故本次8组均没有可恢复checkpoint。
失败记录全部保留；重跑使用新 group `dual_attention_v1_s0_20261023_r1`，从相同初始种子重新训练。

根因：双路模型解码中的 `cache.valid_length or cache.max_len` 会令 Dynamo
把每个递增长度特化成单独版本，超过8个版本后 fullgraph 推理抛错。
此外训练/teacher forcing/prefill/decode 直接编译同一个 forward，共享其版本预算。

修复：

1. 双路解码改用显式 `is None` 判断长度；图缓存上界使用同一已解析长度。
2. 新增 `CompiledInference`，full/prefill/decode 为独立编译入口，与训练隔离。
3. 独立阶段仍触及版本上限时，仅捕获 `RecompileLimitExceeded`，告警并使该阶段后续调用使用 eager。
   不吞模型错误、OOM或其他编译异常；本地自由评估记录 `generation_compile_fallbacks`。
4. 周期 checkpoint 在验证前保存，验证后刷新 best_score，避免验证故障丢失该周期训练状态。
   smoke 的 `_save()` 仍不写文件。

回归覆盖：CPU全套；编译缓存上限定向注入；验证异常前保存顺序；CUDA先编译训练，
再full/prefill和900步decode，batch 8→5→2→1，跨图缓存界限868，最后恢复训练。
长期解码检查要求不触发 eager 回退，间隔核对 eager logits 和有限梯度。

重跑预算与模型、数据均保持本轮设定：8组，140k=40k/30k/30k/40k，batch64，hard DAG，
BF16/SDPA/compile/fused AdamW，teacher forcing5k/free10k，原完整生成上限与最终测试集。

服务器入口：

```bash
cd /mnt/ningxuefei/yangshuo/cot-navigation
.venv/bin/python scripts/train_dual_attention.py --run-name dual_attention_v1_s0_20261023_r1 --background
```

续跑同一新group时使用同一命令加 `--resume`。
