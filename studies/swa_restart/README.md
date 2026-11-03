# Clean record-SWA verification, 320k updates

18 fresh models: 6/9 Transformer layers × FULL/SWA-1/SWA-2 × model seeds 0/1/2.
This is clean in-context DFS-state training, not a correction-model cohort.

All three arms use the existing latest objective from
`studies/correction_reproduction/trace_weight.py`:

`lambda_i = clip(actual_CoT_tokens_i / 128, 1, 16)`

`task_loss_i = 2 * (answer_CE_i + lambda_i * mean_trace_CE_i) / (1 + lambda_i)`

CoT length excludes answer, EOS and padding. Existing trace supervision is kept
unchanged. FULL is the matched length-weighted baseline; SWA arms use the exact
same objective. No MISTAKE data, RL, historical donor or pretrained initialization.

## Fixed protocol

- Simple layered DAG, M=floor(1.5N), shared existing data seed0; 50k training
  graphs per scale. Input edge list remains causal in all arms.
- Four stages, 80k each, total320k: N16 → N36 → N64 → N100. Stage1 uses N16 only.
  Later stages use70% current scale and30% equally among preceding registered
  scales; N24/N32 are included in replay. Exposure seed0 across model seeds.
- d128, one head, FF512, RoPE,4096 context, dropout0; four offline randomized
  search variants, shuffled context edges, effective batch64.
- AdamW, lr1e-3→1e-4 cosine over320k, warmup100, weight decay.01, grad clip1.
- FULL uses full causal history. SWA-1/2 directly read the last1/2 complete
  semicolon-delimited records plus the current partial record; graph/query input
  remains visible. Train and generation use the same mask. Absolute positions
  and total4096 budget stay fixed; this does not remove indirect hidden-state
  history or implement rolling KV eviction.
- Native BF16, fused AdamW,65536-token microbatch budget,30% padding bound and
  compiled loss. FULL retains outer model compilation. SWA uses existing Flex
  record attention with the native SDPA reference path for unsupported shapes
  and cached decode. Evaluation uses the same native, noncompiled cached decoder
  in all three arms because compiled generation does not support record-SWA.
  No new attention implementation is introduced.
- Every20k: teacher-forced and free-generation validation,256 queries rather
  than historical16. Every10k: resumable checkpoint. Every80k: stage checkpoint
  and validation retention on earlier scales. Python rendered-row cache≤2048.
- Primary endpoint: final last/320k checkpoint only; no test-based selection.
  Final tests: six standard scales, N100 deep20×5/wide5×20, N14412×12.
  Each existing test has256 queries. Full prompt and generated trace tokens
  are persisted, with answer accuracy and strict full/positive semantic CoT
  validity. Alternative valid neighbor priorities are accepted; STOP prefixes
  and incomplete searches do not count as complete valid execution.
- Tracking disabled. Datasets, outputs and compiler caches live under the
  repository on `/mnt/workspace`, not ephemeral `/tmp`.

## Reuse and verification

No shared Trainer/model/loss implementation is modified. The study subclasses
the existing FineNodeTrainer for bounded flat-cache loading and correct
stage-boundary evaluation, reuses the established length-loss function, and
uses the public scheduler for idle-GPU queuing and resume.

Preparation verifies NPZ checksums, samples all graph splits and all four search
variants against the real graph, checks native record-boundary truncation, and
hashes paired initial weights for every depth/seed/arm. Worker initialization
must match the preparation receipt. GPU preflight exercises full4096-token BF16
forward/backward for all six depth/window combinations before launching.

```bash
uv sync --frozen --group dev
.venv/bin/python -m pytest studies/swa_restart/test_loss.py tests/test_record_swa.py
.venv/bin/python studies/swa_restart/run.py --prepare
CUDA_VISIBLE_DEVICES=0 .venv/bin/python studies/swa_restart/run.py --verify-gpu
.venv/bin/python studies/swa_restart/run.py --execute --background --wait
```

Resume uses the same final command with `--resume`. A changed data inventory,
objective or study source requires a new reviewed run rather than silently
resuming. The18 jobs queue over the four detected L40 GPUs; busy GPUs are not
interrupted. The formal run name is `swa_length_320k_20261103`.

Generated YAML/receipts are under `configs/generated/swa_length_320k_20261103/`;
training outputs under `outputs/swa_length_320k_20261103/`.
