# Hard DAG experiment, 2026-10-21

Run: `hard_dag_v1_s0`. Server project: `/mnt/ningxuefei/yangshuo/cot-navigation`.

## Graph distribution

`data.graph_family=multiscale_dag_v1` is opt-in. Internal node IDs are a
hidden topological ordering; every edge satisfies u < v. Token identities
are independently randomized, and context edge presentation is shuffled.
For N nodes, M=2N distinct edges are generated. About N/8 vertex-disjoint
diamonds are planted on randomly chosen vertices. Remaining edges are drawn
without replacement from a normalized 85% local / 15% global mixture;
local weight is exp(-(topological_span-1)/(width/2)). Extra edges cannot
remove the planted diamonds, but can add alternative routes/shortcuts.

Shapes such as 6x6 and 10x10 indicate N and the hidden rank-band width,
NOT strict layers. Within-band, neighboring-band, and skip-band edges exist.
This is a specified DAG distribution, not uniform sampling over all DAGs.

The existing query stratifier uses **rank-band gap, not shortest-path hops**.
It samples endpoints in different bands, with s preceding t. There are no
reverse-order negative examples in this suite. Positive and negative context
queries are balanced separately for each band gap. Generating a graph is
conditioned on having both labels in that bucket. No claim of matched
endpoint degrees or minimum required lookahead is made.

Every dataset has `hard_dag_audit.json`, which records actual shortest-path
hops, graph statistics, reference lengths and cache coverage. Do not interpret
legacy `distance` / `train_hops` fields as true graph hops for these datasets.
For fixed graphs, the legacy `one_hop` pool consists of neighboring rank-band
queries, correctly labeled by full reachability; it is not an edge classifier.
Within-band knowledge still appears in WALK, although within-band endpoint
queries are outside this first experiment's query distribution.

## Experiment matrix

| Setting | Graphs | Protocols | Updates |
|---|---|---|---|
| Fixed | N36/M72, graph seed 0 | direct, path, dfs_state, bfs_state, dfs_nodes, bfs_nodes | 75,000 each |
| Fixed | N100/M200, graph seed 0 | same six | 125,000 each |
| Context | main N16 -> N36 -> N64 -> N100 | path, dfs_state, bfs_state, dfs_nodes, bfs_nodes | 200,000 each |

17 task runs and 2 shared fixed knowledge stages. Each fixed graph has a
separate 1,000-update WALK-only stage. Task stages retain the existing
64-query + 16-WALK batches. Query splits are 70%/15%/15% per label/rank-gap
bucket; tiny buckets remain test-only. All neighboring-band query pairs are
train-only, accounting for 25% of task queries. Remaining train queries are
sampled from the full pool with label balancing; no downsampling discards
training pairs.

Context training preserves the previous six-source mixture:
N16, N24, N32, N36, N64, N100; 50,000 train examples and 512 validation / 512
test examples per source. Every main stage lasts 50,000 updates, with 70%
of queries from the current main source and 30% from smaller sources; the
first stage uses N16 only. Datasets and holdouts are shared across protocols.
After the N100 stage, best/last checkpoints are tested on smaller scales,
N100 with local-span scales width=5/20, and N144. These width variants do
not denote actual deep/wide layers; actual depth is measured in the audit.

Architecture unchanged: 6 layers, 1 head, d_model=128, d_ff=512, RoPE,
full attention, trace-loss lambda=1. Max context is 32,768 so long reference
traces can complete. Cached hard-DAG references that would be truncated
cause preparation to fail. Model parameter count is unchanged.

## Preparation and training

All 11 raw datasets, independent checks, WALK caches and protocol token
caches are completed before any new training begins. Context generation
uses deterministic multiprocessing; three dataset preparations run in
parallel with four raw-generation workers each. Four randomized search
traces are cached per query. For fixed positive PATH queries **every legal
path is stored and available for uniform online selection**; negative PATH
queries use four cached legal random walks from their source. The original
label is retained. No graph search or tokenization occurs in training updates.

Efficiency uses BF16, SDPA/Flash attention as available, torch.compile / Triton,
fused AdamW as available, token-limited microbatches and in-memory cached
examples. Validation and saving occur every 5,000 steps. Free-generation
validation uses up to 64 stratified queries; final best/last tests use up to
256 per set. Context best-checkpoint selection starts at update 150,000.
The small validation subset still makes best-checkpoint selection noisy.
Only best and the latest checkpoint are retained, plus required metrics.

W&B uses online mode, project `cot-graph-reasoning`, group `hard_dag_v1_s0`.
Authentication was verified without starting a test W&B run. Missing online
tracking raises an error rather than silently changing to offline mode.

The launcher captures `/proc` start identities for prior launcher PIDs
1004095 and 1059307; it waits for both entire launchers to exit, including
their pending stages. Thereafter the existing scheduler fills GPUs 0-7 with
independent trials, one per free GPU, and replenishes slots as jobs finish.
An occupied GPU is never shared. This is experiment parallelism, not DDP.

## Commands

Already launched in background:

```bash
.venv/bin/python scripts/train_hard_dag.py \
  --run-name hard_dag_v1_s0 --gpus 0 1 2 3 4 5 6 7 \
  --after-pids 1004095 1059307 --tracking online --wait --background
```

Follow the log:

```bash
tail -f outputs/hard_dag_v1/launches/hard_dag_v1_s0.launch.log
```

Status files in `outputs/hard_dag_v1/launches/hard_dag_v1_s0/`:
`preparation_status.json`, `queue_barrier.json`, `queue_status.json` once
preparation completes, and scheduler `status.json` after GPU scheduling.
Frozen resolved configs live in `configs/generated/hard_dag_v1/hard_dag_v1_s0/`.
For recovery, use the same command plus `--resume`; do not launch it twice.

## Verification

54 data, path completeness, prepared-cache, scale and backward-compatibility
tests passed. New tests independently compare reachability to exhaustive
per-query traversal, check deterministic acyclic generation and actual
shortest paths, exercise every protocol cache, verify all fixed positive
PATH variants, and require overflowing references to fail preparation.
All 19 resolved configs passed a dry-run plan validation. No smoke training
was run and no smoke checkpoints or W&B test runs were created.

The graph pilot compares the old N100/M150 layered distribution with the
new N100/M200 distribution; it does not isolate graph structure from density.
Under sampled rank-gap/label queries, mean DFS operations increased from
34.3 to 45.2, mean maximum stack size from 6.2 to 10.6, and mean reference
tokens from 1,067 to 1,822. These are generator pilot statistics, not model
accuracy results or a proof that every new query is harder.
