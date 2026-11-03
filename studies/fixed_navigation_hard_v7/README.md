# Hard fixed-DAG navigation, v7

One matched-size hard graph: 80 nodes, 160 edges, graph seed 0; four random
model initializations (0,1,2,3) for Direct, PATH, DFS State and DFS Nodes.
All 16 runs start from random weights and train for 50,000 joint updates.
No knowledge-only stage. Existing v5/v6 data, code and runs are untouched.

## Graph and held-out queries

Use the previous `multiscale_dag_v1` generator, copied without modification:
random planted diamonds plus 85% local/15% global forward-edge sampling.
Hidden indices are topological; independently randomized tokens reach the model.
Ten bands of width eight are rank bands, not graph layers. The frozen graph has
56 within-band, 45 adjacent-band and 59 skip-band edges, 48 branching and 49
merging nodes, and longest path 27. Generation attempt 1 passes the same
minimum per-label panel acceptance (16; AA validation/test 4). Graph selection
never uses model performance.

Positive queries use TRUE shortest-path hops: short 1–3, long >=4. Negative
queries have no finite path distance; as before, their short/long buckets use
forward rank-band gap 1–3 / >=4. Records retain both `shortest_hops` (null for
negative) and `band_gap`; `gap` is the training bucket, not always a layer gap.
Within-band positive queries are included so every true edge can appear as a
one-hop query. Within-band negative and reverse-order queries are excluded.
Do not reuse the old hard suite's band gap as positive shortest-path hops.

Per-band A/V/T node roles are split 50%/25%/25%. Short queries get only
stratified query holdout (approximately 80/10/10), with eligible endpoint-role
coverage repaired in training. Long AA queries receive the same query split;
long VA/AV/VV are validation and long TA/AT/TT are test. V/T nodes are visible
in short queries, full ADJ knowledge, and intermediate trajectories, but never
as long training endpoints. All protocols use identical query IDs and roles.

Training has 1,316 unique short and 272 unique long queries. Test counts:
short 162, AA 31, TA 158, AT 188, TT 85. Exact label/hop denominators are in
`worlds/data/n80_g0.json` and must accompany downstream hop plots.

## Training, references and evaluation

Logical batch: 48 short + 16 long queries + 16 complete adjacency rows;
query labels balanced within each stratum, then supported hop cells sampled.
Query and variant RNG streams are separate. All four protocols share the
QUERY/ADJ presentation schedule. PATH samples up to eight frozen legal
positive paths or negative random walks; DFS protocols share eight shuffled
neighbor-order traces with no target reachability hint in candidate order.
Every DFS variant is checked against independent DFS and strict replay.

Six layers, one head, width 128, FFN 512, RoPE, tied output head, no dropout.
Trace lambda=1, knowledge weight=1; per-example mean trace loss and answer loss.
AdamW lr=1e-4, 1k warmup, no cosine decay, weight decay .01, grad clipping 1.
BF16, SDPA, torch.compile/Inductor, fused AdamW, resident ragged token banks,
length-sorted microbatches with global loss denominators; no online graph search.
No equal-FLOP claim across protocols.

Max reference lengths: Direct 10, PATH 32, DFS State 5884, DFS Nodes 64.
Contexts: 256 except State 8192. No reference clipping. Free generation can
still fail/truncate; truncation is counted as invalid, never repaired.
Positive PATH legality requires a complete legal goal path; negative PATH
validity requires a legal walk, not exhaustive reachability proof. DFS Nodes
is an expansion sequence with implicit backtracking, not a contiguous path.

Validate every 1k updates on fixed up-to-32 subsets of short/AA/VV; full final
tests cover every saved test query. Report YES/NO separately and strict trace
validity. Save rolling latest every 10k and a fixed 20k checkpoint. Online W&B:
project `cot-graph-reasoning`, group `fixed_navigation_hard_v7`, grouped names
`n80_g0/sSEED/PROTOCOL`; missing authentication does not silently go offline.

## Prepare / verify / launch

From `~/cot-navigation`:

```bash
.venv/bin/python research/fixed_navigation_hard_v7_20261026/worlds.py
.venv/bin/python research/fixed_navigation_hard_v7_20261026/prepare.py --repo-root "$PWD" --mode pilot
.venv/bin/python research/fixed_navigation_hard_v7_20261026/check.py --repo-root "$PWD"
.venv/bin/python research/fixed_navigation_hard_v7_20261026/preflight.py --repo-root "$PWD"
```

Readiness includes dataset hashes, independent BFS/topological-DP agreement,
endpoint/query holdouts, complete reference replay, identical query schedules,
and independent logical-loss/gradient comparisons. Preparation refuses to
replace data once formal jobs have started. Source hashes protect resumes.

```bash
.venv/bin/python research/fixed_navigation_hard_v7_20261026/start.py --repo-root "$PWD" --gpus 0 1 2 3 4 5 6 7 --execute
```

Without `--execute`, print the 16-job plan only. The detached scheduler creates
its output directory before opening logs and fills idle GPUs without stopping
existing work. It resumes matching checkpoints, rejects ambiguous partial
directories, and stops new dispatch on worker failure. Monitor
`outputs/fixed_navigation_hard_v7/pilot/status.json` and `launch.log`.
The memory-only `train_protocol.py --check` writes no run, checkpoint or W&B run.
