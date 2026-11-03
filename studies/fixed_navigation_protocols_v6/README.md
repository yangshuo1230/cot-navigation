# Direct / DFS State / DFS Nodes on the v5 navigation split

Status:80-node pilot launched2026-10-26:12jobs,4initializations x3protocols,
scheduled across8GPUs. The160-node formal matrix has not been launched.
Existing v5 PATH source, datasets, checkpoints and results are unchanged.

## Common design

- Exact same80/160-node fixed graphs, random names and query IDs as v5.
- Short1–3hop QUERY: query-only holdout, all nodes and edges available.
- Long4–9hop QUERY: AA training/query holdout, V validation endpoints,
  T test endpoints; V/T remain visible in short training, ADJ and intermediates.
- Random initialization, one joint stage,50k optimizer updates, seeds0/1/2/3.
- Logical batch:48 short+16 long QUERY+16 complete ADJ rows. No prior checkpoint.
- Direct outputsYES/NO. DFS State emits complete `stack | visited | candidates | action [node] ;`
  records thenYES/NO. DFS Nodes emits unique first-expansion nodes thenYES/NO;
  implicit backtracking means consecutive emitted nodes need not be adjacent.
- Eight fixed randomly shuffled neighbor-order variants/query, shared by both DFS
  protocols, randomly selected in training. Candidates have no reachability hint.
- Positive and negative search traces both complete; no STOP/truncation, no oracle
  states supplied during free generation. Primary correctness is final answer;
  strict trajectory legality is scored separately.

## Context and performance

All dimensions remain6layers,1head,hidden128,FFN512,RoPE,tied output head.
Only the nonparametric context capacity differs to fit entire reference traces:

|graph|Direct / Nodes context|State context|max State reference|
|---|---:|---:|---:|
|80 nodes|256|8192|5082tokens|
|160 nodes,graph0|256|16384|15659tokens|
|160 nodes,graph1|256|16384|11264tokens|
|160 nodes,graph2|256|16384|14929tokens|

Prepared uint16 ragged banks remain compact; token IDs and metadata are GPU-resident.
No graph search or trajectory rendering happens in the optimizer loop. Sort each
logical batch by length, split under32768 padded tokens and30% padding target,
then accumulate gradients with GLOBAL query/knowledge denominators. One optimizer
update per logical batch, not per chunk. BF16+SDPA and Inductor-compiled model/loss.
No equal-FLOP or equal-information claim: full DFS reveals substantially more graph
facts and uses many more tokens than PATH/Direct. Report total real/padded tokens.

## Prepare and verify

From `~/cot-navigation`:

```bash
.venv/bin/python research/fixed_navigation_protocols_v6_20261026/prepare.py --repo-root "$PWD" --mode all
.venv/bin/python research/fixed_navigation_protocols_v6_20261026/check.py --repo-root "$PWD"
.venv/bin/python research/fixed_navigation_protocols_v6_20261026/preflight.py --repo-root "$PWD"
```

Preparation checks every variant against the existing independent DFS, then replays
all state fields/actions and node expansions. Runtime checks compare uneven-microbatch
loss AND gradients to an independent per-row objective, verify paired query/ADJ
schedules and test rejection of illegal state/expansion examples.

Optional memory-only GPU check (no run files, checkpoints or W&B):

```bash
.venv/bin/python research/fixed_navigation_protocols_v6_20261026/train_protocol.py --repo-root "$PWD" --protocol dfs_state --gpu 0 --check
```

## Background training commands (not executed by preparation)

80-node pilot:3protocols×4initializations=12jobs, independent single-GPU workers.

```bash
.venv/bin/python research/fixed_navigation_protocols_v6_20261026/start.py --repo-root "$PWD" --mode pilot --gpus 0 1 2 3 4 5 6 7 --execute
```

Three160-node graphs×3protocols×4initializations=36jobs:

```bash
.venv/bin/python research/fixed_navigation_protocols_v6_20261026/start.py --repo-root "$PWD" --mode formal --gpus 0 1 2 3 4 5 6 7 --execute
```

Use `--protocols direct dfs_nodes` to dispatch a subset. Scheduler claims idle GPUs,
never stops other work, uses locks, resumes matching checkpoints and stops dispatch
after a failed worker. Existing incomplete directories without a checkpoint are
not overwritten. Output directories are created before log redirection.

## Evaluation / logging

Every1000updates: fixed up-to32query validation panels for short, longAA and longVV.
This smaller uniform validation panel limits long-State decode overhead; final
tests still use ALL original unique short/AA/TA/AT/TT queries. Other v5 validation
panels remain saved in the underlying dataset but are not repeatedly decoded.
Direct/Nodes decode batch128, State16. Reference context covers all frozen variants,
but incorrect model loops may still hit the cap; truncation is an error, never
silently converted to a correct answer. ADJ recovery every5000updates is diagnostic.

Rolling latest checkpoint every10k and final; fixed step20k retained. Online W&B
project `cot-graph-reasoning`, group `fixed_navigation_protocols_v6`. Keep visible
loss/answer, loss/trace, loss/knowledge,3validationBA and strict-valid curves;
other details stay in compact JSONL. No activation dumps or per-validation weights.

Actual exposures identify query and search-order variant. Full State records expose
stack ancestry; Nodes exposes an expansion order whose discovery-tree relations
require the known graph to reconstruct. Neither treats arbitrary consecutive
VIS/Nodes tokens as edges. `actual_exposure.json` tags semantic-tree relation
exposure and computes macro relation composition distances; terminal labels and
positive QUERY facts are tracked. Endpoint-role holdout alone does not ensure
complete witnesses or useful intermediate states were unseen.
