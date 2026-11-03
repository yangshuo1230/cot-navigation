# Fixed DAG navigation v5

Remote directory: `~/cot-navigation/research/fixed_navigation_v5_20261026`.
The approved design is `FIXED_DAG_NAVIGATION_SPLIT_V5_20261026.md`.

Current training: **PATH only, random initialization, one joint training stage**.
Every batch contains **48 short +16 long QUERY** and16 complete ADJ rows.
There is no knowledge warmup, parent checkpoint or Direct training job.

## Data

- Short QUERY: gaps 1–3, all nodes, query-only holdout. All edges are in ADJ.
- Long QUERY: gaps 4–9, training endpoints A only. Each layer splits A/V/T=50/25/25%.
- V is validation-only and T test-only for long endpoints. Both remain available in
  short training, ADJ knowledge and intermediate long PATH positions.
- Short and long AA pairs split approximately 80/10/10, stratified by gap/label.
- Negative queries use positive layer-gap strata, not a fictitious shortest path.
- Positive and negative PATH begin source; negatives have legal walks, not empty NO.
- All variants are pre-generated (up to 8/query), checked with BFS + independent DP.
- Graph selection has fixed minimum label counts per panel; exact denominators,
  negative length support shortfall, and potential witness exposure are recorded.
  The pilot is for learning feasibility, not robust cross-graph estimates.

`experiment.json` controls architecture, budgets, mixing, loss weights and tracking.
Changing configuration requires rerunning preparation and verification; changing data
after formal training is refused. Keep a separate versioned suite for changed studies.

## Prepare / verify (no formal training)

From `~/cot-navigation`:

```bash
.venv/bin/python research/fixed_navigation_v5_20261026/data.py
.venv/bin/python research/fixed_navigation_v5_20261026/check.py --repo-root "$PWD"
.venv/bin/python research/fixed_navigation_v5_20261026/preflight.py --repo-root "$PWD"
```

An optional `train.py --check` runs a few memory-only optimizer updates and free
generation. It creates no run directory, checkpoint, training record or W&B run.

## Launch pilot in background

```bash
cd ~/cot-navigation
.venv/bin/python research/fixed_navigation_v5_20261026/start.py --repo-root "$PWD" --mode pilot --gpus 0 1 --execute
```

Directories are created before log redirection. The detached scheduler starts one
four80-node PATH jobs (50k updates each), initialized with seeds0/1/2/3. Short/long QUERY and ADJ
rows are mixed from update1 at48/16/16 rows per batch. Adjacency recovery is a
diagnostic, not a stage-transition gate.
Training and resume are deterministic; counters identify actual query/variant IDs.

The scheduler checks idle GPUs and uses run locks. It resumes existing matching
checkpoints, refuses changed code/data/config, and never kills other workers.
Do not start a second scheduler for the same mode. A failed job without checkpoint
must be inspected; it is not overwritten automatically.

## Formal matrix, after reviewing pilot

```bash
.venv/bin/python research/fixed_navigation_v5_20261026/start.py --repo-root "$PWD" --mode formal --gpus 0 1 2 3 4 5 6 7 --execute
```

Three160-node graphs x4 initializations xPATH=12 independent jobs.
Formal mode never starts automatically when the pilot finishes.

## Outputs and tracking

- Output roots: `outputs/fixed_navigation_v5/{pilot,formal}`.
- Read `status.json`, `launch.log` and per-job logs for progress.
- W&B online: project `cot-graph-reasoning`, group `fixed_navigation_v5`.
- Visible charts: answer/trace/knowledge losses; short, longAA and longVV validation
  BA; PATH valid-goal rate; adjacency exact recovery. Other details stay in JSONL.
- Validation every1000 updates, ADJ recovery every5000. Test is final-only, with
  short/AA/TA/AT/TT reported separately. No test-based model selection.
- Checkpoints: rolling `latest.pt` every10k and on completion; task step20k retained.
  No checkpoint history per validation and no activation dumps.
- `validation.jsonl`, `final.json`, `actual_exposure.json`, `exposure_counts.json`
  are compact reproducible results. Final evaluation uses all unique panel queries;
  large label imbalance is exposed via per-class recall and BA, not hidden by accuracy.

PATH metrics include valid goal paths, legal traces, no trace, wrong start/edge,
wrong branch/stop, passed target and truncation. Potential exposure differs from
actual exposure: only variants supplied to optimizer updates contribute to final
audits. Endpoints held out from QUERY may still have complete witnesses exposed
inside long PATH; exposed/unexposed subgroups are reported, not called equivalent
randomized controls. This design alone cannot establish online graph search.
