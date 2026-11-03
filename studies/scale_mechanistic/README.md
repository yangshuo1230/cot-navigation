# Scale checkpoint mechanistic analysis

`scale_followup.py` installs as `src/cot_navigation/analysis/scale_followup.py` in
the existing server project. It reuses the frozen model, linear-head fitting,
first-error pairing, graph oracle and serialization modules. Historical results
and modules are not rewritten. `report.py` and `verify.py` are standalone tools.

Run from `~/cot-navigation`:

```bash
CUDA_VISIBLE_DEVICES=0 WANDB_MODE=disabled .venv/bin/python -u -m cot_navigation.analysis.scale_followup
.venv/bin/python research/scale_mechanistic_20261019/conditional_probe.py
.venv/bin/python research/scale_mechanistic_20261019/report.py research/scale_mechanistic_20261019/results
.venv/bin/python research/scale_mechanistic_20261019/verify.py . research/scale_mechanistic_20261019/results
```

The default scope is 60 validation-selected best checkpoints: two knowledge
settings, four static scales plus curriculum, six available protocols. No filler
model exists in this batch. DFS/BFS node variants remain separate from full-state
variants. This is not a search for whichever checkpoint performs best on test.

Parameters: `--train 256 --val 96 --test 128 --cases 40 --steps 250
--error-train 32`. Small fixed worlds use every available query without copying
examples. Query identities, split isolation, selected training steps and
checkpoint hashes are recorded. Resume skips completed stages; source/settings
changes require a fresh output directory. Independent processes handle disjoint
`--models` lists. `--inference-gpu 1` places frozen forwards on physical GPU1;
affine head fitting remains on physical GPU0. Source hashes and execution
placement are retained when resuming after the device-placement update.

The exact cached variant-0 reference is used, including WORLD identifiers and
STOP handling. Probe training uses reference prefixes. Natural first-error heads
also use up to 32 generated train/validation queries, capped at 512 new tokens;
test generation uses the native context. Activation arrays and head weights stay
in memory. No W&B run, training checkpoint, or bulk rollout cache is written.

Five readouts: answer, graph successors, PUSH-implied visited history before VIS,
the explicit visited field after VIS, and graph successors minus that explicit
field. The last two are specific to full-state protocols. Input controls use
current-node/token embeddings and position/goal-occurrence features; shuffled
labels preserve graph membership and positive counts. Both positive and negative
scores, within-state ranking and complete-set separation are reported.

The additional conditional probe excludes every non-neighbor during both fitting
and scoring, testing visited filtering among real successors. It uses the full
reference test split and a small saved-natural-state case set. That case set is
selected partly for answer changes and is not a representative test population.

Certified visited attacks block every remaining route from the entire frontier,
with or without inserting the goal. Controls include repeated old entries,
matched distinct dead nodes, goal-only, and all previously unvisited graph nodes.
Candidates/actions are regenerated from the edited VIS boundary. Actual full
insertion sets are audited for repeated PUSH, retention, first candidates and
first state transition. Oversize prefixes, STOP and missing answers are recorded.
The graph label remains YES: a counterfactual NO is state obedience, not improved
task accuracy. Historical inconsistency remains a limitation of these edits.

DFS-node error probing is deliberately omitted because the historical node-only
error oracle accepts BFS distance tiers and would mislabel DFS. Other structural
probes and node interventions still cover DFS-node models. Error detection at +1
uses an identical current token in wrong/repaired arms, with a stricter subset
whose repaired next token is legal. STOP is not treated as an error.

Only one JSON per model, three aggregate figures, a CSV, report, findings and
verification are persisted. A few decoded examples document unexpected behavior;
their first 120 continuation tokens are previews, not complete traces.
