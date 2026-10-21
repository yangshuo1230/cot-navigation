# Frozen navigation-model analysis

This package adds analysis only; it does not modify the model or training code.
The default suite uses validation-selected checkpoints from the completed
`explore_v1_s012` and `retrain_v3_s012_50k` batches, including context training
pools of 10k/50k/150k examples. It keeps all three fixed
worlds. Test performance is never used to select a checkpoint or variant.

```bash
# Plotting dependency (optional for numerical analyses)
uv pip install --python .venv/bin/python matplotlib

# Full suite, or resume missing/outdated analysis stages. Physical GPU 0 only.
.venv/bin/python scripts/analyze_mechanisms.py --gpu 0 --attention-fixed --resume
.venv/bin/python -m cot_navigation.analysis.probe_controls \
  --input research/mechanistic_20261018/results --gpu 0
.venv/bin/python -m cot_navigation.analysis.report \
  --input research/mechanistic_20261018/results
```

`--models context_dfs` limits model selection; `--stages attention probes
patching` selects analyses. Sample counts, probe steps, generation limit, data
root and output folder are CLI arguments. Use a NEW `--output` when changing
sample settings; resume rejects incompatible settings and changed checkpoints.
The default generation limit is the model's native context window. Increasing
an earlier explicit limit safely refreshes only models that hit that limit;
the old cap and outcomes are retained in each model's `short_limit_audit`.
An explicit `--inventory` JSON can select other runs without editing the
analysis methods. The default inventory limits graph sizes and architecture
to comparable runs; change `selection.inventory` or provide an inventory for
a different experiment family.

Reusable components:

- `common.AnalysisModel` adapts checkpoint loading, immutable graph records,
  causal residual capture, and label/distance-stratified query sampling.
- `attention.DistanceBaseline` estimates a separate empirical distance mean
  per layer/head without padding, future tokens or final answers.
- `probes` fits affine heads on frozen late-layer residuals, with input and
  shuffled-label controls. Query-disjoint splits are enforced. Answer probes
  are fit both jointly and separately by progress; structural read positions
  precede the corresponding output field.
- `patching` edits only prefixes and regenerates all suffixes. It records
  matched interventions, paired uncertainty, no-edit replay agreement, state
  repair/revisit behavior and a small number of concrete examples.
- `probe_controls` adds endpoint-only controls and matched pre-DOT readout checks.
- `report` generates five key figures (including the fixed-world attention
  comparison), a CSV summary and a Markdown report.

Results contain JSON summaries, cohort query IDs and at most eight cases per
model. Activations and full attention tensors remain in RAM. No additional
model/probe checkpoints or W&B runs are saved. The analysis stage versions in
the runner must be incremented if a method changes. The current data adapter
targets the project's single-world DAG protocols; new architectures or prompt
formats should implement another adapter rather than altering core methods.

Run pure correctness checks without a GPU or any training:

```bash
CUDA_VISIBLE_DEVICES= .venv/bin/python -m pytest -q tests/test_mechanistic_analysis.py
```
