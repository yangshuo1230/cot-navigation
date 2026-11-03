# Correction-model behavior analysis

For new correction-model cohorts, use the versioned standard workflow documented
in `studies/correction_behavior/PROTOCOL.md`. Add model runs to a cohort manifest;
do not fork the old hardcoded correction_reproduction runners. Run the full
behavior modules together on the same natural generations. Preserve checkpoint
hashes, graph/query identities and per-seed results. Mark missing controls and
incomplete jobs explicitly; never substitute a historical shallower baseline.
Analysis runs use GPU0 unless the user gives a different assignment. Navigation
training, artificial-error intervention and probe training are separate studies.

Deprecated cohort: `dfs_depth_rate_20261031` (6/9 layers, 30/60/90%, seeds0/1)
and every derivative under `depth_rate_20261101` were withdrawn by the user on
2026-11-02. Do not use these checkpoints/results/figures as evidence, baselines,
donors or training initialization. Preserve them only for provenance. The shared
behavior workflow remains available for independently validated future cohorts.
