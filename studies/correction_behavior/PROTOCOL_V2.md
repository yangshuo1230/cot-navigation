# Behavior protocol v2: token-matched final checkpoints

Preserve all v1 graph/query/generation and semantic scoring definitions. v2
changes checkpoint selection only: pin each model's configured final step in
the cohort manifest, verify it equals train.max_steps and the checkpoint step.
Do not select by test scores or truncate all token-matched arms to 200K steps.
0/30/60% arms may have different final steps. Compare within matched depth/seed.

Optional capture_probe_test saves one compressed copy of main-task natural test
tokens for graph-disjoint probes; other graph tokens and activation caches are
not saved. This is an explicit v2 extension, not a claim that v1 saved tokens.
Snapshots are atomic and resumable. Analysis GPU0; frozen navigation weights.

The latest all-position cohort has 12 models (6/9 layers × 0/30/60%, seeds0/1),
and is independent of the withdrawn depth/rate and stopped equal-step cohorts.
Each group has TWO initialization seeds, not four. Source and checkpoint hashes
are retained. The shared natural behavior modules, separate oracle-MISTAKE
intervention, and separate probe/lens studies must identify their own settings.
