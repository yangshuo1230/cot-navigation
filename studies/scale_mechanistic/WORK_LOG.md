# Execution record — 2026-10-19

- Scope: 60 task checkpoints from scale_v1_s0. Selection uses each run's validation-best checkpoint; DFS/BFS variant rankings also use validation only.
- Existing graph datasets, cached trajectories and checkpoint files remained read-only. Models use the real prepared references, WORLD prefix and STOP policy.
- Main probe upper bounds: 256 train, 96 validation, 128 test queries; smaller fixed worlds use available unique queries. Error-head natural training/validation generation uses up to32 queries, capped at512 new tokens; test uses the native4096 context.
- Sources v1/v2/v3 differ only in execution placement and decode batch. Initial execution used GPU0; subsequent frozen inference used GPUs1–7 while every affine-head fit remained on physicalGPU0. Decode batch increased from16 to64. Exact historical source snapshots and hashes are retained.
- Completed all main stages: residual probes, certified route blocking for state models, paired query/node interventions for other models, and applicable first-error probes. DFS-node error detection was omitted because the inherited node oracle is BFS-specific.
- Added neighbor-restricted filtering probes for all20 state models. These cannot obtain high scores merely by rejecting non-neighbors; full reference test splits and selected saved natural states are reported separately.
- 1360 certified blocking states. Independent CPU audit additionally checked301 saved cases, query-split membership/disjointness and all60 checkpoint hashes. Previous-best test prediction overlap:3731/3744 matching answers; differences are recorded rather than treated as exact replay.
- Generated only three aggregate plots, one JSON per model, summary CSV, reports, selection/settings, observations, findings and verification. No navigation training, W&B run, activation dump, probe weights or smoke checkpoint was produced.
- Separate pre-existing/user analysis processes were left untouched. All workers started for this analysis finished normally.
