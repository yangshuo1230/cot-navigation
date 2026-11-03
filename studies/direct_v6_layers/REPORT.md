# v6 Fixed DAG Direct: final results and sublayer answer readouts

One N80 adjacent-layer DAG, four random initializations, final step50000; no test checkpoint selection. Frozen native BF16 inference and all probe fits on GPU0. No saved activations, weights, checkpoints or W&B runs.

Native final BA is computed from full-vocabulary argmax. Intermediate original-head BA compares YES/NO logits after final LayerNorm (uncalibrated logit lens), not causal truncation or proof later layers are unnecessary.

The strict probe fits only original training queries whose two endpoints are A nodes (40); validation uses V/V queries (20 nodes), testing T/T queries (20 nodes). Node sets are disjoint. Main-model training still saw V/T nodes in short QUERY/ADJ. Query-split probes use short+long training queries and short+AA validation; therefore no probe node holdout. Controls shuffle labels separately within each split.

|Panel|n|YES recall|NO recall|BA|
|---|---:|---:|---:|---:|
|short_test|151|84.43%|86.39%|85.41%|
|long_AA_test|31|97.92%|96.43%|97.17%|
|long_TA_test|168|90.48%|97.62%|94.05%|
|long_AT_test|168|89.55%|90.22%|89.88%|
|long_TT_test|84|83.33%|88.89%|86.11%|

|Stage|TT original head|TT query probe|TT node-held probe|
|---|---:|---:|---:|
|E|50.00%|50.00%|50.00%|
|L1A|48.39%|56.29%|58.70%|
|L1F|51.90%|64.47%|62.31%|
|L2A|65.96%|81.65%|74.68%|
|L2F|68.84%|77.61%|70.83%|
|L3A|81.94%|85.19%|80.92%|
|L3F|81.21%|85.99%|79.36%|
|L4A|82.19%|83.04%|83.28%|
|L4F|82.65%|83.28%|84.53%|
|L5A|85.92%|84.87%|86.33%|
|L5F|86.14%|85.06%|86.33%|
|L6A|86.33%|84.60%|87.23%|
|L6F|86.11%|85.67%|87.23%|

Native BF16 maximum BA discrepancy from official report: 0.00000000; observation hooks max logit change: 0.
Shuffled strict-node test mean BA by stage: 50.0%, 50.3%, 50.3%, 49.4%, 52.1%, 51.6%, 51.3%, 51.9%, 51.4%, 51.9%, 51.1%, 50.0%, 51.2%
Four initializations are not four independent graphs. AA has only seven negatives; TT has 57 positives/27 negatives. This v6 design is endpoint-role holdout, not the prior strictly witness-restricted R0/R2/R3 setup.