# v6 Direct: A/T node-held-out structural probes

Four final step50000 Direct checkpoints, one N80 graph. A40/T20: only A nodes have long-query endpoint training. Original model saw all nodes in short queries and adjacency knowledge. V is not used for probe fitting; independent within-A/T folds supply validation.

## Design

- Frozen native BF16 QUERY forward. One focal-node vector per stage, averaging over all legal forward query contexts; nodes in the same layer have the same other-endpoint pool. This includes train/held-out query INPUTS, never output labels. Source state is causally independent of later target. Target context average can wash out query-specific information. E is the raw embedding.
- Exclude source nodes in last layer and target nodes in first layer, which have no forward contexts; A36/T18 per role remain. Test focal nodes never enter standardization, fit, ridge or threshold selection. Candidate output identities are fixed all80 nodes; this is not unseen-candidate decoding.
- Three node assignments17/31/47, five rotating folds: three folds train, one val, one test. Each A test fold has7–8 nodes; T3–4. A_matched has identical train/val sample counts to T-only, about10–11 train and3–4 val; A_full has about21–22 train and7–8 val. Matching is on sample counts, not every graph attribute. Pooled fits A+T jointly, then evaluates each group separately.
- Cross-group curves use the same fitted decoder: A->T and T->A. Within-group test focal nodes are also held out. Results averaged within each model first, then across four initializations; fold repetitions do not create new nodes.
- Ancestors, strict reachable descendants, direct successors; BFS independently verifies DP labels. 128->80 affine ridge, FP64 dual solve on GPU0, train-only scaling, validation MSE selection. Predicted scores are not probabilities.
- Main fixed-candidate AUROC compares different held-out focal nodes for a fixed candidate, then averages eligible candidates WITHIN each fold. Scores are never pooled across folds. Restrict to correct layer direction; successors next-layer only. Long metrics additionally restrict layer distance>=4, not a different decoder. Few T test nodes limit eligible candidate support.
- Row AUROC and top-k separately check true members outrank wrong candidates. Top-k uses true set size; empty/all-positive rows excluded. Full set recovery uses val-F1-selected threshold and excludes self, but no layer mask.
- Controls: three fixed random128-dimensional identity vectors, oracle layer one-hot, candidate prior only, and training labels permuted within each layer (validation/test true). Random identity may learn candidate priors, so high row AUC alone is not a node-specific signal.
- Intermediate set readout is not a causal-use test. Higher ancestor AUROC does not by itself show native YES/NO uses ancestors.

## Equal-sample within-group main result

|Role/task|Stage|A AUROC|T AUROC|A long AUROC|T long AUROC|
|---|---|---:|---:|---:|---:|
|target/ancestors|E|0.688|0.739|0.705|0.771|
|target/ancestors|L1A|0.665|0.711|0.686|0.731|
|target/ancestors|L1F|0.779|0.754|0.846|0.746|
|target/ancestors|L2F|0.783|0.759|0.828|0.737|
|target/ancestors|L3F|0.759|0.768|0.805|0.766|
|target/ancestors|L4F|0.760|0.758|0.802|0.748|
|target/ancestors|L5F|0.738|0.673|0.766|0.635|
|target/ancestors|L6F|0.711|0.638|0.728|0.589|
|source/descendants|E|0.697|0.704|0.695|0.676|
|source/descendants|L1A|0.749|0.771|0.829|0.850|
|source/descendants|L1F|0.796|0.820|0.856|0.948|
|source/descendants|L2F|0.737|0.825|0.807|0.869|
|source/descendants|L3F|0.725|0.769|0.779|0.829|
|source/descendants|L4F|0.690|0.780|0.761|0.886|
|source/descendants|L5F|0.707|0.823|0.786|0.915|
|source/descendants|L6F|0.691|0.800|0.763|0.918|

## Common pooled decoder, separate A/T test nodes

|Role/task|Stage|A AUROC|T AUROC|
|---|---|---:|---:|
|target/ancestors|E|0.834|0.793|
|target/ancestors|L1F|0.861|0.791|
|target/ancestors|L3F|0.817|0.807|
|target/ancestors|L6F|0.797|0.720|
|source/descendants|E|0.828|0.832|
|source/descendants|L1F|0.842|0.856|
|source/descendants|L3F|0.810|0.866|
|source/descendants|L6F|0.780|0.870|

## Embedding controls, equal-sample within-group

|Role/task|Group|Embedding column AUC|Random column AUC|Layer column AUC|Shuffled column AUC|Embedding row AUC|Prior row AUC|Top-k recall|Uniform recall|
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
|target/ancestors|A|0.688|0.510|0.626|0.619|0.798|0.797|0.716|0.560|
|target/ancestors|T|0.739|0.522|0.536|0.716|0.740|0.743|0.628|0.497|
|source/descendants|A|0.697|0.555|0.615|0.648|0.799|0.790|0.727|0.536|
|source/descendants|T|0.704|0.456|0.528|0.671|0.862|0.857|0.772|0.570|

## Cross-group transfer (sample-matched A training)

|Role/task|Stage|A->A|A->T|T->T|T->A|
|---|---|---:|---:|---:|---:|
|target/ancestors|E|0.688|0.713|0.739|0.674|
|target/ancestors|L1F|0.779|0.721|0.754|0.736|
|target/ancestors|L3F|0.759|0.736|0.768|0.687|
|target/ancestors|L6F|0.711|0.636|0.638|0.627|
|source/descendants|E|0.697|0.756|0.704|0.718|
|source/descendants|L1F|0.796|0.814|0.820|0.793|
|source/descendants|L3F|0.725|0.761|0.769|0.717|
|source/descendants|L6F|0.691|0.768|0.800|0.708|

## Interpretation and limitations

- T does not lack linearly readable ancestor signal: matched within-group E is T .739 vs A .688, L1F .754 vs .779. A-trained decoder transfers to T at E .713 and L1F .721. Neither a missing representation nor completely incompatible coding is established.
- The SAME pooled decoder reads target ancestors better on A than T after L1F (.861 vs .791), but this gap is not uniform across all four initializations. The matched A/T training experiment gives a much smaller L1F difference. Training support and group structure matter.
- Source descendants remain readable on T, including L1F matched .820 vs A .796; no ancestor-only conclusion. Ancestor/descendant AUCs are different classification problems, not interchangeable information quantities.
- Main fixed-candidate AUC exceeds random-vector and layer-only controls in pooled L1F probes. Nevertheless row ranking can be dominated by candidate priors: at E, matched target ancestor row AUC is A .798 vs prior .797, T .740 vs prior .743. High row AUC does not establish complete node-specific sets.
- Complete nonempty-set recovery is near zero, including pooled target ancestors E 0% and L1F A1.7%/T0%. These probes recover partial ranking signal, not exact stored tables.
- Long-only T AUROC can be very high but is based on only about6-8 eligible candidate columns per fold, each with3-4 test focal nodes; it is unstable and must not be described as near-perfect general decoding. Same-layer shuffled controls are weak for T-only training because many layer groups are singletons; some labels cannot change. Pooled shuffled controls are more informative.
- Context means may hide query-specific representations, and late target states mix source information. Probe failure or decreasing AUC is not evidence that information is absent. None of these tests establish causal mediation of native answers.
Detailed support, row rankings, direct successors, reciprocal role/task probes, full-A fits, exact-set recovery and per-model values are in summary.json and four seed JSONs. Only one key figure, no activation/probe weights/checkpoints or W&B runs.