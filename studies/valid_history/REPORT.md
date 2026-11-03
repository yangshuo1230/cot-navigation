# Native trace replay and legal history pairs

60 validation-selected scale checkpoints; same test query IDs as the previous analysis. Each static scale and curriculum model is reported separately. No new probe fitting or checkpoint edits.

## Full replay

| Model | n | Answer | Answer + valid full trace | Correct answer / invalid trace | Non-edge PUSH query | State jump query | STOP |
|---|---:|---:|---:|---:|---:|---:|---:|
| context_curriculum_bfs_nodes | 128 | 66.4% | 24.2% | 42.2% | 0.0% | 0.0% | 0.0% |
| context_curriculum_bfs_state | 128 | 82.8% | 41.4% | 41.4% | 42.2% | 15.6% | 3.9% |
| context_curriculum_dfs_nodes | 128 | 77.3% | 44.5% | 32.8% | 0.0% | 0.0% | 0.0% |
| context_curriculum_dfs_state | 128 | 82.8% | 49.2% | 33.6% | 48.4% | 14.8% | 3.9% |
| context_curriculum_direct | 128 | 50.8% | 50.8% | 0.0% | 0.0% | 0.0% | 0.0% |
| context_curriculum_path | 128 | 60.2% | 57.0% | 3.1% | 0.0% | 0.0% | 0.0% |
| context_l10_bfs_nodes | 128 | 50.0% | 18.0% | 32.0% | 0.0% | 0.0% | 0.0% |
| context_l10_bfs_state | 128 | 50.8% | 15.6% | 35.2% | 0.0% | 0.0% | 0.0% |
| context_l10_dfs_nodes | 128 | 50.8% | 0.0% | 50.8% | 0.0% | 0.0% | 0.0% |
| context_l10_dfs_state | 128 | 73.4% | 28.9% | 44.5% | 66.4% | 9.4% | 0.0% |
| context_l10_direct | 128 | 50.8% | 50.8% | 0.0% | 0.0% | 0.0% | 0.0% |
| context_l10_path | 128 | 53.1% | 10.2% | 43.0% | 0.0% | 0.0% | 0.0% |
| context_l4_bfs_nodes | 128 | 49.2% | 1.6% | 47.7% | 0.0% | 0.0% | 0.0% |
| context_l4_bfs_state | 128 | 51.6% | 25.8% | 25.8% | 45.3% | 0.0% | 0.0% |
| context_l4_dfs_nodes | 128 | 89.1% | 64.1% | 25.0% | 0.0% | 0.0% | 0.0% |
| context_l4_dfs_state | 128 | 74.2% | 45.3% | 28.9% | 49.2% | 0.8% | 0.0% |
| context_l4_direct | 128 | 49.2% | 49.2% | 0.0% | 0.0% | 0.0% | 0.0% |
| context_l4_path | 128 | 75.8% | 45.3% | 30.5% | 0.0% | 0.0% | 0.0% |
| context_l6_bfs_nodes | 128 | 48.4% | 2.3% | 46.1% | 0.0% | 0.0% | 0.0% |
| context_l6_bfs_state | 128 | 65.6% | 34.4% | 31.2% | 64.1% | 0.0% | 0.0% |
| context_l6_dfs_nodes | 128 | 73.4% | 37.5% | 35.9% | 0.0% | 0.0% | 0.0% |
| context_l6_dfs_state | 128 | 71.1% | 28.1% | 43.0% | 62.5% | 0.8% | 0.0% |
| context_l6_direct | 128 | 49.2% | 49.2% | 0.0% | 0.0% | 0.0% | 0.0% |
| context_l6_path | 128 | 57.0% | 10.9% | 46.1% | 0.0% | 0.0% | 0.0% |
| context_l8_bfs_nodes | 128 | 50.8% | 1.6% | 49.2% | 0.0% | 0.0% | 0.0% |
| context_l8_bfs_state | 128 | 50.0% | 10.9% | 39.1% | 28.1% | 0.0% | 0.0% |
| context_l8_dfs_nodes | 128 | 50.8% | 0.0% | 50.8% | 0.0% | 0.0% | 0.0% |
| context_l8_dfs_state | 128 | 89.8% | 49.2% | 40.6% | 7.8% | 0.8% | 0.0% |
| context_l8_direct | 128 | 50.8% | 50.8% | 0.0% | 0.0% | 0.0% | 0.0% |
| context_l8_path | 128 | 60.2% | 24.2% | 35.9% | 0.0% | 0.0% | 0.0% |
| fixed_curriculum_bfs_nodes | 128 | 89.8% | 58.6% | 31.2% | 0.0% | 0.0% | 0.0% |
| fixed_curriculum_bfs_state | 128 | 81.2% | 55.5% | 25.8% | 2.3% | 8.6% | 32.0% |
| fixed_curriculum_dfs_nodes | 128 | 93.8% | 75.8% | 18.0% | 0.0% | 0.0% | 0.0% |
| fixed_curriculum_dfs_state | 128 | 94.5% | 72.7% | 21.9% | 8.6% | 14.1% | 5.5% |
| fixed_curriculum_direct | 128 | 82.0% | 82.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| fixed_curriculum_path | 128 | 87.5% | 86.7% | 0.8% | 0.0% | 0.0% | 0.0% |
| fixed_l10_bfs_nodes | 128 | 96.1% | 88.3% | 7.8% | 0.0% | 0.0% | 0.0% |
| fixed_l10_bfs_state | 128 | 99.2% | 86.7% | 12.5% | 0.0% | 0.8% | 10.9% |
| fixed_l10_dfs_nodes | 128 | 94.5% | 76.6% | 18.0% | 0.0% | 0.0% | 0.0% |
| fixed_l10_dfs_state | 128 | 98.4% | 87.5% | 10.9% | 1.6% | 3.9% | 7.0% |
| fixed_l10_direct | 128 | 92.2% | 92.2% | 0.0% | 0.0% | 0.0% | 0.0% |
| fixed_l10_path | 128 | 96.1% | 96.1% | 0.0% | 0.0% | 0.0% | 0.0% |
| fixed_l4_bfs_nodes | 19 | 84.2% | 63.2% | 21.1% | 0.0% | 0.0% | 0.0% |
| fixed_l4_bfs_state | 19 | 94.7% | 42.1% | 52.6% | 0.0% | 0.0% | 0.0% |
| fixed_l4_dfs_nodes | 19 | 84.2% | 68.4% | 15.8% | 0.0% | 0.0% | 0.0% |
| fixed_l4_dfs_state | 19 | 84.2% | 52.6% | 31.6% | 5.3% | 10.5% | 0.0% |
| fixed_l4_direct | 19 | 78.9% | 78.9% | 0.0% | 0.0% | 0.0% | 0.0% |
| fixed_l4_path | 19 | 94.7% | 94.7% | 0.0% | 0.0% | 0.0% | 0.0% |
| fixed_l6_bfs_nodes | 114 | 93.0% | 82.5% | 10.5% | 0.0% | 0.0% | 0.0% |
| fixed_l6_bfs_state | 114 | 73.7% | 45.6% | 28.1% | 0.0% | 0.9% | 0.0% |
| fixed_l6_dfs_nodes | 114 | 93.0% | 85.1% | 7.9% | 0.0% | 0.0% | 0.0% |
| fixed_l6_dfs_state | 114 | 78.1% | 64.9% | 13.2% | 2.6% | 3.5% | 0.0% |
| fixed_l6_direct | 114 | 82.5% | 82.5% | 0.0% | 0.0% | 0.0% | 0.0% |
| fixed_l6_path | 114 | 91.2% | 88.6% | 2.6% | 0.0% | 0.0% | 0.0% |
| fixed_l8_bfs_nodes | 128 | 97.7% | 91.4% | 6.2% | 0.0% | 0.0% | 0.0% |
| fixed_l8_bfs_state | 128 | 99.2% | 93.8% | 5.5% | 2.3% | 0.0% | 0.8% |
| fixed_l8_dfs_nodes | 128 | 96.9% | 89.8% | 7.0% | 0.0% | 0.0% | 0.0% |
| fixed_l8_dfs_state | 128 | 96.9% | 89.8% | 7.0% | 3.1% | 3.1% | 0.0% |
| fixed_l8_direct | 128 | 93.0% | 93.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| fixed_l8_path | 128 | 99.2% | 99.2% | 0.0% | 0.0% | 0.0% | 0.0% |

Semantic replay stops at the first error. Non-edge PUSH and state-jump columns are local checks across printed records, including records after the first semantic error; these are not independently valid execution histories. STOP traces count as incomplete, not automatically as algorithm errors. Node-only traces are expansion orders, not edge-by-edge paths. Direct has no intermediate trace.

## Legal history pairs

| Model | Pair type | Match | n | Same length | Candidate set A/B | Legal action A/B | Focus included eligible/visited |
|---|---|---|---:|---:|---:|---:|---:|
| context_curriculum_bfs_state | changed | relaxed_queue | 40 | 40 | 80.0%/87.5% | 95.0%/95.0% | 90.0%/2.5% |
| context_curriculum_bfs_state | same_state | exact_state | 40 | 37 | 92.5%/90.0% | 92.5%/92.5% | — |
| context_curriculum_dfs_state | changed | exact_frontier | 40 | 38 | 82.5%/97.5% | 85.0%/100.0% | 87.5%/0.0% |
| context_curriculum_dfs_state | same_state | exact_state | 40 | 40 | 100.0%/100.0% | 100.0%/100.0% | — |
| context_l10_bfs_state | changed | relaxed_queue | 40 | 40 | 30.0%/55.0% | 32.5%/57.5% | 32.5%/0.0% |
| context_l10_bfs_state | same_state | exact_state | 40 | 37 | 25.0%/25.0% | 27.5%/27.5% | — |
| context_l10_dfs_state | changed | exact_frontier | 40 | 38 | 57.5%/85.0% | 72.5%/92.5% | 62.5%/0.0% |
| context_l10_dfs_state | same_state | exact_state | 40 | 40 | 100.0%/100.0% | 100.0%/100.0% | — |
| context_l4_bfs_state | changed | relaxed_queue | 40 | 40 | 92.5%/95.0% | 97.5%/95.0% | 97.5%/2.5% |
| context_l4_bfs_state | same_state | exact_state | 40 | 22 | 87.5%/85.0% | 92.5%/90.0% | — |
| context_l4_dfs_state | changed | exact_frontier | 40 | 32 | 95.0%/97.5% | 100.0%/97.5% | 97.5%/0.0% |
| context_l4_dfs_state | same_state | exact_state | 40 | 40 | 100.0%/100.0% | 100.0%/100.0% | — |
| context_l6_bfs_state | changed | relaxed_queue | 40 | 40 | 85.0%/90.0% | 87.5%/90.0% | 90.0%/0.0% |
| context_l6_bfs_state | same_state | exact_state | 40 | 34 | 90.0%/87.5% | 92.5%/90.0% | — |
| context_l6_dfs_state | changed | exact_frontier | 40 | 37 | 92.5%/97.5% | 97.5%/100.0% | 95.0%/0.0% |
| context_l6_dfs_state | same_state | exact_state | 40 | 40 | 100.0%/100.0% | 100.0%/100.0% | — |
| context_l8_bfs_state | changed | relaxed_queue | 40 | 40 | 45.0%/62.5% | 45.0%/62.5% | 45.0%/0.0% |
| context_l8_bfs_state | same_state | exact_state | 40 | 34 | 37.5%/37.5% | 37.5%/37.5% | — |
| context_l8_dfs_state | changed | exact_frontier | 40 | 37 | 80.0%/87.5% | 95.0%/95.0% | 90.0%/12.5% |
| context_l8_dfs_state | same_state | exact_state | 40 | 40 | 100.0%/100.0% | 100.0%/100.0% | — |
| fixed_curriculum_bfs_state | changed | relaxed_queue | 40 | 40 | 65.0%/92.5% | 100.0%/100.0% | 87.5%/5.0% |
| fixed_curriculum_bfs_state | same_state | exact_state | 40 | 32 | 100.0%/100.0% | 100.0%/100.0% | — |
| fixed_curriculum_dfs_state | changed | exact_frontier | 40 | 37 | 92.5%/100.0% | 100.0%/100.0% | 95.0%/0.0% |
| fixed_curriculum_dfs_state | same_state | exact_state | 40 | 40 | 92.5%/97.5% | 92.5%/100.0% | — |
| fixed_l10_bfs_state | changed | relaxed_queue | 40 | 40 | 97.5%/100.0% | 100.0%/100.0% | 100.0%/0.0% |
| fixed_l10_bfs_state | same_state | exact_state | 40 | 32 | 100.0%/100.0% | 100.0%/100.0% | — |
| fixed_l10_dfs_state | changed | exact_frontier | 40 | 37 | 100.0%/100.0% | 100.0%/100.0% | 100.0%/0.0% |
| fixed_l10_dfs_state | same_state | exact_state | 40 | 40 | 100.0%/100.0% | 100.0%/100.0% | — |
| fixed_l4_bfs_state | changed | relaxed_queue | 14 | 14 | 100.0%/100.0% | 100.0%/100.0% | 100.0%/0.0% |
| fixed_l4_bfs_state | same_state | exact_state | 12 | 9 | 100.0%/100.0% | 91.7%/83.3% | — |
| fixed_l4_dfs_state | changed | exact_frontier | 14 | 13 | 100.0%/100.0% | 100.0%/100.0% | 100.0%/0.0% |
| fixed_l4_dfs_state | same_state | exact_state | 13 | 13 | 100.0%/100.0% | 100.0%/100.0% | — |
| fixed_l6_bfs_state | changed | relaxed_queue | 40 | 40 | 100.0%/100.0% | 87.5%/100.0% | 100.0%/0.0% |
| fixed_l6_bfs_state | same_state | exact_state | 40 | 23 | 100.0%/100.0% | 97.5%/87.5% | — |
| fixed_l6_dfs_state | changed | exact_frontier | 40 | 40 | 100.0%/100.0% | 100.0%/100.0% | 100.0%/0.0% |
| fixed_l6_dfs_state | same_state | exact_state | 40 | 40 | 100.0%/100.0% | 100.0%/100.0% | — |
| fixed_l8_bfs_state | changed | relaxed_queue | 40 | 40 | 100.0%/100.0% | 100.0%/100.0% | 100.0%/0.0% |
| fixed_l8_bfs_state | same_state | exact_state | 40 | 33 | 100.0%/100.0% | 100.0%/100.0% | — |
| fixed_l8_dfs_state | changed | exact_frontier | 40 | 39 | 100.0%/100.0% | 100.0%/100.0% | 100.0%/0.0% |
| fixed_l8_dfs_state | same_state | exact_state | 40 | 40 | 100.0%/100.0% | 100.0%/100.0% | — |

Both prefixes are independently replay-certified legal histories of the same graph/query, generated from 64 seeded neighbor permutations. Each query contributes at most one pair per type (maximum 40). Selection never uses model outputs. Prefixes end after the VIS separator, before CAND. We score the next-token legal probability mass and free-generate up to 32 tokens to inspect the current candidate/action record. This 32-token local diagnostic is not a final-answer test.

DFS changed pairs have the identical full frontier and current node. A focus successor is unvisited in A and visited in B. Prefix length and VIS count are matched where possible, not universally. Changes include the entire legal history and VIS, so this does not isolate a single internal representation or a single token.

On this layered DAG family, BFS changed pairs require relaxed queue matching: the current node, queue length and VIS count match, but queue contents can differ. These demonstrate response to a valid state/history change and cannot attribute the effect to VIS alone. Same-state controls preserve frontier and VIS set across histories. Candidate filtering success is evidence of local state use, not proof of complete search or planning.

Small fixed graphs have fewer held-out queries (N16: 19; N36: 114); other cells use 128. Results are one graph seed per scale and must not be treated as replicated graph-level confidence estimates. No Filler checkpoint is present in this scale selection.

Raw per-query counts, first errors, limited decoded examples, exact checkpoints/hashes, and all paired measurements are in results/models/*.json. Only two key figures are produced; no activation dumps or probe weights.
