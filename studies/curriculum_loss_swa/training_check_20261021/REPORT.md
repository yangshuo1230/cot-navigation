# Curriculum DFS loss/SWA training snapshot

Three full-attention runs completed 200k updates. SWA-1 and SWA-2 are still near 100.1k; active compilation/autotuning logs do not indicate a process crash. No runs were interrupted.

Answer accuracy below is measured on the native final evaluation: 256 queries per graph condition, 128 reachable and 128 unreachable. This differs from the earlier stratified diagnostic panel. Best checkpoints are selected on validation; last means 200k.

## best

Cells: overall / reachable / unreachable accuracy (%).

| Graph | lambda=1 | lambda=4 | lambda=16 |
|---|---|---|---|
| n16 | 99.2 / 98.4 / 100.0 | 97.7 / 95.3 / 100.0 | 97.3 / 95.3 / 99.2 |
| n24 | 95.7 / 91.4 / 100.0 | 96.1 / 93.0 / 99.2 | 94.9 / 89.8 / 100.0 |
| n32 | 95.3 / 91.4 / 99.2 | 97.3 / 95.3 / 99.2 | 93.8 / 88.3 / 99.2 |
| n36 | 95.7 / 92.2 / 99.2 | 94.9 / 89.8 / 100.0 | 93.8 / 88.3 / 99.2 |
| n64 | 95.3 / 92.2 / 98.4 | 95.3 / 91.4 / 99.2 | 91.4 / 87.5 / 95.3 |
| main | 91.8 / 84.4 / 99.2 | 84.0 / 72.7 / 95.3 | 90.6 / 82.8 / 98.4 |
| deep | 82.4 / 67.2 / 97.7 | 75.0 / 54.7 / 95.3 | 78.1 / 57.8 / 98.4 |
| wide | 89.1 / 82.0 / 96.1 | 89.8 / 81.2 / 98.4 | 89.8 / 81.2 / 98.4 |
| larger | 83.6 / 69.5 / 97.7 | 75.8 / 60.2 / 91.4 | 78.9 / 63.3 / 94.5 |
## last

Cells: overall / reachable / unreachable accuracy (%).

| Graph | lambda=1 | lambda=4 | lambda=16 |
|---|---|---|---|
| n16 | 98.4 / 96.9 / 100.0 | 96.1 / 92.2 / 100.0 | 96.9 / 94.5 / 99.2 |
| n24 | 95.7 / 92.2 / 99.2 | 96.9 / 93.8 / 100.0 | 95.7 / 91.4 / 100.0 |
| n32 | 96.5 / 93.8 / 99.2 | 93.4 / 88.3 / 98.4 | 93.8 / 88.3 / 99.2 |
| n36 | 96.9 / 94.5 / 99.2 | 95.7 / 91.4 / 100.0 | 94.9 / 90.6 / 99.2 |
| n64 | 94.5 / 90.6 / 98.4 | 94.9 / 89.8 / 100.0 | 94.5 / 91.4 / 97.7 |
| main | 90.6 / 85.2 / 96.1 | 90.6 / 86.7 / 94.5 | 93.4 / 88.3 / 98.4 |
| deep | 82.8 / 66.4 / 99.2 | 74.2 / 53.9 / 94.5 | 78.5 / 58.6 / 98.4 |
| wide | 92.2 / 85.2 / 99.2 | 88.3 / 78.9 / 97.7 | 92.2 / 86.7 / 97.7 |
| larger | 86.3 / 75.0 / 97.7 | 81.2 / 69.5 / 93.0 | 85.2 / 72.7 / 97.7 |

Best selection used only 16 free-generation validation queries, and all three runs reached a tied 100% score at different steps (155k,150k,160k). The single-seed best-checkpoint comparison is noisy; last-checkpoint results are also included.

Increasing lambda has not consistently improved final answer accuracy. Complete trace legality has not yet been replayed for these new checkpoints, so this check does not establish whether the reasoning trajectories improved.

SWA cannot yet be compared at the final training budget; its current validation also contains only 16 queries. Static-shape kernel compilation/autotuning is a significant runtime cost.
