"""Reference BFS using the existing PUSH/POP/FOUND/EXHAUST vocabulary.

Step.stack stores the FIFO queue for serialization compatibility. Its head is
the current node. PUSH enqueues one unvisited neighbor, POP removes the head.
Nodes are marked on enqueue, so merges never cause duplicate visits. FOUND
is checked on dequeue/expansion, before expanding the target's neighbors.
"""
from collections import deque
from cot_navigation.protocols.dfs import Step


def bfs_search(adj: list[list[int]], src: int, dst: int) -> list[Step]:
    queue = deque([src])
    visited, order = {src}, [src]
    steps = []
    while queue:
        cur = queue[0]
        cands = tuple(v for v in adj[cur] if v not in visited)
        if cur == dst:
            action, arg = "FOUND", None
        elif cands:
            action, arg = "PUSH", cands[0]
        elif len(queue) > 1:
            action, arg = "POP", None
        else:
            action, arg = "EXHAUST", None
        steps.append(Step(cur, cands, tuple(queue), tuple(order), action, arg))
        if action in {"FOUND", "EXHAUST"}:
            return steps
        if action == "PUSH":
            visited.add(arg)
            order.append(arg)
            queue.append(arg)
        else:
            queue.popleft()
    raise AssertionError("BFS must emit a terminal record")


def first_expansion_order(steps: list[Step]) -> list[int]:
    """First occurrence of each expanded current node; no backtrack repeats."""
    return list(dict.fromkeys(step.cur for step in steps))
