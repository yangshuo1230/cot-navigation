"""Reference DFS used only to synthesize and audit traces.

Neighbor order is the adjacency list order. VIS is first-visit order. CAND is
the unvisited out-neighbors in that order, with no reachability hint. The
search marks a node visited when it is pushed, so merges are not re-entered.

Action priority at the current node, before the action mutates the stack:
FOUND if the current node is the target; else PUSH the first candidate; else
POP when the stack has a parent; else EXHAUST. The start node is visited
before the first record. ``s == t`` is not a generated query; the machine
still emits FOUND immediately if it is asked.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Step:
    cur: int
    cands: tuple[int, ...]
    stack: tuple[int, ...]
    visited: tuple[int, ...]
    action: str
    arg: int | None = None


def dfs_search(adj: list[list[int]], src: int, dst: int) -> list[Step]:
    visited_order = [src]
    visited = {src}
    stack = [src]
    steps: list[Step] = []
    while True:
        cur = stack[-1]
        cands = tuple(v for v in adj[cur] if v not in visited)
        if cur == dst:
            action, arg = "FOUND", None
        elif cands:
            action, arg = "PUSH", cands[0]
        elif len(stack) > 1:
            action, arg = "POP", None
        else:
            action, arg = "EXHAUST", None
        steps.append(
            Step(
                cur=cur,
                cands=cands,
                stack=tuple(stack),
                visited=tuple(visited_order),
                action=action,
                arg=arg,
            )
        )
        if action in {"FOUND", "EXHAUST"}:
            return steps
        if action == "PUSH":
            assert arg is not None
            visited.add(arg)
            visited_order.append(arg)
            stack.append(arg)
        else:
            stack.pop()


def witness_path(steps: list[Step]) -> list[int] | None:
    """The stack at FOUND, which is the first successful DFS path. Not shortest."""
    if not steps or steps[-1].action != "FOUND":
        return None
    return list(steps[-1].stack)


def dfs_summary(steps: list[Step]) -> dict[str, int]:
    return {
        "steps": len(steps),
        "pops": sum(1 for step in steps if step.action == "POP"),
        "max_stack": max((len(step.stack) for step in steps), default=0),
        "found": int(steps[-1].action == "FOUND") if steps else 0,
        "trace_tokens": sum(len(s.stack) + len(s.visited) + len(s.cands) + 5
                            + int(s.arg is not None) for s in steps),
        "visited_nodes": max((len(s.visited) for s in steps), default=0),
        "branch_records": sum(len(s.cands) > 1 for s in steps),
    }
