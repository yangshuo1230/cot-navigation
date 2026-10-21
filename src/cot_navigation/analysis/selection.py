"""Select checkpoint families using validation only, never test performance."""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import json
import re

import numpy as np
import torch
import yaml

from .common import family


DEFAULT_BATCHES = ("outputs/explore_v1/launches/explore_v1_s012",
                   "outputs/retrain_v3/launches/retrain_v3_s012_50k")


def inventory(root: Path, batches=DEFAULT_BATCHES):
    result = []
    for batch in batches:
        for checkpoint in sorted((root/batch).rglob("best.pt")):
            run = checkpoint.parent.parent
            config = yaml.safe_load((run/"resolved_config.yaml").read_text())
            d, t, m = config["data"], config["train"], config["model"]
            if d["knowledge"] == "context_graph" and (d["n_nodes"], d["n_edges"]) != (16, 24):
                continue
            if d["knowledge"] == "parametric_graph" and (d["n_nodes"], d["n_edges"]) != (32, 48):
                continue
            if t.get("query_hops") == "one_hop" or t["max_steps"] != 50000:
                continue
            if (m["layers"], m["heads"], m["d_model"], m["d_ff"]) != (6, 1, 128, 512):
                continue
            state = torch.load(checkpoint, map_location="cpu", weights_only=False)
            result.append({"run": str(run.relative_to(root)), "protocol": config["protocol"]["name"],
                           "kind": d["knowledge"], "seed": d.get("graph_seed"),
                           "step": int(state["step"]), "score": float(state["best_score"]),
                           "selection_metric": config["evaluation"].get("selection_metric", "answer_accuracy"),
                           "data_root": d["root"]})
    return result


def select(rows):
    groups = defaultdict(list)
    for row in rows:
        variant = re.sub(r"/seed[012]/", "/seedX/", row["run"])
        groups[(row["kind"], family(row["protocol"]), variant)].append(row)
    selected, ranking = [], []
    for kind in ("context_graph", "parametric_graph"):
        for protocol in ("direct", "path", "filler", "dfs", "bfs"):
            options = [(float(np.mean([r["score"] for r in v])), k, v) for k, v in groups.items()
                       if k[:2] == (kind, protocol) and
                       (kind == "context_graph" or {r["seed"] for r in v} == {0, 1, 2})]
            options.sort(key=lambda item: (-item[0], item[1][2]))
            if not options:
                raise ValueError(f"No completed comparable candidate for {kind}/{protocol}")
            for score, key, _ in options:
                ranking.append({"kind": kind, "family": protocol, "variant": key[2], "mean_validation": score})
            for row in sorted(options[0][2], key=lambda r: r["seed"] if r["seed"] is not None else -1):
                prefix = "context" if kind == "context_graph" else f"fixed_seed{row['seed']}"
                selected.append(dict(row, family=protocol, auxiliary=False, id=f"{prefix}_{protocol}"))
            if kind == "context_graph" and protocol == "filler":
                control = [item for item in options if "/train50000/" in item[2][0]["run"]]
                if control and control[0][2][0]["run"] != options[0][2][0]["run"]:
                    selected.append(dict(control[0][2][0], family=protocol, auxiliary=True,
                                         id="context_filler_50k_control"))
            # Best node-only BFS has no visited field. Include its full-state
            # counterpart for explicit VIS interventions, without relabeling it
            # as the best BFS or replacing the main result.
            if protocol == "bfs" and options[0][2][0]["protocol"] == "bfs_nodes":
                states = [item for item in options if item[2][0]["protocol"] == "bfs_state"]
                if states:
                    for row in sorted(states[0][2], key=lambda r: r["seed"]):
                        selected.append(dict(row, family=protocol, auxiliary=True,
                                             id=f"fixed_seed{row['seed']}_bfs_state_control"))
    return {"selection_rule": "validation best checkpoint; fixed variant chosen by equal-weight mean across graph seeds 0/1/2",
            "scope": list(DEFAULT_BATCHES), "context_training_pools": [10000, 50000, 150000],
            "no_test_used_for_selection": True,
            "candidates": ranking, "models": selected}
