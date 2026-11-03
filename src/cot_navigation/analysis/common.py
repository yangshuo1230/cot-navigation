"""Shared checkpoint, graph, trajectory and sampling adapters.

All positions are zero-based INPUT positions. An activation at position i
contains tokens through i and predicts token i+1. Answer tokens are excluded
from every analysis prefix. No activations or full attention maps are saved.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import hashlib
import json

import numpy as np
import torch

from cot_navigation.config import load_config
from cot_navigation.data.dataset import ContextStore, ParametricWorld
from cot_navigation.models.transformer import build_model
from cot_navigation.protocols.render import render_task
from cot_navigation.protocols.tokenizer import Tokenizer
from cot_navigation.training.augmentation import transform


SEED = 20261018
ROLES = ("edge_src", "edge_dst", "query_src", "query_dst", "syntax",
         "frontier", "visited", "candidates", "action", "cot_node", "dot")


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def family(protocol: str) -> str:
    return protocol.split("_")[0] if protocol.startswith(("dfs_", "bfs_")) else protocol


@dataclass
class Example:
    index: int
    split: str
    label: int
    adj: list[list[int]]
    nodes: list[int]
    source: int
    target: int
    row: object

    @property
    def prompt(self):
        return self.row.tokens[:self.row.prompt_len]


class AnalysisModel:
    def __init__(self, entry: dict, root: Path, device: str = "cuda:0"):
        self.entry, self.root = entry, root
        self.run = root / entry["run"]
        self.cfg = load_config([self.run / "resolved_config.yaml"])
        self.tok = Tokenizer.from_config(self.cfg)
        self.protocol = self.cfg.protocol.name
        self.filler = self.cfg.protocol.filler_tokens or self.cfg.data.n_nodes
        self.model = build_model(self.cfg.model, len(self.tok), self.tok.pad).to(device).eval()
        checkpoint = self.run / "checkpoints" / entry.get("checkpoint_name", "best.pt")
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        self.model.load_state_dict(state["model"], strict=True)
        self.model.requires_grad_(False)
        self.device = torch.device(device)
        self.layers = list(range(max(0, self.model.n_layers - 3), self.model.n_layers))
        self.checkpoint = {"sha256": sha256(checkpoint), "step": int(state["step"]),
                           "best_score": float(state.get("best_score",0))}
        if "checkpoint_step" in entry:assert self.checkpoint["step"]==entry["checkpoint_step"]
        del state
        self.context = self.cfg.data.knowledge == "context_graph"
        data_root = root / self.cfg.data.root
        if self.context:
            self.store = ContextStore(data_root)
        else:
            # The trainer's folder layout is the authoritative world resolver.
            candidates = list(data_root.glob(f"*/seed_{self.cfg.data.graph_seed}/world.json"))
            candidates += list(data_root.glob(f"*/seed{self.cfg.data.graph_seed}/world.json"))
            candidates += [p for p in data_root.rglob("world.json")
                           if json.loads(p.read_text())["graph_seed"] == self.cfg.data.graph_seed]
            if not candidates:
                raise FileNotFoundError(f"world seed {self.cfg.data.graph_seed}: {data_root}")
            self.store = ParametricWorld(candidates[0].parent)

    def indices(self, split):
        if self.context:
            return self.store.indices(split)
        view = self.cfg.evaluation.fixed_query_view
        return self.store.indices(split, view=view)

    def sample_ids(self, split: str, count: int) -> list[int]:
        """Label/distance stratified; entire queries stay in one probe split."""
        ids = self.indices(split)
        a = self.store.arrays if self.context else self.store.queries
        buckets = {}
        rng = np.random.default_rng(SEED + {"train": 0, "val": 1, "test": 2}[split])
        for idx in ids:
            buckets.setdefault((int(a["label"][idx]), int(a["distance"][idx])), []).append(int(idx))
        buckets = [list(rng.permutation(v)) for _, v in sorted(buckets.items())]
        chosen = []
        while len(chosen) < min(count, len(ids)):
            for b in buckets:
                if b and len(chosen) < count:
                    chosen.append(int(b.pop()))
        return chosen

    def example(self, index: int, split: str) -> Example:
        a = self.store.arrays if self.context else self.store.queries
        adj = self.store.adj(index) if self.context else self.store.adj()
        nodes = (self.store.node_vocab_ids(index, self.tok) if self.context
                 else self.store.node_vocab_ids(self.tok))
        source, target = int(a["s_topo"][index]), int(a["t_topo"][index])
        if self.context:
            row = self.store.render(index, self.protocol, self.tok, filler_tokens=self.filler)
        else:
            row = self.store.render_query(index, self.protocol, self.tok, filler_tokens=self.filler)
        # Trainer evaluation keeps canonical edges/search order; only negative
        # Path uses a deterministic random path. Online shuffling is TRAIN-only.
        negative = (np.random.default_rng(self.cfg.evaluation.seed + index)
                    if self.cfg.protocol.negative_random_path else None)
        row = transform(row, self.protocol, self.tok, adj, nodes, source, target,
                        negative_rng=negative)
        if self.context:
            # Recover neighbor order from the displayed prompt for diagnostics.
            inverse = {node: k for k, node in enumerate(nodes)}
            adj = [[] for _ in nodes]
            for _, u, v in np.asarray(row.tokens[:row.prompt_len - 4]).reshape(-1, 3):
                adj[inverse[int(u)]].append(inverse[int(v)])
        return Example(index, split, row.label, adj, nodes, source, target, row)

    def examples(self, split, count):
        return [self.example(i, split) for i in self.sample_ids(split, count)]

    @torch.inference_mode()
    def forward(self, sequences: list[list[int]], *, attention=False):
        lengths = [len(s) for s in sequences]
        ids = torch.full((len(sequences), max(lengths)), self.tok.pad, device=self.device, dtype=torch.long)
        for i, s in enumerate(sequences):
            ids[i, :len(s)] = torch.tensor(s, device=self.device)
        capture = {}
        with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
            result = self.model(ids, capture=capture, return_attn=attention)
        if attention:
            logits, weights = result
        else:
            logits, weights = result, None
        return logits, capture["resid"], weights


def records(tokens: list[int], start: int, tok: Tokenizer) -> list[dict]:
    """Parse complete state records while retaining absolute token positions."""
    result, begin = [], start
    for end in range(start, len(tokens)):
        if tokens[end] != tok.id(";"):
            continue
        separators = [j for j in range(begin, end) if tokens[j] == tok.id("|")]
        if len(separators) == 3:
            a, b, c = separators
            state = tokens[begin:a]
            action = tokens[c + 1:end]
            if state and action:
                result.append({"start": begin, "end": end, "separators": separators,
                               "frontier": state, "visited": tokens[a+1:b],
                               "candidates": tokens[b+1:c], "action": action})
        begin = end + 1
    return result


def role_ids(tokens, example: Example, protocol, tok):
    roles = np.full(len(tokens), ROLES.index("syntax"), dtype=np.int64)
    p = example.row.prompt_len
    if p > 4:
        roles[1:p-4:3] = ROLES.index("edge_src")
        roles[2:p-4:3] = ROLES.index("edge_dst")
    roles[p-3], roles[p-2] = ROLES.index("query_src"), ROLES.index("query_dst")
    if protocol == "filler":
        roles[p:] = ROLES.index("dot")
    elif protocol in {"path", "dfs_nodes", "bfs_nodes"}:
        roles[p:] = ROLES.index("cot_node")
    else:
        for record in records(tokens, p, tok):
            a, b, c = record["separators"]
            roles[record["start"]:a] = ROLES.index("frontier")
            roles[a+1:b] = ROLES.index("visited")
            roles[b+1:c] = ROLES.index("candidates")
            roles[c+1:record["end"]] = ROLES.index("action")
    return roles


def anchors(tokens, ex: Example, protocol: str, tok: Tokenizer):
    """Positions used by probes; never select a final answer or terminal action."""
    p = ex.row.prompt_len
    available = list(range(p, len(tokens)))
    terminal = {tok.id("<FOUND>"), tok.id("<EXHAUST>"), tok.yes, tok.no, tok.id("<EOS>")}
    cut = next((i for i in available if tokens[i] in terminal), len(tokens))
    available = [i for i in available if i < cut]
    answer = [("prompt", p-1)]
    if available:
        for name, frac in [("early", .25), ("middle", .5), ("late", .75)]:
            answer.append((name, available[min(len(available)-1, int(frac*(len(available)-1)))]))
    structures = []
    inverse = {token: i for i, token in enumerate(ex.nodes)}
    if protocol in {"dfs_state", "bfs_state"}:
        implied = {ex.nodes[ex.source]}
        for r in records(tokens, p, tok):
            current = r["frontier"][0 if protocol == "bfs_state" else -1]
            if current in inverse and r["separators"][0] < cut:
                structures.append({"pos": r["separators"][0], "current": current,
                                   "successor_pos": r["separators"][1],
                                   "visited": sorted(implied), "stage": "pre_visited"})
            if r["action"][0] == tok.id("<PUSH>") and len(r["action"]) == 2:
                if r["action"][1] in inverse:
                    implied.add(r["action"][1])
    elif protocol in {"path", "dfs_nodes", "bfs_nodes"}:
        seen = set()
        for i in available:
            if tokens[i] in inverse:
                seen.add(tokens[i])
                structures.append({"pos": i, "current": tokens[i], "visited": sorted(seen),
                                   "stage": "node_history"})
    else:
        structures.append({"pos": p-3, "current": ex.nodes[ex.source], "visited": [ex.nodes[ex.source]],
                           "stage": "query_source"})
    # Long trajectories must not dominate a query's probe contribution.
    if len(structures) > 4:
        structures = [structures[i] for i in np.linspace(0, len(structures)-1, 4).astype(int)]
    return answer, structures
