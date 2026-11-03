"""Offline data generation.

Each ``sample_id`` draws its graph from ``derive_seed(data_seed, split, sample_id, attempt)``.
Changing the worker count or restarting a finished shard does not change that
sample. Previous splits' structure hashes are forbidden so a relabeling of the
same pre-rename edge set cannot cross splits. The hash is not an isomorphism
certificate.
"""

from __future__ import annotations

import os
from collections import defaultdict
from pathlib import Path

import numpy as np

from cot_navigation.config import DataConfig, to_plain
from cot_navigation.data.graph import (
    adj_from_presentation,
    edges_of,
    sample_edges,
    shuffle_neighbors,
    structure_hash,
)
from cot_navigation.data.reachability import build_reach, pool_index, reaches, sample_pair
from cot_navigation.data.shards import SCHEMA_VERSION, atomic_json, atomic_savez, is_complete, load_json, load_npz, mark_complete
from cot_navigation.data.splits import balanced_ids, iter_sample_plan, materialize_split_pairs
from cot_navigation.data.walks import generate_walks, subpath_set
from cot_navigation.data.provenance import prepare_generation
from cot_navigation.data.paths import PATH_TABLE_VERSION, prepare_paths, query_paths
from cot_navigation.data.shards import file_sha256
from cot_navigation.protocols.dfs import dfs_search, dfs_summary
from cot_navigation.seeds import make_rng

SPLIT_CODE = {"train": 1, "val": 2, "test": 3}


class GenerationError(RuntimeError):
    pass


def _empty_stats():
    return {"attempts": 0, "accepted": 0, "empty_pool": 0, "hash_collision": 0}


def generate_context(cfg: DataConfig, resume: bool = True) -> dict:
    generation = prepare_generation(cfg)
    root = Path(cfg.root)
    root.mkdir(parents=True, exist_ok=True)
    shape_names = list(cfg.shapes)
    forbidden: set[int] = set()
    split_info = {}
    order = ("val", "test", "train") if cfg.context_holdout_first else ("train", "val", "test")
    for split in order:
        info = _generate_context_split(cfg, split, shape_names, forbidden, resume)
        split_info[split] = info
        forbidden |= set(info["hashes"])
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "kind": "context_graph",
        "generation_fingerprint": generation["fingerprint"],
        "config": to_plain(cfg),
        "shape_names": shape_names,
        "splits": {name: {k: v for k, v in info.items() if k != "hashes"} for name, info in split_info.items()},
        "n_forbidden_hashes": len(forbidden),
    }
    atomic_json(root / "manifest.json", manifest)
    mark_complete(root / "manifest.json", file_sha256(root / "manifest.json"))
    return manifest


def _generate_context_split(cfg: DataConfig, split: str, shape_names: list[str], forbidden: set[int], resume: bool) -> dict:
    root = Path(cfg.root) / split
    root.mkdir(parents=True, exist_ok=True)
    per_shape = {"train": cfg.splits.train, "val": cfg.splits.val, "test": cfg.splits.test}[split] // len(shape_names)
    layers = {name: cfg.shapes[name].layers for name in shape_names}
    plan = list(iter_sample_plan(shape_names, layers, per_shape))
    chunks = [plan[i : i + cfg.chunk_size] for i in range(0, len(plan), cfg.chunk_size)]
    payloads = []
    for chunk in chunks:
        start = chunk[0][0]
        path = root / f"shard-{start:08d}.npz"
        payloads.append(
            {
                "path": str(path),
                "split": split,
                "chunk": chunk,
                "n_nodes": cfg.n_nodes,
                "n_edges": cfg.n_edges,
        "graph_family": cfg.graph_family,
                "node_vocab": cfg.node_vocab,
                "data_seed": cfg.data_seed,
                "max_attempts": cfg.max_attempts,
                "shapes": {name: (cfg.shapes[name].layers, cfg.shapes[name].width) for name in shape_names},
                "shape_names": shape_names,
                "forbidden": forbidden,
            }
        )
    pending = [p for p in payloads if not (resume and is_complete(Path(p["path"])))]
    if pending:
        if cfg.workers <= 1:
            for payload in pending:
                _context_shard_worker(payload)
        else:
            import multiprocessing as mp

            # spawn: the CLI imports torch, so fork would copy a CUDA context.
            # Workers receive only the small plan payload, not the graphs.
            ctx = mp.get_context("spawn")
            with ctx.Pool(processes=cfg.workers) as pool:
                pool.map(_context_shard_worker, pending)
    _dedupe_split(cfg, split, shape_names, forbidden)
    hashes: list[int] = []
    stats = defaultdict(_empty_stats)
    shards = []
    for payload in payloads:
        path = Path(payload["path"])
        if not is_complete(path):
            raise GenerationError(f"shard missing after generation: {path}")
        arrays = load_npz(path)
        hashes.extend(int(h) for h in arrays["struct_hash"].tolist())
        side = load_json(Path(str(path) + ".stats.json"))
        for key, value in side["by_shape_distance"].items():
            bucket = stats[key]
            for stat_name, stat_value in value.items():
                bucket[stat_name] += stat_value
        shards.append(
            {
                "file": path.name,
                "count": int(arrays["sample_id"].shape[0]),
                "checksum": Path(str(path) + ".complete").read_text().strip(),
            }
        )
    _raise_if_duplicate(hashes, split)
    return {
        "count": len(plan),
        "n_shards": len(shards),
        "shards": shards,
        "acceptance": dict(stats),
        "hashes": hashes,
    }


def _dedupe_split(cfg: DataConfig, split: str, shape_names: list[str], forbidden: set[int]) -> None:
    """Rewrite later sample_ids whose pre-rename edge set collides.

    Repair walks sample ids in order after the parallel shards exist, so the
    result does not depend on which worker wrote which shard. A collision is
    retried inside that sample's own attempt stream; it is not moved to
    another distance bucket.
    """
    root = Path(cfg.root) / split
    paths = sorted(root.glob("shard-*.npz"))
    loaded = []
    for path in paths:
        loaded.append((path, {key: np.array(value, copy=True) for key, value in load_npz(path).items()}))
    seen = set(forbidden)
    payload = {
        "split": split,
        "n_nodes": cfg.n_nodes,
        "n_edges": cfg.n_edges,
        "graph_family": cfg.graph_family,
        "node_vocab": cfg.node_vocab,
        "data_seed": cfg.data_seed,
        "max_attempts": cfg.max_attempts,
        "shapes": {name: (cfg.shapes[name].layers, cfg.shapes[name].width) for name in shape_names},
        "shape_names": shape_names,
        "forbidden": seen,
    }
    dirty = [False] * len(loaded)
    for shard_index, (_path, arrays) in enumerate(loaded):
        for local in range(int(arrays["sample_id"].shape[0])):
            digest = int(arrays["struct_hash"][local])
            if digest not in seen:
                seen.add(digest)
                continue
            plan = (
                int(arrays["sample_id"][local]),
                shape_names[int(arrays["shape_id"][local])],
                int(arrays["distance"][local]),
                int(arrays["label"][local]),
            )
            payload["forbidden"] = set(seen)
            row = _generate_context_row(payload, plan)
            _install_row(arrays, local, row)
            seen.add(int(row["struct_hash"]))
            dirty[shard_index] = True
    for (path, arrays), changed in zip(loaded, dirty):
        if not changed:
            continue
        checksum = atomic_savez(path, **arrays)
        mark_complete(path, checksum)


def _install_row(arrays: dict, index: int, row: dict) -> None:
    arrays["sample_id"][index] = row["sample_id"]
    arrays["shape_id"][index] = row["shape_id"]
    arrays["distance"][index] = row["distance"]
    arrays["label"][index] = row["label"]
    arrays["s_topo"][index] = row["s"]
    arrays["t_topo"][index] = row["t"]
    arrays["struct_hash"][index] = np.uint64(row["struct_hash"])
    arrays["node_token"][index] = np.asarray(row["node_token"], dtype=np.int16)
    arrays["node_order"][index] = np.asarray(row["node_order"], dtype=np.int16)
    arrays["edge_src"][index] = np.asarray(row["edge_src"], dtype=np.int16)
    arrays["edge_dst"][index] = np.asarray(row["edge_dst"], dtype=np.int16)
    arrays["attempts"][index] = row["attempts"]
    arrays["dfs_steps"][index] = row["dfs_steps"]
    arrays["dfs_pops"][index] = row["dfs_pops"]
    arrays["dfs_max_stack"][index] = row["dfs_max_stack"]
    arrays["reach_count"][index] = row["reach_count"]
    for key in ("dfs_trace_tokens", "dfs_visited_nodes", "dfs_branch_records"):
        arrays[key][index] = row[key]


def _raise_if_duplicate(hashes: list[int], split: str) -> None:
    seen: dict[int, int] = {}
    for index, value in enumerate(hashes):
        if value in seen:
            raise GenerationError(
                f"duplicate pre-rename structure hash in split {split}: "
                f"rows {seen[value]} and {index} hash={value}. "
                "This is an exact edge-set collision, not an isomorphism claim. "
                "Increase the graph or lower the sample count."
            )
        seen[value] = index


def _context_shard_worker(payload: dict) -> None:
    path = Path(payload["path"])
    shape_names = payload["shape_names"]
    rows = [_generate_context_row(payload, row) for row in payload["chunk"]]
    n = payload["n_nodes"]
    m = payload["n_edges"]
    arrays = {
        "sample_id": np.asarray([r["sample_id"] for r in rows], dtype=np.int64),
        "shape_id": np.asarray([r["shape_id"] for r in rows], dtype=np.int16),
        "distance": np.asarray([r["distance"] for r in rows], dtype=np.int16),
        "label": np.asarray([r["label"] for r in rows], dtype=np.int8),
        "s_topo": np.asarray([r["s"] for r in rows], dtype=np.int16),
        "t_topo": np.asarray([r["t"] for r in rows], dtype=np.int16),
        "struct_hash": np.asarray([r["struct_hash"] for r in rows], dtype=np.uint64),
        "node_token": np.asarray([r["node_token"] for r in rows], dtype=np.int16),
        "node_order": np.asarray([r["node_order"] for r in rows], dtype=np.int16),
        "edge_src": np.asarray([r["edge_src"] for r in rows], dtype=np.int16).reshape(len(rows), m),
        "edge_dst": np.asarray([r["edge_dst"] for r in rows], dtype=np.int16).reshape(len(rows), m),
        "attempts": np.asarray([r["attempts"] for r in rows], dtype=np.int16),
        "dfs_steps": np.asarray([r["dfs_steps"] for r in rows], dtype=np.int16),
        "dfs_pops": np.asarray([r["dfs_pops"] for r in rows], dtype=np.int16),
        "dfs_trace_tokens": np.asarray([r["dfs_trace_tokens"] for r in rows], dtype=np.int32),
        "dfs_visited_nodes": np.asarray([r["dfs_visited_nodes"] for r in rows], dtype=np.int16),
        "dfs_branch_records": np.asarray([r["dfs_branch_records"] for r in rows], dtype=np.int16),
        "dfs_max_stack": np.asarray([r["dfs_max_stack"] for r in rows], dtype=np.int16),
        "reach_count": np.asarray([r["reach_count"] for r in rows], dtype=np.int16),
    }
    if arrays["node_token"].shape != (len(rows), n):
        raise GenerationError(f"node_token shape {arrays['node_token'].shape}")
    checksum = atomic_savez(path, **arrays)
    by_key: dict[str, dict[str, int]] = defaultdict(_empty_stats)
    for row in rows:
        bucket = by_key[f"{row['shape']}|d={row['distance']}"]
        bucket["attempts"] += row["attempt_events"]["attempts"]
        bucket["accepted"] += 1
        bucket["empty_pool"] += row["attempt_events"]["empty_pool"]
        bucket["hash_collision"] += row["attempt_events"]["hash_collision"]
    atomic_json(Path(str(path) + ".stats.json"), {"by_shape_distance": dict(by_key)})
    mark_complete(path, checksum)


def _generate_context_row(payload: dict, plan_row: tuple) -> dict:
    sample_id, shape, distance, label = plan_row
    layers, width = payload["shapes"][shape]
    n_nodes = payload["n_nodes"]
    n_edges = payload["n_edges"]
    events = _empty_stats()
    forbidden = payload["forbidden"]
    for attempt in range(payload["max_attempts"]):
        events["attempts"] += 1
        rng = make_rng(payload["data_seed"], SPLIT_CODE[payload["split"]], sample_id, attempt, 1)
        edges = sample_edges(layers, width, n_edges, rng, payload.get("graph_family", "layered"))
        identity = structure_hash(layers, width, edges)
        if identity in forbidden:
            events["hash_collision"] += 1
            continue
        present_rng = make_rng(payload["data_seed"], SPLIT_CODE[payload["split"]], sample_id, attempt, 2)
        order = present_rng.permutation(len(edges))
        presented = [edges[int(i)] for i in order]
        adj = adj_from_presentation(n_nodes, presented)
        reach = build_reach(adj)
        _pos_nodes, _c, _p, n_pos = pool_index(reach, width, layers, distance, True)
        _neg_nodes, _c2, _p2, n_neg = pool_index(reach, width, layers, distance, False)
        if n_pos == 0 or n_neg == 0:
            events["empty_pool"] += 1
            continue
        query_rng = make_rng(payload["data_seed"], SPLIT_CODE[payload["split"]], sample_id, attempt, 3)
        src, dst = sample_pair(reach, width, layers, distance, label == 1, query_rng)
        if reaches(reach, src, dst) != (label == 1):
            raise GenerationError(f"sampler/oracle mismatch sample {sample_id}")
        token_rng = make_rng(payload["data_seed"], SPLIT_CODE[payload["split"]], sample_id, attempt, 4)
        raw = token_rng.choice(payload["node_vocab"], size=n_nodes, replace=False)
        # Vocab ids are filled by the dataset using Tokenizer.node_id. Store raw ids.
        node_token = [int(x) for x in raw.tolist()]
        node_order = [int(x) for x in token_rng.permutation(n_nodes).tolist()]
        steps = dfs_search(adj, src, dst)
        expect = "FOUND" if label == 1 else "EXHAUST"
        if steps[-1].action != expect:
            raise GenerationError(
                f"DFS/label mismatch sample {sample_id} shape {shape} d={distance} label={label}"
            )
        summary = dfs_summary(steps)
        return {
            "sample_id": sample_id,
            "shape": shape,
            "shape_id": payload["shape_names"].index(shape),
            "distance": distance,
            "label": label,
            "s": src,
            "t": dst,
            "struct_hash": identity,
            "node_token": node_token,
            "node_order": node_order,
            "edge_src": [u for u, _v in presented],
            "edge_dst": [v for _u, v in presented],
            "attempts": attempt + 1,
            "attempt_events": events,
            "dfs_steps": summary["steps"],
            "dfs_pops": summary["pops"],
            "dfs_max_stack": summary["max_stack"],
            "reach_count": reach[src].bit_count(),
            "dfs_trace_tokens": summary["trace_tokens"],
            "dfs_visited_nodes": summary["visited_nodes"],
            "dfs_branch_records": summary["branch_records"],
        }
    raise GenerationError(
        f"rejected sample_id={sample_id} split={payload['split']} shape={shape} "
        f"d={distance} label={label} after {payload['max_attempts']} attempts "
        f"events={events}. Refusing to switch buckets."
    )


def generate_parametric(cfg: DataConfig, resume: bool = True) -> dict:
    generation = prepare_generation(cfg)
    root = Path(cfg.root)
    root.mkdir(parents=True, exist_ok=True)
    worlds = []
    for shape in cfg.shapes:
        for graph_seed in cfg.graph_seeds:
            worlds.append(_generate_world(cfg, shape, int(graph_seed), resume))
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "kind": "parametric_graph",
        "generation_fingerprint": generation["fingerprint"],
        "config": to_plain(cfg),
        "shape_names": list(cfg.shapes),
        "worlds": worlds,
        "note": (
            "Query splits withhold task supervision. Walks may still contain a "
            "held-out witness of length <= 4. d>=5 cannot be fully shown by those walks. "
            "Rows are not filtered on witness exposure."
        ),
    }
    atomic_json(root / "manifest.json", manifest)
    mark_complete(root / "manifest.json", file_sha256(root / "manifest.json"))
    return manifest


def _generate_world(cfg: DataConfig, shape: str, graph_seed: int, resume: bool) -> dict:
    spec = cfg.shapes[shape]
    world_dir = Path(cfg.root) / f"{shape}-seed{graph_seed}"
    done = world_dir / "world.json"
    if resume and is_complete(done):
        return load_json(done)
    layers, width = spec.layers, spec.width
    n = cfg.n_nodes
    events = _empty_stats()
    accepted = None
    for attempt in range(cfg.max_attempts):
        events["attempts"] += 1
        rng = make_rng(cfg.data_seed, 100, graph_seed, attempt, _shape_code(cfg, shape))
        edges = sample_edges(layers, width, cfg.n_edges, rng, cfg.graph_family)
        adj = adj_from_presentation(n, edges)
        reach = build_reach(adj)
        distance = layers - 1
        _a, _b, _c, n_pos = pool_index(reach, width, layers, distance, True)
        _d, _e, _f, n_neg = pool_index(reach, width, layers, distance, False)
        if n_pos == 0 or n_neg == 0:
            events["empty_pool"] += 1
            continue
        accepted = (edges, adj, reach, attempt + 1)
        break
    if accepted is None:
        raise GenerationError(
            f"fixed graph {shape} seed {graph_seed} has no positive and negative "
            f"pair at d={layers - 1} after {cfg.max_attempts} attempts events={events}"
        )
    edges, adj, reach, attempts_used = accepted
    order_rng = make_rng(cfg.data_seed, 101, graph_seed, _shape_code(cfg, shape))
    adj = shuffle_neighbors(adj, order_rng)
    token_rng = make_rng(cfg.data_seed, 102, graph_seed, _shape_code(cfg, shape))
    node_token = [int(x) for x in token_rng.choice(cfg.node_vocab, size=n, replace=False).tolist()]
    walk_rng = make_rng(cfg.data_seed, 103, graph_seed, _shape_code(cfg, shape))
    walks, counts, supplements = generate_walks(
        adj, cfg.n_walks, walk_rng, cfg.min_edge_coverage, cfg.walk_lengths
    )
    query_rng = make_rng(cfg.data_seed, 104, graph_seed, _shape_code(cfg, shape))
    rows, notes = materialize_split_pairs(reach, width, layers, 2, query_rng, cfg.query_ratios)
    # Append after drawing multi-hop splits; no RNG draws here. The existing
    # graph, walks, multi-hop rows and held-out query identities stay aligned.
    if cfg.include_one_hop_queries:
        rows.extend((0, 1, int(reaches(reach, u, v)) if cfg.graph_family != "layered" else int(v in adj[u]), u, v)
                    for layer in range(layers - 1)
                    for u in range(layer * width, (layer + 1) * width)
                    for v in range((layer + 1) * width, (layer + 2) * width))
    paths = prepare_paths(adj, [(src, dst) for _, _, _, src, dst in rows],
                          max_paths=cfg.max_prepared_paths, max_nodes=cfg.max_prepared_path_nodes)
    subpaths = subpath_set(walks, min_edges=1 if cfg.include_one_hop_queries else 2)
    witness_flags = []
    for q, (split, distance, label, src, dst) in enumerate(rows):
        if label != 1:
            witness_flags.append(-1)
            continue
        if paths["query_offsets"][q + 1] == paths["query_offsets"][q]:
            raise GenerationError(f"positive fixed query without witness {shape} {graph_seed}")
        # Audit any valid witness, not just the first DFS path.
        seen = any(tuple(path) in subpaths for path in query_paths(paths, q))
        if cfg.graph_family == "layered" and distance >= 5 and seen:
            raise GenerationError("d>=5 witness appeared in a walk of at most 4 edges")
        witness_flags.append(int(seen))
    flat: list[int] = []
    offsets = [0]
    supplement_flags = []
    # Supplements were appended and are exactly the one-hop top-ups. Recover
    # the flag by length and position: generate_walks appends them at the end
    # and each has length 2. Count from the end.
    n_random = len(walks) - supplements
    for index, walk in enumerate(walks):
        flat.extend(walk)
        offsets.append(len(flat))
        supplement_flags.append(1 if index >= n_random else 0)
    ptr = [0]
    flat_nbr: list[int] = []
    for nbrs in adj:
        flat_nbr.extend(nbrs)
        ptr.append(len(flat_nbr))
    world_dir.mkdir(parents=True, exist_ok=True)
    atomic_savez(
        world_dir / "graph.npz",
        node_token=np.asarray(node_token, dtype=np.int16),
        nbr_ptr=np.asarray(ptr, dtype=np.int32),
        nbr_flat=np.asarray(flat_nbr, dtype=np.int16),
        struct_hash=np.asarray([structure_hash(layers, width, edges_of(adj))], dtype=np.uint64),
        attempts=np.asarray([attempts_used], dtype=np.int16),
    )
    atomic_savez(
        world_dir / "walks.npz",
        offset=np.asarray(offsets, dtype=np.int64),
        nodes=np.asarray(flat, dtype=np.int16),
        is_supplement=np.asarray(supplement_flags, dtype=np.int8),
    )
    atomic_savez(world_dir / "paths.npz", **paths)
    if rows:
        table = np.asarray(rows, dtype=np.int16)
    else:
        table = np.zeros((0, 5), dtype=np.int16)
    atomic_savez(
        world_dir / "queries.npz",
        split=table[:, 0] if len(table) else np.zeros(0, dtype=np.int16),
        distance=table[:, 1] if len(table) else np.zeros(0, dtype=np.int16),
        label=table[:, 2] if len(table) else np.zeros(0, dtype=np.int16),
        s_topo=table[:, 3] if len(table) else np.zeros(0, dtype=np.int16),
        t_topo=table[:, 4] if len(table) else np.zeros(0, dtype=np.int16),
        witness_seen=np.asarray(witness_flags, dtype=np.int8),
    )
    balance_notes = {}
    for split_id, split_name in ((0, "train"), (1, "val"), (2, "test")):
        mask = table[:, 0] == split_id if len(table) else np.zeros(0, dtype=bool)
        # Preserve the legacy multi-hop balanced subset exactly. One-hop rows
        # are sampled separately by the trainer, with all 112 rows available.
        idx = np.flatnonzero(mask & (table[:, 1] >= 2))
        if idx.size == 0:
            chosen = np.zeros(0, dtype=np.int64)
        else:
            local = balanced_ids(table[idx, 1], table[idx, 2], make_rng(cfg.data_seed, 105, graph_seed, split_id))
            chosen = idx[local] if local.size else np.zeros(0, dtype=np.int64)
        np.save(world_dir / f"balanced_{split_name}.npy", chosen)
        balance_notes[split_name] = {"full": int(mask.sum()), "balanced": int(chosen.size)}
    min_count = min(counts.values()) if counts else 0
    info = {
        "shape": shape,
        "graph_seed": graph_seed,
        "dir": world_dir.name,
        "attempts": attempts_used,
        "events": events,
        "n_walks": len(walks),
        "n_supplements": supplements,
        "min_edge_count": int(min_count),
        "n_queries": len(rows),
        "one_hop_queries": int(sum(row[1] == 1 for row in rows)),
        "path_table_version": PATH_TABLE_VERSION,
        "prepared_paths": {"total": int(paths["path_offsets"].size - 1),
                           "node_entries": int(paths["nodes"].size),
                           "max_per_query": int(np.diff(paths["query_offsets"]).max(initial=0))},
        "split_buckets": notes,
        "balanced": balance_notes,
        "witness_positive": int(sum(1 for flag in witness_flags if flag == 1)),
        "witness_positive_d_ge_5": int(
            sum(1 for row, flag in zip(rows, witness_flags) if flag == 1 and row[1] >= 5)
        ),
        "positive_d_ge_5": int(sum(1 for row in rows if row[2] == 1 and row[1] >= 5)),
        "struct_hash": int(structure_hash(layers, width, edges_of(adj))),
    }
    atomic_json(done, info)
    mark_complete(done, file_sha256(done))
    return info


def _shape_code(cfg: DataConfig, shape: str) -> int:
    return list(cfg.shapes).index(shape) + 1


def load_manifest(root: str | os.PathLike) -> dict:
    path = Path(root) / "manifest.json"
    if not is_complete(path):
        raise GenerationError(f"manifest is incomplete at {path}")
    return load_json(path)
