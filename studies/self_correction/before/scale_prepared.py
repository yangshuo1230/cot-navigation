"""Bounded offline randomized CoT variants for the scale experiments.

Legacy PreparedRows is unchanged. Cache identity includes all rendering flags,
the source fingerprint, vocab, context budget and fixed-world prefix reserve.
"""
import hashlib
import json

import numpy as np

from cot_navigation.data.dataset import ContextStore
from cot_navigation.data.prepared import PreparedRows
from cot_navigation.data.provenance import digest
from cot_navigation.data.shards import atomic_json, atomic_savez, file_sha256, is_complete, load_json, load_npz, mark_complete
from cot_navigation.protocols.render import truncate_example
from cot_navigation.training.augmentation import transform


def identity(source, cfg, tok, protocol=None):
    protocol = protocol or cfg.protocol.name
    return dict(**({"fixed_path_policy": "all_paths_v1"} if getattr(getattr(cfg, "data", None), "graph_family", "layered") != "layered" and not isinstance(source, ContextStore) and protocol == "path" else {}),
                source=source.fingerprint, protocol=protocol, vocab=tok.hash(), version=1,
                variants=1 if protocol == 'direct' else cfg.protocol.trajectory_variants,
                max_seq_len=cfg.model.max_seq_len, margin=cfg.protocol.cot_margin,
                prefix_reserve=int(bool(cfg.train.world_roots)),
                random_search=protocol in {'dfs_state', 'bfs_state', 'dfs_nodes', 'bfs_nodes'},
                negative_path=protocol == 'path')


def cache_path(source, spec):
    return source.root / 'prepared' / f"scale_{spec['protocol']}_{digest(spec)[:20]}.npz"


def prepare_variants(source, cfg, tok, protocol):
    spec = identity(source, cfg, tok, protocol)
    path = cache_path(source, spec)
    if is_complete(path) and path.with_suffix('.json').exists() and load_json(path.with_suffix('.json')) == spec:
        return path
    tokens, offsets, fields, metadata, queries = [], [0], [], [], [0]
    count = len(source) if isinstance(source, ContextStore) else len(source.queries['label'])
    # These protocol IDs only seed independent offline priorities.
    seed = int(hashlib.sha256(protocol.encode()).hexdigest()[:8], 16)
    for index in range(count):
        adj = source.adj(index) if isinstance(source, ContextStore) else source.adj()
        nodes = source.node_vocab_ids(index, tok) if isinstance(source, ContextStore) else source.node_vocab_ids(tok)
        arrays = source.arrays if isinstance(source, ContextStore) else source.queries
        s, t = int(arrays['s_topo'][index]), int(arrays['t_topo'][index])
        all_paths = spec.get('fixed_path_policy') and source.path_count(index) > 0
        n_variants = source.path_count(index) if all_paths else spec['variants']
        for variant in range(n_variants):
            rng = np.random.default_rng(seed + index * 104729 + variant)
            if isinstance(source, ContextStore):
                row = source.render(index, protocol, tok)
            else:
                selected = variant if all_paths else int(rng.integers(source.path_count(index))) if protocol == 'path' and source.path_count(index) else 0
                row = source.render_query(index, protocol, tok, path_index=selected)
            row = transform(row, protocol, tok, adj, nodes, s, t,
                            search_rng=rng if spec['random_search'] else None,
                            negative_rng=rng if spec['negative_path'] and row.label == 0 else None)
            row = truncate_example(row, tok, cfg.model.max_seq_len,
                                   cfg.protocol.cot_margin + spec['prefix_reserve'])
            if getattr(getattr(cfg, "data", None), "graph_family", "layered") != "layered" and row.meta.get("truncated", False):
                raise ValueError(f"hard DAG reference exceeds context: {source.root} {protocol} query={index}; increase max_seq_len")
            tokens.extend(row.tokens)
            offsets.append(len(tokens))
            fields.append((row.prompt_len, row.answer_index, row.label, row.filler_tokens))
            meta = {k: v for k, v in row.meta.items() if k not in {'reference_steps', 'node_token', 'full_body'}}
            metadata.append(json.dumps(meta, separators=(',', ':')))
        queries.append(len(fields))
    atomic_savez(path, tokens=np.asarray(tokens, dtype=np.int32), offsets=np.asarray(offsets, dtype=np.int64),
                 fields=np.asarray(fields, dtype=np.int32), metadata=np.asarray(metadata, dtype=str),
                 query_offsets=np.asarray(queries, dtype=np.int64))
    atomic_json(path.with_suffix('.json'), spec)
    mark_complete(path, file_sha256(path))
    return path


class ScalePreparedRows(PreparedRows):
    def __init__(self, source, cfg, tok):
        spec = identity(source, cfg, tok)
        path = cache_path(source, spec)
        if not is_complete(path) or load_json(path.with_suffix('.json')) != spec:
            raise FileNotFoundError(f'missing scale variants at {path}; run scale --prepare-only')
        self.protocol = cfg.protocol.name
        self.arrays = load_npz(path)
