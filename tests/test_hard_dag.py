"""Offline correctness checks; no GPU training, checkpoints, or W&B runs."""
import json
from pathlib import Path

import numpy as np
import pytest

from cot_navigation.config import load_config
from cot_navigation.data.graph import sample_edges, adj_from_presentation
from cot_navigation.data.hard_graph import graph_statistics, shortest_distances
from cot_navigation.data.reachability import build_reach, bfs_reaches, reaches
from cot_navigation.data.prepared import prepare_dataset
from cot_navigation.data.dataset import ContextStore, ParametricWorld
from cot_navigation.data.scale_prepared import prepare_variants, ScalePreparedRows
from cot_navigation.data.validate import validate_context, validate_parametric
from cot_navigation.protocols.tokenizer import Tokenizer

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('size', [4,6,10])
def test_hard_dag_structure_and_independent_oracle(size):
    for seed in range(4):
        edges = sample_edges(size,size,2*size*size,np.random.default_rng(seed),'multiscale_dag_v1')
        assert edges == sample_edges(size,size,2*size*size,np.random.default_rng(seed),'multiscale_dag_v1')
        assert len(edges) == len(set(edges)) == 2*size*size
        assert all(0 <= u < v < size*size for u,v in edges)
        adj = adj_from_presentation(size*size,edges)
        reach = build_reach(adj)
        stats = graph_statistics(adj,size)
        assert stats['within_band'] > 0 and stats['skip_band'] > 0
        assert stats['merging_nodes'] > 0 and stats['branching_nodes'] > 0
        for s in range(size*size):
            d = shortest_distances(adj,s)
            for t in range(size*size):
                assert reaches(reach,s,t) == bfs_reaches(adj,s,t)
                if s != t:
                    assert (d[t] > 0) == reaches(reach,s,t)


def config(tmp_path, kind):
    preset = 'retrain_context' if kind == 'context' else 'retrain_stage_b'
    return load_config([ROOT/f'configs/experiments/{preset}.yaml'],[
        f'data.root={tmp_path/kind}', 'data.graph_family=multiscale_dag_v1',
        'data.n_nodes=16','data.n_edges=32','data.shapes={main: {layers: 4,width: 4}}',
        'data.graph_shape=main','data.graph_seed=0','data.graph_seeds=[0]',
        'data.splits.train=24','data.splits.val=12','data.splits.test=12',
        'data.workers=1','data.n_walks=100','data.min_edge_coverage=2',
        'data.include_one_hop_queries=true','data.query_ratios=[0.7,0.15,0.15]',
        'model.max_seq_len=16384','protocol.cot_margin=8','protocol.trajectory_variants=4',
        'data.audit_fraction=1.0','protocol.name=path','tracking.mode=disabled'])


@pytest.mark.parametrize('kind', ['context','fixed'])
def test_raw_validation_and_every_protocol_cache(tmp_path, kind):
    cfg = config(tmp_path,kind)
    prepare_dataset(cfg,['direct'])
    source = ContextStore(cfg.data.root) if kind == 'context' else ParametricWorld(Path(cfg.data.root)/'main-seed0')
    report = validate_context(cfg.data) if kind == 'context' else validate_parametric(cfg.data)
    tok = Tokenizer.from_config(cfg)
    for protocol in ('direct','path','dfs_state','bfs_state','dfs_nodes','bfs_nodes'):
        cfg.protocol.name = protocol
        path = prepare_variants(source,cfg,tok,protocol)
        cached = ScalePreparedRows(source,cfg,tok)
        assert all(not json.loads(str(x)).get('truncated') for x in cached.arrays['metadata'])
        for i in range(len(source) if kind == 'context' else len(source.queries['label'])):
            for v in range(cached.variants(i)):
                row = cached.row(i,v)
                assert row.tokens[row.answer_index] == tok.id('<YES>' if row.label else '<NO>')
                assert len(row.tokens) < cfg.model.max_seq_len
            if kind == 'fixed' and protocol == 'path' and source.queries['label'][i]:
                assert cached.variants(i) == source.path_count(i)
                for v in range(cached.variants(i)):
                    row = cached.row(i,v)
                    expected = [source.node_vocab_ids(tok)[u] for u in source.path_nodes_at(i,v)]
                    assert row.tokens[row.prompt_len:row.answer_index] == expected
    if kind == 'context':
        assert report['bfs_checked'] == 48
    else:
        assert report['worlds'][0]['bfs_checked'] > 0


def test_default_layered_generation_identity_unchanged(tmp_path):
    from cot_navigation.data.provenance import prepare_generation, digest
    from cot_navigation.config import to_plain
    cfg = config(tmp_path,'context')
    cfg.data.graph_family = 'layered'
    actual = prepare_generation(cfg.data)
    assert 'graph_family' not in actual['semantic']
    expected = {k:v for k,v in actual.items() if k not in ('fingerprint','chunk_size')}
    assert actual['fingerprint'] == digest(expected)


def test_truncation_rejected_for_hard_dag(tmp_path):
    cfg = config(tmp_path,'context')
    prepare_dataset(cfg,['direct'])
    source = ContextStore(cfg.data.root)
    cfg.model.max_seq_len = 120
    with pytest.raises(ValueError,match='exceeds context'):
        prepare_variants(source,cfg,Tokenizer.from_config(cfg),'dfs_state')
