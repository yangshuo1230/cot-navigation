"""Graph, reachability, walks, quotas, and generation invariants."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from cot_navigation.config import ConfigError, load_config
from cot_navigation.data.generate import generate_context, generate_parametric
from cot_navigation.data.graph import sample_edges, structure_hash
from cot_navigation.data.reachability import bfs_reaches, build_reach, pair_at_rank, pool_index, reaches
from cot_navigation.data.splits import balanced_ids, per_distance_quotas, split_small_bucket
from cot_navigation.data.validate import validate_context, validate_parametric
from cot_navigation.data.walks import generate_walks
from cot_navigation.seeds import derive_seed, make_rng


def test_bitset_matches_bfs_on_a_small_layered_graph():
    rng = make_rng(11, 3)
    edges = sample_edges(4, 3, 8, rng)
    n = 12
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
    reach = build_reach(adj)
    for src in range(n):
        for dst in range(n):
            assert reaches(reach, src, dst) == bfs_reaches(adj, src, dst)


def test_rank_sampling_is_uniform_and_exhaustive():
    adj = [[] for _ in range(6)]
    adj[0] += [2, 3]
    adj[1] += [3]
    reach = build_reach(adj)
    nodes, _counts, prefix, total = pool_index(reach, width=2, n_layers=3, distance=1, positive=True)
    pairs = [pair_at_rank(reach, 2, 1, True, nodes, prefix, rank) for rank in range(total)]
    assert len(pairs) == len(set(pairs)) == total
    assert set(pairs) == {(0, 2), (0, 3), (1, 3)}
    neg_nodes, _c, neg_prefix, neg_total = pool_index(reach, 2, 3, 1, False)
    neg_pairs = [pair_at_rank(reach, 2, 1, False, neg_nodes, neg_prefix, rank) for rank in range(neg_total)]
    assert (1, 2) in neg_pairs
    assert set(pairs).isdisjoint(neg_pairs)


def test_quotas_are_exact_and_balanced():
    quotas = per_distance_quotas(50000, 9)
    assert sum(pos + neg for _d, pos, neg in quotas) == 50000
    assert sum(pos for _d, pos, neg in quotas) == sum(neg for _d, pos, neg in quotas) == 25000
    counts = [pos for _d, pos, neg in quotas]
    assert max(counts) - min(counts) <= 1


def test_small_bucket_prefers_test_and_does_not_copy():
    _train, _val, test, meta = split_small_bucket(2)
    assert meta["insufficient"] and len(test) == 2
    train, val, test, meta = split_small_bucket(10)
    assert (len(train), len(val), len(test)) == (2, 1, 7)
    assert len(set(train.tolist() + val.tolist() + test.tolist())) == 10


def test_edges_are_exactly_m_and_adjacent():
    edges = sample_edges(5, 4, 7, make_rng(4))
    assert len(edges) == len(set(edges)) == 7
    for u, v in edges:
        assert u // 4 + 1 == v // 4


def test_structure_hash_ignores_order_and_is_not_claimed_as_isomorphism():
    assert structure_hash(3, 2, [(0, 2), (1, 3)]) == structure_hash(3, 2, [(1, 3), (0, 2)])
    assert structure_hash(3, 2, [(0, 2)]) != structure_hash(3, 2, [(0, 3)])


def test_walks_are_legal_and_cover_edges():
    adj = [[1, 2], [], [3], []]
    walks, counts, _extra = generate_walks(adj, n_walks=30, rng=make_rng(8), min_coverage=5, lengths=(1, 2, 3))
    for walk in walks:
        assert len(walk) >= 2
        for u, v in zip(walk, walk[1:]):
            assert v in adj[u]
    for u, nbrs in enumerate(adj):
        for v in nbrs:
            assert counts[(u, v)] >= 5


def test_config_rejects_unknown_keys_bad_shapes_and_budget_overflow(tmp_path):
    text = """
output_dir: out
data:
  knowledge: context_graph
  n_nodes: 6
  n_edges: 2
  shapes:
    bad: {layers: 3, width: 3}
  splits: {train: 4, val: 2, test: 2}
  root: data/x
model: {layers: 2, heads: 1, d_model: 32, d_ff: 64, max_seq_len: 64, tie_embeddings: true}
protocol: {name: direct, budgets: [16]}
train: {max_steps: 1, precision: fp32, stage: context}
tracking: {mode: disabled}
runtime: {device: cpu}
"""
    path = tmp_path / "bad-shape.yaml"
    path.write_text(text)
    with pytest.raises(ConfigError, match="layers\\*width"):
        load_config([path])
    path.write_text(text.replace("bad: {layers: 3, width: 3}", "ok: {layers: 3, width: 2}").replace("root: data/x", "root: data/x\n  extra: 1"))
    with pytest.raises(ConfigError, match="unknown"):
        load_config([path])


def _tiny_context_config(root: Path, workers: int) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    path = root / "cfg.yaml"
    data = root / "data"
    path.write_text(
        f"""
output_dir: {root / "out"}
data:
  knowledge: context_graph
  n_nodes: 6
  n_edges: 4
  node_vocab: 32
  shapes:
    balanced: {{layers: 3, width: 2}}
  splits: {{train: 8, val: 4, test: 4}}
  root: {data}
  data_seed: 7
  max_attempts: 300
  workers: {workers}
  chunk_size: 4
  audit_fraction: 1.0
  graph_seeds: [0]
  min_edge_coverage: 1
  n_walks: 10
  walk_lengths: [1, 2]
  query_ratios: [0.2, 0.1, 0.7]
model:
  layers: 2
  heads: 1
  d_model: 32
  d_ff: 64
  max_seq_len: 256
  tie_embeddings: true
  attn_impl: math
protocol: {{name: direct, budgets: [16]}}
train:
  max_steps: 2
  effective_query_batch_size: 4
  precision: fp32
  stage: context
  eval_interval: 0
  save_interval: 2
  log_interval: 1
  warmup_steps: 0
tracking: {{mode: disabled}}
runtime: {{device: cpu, fused_adamw: disabled}}
"""
    )
    return path


def test_worker_count_and_resume_do_not_change_samples(tmp_path: Path):
    cfg1 = load_config([_tiny_context_config(tmp_path / "w1", 1)])
    cfg2 = load_config([_tiny_context_config(tmp_path / "w2", 2)])
    generate_context(cfg1.data, resume=False)
    generate_context(cfg2.data, resume=False)
    from cot_navigation.data.dataset import ContextStore

    a = ContextStore(cfg1.data.root)
    b = ContextStore(cfg2.data.root)
    for key in ("sample_id", "edge_src", "edge_dst", "node_token", "node_order", "s_topo", "t_topo", "label", "struct_hash"):
        assert np.array_equal(a.arrays[key], b.arrays[key])
    # Drop one shard and the manifest, then resume. sample_id content stays the same.
    split = Path(cfg2.data.root) / "train"
    shards = sorted(split.glob("shard-*.npz"))
    shards[-1].unlink()
    Path(str(shards[-1]) + ".complete").unlink()
    Path(str(shards[-1]) + ".stats.json").unlink(missing_ok=True)
    (Path(cfg2.data.root) / "manifest.json").unlink()
    Path(str(Path(cfg2.data.root) / "manifest.json") + ".complete").unlink(missing_ok=True)
    generate_context(cfg2.data, resume=True)
    c = ContextStore(cfg2.data.root)
    assert np.array_equal(a.arrays["edge_src"], c.arrays["edge_src"])
    report = validate_context(cfg1.data)
    assert report["bfs_mismatch"] == 0
    orders = a.arrays["node_order"]
    assert any(not np.array_equal(order, np.arange(order.shape[0])) for order in orders)
    # Same seed parts, same seed, independent of process hash randomization.
    assert derive_seed(1, 2, 3) == derive_seed(1, 2, 3)


def test_parametric_holdout_and_witness_audit(tmp_path: Path):
    root = tmp_path / "para"
    root.mkdir(parents=True, exist_ok=True)
    cfg_path = root / "cfg.yaml"
    cfg_path.write_text(
        f"""
output_dir: {root / "out"}
data:
  knowledge: parametric_graph
  n_nodes: 8
  n_edges: 6
  node_vocab: 16
  shapes:
    balanced: {{layers: 4, width: 2}}
  splits: {{train: 0, val: 0, test: 0}}
  root: {root / "data"}
  data_seed: 3
  graph_seeds: [0, 1]
  max_attempts: 400
  workers: 1
  chunk_size: 8
  audit_fraction: 1.0
  min_edge_coverage: 3
  n_walks: 40
  walk_lengths: [1, 2, 3, 4]
  query_ratios: [0.2, 0.1, 0.7]
  graph_shape: balanced
  graph_seed: 0
model:
  layers: 2
  heads: 1
  d_model: 32
  d_ff: 64
  max_seq_len: 256
  tie_embeddings: true
protocol: {{name: direct, budgets: [16]}}
train: {{max_steps: 1, precision: fp32, stage: A, walk_batch_size: 4}}
tracking: {{mode: disabled}}
runtime: {{device: cpu}}
"""
    )
    cfg = load_config([cfg_path])
    generate_parametric(cfg.data, resume=False)
    report = validate_parametric(cfg.data)
    assert report["worlds"]
    from cot_navigation.data.dataset import ParametricWorld

    world = ParametricWorld(Path(cfg.data.root) / "balanced-seed0")
    splits = [set(), set(), set()]
    for i in range(len(world.queries["s_topo"])):
        pair = (int(world.queries["s_topo"][i]), int(world.queries["t_topo"][i]))
        splits[int(world.queries["split"][i])].add(pair)
        if int(world.queries["distance"][i]) >= 5:
            assert int(world.queries["witness_seen"][i]) != 1
    assert splits[0].isdisjoint(splits[1]) and splits[0].isdisjoint(splits[2])
    distances = world.queries["distance"][world.balanced["train"]]
    labels = world.queries["label"][world.balanced["train"]]
    if len(distances):
        for distance in set(int(d) for d in distances.tolist()):
            pos = int(((distances == distance) & (labels == 1)).sum())
            neg = int(((distances == distance) & (labels == 0)).sum())
            assert pos == neg
    # Rebalancing does not invent duplicate query copies beyond the chosen ids.
    ids = balanced_ids(world.queries["distance"], world.queries["label"], make_rng(1))
    assert len(ids) == len(set(ids.tolist()))
