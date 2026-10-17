"""All-path preparation and training sampler contracts; no model training."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from cot_navigation.config import load_config
from cot_navigation.data.dataset import ParametricWorld
from cot_navigation.data.generate import generate_parametric
from cot_navigation.data.paths import prepare_paths, query_paths
from cot_navigation.data.validate import ValidationError, _validate_paths, validate_parametric
from cot_navigation.protocols.tokenizer import Tokenizer
from cot_navigation.training.trainer import Trainer, TrainError, _arch_snapshot


ROOT = Path(__file__).resolve().parents[1]


def test_all_paths_match_independent_enumeration():
    def reference(adj, src, dst):
        if src == dst:
            return [(src,)]
        return [(src, *tail) for v in adj[src] for tail in reference(adj, v, dst)]

    rng = np.random.default_rng(73)
    for _ in range(12):
        adj = [[v for v in range(u + 1, 8) if rng.random() < .4] for u in range(8)]
        pairs = [(s, t) for s in range(8) for t in range(8)]
        table = prepare_paths(adj, pairs)
        for i, (s, t) in enumerate(pairs):
            actual = [tuple(p) for p in query_paths(table, i)]
            assert actual == reference(adj, s, t)
            assert len(actual) == len(set(actual))


def test_limits_fail_before_enumeration(monkeypatch):
    def forbidden(*args):
        raise AssertionError("must count and reject before enumeration")

    monkeypatch.setattr("cot_navigation.data.paths._paths", forbidden)
    adj = [[1, 2], [3], [3], []]
    for limits in ({"max_paths": 1}, {"max_nodes": 5}):
        with pytest.raises(ValueError, match="no paths were silently omitted"):
            prepare_paths(adj, [(0, 3)], **limits)
    with pytest.raises(ValueError, match="unique forward edges"):
        prepare_paths([[1], [0]], [(0, 1)])


@pytest.fixture
def prepared_world(tmp_path):
    cfg = load_config([ROOT / "configs/experiments/main_stage_b_path.yaml"],
                      overrides=[f"data.root={tmp_path / 'data'}", "data.graph_seeds=[0]",
                                 "data.n_walks=40", "data.min_edge_coverage=2"])
    generate_parametric(cfg.data, resume=False)
    report = validate_parametric(cfg.data)
    world = ParametricWorld(Path(cfg.data.root) / "balanced-seed0")
    assert report["worlds"][0]["prepared_paths"] == world.info["prepared_paths"]
    return cfg, world


def _sampler_trainer(cfg, world):
    # Build just the data/sampling side. Never construct a model or tracker.
    trainer = Trainer.__new__(Trainer)
    trainer.cfg, trainer.world, trainer.store = cfg, world, None
    trainer.tok = Tokenizer(cfg.data.node_vocab)
    trainer.train_ids = world.balanced["train"]
    trainer.render_cache = {}
    trainer.loss_balance_ema = None
    trainer._init_samplers()
    return trainer


def test_random_paths_cache_and_exposure_alignment(prepared_world):
    cfg, world = prepared_world
    path = _sampler_trainer(cfg, world)
    direct_cfg = deepcopy(cfg)
    direct_cfg.protocol.name = "direct"
    direct = _sampler_trainer(direct_cfg, world)
    choices = [int(i) for i in path.train_ids if world.path_count(int(i)) > 1]
    assert choices, "fixture must include a train query with multiple paths"
    index = choices[0]
    counts = np.zeros(world.path_count(index), dtype=int)
    for _ in range(2000):
        row = path._training_task(index)
        j = row.meta["path_index"]
        counts[j] += 1
        nodes = world.path_nodes_at(index, j)
        assert row.tokens[row.prompt_len:row.answer_index] == [
            path.tok.node_id(int(world.node_token_raw[v])) for v in nodes]
    assert np.all(np.abs(counts / counts.sum() - 1 / len(counts)) < .05)
    assert len({tuple(path._render_task(index, path_index=j).tokens) for j in range(len(counts))}) == len(counts)
    # Query/walk selection is unaffected by any number of path draws.
    for _ in range(5):
        assert path.query_sampler.next() == direct.query_sampler.next()
        assert path.walk_sampler.next() == direct.walk_sampler.next()
    state = deepcopy(path.path_rng.bit_generator.state)
    for i in world.balanced["val"]:
        path._render_task(int(i))
    assert path.path_rng.bit_generator.state == state
    held_out = int(np.flatnonzero(world.queries["split"] != 0)[0])
    with pytest.raises(TrainError, match="held-out"):
        path._training_task(held_out)


def test_path_checkpoint_roundtrip(prepared_world, tmp_path):
    cfg, world = prepared_world
    trainer = _sampler_trainer(cfg, world)
    def attach_checkpoint_fields(t):
        empty = lambda: {}
        load = lambda state: None
        t.raw = SimpleNamespace(state_dict=empty, load_state_dict=load)
        t.opt = SimpleNamespace(state={}, state_dict=empty, load_state_dict=load)
        t.schedule = SimpleNamespace(state_dict=empty, load_state_dict=load)
        t.device, t.persist = torch.device("cpu"), False
        t.snapshot = _arch_snapshot(cfg, t.tok, world.fingerprint)
        t.run_id, t.parent_run_id = "sampler-test", None
        t.completed = t.queries_seen = t.walks_seen = t.tokens_seen = 0
        t.best_score = -1.0

    attach_checkpoint_fields(trainer)
    for i in trainer.train_ids:
        trainer._training_task(int(i))
    checkpoint = tmp_path / "sampler.pt"
    torch.save(trainer._payload(), checkpoint)
    def sequence(t):
        rows = []
        for _ in range(4):
            ids = t.train_ids[t.query_sampler.next()]
            rows.extend(tuple(t._training_task(int(i)).tokens) for i in ids)
            rows.append(tuple(t.walk_sampler.next()))
        return rows
    expected = sequence(trainer)
    restored = _sampler_trainer(cfg, world)
    attach_checkpoint_fields(restored)
    restored._load_resume(str(checkpoint))
    assert sequence(restored) == expected


def test_path_validation_rejects_incomplete_or_duplicate(prepared_world):
    _cfg, world = prepared_world
    i = next(i for i in range(len(world.queries["label"])) if world.path_count(i) > 1)
    start = int(world.paths["query_offsets"][i])
    a, b, c = world.paths["path_offsets"][start:start + 3]
    assert b - a == c - b  # Adjacent-layer DAG: equal length paths.
    world.paths["nodes"][b:c] = world.paths["nodes"][a:b]
    with pytest.raises(ValidationError, match="duplicate prepared path"):
        _validate_paths(world, world.adj())


def test_path_validation_rejects_missing_paths(prepared_world):
    _cfg, world = prepared_world
    i = next(i for i in range(len(world.queries["label"])) if world.path_count(i) > 1)
    # Transfer one path to the next query segment, keeping offsets monotonic.
    # Counts must still expose the missing route in the original segment.
    world.paths["query_offsets"][i + 1] -= 1
    with pytest.raises(ValidationError, match="incomplete prepared paths"):
        _validate_paths(world, world.adj())


def test_knowledge_stage_does_not_load_paths(prepared_world):
    _cfg, world = prepared_world
    knowledge = ParametricWorld(world.root, include_queries=False)
    assert knowledge.queries is None and knowledge.paths is None and knowledge.balanced == {}
    with pytest.raises(ValueError, match="knowledge-only"):
        knowledge.path_count(0)
