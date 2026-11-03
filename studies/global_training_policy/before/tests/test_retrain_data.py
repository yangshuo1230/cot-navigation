"""Independent oracles for new data coverage, holdouts, sampling and reports."""
import copy
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

from cot_navigation.config import (AppConfig, DataConfig, ShapeSpec, SplitsSpec, ModelConfig,
                                   TrainConfig, RuntimeConfig, EvaluationConfig)
from cot_navigation.data.dataset import ContextStore, ParametricWorld
from cot_navigation.data.generate import generate_context, generate_parametric
from cot_navigation.data.provenance import prepare_generation
from cot_navigation.data.validate import validate_parametric, validate_context, ValidationError
from cot_navigation.training.sampler import FixedQuerySampler
from cot_navigation.training.trainer import Trainer, _arch_snapshot, _check_resume, TrainError
from cot_navigation.protocols.tokenizer import Tokenizer

ROOT = Path(__file__).resolve().parents[1]


def fixed_config(root, one_hop=True):
    return AppConfig(output_dir=str(root / "out"), data=DataConfig(
        knowledge="parametric_graph", n_nodes=32, n_edges=48,
        shapes={"balanced": ShapeSpec(8, 4)}, splits=SplitsSpec(0, 0, 0), root=str(root / "data"),
        graph_seeds=(0, 1, 2), data_seed=0, graph_shape="balanced", graph_seed=0,
        n_walks=64, min_edge_coverage=2, include_one_hop_queries=one_hop, audit_fraction=1.0),
        model=ModelConfig(layers=1, d_model=16, d_ff=32),
        train=TrainConfig(max_steps=2, warmup_steps=0, precision="fp32", stage="A", walk_batch_size=4),
        runtime=RuntimeConfig(device="cpu", fused_adamw="disabled"))


def test_complete_one_hop_preserves_multi_hop_splits_and_legacy_subsets(tmp_path):
    old, new = fixed_config(tmp_path / "old", False), fixed_config(tmp_path / "new")
    generate_parametric(old.data)
    generate_parametric(new.data)
    report = validate_parametric(new.data)
    assert all(w["one_hop_train"] == 112 for w in report["worlds"])
    for seed, raw_train in enumerate((164, 166, 166)):
        a = ParametricWorld(Path(old.data.root) / f"balanced-seed{seed}")
        b = ParametricWorld(Path(new.data.root) / f"balanced-seed{seed}")
        assert np.array_equal(a.nbr_flat, b.nbr_flat)
        assert np.array_equal(a.walk_nodes, b.walk_nodes)
        for key in a.queries:
            assert np.array_equal(a.queries[key], b.queries[key][:336])
        for split in ("train", "val", "test"):
            assert np.array_equal(a.balanced[split], b.balanced[split])
        assert len(b.indices("train", "full")) == raw_train + 112
        ids = b.indices("train", "one_hop")
        assert sum(b.queries["label"][ids]) == 48
        assert not len(b.indices("test", "one_hop"))
        # Independent coverage enumeration, not the generation helper.
        pairs = {(int(b.queries["s_topo"][i]), int(b.queries["t_topo"][i])) for i in ids}
        expected = {(u, v) for u in range(28) for v in range(32) if v // 4 == u // 4 + 1}
        assert pairs == expected
    # A semantic change must never relabel an existing dataset root.
    old.data.include_one_hop_queries = True
    with pytest.raises(ValueError, match="configuration mismatch"):
        prepare_generation(old.data)


def test_query_sampler_balances_without_losing_items_and_resumes():
    distances = np.array([1]*7 + [2]*13)
    labels = np.array([0]*5 + [1]*2 + [0]*3 + [1]*10)
    sampler = FixedQuerySampler(distances, labels, 64, 11, .25, True)
    seen = set()
    for _ in range(7):
        ids = sampler.next()
        seen.update(ids)
        assert len(ids) == 64 and sum(distances[ids] == 1) == 16
        for d, count in [(1, 8), (2, 24)]:
            assert sum((distances[ids] == d) & (labels[ids] == 1)) == count
    assert seen == set(range(20))
    state = copy.deepcopy(sampler.state_dict())
    expected = [sampler.next() for _ in range(10)]
    restored = FixedQuerySampler(distances, labels, 64, 11, .25, True)
    restored.load_state_dict(state)
    assert expected == [restored.next() for _ in range(10)]
    fractional = FixedQuerySampler(distances, labels, 3, 11, .25, True)
    assert sum(sum(distances[fractional.next()] == 1) for _ in range(8)) == 6


def test_context_size_sweep_shares_exact_holdouts(tmp_path):
    stores = []
    for n in (36, 72):
        cfg = fixed_config(tmp_path / str(n))
        cfg.data.knowledge = "context_graph"
        cfg.data.n_nodes, cfg.data.n_edges = 16, 24
        cfg.data.shapes = {"balanced": ShapeSpec(4, 4)}
        cfg.data.splits = SplitsSpec(n, 18, 18)
        cfg.data.context_holdout_first = True
        cfg.data.include_one_hop_queries = False
        generate_context(cfg.data)
        validate_context(cfg.data)
        stores.append(ContextStore(cfg.data.root))
    a, b = stores
    for split in ("val", "test"):
        for key in a.arrays:
            assert np.array_equal(a.arrays[key][a.indices(split)], b.arrays[key][b.indices(split)])


def test_actual_training_pool_and_new_sampler_resume_guard(tmp_path):
    cfg = fixed_config(tmp_path)
    cfg.data.graph_seeds = (0,)
    generate_parametric(cfg.data)
    cfg.train.persist_outputs = False
    a = Trainer(cfg)
    initial = a.initial_state()
    cfg.train.stage = "B"
    cfg.train.fixed_query_pool = "full"
    cfg.train.balance_query_labels = True
    cfg.train.one_hop_fraction = .25
    b = Trainer(cfg, init_state=initial)
    assert len(b.train_ids) == 276
    assert set(b.train_ids) == set(b.world.indices("train"))
    ids = b.train_ids[b.query_sampler.next()]
    assert sum(b.world.queries["distance"][ids] == 1) == 16
    snap = dict(b.snapshot)
    cfg.train.one_hop_fraction = .5
    with pytest.raises(TrainError, match="one_hop_fraction"):
        _check_resume(snap, _arch_snapshot(cfg, Tokenizer(256), b.manifest_hash))


def test_final_test_reports_keep_tags_steps_predictions_and_last_weights(tmp_path):
    cfg = fixed_config(tmp_path)
    cfg.data.knowledge = "context_graph"
    cfg.data.n_nodes, cfg.data.n_edges = 16, 24
    cfg.data.shapes = {"balanced": ShapeSpec(4, 4)}
    cfg.data.splits = SplitsSpec(12, 8, 8)
    cfg.data.include_one_hop_queries = False
    cfg.train.stage = "context"
    cfg.train.effective_query_batch_size = 4
    cfg.train.eval_interval = cfg.train.save_interval = 1
    cfg.train.free_gen_samples = cfg.train.eval_samples = 8
    cfg.evaluation.final_checkpoints = ("best", "last")
    generate_context(cfg.data)
    tr = Trainer(cfg)
    tr.fit()
    for tag in ("best", "last"):
        p = Path(cfg.output_dir) / "eval" / f"free_test_{tag}.json"
        report = json.loads(p.read_text())
        assert report["checkpoint_tag"] == tag and report["overall"]["n"] == 8
        predictions = [json.loads(x) for x in p.with_name(f"predictions_test_{tag}.jsonl").read_text().splitlines()]
        assert len(predictions) == 8 and len({p["query_id"] for p in predictions}) == 8
        assert set(p["query_id"] for p in predictions).isdisjoint(tr.train_ids)
    last = torch.load(Path(cfg.output_dir) / "checkpoints/latest.pt", weights_only=False, map_location="cpu")
    assert all(torch.equal(v.cpu(), last["model"][k]) for k,v in tr.raw.state_dict().items())
    cfg.train.persist_outputs = False
    evaluation = Trainer(cfg, evaluation_checkpoint=str(Path(cfg.output_dir) / "checkpoints/best.pt"))
    assert evaluation.completed == json.loads((Path(cfg.output_dir) / "eval/free_test_best.json").read_text())["checkpoint_step"]


def test_fixed_path_new_sampling_resume_matches_uninterrupted_weights(tmp_path):
    cfg = fixed_config(tmp_path)
    cfg.data.graph_seeds = (0,)
    generate_parametric(cfg.data)
    cfg.train.persist_outputs = False
    initial = Trainer(cfg).initial_state()
    cfg.train.stage = "B"
    cfg.protocol.name = "path"
    cfg.train.max_steps = 4
    cfg.train.effective_query_batch_size = 4
    cfg.train.fixed_query_pool = "full"
    cfg.train.balance_query_labels = True
    cfg.train.one_hop_fraction = .25
    cfg.train.eval_interval = 0
    cfg.train.save_interval = 2
    cfg.train.log_interval = 1
    cfg.train.persist_outputs = True
    cfg.output_dir = str(tmp_path / "full")
    full = Trainer(cfg, init_state=initial)
    full.fit()
    cfg.output_dir = str(tmp_path / "interrupted")
    partial = Trainer(cfg, init_state=initial)
    update = partial._update
    def interrupted(*args, **kwargs):
        if partial.completed == 2:
            raise InterruptedError("simulated stop after saved update")
        return update(*args, **kwargs)
    partial._update = interrupted
    with pytest.raises(InterruptedError):
        partial.fit()
    partial.tracker.finish()
    resumed = Trainer(cfg, resume=str(Path(cfg.output_dir) / "checkpoints/latest.pt"))
    resumed.fit()
    assert resumed.queries_seen == full.queries_seen
    assert all(torch.equal(v, resumed.raw.state_dict()[k]) for k,v in full.raw.state_dict().items())


def suite():
    spec = importlib.util.spec_from_file_location("retrain_suite", ROOT / "scripts/retrain.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_retrain_plan_shared_parents_variants_and_dry_run(tmp_path, monkeypatch):
    m = suite()
    args = m.parse_args(["--run-name", "unit", "--dry-run", "--output-root", str(tmp_path / "absent")])
    output, jobs = m.build_jobs(args)
    assert args.protocols == ["direct", "path", "dfs_state"]
    assert len(jobs) == 21
    assert len({j.depends_on for j in jobs if j.depends_on}) == 3
    for j in jobs:
        cfg = m.load_config([j.config], j.overrides)
        if cfg.train.stage == "B":
            assert cfg.train.fixed_query_pool == "full" and cfg.train.one_hop_fraction == .25
        if cfg.train.stage != "A":
            assert cfg.evaluation.final_checkpoints == ("best", "last")
    monkeypatch.setattr(m.engine, "idle_gpus", lambda: [{"index": 0, "idle": True}])
    assert m.main(["--run-name", "unit", "--dry-run", "--background", "--output-root", str(tmp_path / "absent")]) == 0
    assert not output.parent.exists()
    args = m.parse_args(["--setting", "fixed", "--fixed-variants", *m.VARIANTS, "--run-name", "four"])
    assert len(m.build_jobs(args)[1]) == 39


def test_completion_requires_final_test_not_just_a_finished_checkpoint(tmp_path, monkeypatch):
    m = suite()
    args = m.parse_args(["--run-name", "missing_test", "--setting", "context", "--protocols", "direct",
                         "--context-sizes", "10000", "--output-root", str(tmp_path)])
    _, jobs = m.build_jobs(args)
    job = jobs[0]
    (job.output / "checkpoints").mkdir(parents=True)
    (job.output / "checkpoints/latest.pt").write_bytes(b"mock")
    monkeypatch.setattr(m.engine, "checkpoint_step", lambda p: job.steps)
    assert not m.engine.job_complete(job)
    (job.output / "eval").mkdir()
    for tag in ("best", "last"):
        (job.output / "eval" / f"free_test_{tag}.json").write_text(json.dumps(
            {"training_step": job.steps, "overall": {"n": 10000}}))
    assert m.engine.job_complete(job)
