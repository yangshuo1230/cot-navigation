"""Regression coverage for the framework audit and ephemeral smoke policy."""

import copy
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
import torch

from cot_navigation.config import (AppConfig, DataConfig, EvaluationConfig, ModelConfig,
                                  ProtocolConfig, RuntimeConfig, ShapeSpec, SplitsSpec, TrainConfig)
from cot_navigation.data.dataset import ContextStore
from cot_navigation.data.generate import generate_context, generate_parametric
from cot_navigation.data.shards import is_complete, load_npz
from cot_navigation.data.splits import iter_sample_plan
from cot_navigation.evaluation.rollout import generate
from cot_navigation.evaluation.sampling import stratified_ids
from cot_navigation.protocols.render import render_task
from cot_navigation.protocols.tokenizer import Tokenizer
from cot_navigation.tracking.logger import Tracker
from cot_navigation.training.trainer import Trainer, TrainError
import cot_navigation.training.trainer as trainer_module


@pytest.mark.parametrize("path", sorted((Path(__file__).resolve().parents[1] / "configs/experiments").glob("*.yaml")))
def test_shipped_experiment_presets_load(path):
    from cot_navigation.config import load_config
    cfg = load_config([path], ["evaluation.batch_size=2"])
    assert cfg.evaluation.batch_size == 2
    if path.name.startswith("smoke_"):
        assert cfg.runtime.smoke and not cfg.train.persist_outputs and cfg.tracking.mode == "disabled"


def tiny_config(root: Path):
    cfg = AppConfig(
        output_dir=str(root / "out"),
        data=DataConfig(knowledge="context_graph", n_nodes=32, n_edges=32,
                        shapes={"balanced": ShapeSpec(8, 4)}, splits=SplitsSpec(16, 8, 8),
                        root=str(root / "data"), node_vocab=64, data_seed=7,
                        workers=1, chunk_size=8, n_walks=32, min_edge_coverage=1),
        model=ModelConfig(layers=2, heads=1, d_model=16, d_ff=32, max_seq_len=4096, attn_impl="math"),
        protocol=ProtocolConfig(name="direct"),
        train=TrainConfig(max_steps=2, effective_query_batch_size=4, walk_batch_size=4,
                          microbatch_size=2, precision="fp32", eval_interval=0,
                          log_interval=1, bucket_multiple=8, warmup_steps=1),
        runtime=RuntimeConfig(device="cpu", fused_adamw="disabled"),
        evaluation=EvaluationConfig(batch_size=2),
    )
    cfg.validate()
    return cfg


def test_direct_free_answer_matches_teacher_forcing_slot(tmp_path):
    cfg = tiny_config(tmp_path)
    generate_context(cfg.data)
    cfg.train.persist_outputs = False
    trainer = Trainer(cfg)
    row = trainer._render_task(0)
    with torch.no_grad():
        logits = trainer.raw(torch.tensor([row.tokens]))
    expected = int(logits[0, row.answer_index - 1].argmax())
    [result] = generate(trainer.raw, [row.tokens[:row.prompt_len]], "direct", trainer.tok)
    assert result["answer_token"] == expected
    assert result["generated"] == [expected]
    assert row.trace_indices == []
    assert not Path(cfg.output_dir).exists()


def test_main_evaluation_subset_covers_all_buckets_and_small_subset_covers_shapes_labels():
    names = ["wide", "balanced", "deep"]
    plan = list(iter_sample_plan(names, {"wide": 5, "balanced": 10, "deep": 20}, 5000))
    shape = [row[1] for row in plan]
    distances = [row[2] for row in plan]
    labels = [row[3] for row in plan]
    ids = np.arange(len(plan))
    selected = stratified_ids(ids, shape, distances, labels, 64, 17)
    assert len({(shape[i], distances[i], labels[i]) for i in selected}) == 64
    small = stratified_ids(ids, shape, distances, labels, 16, 17)
    assert {(shape[i], labels[i]) for i in small} == {(s,y) for s in names for y in (0,1)}
    assert selected == stratified_ids(ids, shape, distances, labels, 64, 17)
    assert len(stratified_ids(ids, shape, distances, labels, 0, 17)) == len(ids)


def test_generation_resume_rejects_changed_seed_without_rewriting_manifest(tmp_path):
    cfg = tiny_config(tmp_path)
    generate_context(cfg.data)
    manifest = Path(cfg.data.root) / "manifest.json"
    old = manifest.read_bytes()
    cfg.data.data_seed += 1
    with pytest.raises(ValueError, match="configuration mismatch"):
        generate_context(cfg.data, resume=True)
    assert manifest.read_bytes() == old


def test_generation_resume_worker_change_keeps_identity_and_corruption_is_detected(tmp_path):
    cfg = tiny_config(tmp_path)
    generate_context(cfg.data)
    before = ContextStore(cfg.data.root)
    cfg.data.workers = 2
    generate_context(cfg.data, resume=True)
    after = ContextStore(cfg.data.root)
    assert before.fingerprint == after.fingerprint
    path = next((Path(cfg.data.root) / "train").glob("*.npz"))
    b = bytearray(path.read_bytes())
    b[-1] ^= 1
    path.write_bytes(b)
    assert not is_complete(path)
    with pytest.raises(FileNotFoundError, match="incomplete shard"):
        ContextStore(cfg.data.root)
    cfg.data.workers = 1
    generate_context(cfg.data, resume=True)
    assert np.array_equal(before.arrays["edge_src"], ContextStore(cfg.data.root).arrays["edge_src"])


def test_same_update_all_metrics_reach_wandb_history(tmp_path, monkeypatch):
    cfg = tiny_config(tmp_path)
    cfg.tracking.mode = "offline"
    monkeypatch.setenv("WANDB_SILENT", "true")
    tracker = Tracker(cfg, "audit-regression", None, {})
    tracker.log(1, {"train_unique_probe": 1.0})
    tracker.log(1, {"teacher_forced_unique_probe": 2.0})
    tracker.log(1, {"free_unique_probe": 3.0})
    tracker.finish()
    streams = list(Path(cfg.output_dir).rglob("*.wandb"))
    assert streams
    raw = b"".join(p.read_bytes() for p in streams)
    for key in (b"train_unique_probe", b"teacher_forced_unique_probe", b"free_unique_probe"):
        assert key in raw
    assert len((Path(cfg.output_dir) / "metrics.jsonl").read_text().splitlines()) == 3


def test_best_checkpoint_uses_free_answer_accuracy_not_gold_trace(tmp_path, monkeypatch):
    cfg = tiny_config(tmp_path)
    generate_context(cfg.data)
    trainer = Trainer(cfg)
    monkeypatch.setattr(trainer, "_teacher_forced", lambda rows: {"overall": {"answer_accuracy": 1.0}})
    monkeypatch.setattr(trainer, "_free", lambda rows: {"overall": {"answer_accuracy": 0.25}})
    trainer._evaluate("teacher_forced")
    assert trainer.best_score == -1 and not (Path(cfg.output_dir) / "checkpoints/best.pt").exists()
    trainer._evaluate("free")
    assert trainer.best_score == 0.25
    assert (Path(cfg.output_dir) / "checkpoints/best.pt").exists()
    trainer.tracker.finish()


def test_full_test_is_batched_independent_of_training_sample_limit(tmp_path, monkeypatch):
    cfg = tiny_config(tmp_path)
    generate_context(cfg.data)
    cfg.train.free_gen_samples = 1
    cfg.train.persist_outputs = False
    trainer = Trainer(cfg)
    batches = []
    def fake(model, prompts, protocol, tok, precision, **kwargs):
        batches.append(len(prompts))
        return [{"pred": 1, "format_fail": False, "generation_limit_reached": False,
                 "trace_tokens": [], "tokens": p + [tok.yes]} for p in prompts]
    monkeypatch.setattr(trainer_module, "generate", fake)
    report = trainer.evaluate_split("test")
    assert report["overall"]["n"] == cfg.data.splits.test
    assert report["unique_queries"] == cfg.data.splits.test
    assert max(batches) <= 2 and len(batches) == 4
    assert report["overall"]["n_yes"] == report["overall"]["n_no"] == 4
    assert not Path(cfg.output_dir).exists()


def test_perfect_dfs_has_correct_field_metrics(tmp_path, monkeypatch):
    cfg = tiny_config(tmp_path)
    generate_context(cfg.data)
    cfg.protocol.name = "dfs_state"
    cfg.train.persist_outputs = False
    trainer = Trainer(cfg)
    row = trainer._render_task(0)
    def perfect(*args, **kwargs):
        return [{"pred": row.label, "format_fail": False, "generation_limit_reached": False,
                 "trace_tokens": row.meta["full_body"], "tokens": row.tokens}]
    monkeypatch.setattr(trainer_module, "generate", perfect)
    report = trainer._free([row])
    assert report["overall"]["answer_accuracy"] == 1
    assert report["trace"]["completed_records_match"] == report["trace"]["completed_records"] > 0
    assert all(v == 1 for v in report["trace"]["field_accuracy"].values())


def test_stage_a_wrong_world_is_rejected_and_in_memory_handoff_works(tmp_path):
    cfg = tiny_config(tmp_path)
    cfg.data.knowledge = "parametric_graph"
    cfg.data.graph_seeds = (0,1)
    cfg.data.graph_shape = "balanced"
    cfg.data.graph_seed = 0
    cfg.train.stage = "A"
    cfg.train.persist_outputs = False
    generate_parametric(cfg.data)
    stage_a = Trainer(cfg)
    initial = stage_a.initial_state()
    other = copy.deepcopy(cfg)
    other.train.stage = "B"
    other.data.graph_seed = 1
    with pytest.raises(TrainError, match="world/provenance mismatch"):
        Trainer(other, init_state=initial)
    correct = copy.deepcopy(cfg)
    correct.train.stage = "B"
    stage_b = Trainer(correct, init_state=initial)
    for key, value in initial["model"].items():
        assert torch.equal(value, stage_b.raw.state_dict()[key])
    assert not Path(cfg.output_dir).exists()


def test_smoke_cannot_enable_persistence_or_wandb(tmp_path):
    cfg = tiny_config(tmp_path)
    generate_context(cfg.data)
    cfg.runtime.smoke = True
    cfg.tracking.mode = "online"
    cfg.tracking.upload_artifacts = True
    cfg.train.persist_outputs = True
    trainer = Trainer(cfg)
    trainer.fit()
    trainer.evaluate_split("test")
    assert not cfg.train.persist_outputs and cfg.tracking.mode == "disabled"
    assert trainer.tracker.run is None
    assert not Path(cfg.output_dir).exists()
    assert not list(tmp_path.rglob("*.pt"))


def test_complete_smoke_cleans_temporary_data(tmp_path):
    from cot_navigation.runtime.smoke import run_smoke
    cfg = tiny_config(tmp_path)
    result = run_smoke(cfg, steps=1)
    assert result["temporary_data_cleaned"]
    assert not result["persist_outputs"] and result["checkpoints_saved"] == 0
    assert len(result["runs"]) == 9
    assert not Path(cfg.output_dir).exists()


def test_resume_extension_preserves_original_lr_floor():
    from cot_navigation.training.schedule import WarmupCosine
    param = torch.nn.Parameter(torch.tensor(1.0))
    opt = torch.optim.AdamW([param], lr=.01)
    original = WarmupCosine(opt, 1, 2, .01, .001)
    original.updates = 2
    state = original.state_dict()
    extended = WarmupCosine(opt, 1, 10, .01, .001)
    extended.load_state_dict(state)
    assert extended.lr_at(2) == pytest.approx(.001)
    assert extended.lr_at(5) == pytest.approx(.001)


def test_stage_a_never_loads_or_samples_queries_and_b_mixes_both(tmp_path):
    cfg = tiny_config(tmp_path)
    cfg.data.knowledge = "parametric_graph"
    cfg.data.graph_shape, cfg.data.graph_seed = "balanced", 0
    cfg.data.graph_seeds = (0,)
    cfg.train.stage = "A"
    cfg.train.persist_outputs = False
    generate_parametric(cfg.data)
    a = Trainer(cfg)
    assert a.world.queries is None and a.world.balanced == {}
    assert a.query_sampler is None and a.walk_sampler is not None
    a.fit()
    assert a.queries_seen == 0 and a.walks_seen > 0
    b_cfg = copy.deepcopy(cfg);b_cfg.train.stage = "B"
    with pytest.raises(TrainError, match="shared stage-A"):
        Trainer(b_cfg)
    b = Trainer(b_cfg,init_state=a.initial_state())
    b.fit()
    assert b.queries_seen == b_cfg.train.max_steps*b_cfg.train.effective_query_batch_size
    assert b.walks_seen == b_cfg.train.max_steps*b_cfg.train.walk_batch_size
    assert not Path(cfg.output_dir).exists()


def test_old_checkpoint_version_is_rejected():
    from cot_navigation.training.trainer import _check_resume
    with pytest.raises(TrainError,match="incompatible"):
        _check_resume({"checkpoint_version":2},{"checkpoint_version":3})


def test_first_round_dimensions_filler_and_complete_trace_fit():
    from cot_navigation.config import load_config, filler_length, _worst_sequence_len
    root = Path(__file__).resolve().parents[1]
    for base,n,m,k in [("main_context",16,24,16),("main_stage_b",32,48,32)]:
        for mode in ("direct","path","dfs_state","filler"):
            cfg=load_config([root/f"configs/experiments/{base}_{mode}.yaml"])
            assert (cfg.data.n_nodes,cfg.data.n_edges)==(n,m)
            assert filler_length(cfg)==k
            assert _worst_sequence_len(cfg)<=4096
            assert cfg.protocol.name==mode
