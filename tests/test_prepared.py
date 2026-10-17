"""Offline tokens, loss masks and zero-render/zero-IO training data access."""
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from cot_navigation.config import filler_length, load_config
from cot_navigation.data.dataset import ContextStore, ParametricWorld, collate_examples
from cot_navigation.data.prepared import PROTOCOLS, PreparedRows, prepare_dataset
from cot_navigation.protocols.tokenizer import Tokenizer
from cot_navigation.training.trainer import Trainer

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(params=["context", "fixed"])
def prepared(request, tmp_path):
    preset = "main_context_direct" if request.param == "context" else "main_stage_b_path"
    overrides = [f"data.root={tmp_path / 'data'}", "data.workers=1", "data.audit_fraction=1.0"]
    if request.param == "context":
        overrides += ["data.splits.train=24", "data.splits.val=12", "data.splits.test=12", "data.chunk_size=12"]
    else:
        overrides += ["data.graph_seeds=[0]", "data.n_walks=40", "data.min_edge_coverage=2"]
    cfg = load_config([ROOT / f"configs/experiments/{preset}.yaml"], overrides=overrides)
    # This compatibility test deliberately also covers retired Filler caches.
    report = prepare_dataset(cfg, PROTOCOLS)
    assert not report["reused"]
    source = (ContextStore(cfg.data.root) if request.param == "context" else
              ParametricWorld(Path(cfg.data.root) / "balanced-seed0"))
    return cfg, source


def _assert_equal(raw, cached, tok):
    for key in ("tokens", "prompt_len", "answer_index", "label", "trace_indices", "walk_indices", "filler_tokens"):
        assert getattr(raw, key) == getattr(cached, key), key
    a, b = collate_examples([raw], tok.pad), collate_examples([cached], tok.pad)
    for key in a:
        assert np.array_equal(a[key], b[key]), key


def test_offline_tokens_preserve_all_protocols_paths_and_masks(prepared):
    cfg, source = prepared
    tok = Tokenizer(cfg.data.node_vocab)
    is_context = isinstance(source, ContextStore)
    for protocol in PROTOCOLS:
        cached = PreparedRows(source, protocol, tok, filler_length(cfg))
        count = len(source) if is_context else len(source.queries["label"])
        for index in range(count):
            variants = max(1, source.path_count(index)) if not is_context and protocol == "path" else 1
            assert cached.variants(index) == variants
            for variant in range(variants):
                raw = (source.render(index, protocol, tok, filler_tokens=filler_length(cfg)) if is_context else
                       source.render_query(index, protocol, tok, filler_tokens=filler_length(cfg), path_index=variant))
                _assert_equal(raw, cached.row(index, variant), tok)
    if not is_context:
        walks = PreparedRows(source, "walk", tok)
        for index in range(source.n_walks()):
            _assert_equal(source.render_walk(index, tok), walks.row(index), tok)


def test_reuse_receipt_skips_generation_and_validation(prepared, monkeypatch):
    cfg, _source = prepared
    def forbidden(*args, **kwargs):
        raise AssertionError("reuse must not generate or revalidate")
    for fn in ("generate_context", "generate_parametric", "validate_context", "validate_parametric"):
        monkeypatch.setattr(f"cot_navigation.data.prepared.{fn}", forbidden)
    before = (Path(cfg.data.root) / "prepared_data.json").stat().st_mtime_ns
    assert prepare_dataset(cfg, ["path"])["reused"]
    assert (Path(cfg.data.root) / "prepared_data.json").stat().st_mtime_ns == before


def test_warmed_training_never_reads_disk_or_renders(prepared, monkeypatch):
    cfg, source = prepared
    def forbidden(*args, **kwargs):
        raise AssertionError("training row lookup must not render or access files")
    # Persisted tokens must suffice even for startup (no raw DFS/render calls).
    monkeypatch.setattr(ContextStore, "render", forbidden)
    monkeypatch.setattr(ParametricWorld, "render_query", forbidden)
    monkeypatch.setattr(ParametricWorld, "render_walk", forbidden)
    protocols = PROTOCOLS if isinstance(source, ContextStore) else (*PROTOCOLS, "knowledge")
    for protocol in protocols:
        trainer = Trainer.__new__(Trainer)
        trainer.cfg = deepcopy(cfg)
        trainer.cfg.train.stage = "context" if isinstance(source, ContextStore) else "B"
        if protocol == "knowledge":
            trainer.cfg.train.stage = "A"
        trainer.cfg.protocol.name = "direct" if protocol == "knowledge" else protocol
        trainer.tok = Tokenizer(cfg.data.node_vocab)
        trainer.store = source if isinstance(source, ContextStore) else None
        trainer.world = source if isinstance(source, ParametricWorld) else None
        trainer.train_ids = (source.indices("train") if trainer.store is not None else
                             np.arange(source.n_walks()) if protocol == "knowledge" else source.balanced["train"])
        trainer.render_cache = {}
        trainer._preload_training_rows()
        trainer._init_samplers()
        with monkeypatch.context() as hot:
            hot.setattr(PreparedRows, "row", forbidden)
            hot.setattr(Path, "open", forbidden)
            for _ in range(4):
                if trainer.query_sampler is not None:
                    for i in trainer.query_sampler.next():
                        trainer._training_task(int(trainer.train_ids[i]))
                if trainer.walk_sampler is not None:
                    for i in trainer.walk_sampler.next():
                        trainer._render_walk(i)


def test_receipt_invalidated_by_file_mutation(prepared, monkeypatch):
    cfg, _source = prepared
    manifest = Path(cfg.data.root) / "manifest.json"
    manifest.write_bytes(manifest.read_bytes() + b"\n")
    def stop(*args, **kwargs):
        raise RuntimeError("full preparation requested")
    monkeypatch.setattr("cot_navigation.data.prepared.generate_context", stop)
    monkeypatch.setattr("cot_navigation.data.prepared.generate_parametric", stop)
    with pytest.raises(RuntimeError, match="full preparation requested"):
        prepare_dataset(cfg)


def test_cached_dfs_keeps_evaluation_field_diagnostics(prepared, monkeypatch):
    cfg, source = prepared
    trainer = Trainer.__new__(Trainer)
    trainer.cfg = deepcopy(cfg)
    trainer.cfg.protocol.name = "dfs_state"
    trainer.tok = Tokenizer(cfg.data.node_vocab)
    trainer.store = source if isinstance(source, ContextStore) else None
    trainer.world = source if isinstance(source, ParametricWorld) else None
    trainer.persist, trainer.raw = False, None
    row = PreparedRows(source, "dfs_state", trainer.tok).row(0)
    assert "reference_steps" not in row.meta
    def perfect(*args, **kwargs):
        return [{"pred": row.label, "format_fail": False, "generation_limit_reached": False,
                 "trace_tokens": row.tokens[row.prompt_len:row.answer_index], "tokens": row.tokens}]
    monkeypatch.setattr("cot_navigation.training.trainer.generate", perfect)
    report = trainer._free([row])
    assert report["overall"]["answer_accuracy"] == 1
    assert report["trace"]["completed_records_match"] == report["trace"]["completed_records"] > 0
    assert all(v == 1 for v in report["trace"]["field_accuracy"].values())
