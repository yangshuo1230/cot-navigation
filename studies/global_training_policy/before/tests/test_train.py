"""Short CPU training, resume, and offline tracking."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from cot_navigation.config import load_config
from cot_navigation.data.generate import generate_context
from cot_navigation.tracking.logger import Tracker
from cot_navigation.training.checkpoint import load_checkpoint
from cot_navigation.training.trainer import Trainer


def _write_cfg(root: Path, out: Path, steps: int, data_root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    path = root / "train.yaml"
    path.write_text(
        f"""
output_dir: {out}
data:
  knowledge: context_graph
  n_nodes: 6
  n_edges: 4
  node_vocab: 32
  shapes:
    balanced: {{layers: 3, width: 2}}
  splits: {{train: 8, val: 4, test: 4}}
  root: {data_root}
  data_seed: 9
  max_attempts: 400
  workers: 1
  chunk_size: 8
  audit_fraction: 1
  graph_seeds: [0]
  min_edge_coverage: 1
  n_walks: 8
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
  compile: false
protocol: {{name: direct}}
train:
  max_steps: {steps}
  effective_query_batch_size: 4
  microbatch_size: 2
  precision: fp32
  stage: context
  eval_interval: 0
  save_interval: 2
  log_interval: 1
  warmup_steps: 1
  lr: 0.001
  min_lr: 0.0001
  model_seed: 5
  exposure_seed: 11
  keep_last: 2
tracking: {{mode: disabled, project: cot-graph-reasoning}}
runtime: {{device: cpu, fused_adamw: disabled}}
"""
    )
    return path


def _losses(path: Path) -> list[float]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return [row["total_loss"] for row in rows]


def test_cpu_resume_matches_uninterrupted_run(tmp_path: Path):
    data = tmp_path / "data"
    cfg = load_config([_write_cfg(tmp_path / "cfg", tmp_path / "full", 4, data)])
    generate_context(cfg.data, resume=False)
    full = Trainer(load_config([_write_cfg(tmp_path / "cfg", tmp_path / "full", 4, data)]))
    full.fit()
    # A second run stops conceptually at the step-2 checkpoint by resuming that file.
    resumed = Trainer(
        load_config([_write_cfg(tmp_path / "cfg2", tmp_path / "resume", 4, data)]),
        resume=str(tmp_path / "full" / "checkpoints" / "step_00000002.pt"),
    )
    assert resumed.completed == 2
    resumed.fit()
    full_losses = _losses(tmp_path / "full" / "metrics.jsonl")
    resume_losses = _losses(tmp_path / "resume" / "metrics.jsonl")
    assert len(resume_losses) == 2
    assert torch.allclose(torch.tensor(full_losses[2:]), torch.tensor(resume_losses), atol=1e-5, rtol=1e-5)
    a = load_checkpoint(tmp_path / "full" / "checkpoints" / "latest.pt")
    b = load_checkpoint(tmp_path / "resume" / "checkpoints" / "latest.pt")
    assert a["step"] == b["step"] == 4
    for key in a["model"]:
        assert torch.equal(a["model"][key], b["model"][key])
    assert a["query_sampler"]["cursor"] == b["query_sampler"]["cursor"]
    assert torch.equal(a["rng"]["torch"], b["rng"]["torch"])


def test_tracking_disabled_and_offline(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    cfg = load_config([_write_cfg(tmp_path / "c", tmp_path / "o", 1, tmp_path / "d")])
    tracker = Tracker(cfg, run_id="abc", parent_run_id=None, env={"torch": "test"})
    tracker.log(1, {"total_loss": 1.25, "trace_loss": None})
    tracker.finish()
    text = (tmp_path / "o" / "metrics.jsonl").read_text()
    assert "total_loss" in text and "trace_loss" not in text
    offline = load_config([_write_cfg(tmp_path / "c2", tmp_path / "off", 1, tmp_path / "d")])
    offline.tracking.mode = "offline"
    offline.tracking.project = "cot-graph-reasoning"
    tracker = Tracker(offline, run_id="offline123", parent_run_id="parent", env={"torch": "test"})
    tracker.log(1, {"total_loss": 0.5})
    tracker.finish()
    status = json.loads((tmp_path / "off" / "wandb_status.json").read_text())
    assert status["effective"] == "offline"
    assert status["run_id"] == "offline123"
    assert "WANDB_API_KEY" not in (tmp_path / "off" / "wandb_status.json").read_text()
