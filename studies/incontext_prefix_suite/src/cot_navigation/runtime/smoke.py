"""Ephemeral end-to-end checks: no run records, checkpoints or W&B uploads."""

from __future__ import annotations

import copy
import gc
from pathlib import Path
from tempfile import TemporaryDirectory

from cot_navigation.config import ShapeSpec
from cot_navigation.data.generate import generate_context, generate_parametric
from cot_navigation.data.validate import validate_context, validate_parametric
from cot_navigation.training.trainer import Trainer


def run_smoke(config, steps: int = 2) -> dict:
    """Stage A hands CPU weights to B in memory; temporary fixtures are removed.

    The caller receives only scalar summaries to print, not a saved run. This
    intentionally does not test checkpoint persistence; pytest covers resume.
    """
    if steps < 1:
        raise ValueError("smoke steps must be positive")
    result = {"persist_outputs": False, "wandb": "disabled", "checkpoints_saved": 0, "runs": {}}
    with TemporaryDirectory(prefix="cot-navigation-smoke-") as scratch:
        root = Path(scratch)
        cfg = copy.deepcopy(config)
        cfg.runtime.smoke = True
        cfg.model.max_seq_len = 4096
        cfg.data.root = str(root / "context")
        cfg.data.n_nodes, cfg.data.n_edges = 16, 24
        cfg.data.shapes = {"balanced": ShapeSpec(4, 4)}
        cfg.data.workers = 1
        cfg.data.chunk_size = 64
        n_shapes = len(cfg.data.shapes)
        cfg.data.splits.train, cfg.data.splits.val, cfg.data.splits.test = 64*n_shapes, 32*n_shapes, 32*n_shapes
        cfg.train.effective_query_batch_size = 4
        cfg.train.walk_batch_size = 4
        cfg.train.microbatch_size = 2
        cfg.train.max_steps = steps
        # Smoke validates all protocol pipelines on its own tiny geometry;
        # experimental schedules/format flags are covered by dedicated tests.
        cfg.train.query_hops = "all"
        cfg.train.curriculum_stage_steps = ()
        cfg.protocol.negative_random_path = False
        cfg.protocol.random_search_order = False
        cfg.train.log_interval = 1
        cfg.train.eval_interval = 0
        cfg.train.stage = "context"
        cfg.data.knowledge = "context_graph"
        cfg.evaluation.test_samples = 8
        # Execution check only: untrained models may hit this failure guard.
        cfg.evaluation.max_new_tokens = 64
        cfg.validate()
        generate_context(cfg.data, resume=False)
        validate_context(cfg.data)

        def train_one(current, name, initial=None):
            current.output_dir = str(root / "would-be-output" / name)
            current.validate()
            trainer = Trainer(current, init_state=initial)
            metrics = trainer.fit()
            evaluation = None if current.train.stage == "A" else trainer.evaluate_split("test")["overall"]
            result["runs"][name] = {"step": trainer.completed, "loss": metrics.get("total_loss"), "evaluation": evaluation}
            initial = trainer.initial_state() if current.train.stage == "A" else None
            trainer.tracker.finish()
            del trainer
            gc.collect()
            return initial

        for protocol in ("direct", "path", "dfs_state", "bfs_state", "dfs_nodes", "bfs_nodes"):
            current = copy.deepcopy(cfg)
            current.protocol.name = protocol
            train_one(current, f"context_{protocol}")

        parametric = copy.deepcopy(cfg)
        parametric.data.knowledge = "parametric_graph"
        parametric.data.n_nodes, parametric.data.n_edges = 32, 48
        parametric.data.shapes = {"balanced": ShapeSpec(8, 4)}
        parametric.data.root = str(root / "parametric")
        parametric.data.graph_seeds = (0,)
        parametric.data.graph_seed = 0
        parametric.data.graph_shape = next(iter(parametric.data.shapes))
        parametric.data.n_walks = 64
        parametric.data.min_edge_coverage = 2
        parametric.train.stage = "A"
        generate_parametric(parametric.data, resume=False)
        validate_parametric(parametric.data)
        stage_a = train_one(parametric, "stage_a")
        for protocol in ("direct", "path", "dfs_state", "bfs_state", "dfs_nodes", "bfs_nodes"):
            current = copy.deepcopy(parametric)
            current.train.stage = "B"
            current.protocol.name = protocol
            train_one(current, f"stage_b_{protocol}", stage_a)
        if (root / "would-be-output").exists() or list(root.rglob("*.pt")):
            raise RuntimeError("smoke unexpectedly persisted training outputs")
    result["temporary_data_cleaned"] = not root.exists()
    return result
