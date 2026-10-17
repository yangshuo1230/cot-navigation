"""Single-device trainer.

``step`` counts optimizer updates, not microbatches. Microbatch losses are
divided by the logical query count and the logical walk count, so splitting a
batch or a walk set does not change example weight. Stage B can use fixed,
EMA-balanced, or no WALK supervision. Evaluation uses its own forwards and
does not draw training RNGs.
"""

from __future__ import annotations

import json
import math
import random
import time
import uuid
from itertools import islice
from pathlib import Path

import numpy as np
import torch

from cot_navigation.config import AppConfig, dump_config, filler_length
from cot_navigation.data.dataset import ContextStore, ParametricWorld, collate_examples
from cot_navigation.data.prepared import PreparedRows
from cot_navigation.data.shards import file_sha256
from cot_navigation.evaluation.metrics import MetricAccumulator, prefix_length
from cot_navigation.evaluation.sampling import stratified_ids
from cot_navigation.evaluation.rollout import generate
from cot_navigation.models.transformer import build_model
from cot_navigation.protocols.tokenizer import Tokenizer
from cot_navigation.protocols.validate_trace import parse_completed_steps, compare_completed
from cot_navigation.runtime.envcheck import probe_environment
from cot_navigation.tracking.logger import Tracker
from cot_navigation.training.checkpoint import (
    capture_rng,
    load_checkpoint,
    restore_rng,
    save_best,
    save_training_checkpoint,
)
from cot_navigation.training.loss import combine_losses
from cot_navigation.training.sampler import FixedQuerySampler, IndexSampler, HopCurriculumSampler
from cot_navigation.training.augmentation import transform
from cot_navigation.training.schedule import WarmupCosine


class TrainError(RuntimeError):
    pass


def _optimizer(model, cfg, device: torch.device):
    decay, nodecay = [], []
    seen: set[int] = set()
    for _name, param in model.named_parameters():
        if not param.requires_grad or id(param) in seen:
            continue
        seen.add(id(param))
        (decay if param.ndim >= 2 else nodecay).append(param)
    groups = [
        {"params": decay, "weight_decay": cfg.train.weight_decay},
        {"params": nodecay, "weight_decay": 0.0},
    ]
    kwargs = {"lr": cfg.train.lr, "betas": tuple(cfg.train.betas)}
    impl = "adamw"
    if cfg.runtime.fused_adamw != "disabled" and device.type == "cuda":
        try:
            opt = torch.optim.AdamW(groups, fused=True, **kwargs)
            return opt, {"impl": "fused_adamw", "n_param_tensors": len(seen)}
        except (RuntimeError, TypeError) as exc:
            if cfg.runtime.fused_adamw == "enabled":
                raise TrainError(f"fused AdamW was requested and is unavailable: {exc}") from exc
            impl = f"foreach_fallback ({exc.__class__.__name__})"
    try:
        opt = torch.optim.AdamW(groups, foreach=True, **kwargs)
        impl = "foreach_adamw" if impl == "adamw" else impl
    except TypeError:
        opt = torch.optim.AdamW(groups, **kwargs)
        impl = "adamw"
    return opt, {"impl": impl, "n_param_tensors": len(seen)}


def _arch_snapshot(cfg: AppConfig, tok: Tokenizer, manifest_hash: str) -> dict:
    return {
        "checkpoint_version": 4 if cfg.data.knowledge == "parametric_graph" else 3,
        "vocab_hash": tok.hash(),
        "vocab_size": len(tok),
        "manifest_hash": manifest_hash,
        "layers": cfg.model.layers,
        "heads": cfg.model.heads,
        "d_model": cfg.model.d_model,
        "d_ff": cfg.model.d_ff,
        "ffn": cfg.model.ffn,
        "max_seq_len": cfg.model.max_seq_len,
        "protocol": cfg.protocol.name,
        "filler_tokens": filler_length(cfg) if cfg.protocol.name == "filler" else 0,
        "lr": cfg.train.lr,
        "min_lr": cfg.train.min_lr,
        "betas": list(cfg.train.betas),
        "weight_decay": cfg.train.weight_decay,
        "precision": cfg.train.precision,
        "query_batch": cfg.train.effective_query_batch_size,
        "walk_batch": cfg.train.walk_batch_size,
        "stage": cfg.train.stage,
        "knowledge": cfg.data.knowledge,
        "graph_shape": cfg.data.graph_shape,
        "graph_seed": cfg.data.graph_seed,
        "max_steps": cfg.train.max_steps,
        "rope_base": cfg.model.rope_base,
        "dropout": cfg.model.dropout,
        "activation": cfg.model.activation,
        "position_encoding": cfg.model.position_encoding,
        "tie_embeddings": cfg.model.tie_embeddings,
        "exposure_seed": cfg.train.exposure_seed,
        "warmup_steps": cfg.train.warmup_steps,
        "grad_clip": cfg.train.grad_clip,
        "fixed_query_pool": cfg.train.fixed_query_pool,
        "balance_query_labels": cfg.train.balance_query_labels,
        "one_hop_fraction": cfg.train.one_hop_fraction,
        "selection_metric": cfg.evaluation.selection_metric,
        "fixed_query_view": cfg.evaluation.fixed_query_view,
        "shuffle_context_edges": cfg.train.shuffle_context_edges,
        "query_hops": cfg.train.query_hops,
        "curriculum_stage_steps": list(cfg.train.curriculum_stage_steps),
        "walk_loss_mode": cfg.train.walk_loss_mode,
        "walk_loss_weight": cfg.train.walk_loss_weight,
        "walk_balance_decay": cfg.train.walk_balance_decay,
        "negative_random_path": cfg.protocol.negative_random_path,
        "random_search_order": cfg.protocol.random_search_order,
    }


def _check_resume(saved: dict, current: dict) -> list[str]:
    # Old runs remain resumable only with their original sampling/eval defaults.
    for key, default in {"fixed_query_pool": "balanced", "balance_query_labels": False,
                         "one_hop_fraction": 0.0, "selection_metric": "answer_accuracy",
                         "fixed_query_view": "balanced", "shuffle_context_edges": False,
                         "query_hops": "all", "curriculum_stage_steps": [],
                         "walk_loss_mode": "fixed", "walk_loss_weight": 1.0,
                         "walk_balance_decay": 0.98, "negative_random_path": False,
                         "random_search_order": False}.items():
        saved.setdefault(key, default)
    if saved.get("checkpoint_version") != current["checkpoint_version"]:
        raise TrainError("legacy checkpoint is incompatible with this data/path sampling version; start a new run")
    allowed = []
    for key, value in saved.items():
        if key == "max_steps":
            if current["max_steps"] < value:
                raise TrainError("max_steps can only increase on resume")
            if current["max_steps"] != value:
                allowed.append(f"max_steps {value} -> {current['max_steps']}")
            continue
        if current[key] != value:
            raise TrainError(f"resume mismatch on {key}: checkpoint={value} config={current[key]}")
    return allowed


def _chunk(rows: list, micro: int, max_tokens: int = 0, multiple: int = 1) -> list[list]:
    if not rows:
        return []
    size = len(rows) if micro <= 0 else micro
    chunks, current = [], []
    # Reorder only inside this optimizer update. Logical query/walk exposure
    # and their separately normalized losses remain unchanged.
    for row in sorted(rows, key=lambda r: len(r.tokens)):
        padded = ((len(row.tokens) + multiple - 1) // multiple) * multiple
        if current and (len(current) >= size or (max_tokens and padded * (len(current) + 1) > max_tokens)):
            chunks.append(current)
            current = []
        current.append(row)
    if current:
        chunks.append(current)
    return chunks


class Trainer:
    def __init__(self, cfg: AppConfig, resume: str | None = None, init_from: str | None = None,
                 *, init_state: dict | None = None, evaluation_checkpoint: str | None = None):
        cfg.validate()
        if sum(x is not None for x in (resume, init_from, init_state, evaluation_checkpoint)) > 1:
            raise TrainError("resume, stage initialization and evaluation checkpoint are distinct operations")
        if cfg.train.stage == "B" and all(x is None for x in (resume, init_from, init_state, evaluation_checkpoint)):
            raise TrainError("stage B requires a shared stage-A checkpoint via init_from/init_state, or resume")
        self.cfg = cfg
        self.persist = cfg.train.persist_outputs
        self.out = Path(cfg.output_dir)
        if self.persist:
            self.out.mkdir(parents=True, exist_ok=True)
            (self.out / "resolved_config.yaml").write_text(dump_config(cfg))
        self.device = _resolve_device(cfg)
        if cfg.train.precision == "bf16" and self.device.type != "cuda":
            raise TrainError("precision bf16 requires CUDA; set train.precision=fp32 for CPU")
        if cfg.train.deterministic:
            torch.use_deterministic_algorithms(True)
        if self.device.type == "cuda" and cfg.train.precision == "fp32":
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
        self.tok = Tokenizer(cfg.data.node_vocab)
        self.store = None
        self.world = None
        self._load_data()
        self.render_cache: dict = {}
        self._preload_training_rows()
        self.manifest_hash = self.store.fingerprint if self.store is not None else self.world.fingerprint
        torch.manual_seed(cfg.train.model_seed)
        np.random.seed(cfg.train.model_seed)
        random.seed(cfg.train.model_seed)
        self.raw = build_model(cfg.model, len(self.tok), pad_id=self.tok.pad).to(self.device)
        self.opt, self.opt_info = _optimizer(self.raw, cfg, self.device)
        self.schedule = WarmupCosine(self.opt, cfg.train.warmup_steps, cfg.train.max_steps, cfg.train.lr, cfg.train.min_lr)
        self.snapshot = _arch_snapshot(cfg, self.tok, self.manifest_hash)
        self.parent_run_id = None
        self.run_id = uuid.uuid4().hex
        self.completed = 0
        self.best_score = -1.0
        self.queries_seen = 0
        self.walks_seen = 0
        self.tokens_seen = 0
        self.loss_balance_ema = None
        self._init_samplers()
        if init_from and resume:
            raise TrainError("stage init-from and resume are different operations; pass only one")
        if init_from:
            self._load_init(init_from)
        if init_state is not None:
            self._load_initial_payload(init_state, "in-memory")
        if evaluation_checkpoint:
            self._load_evaluation(evaluation_checkpoint)
        if resume:
            self._load_resume(resume)
        self.model = self.raw
        if cfg.model.compile:
            if self.device.type != "cuda":
                raise TrainError("torch.compile training path is implemented for CUDA")
            self.model = torch.compile(self.raw, mode=cfg.model.compile_mode)
        env = probe_environment(probe_kernels=cfg.runtime.probe_kernels and self.device.type == "cuda")
        env["optimizer"] = self.opt_info
        env["parameters"] = self.raw.report()
        if self.persist:
            (self.out / "env.json").write_text(json.dumps(env, indent=2, default=str) + "\n")
        self.tracker = Tracker(cfg, self.run_id, self.parent_run_id, env, resume=resume is not None)
        self.schedule.set_lr()

    def _manifest_path(self) -> Path:
        if self.cfg.data.knowledge == "context_graph":
            return Path(self.cfg.data.root) / "manifest.json"
        return Path(self.cfg.data.root) / "manifest.json"

    def _load_data(self) -> None:
        cfg = self.cfg
        if cfg.data.knowledge == "context_graph":
            self.store = ContextStore(cfg.data.root)
            self.train_ids = self.store.indices("train")
            if cfg.train.query_hops == "one_hop":
                self.train_ids = self.train_ids[self.store.arrays["distance"][self.train_ids] == 1]
            self.val_ids = self.store.indices("val")
            return
        if cfg.data.graph_shape is None or cfg.data.graph_seed is None:
            raise TrainError("parametric training requires data.graph_shape and data.graph_seed")
        world = Path(cfg.data.root) / f"{cfg.data.graph_shape}-seed{cfg.data.graph_seed}"
        self.world = ParametricWorld(world, include_queries=cfg.train.stage != "A")
        if cfg.train.stage == "A":
            self.train_ids = np.arange(self.world.n_walks(), dtype=np.int64)
            self.val_ids = np.zeros(0, dtype=np.int64)
            return
        ids = self.world.indices("train", "multi_hop" if cfg.train.fixed_query_pool == "full" else "balanced")
        if cfg.train.one_hop_fraction > 0 or cfg.train.query_hops != "all":
            ids = np.concatenate([ids, self.world.indices("train", "one_hop")])
        if cfg.train.query_hops == "one_hop":
            ids = ids[self.world.queries["distance"][ids] == 1]
        if ids.size == 0:
            raise TrainError(f"balanced train query pool is empty in {world}")
        self.train_ids = ids.astype(np.int64)
        self.val_ids = self.world.indices("val", cfg.evaluation.fixed_query_view)

    def _init_samplers(self) -> None:
        seed = self.cfg.train.exposure_seed
        stage = self.cfg.train.stage
        self.query_sampler = None
        self.walk_sampler = None
        # Independent streams: adding a trace augmentation cannot change query,
        # walk, positive-path, or edge-presentation exposure in another branch.
        self.augmentation_rngs = {name: np.random.default_rng(seed + offset) for name, offset in
                                 (("edges", 701), ("search", 709), ("negative", 719))}
        # Path choices must not change query/walk exposure across protocols.
        self.path_rng = (np.random.default_rng(seed + 29)
                         if stage == "B" and self.cfg.protocol.name == "path" else None)
        if stage == "A":
            self.walk_sampler = IndexSampler(len(self.train_ids), self.cfg.train.walk_batch_size, seed + 17)
        elif stage == "B":
            if self.cfg.train.query_hops != "all":
                self.query_sampler = self._hop_sampler(self.world.queries)
            elif self.cfg.train.one_hop_fraction > 0 or self.cfg.train.balance_query_labels:
                self.query_sampler = FixedQuerySampler(
                    self.world.queries["distance"][self.train_ids], self.world.queries["label"][self.train_ids],
                    self.cfg.train.effective_query_batch_size, seed + 3,
                    self.cfg.train.one_hop_fraction, self.cfg.train.balance_query_labels)
            else:
                self.query_sampler = IndexSampler(len(self.train_ids), self.cfg.train.effective_query_batch_size, seed + 3)
            if self.cfg.train.walk_loss_mode != "none":
                self.walk_sampler = IndexSampler(self.world.n_walks(), self.cfg.train.walk_batch_size, seed + 17)
        else:
            self.query_sampler = (self._hop_sampler(self.store.arrays) if self.cfg.train.query_hops != "all"
                                  else IndexSampler(len(self.train_ids), self.cfg.train.effective_query_batch_size, seed + 3))

    def _hop_sampler(self, arrays):
        stages = (self.cfg.train.max_steps,) if self.cfg.train.query_hops == "one_hop" else self.cfg.train.curriculum_stage_steps
        return HopCurriculumSampler(arrays["distance"][self.train_ids], arrays["label"][self.train_ids],
                                    self.cfg.train.effective_query_batch_size, self.cfg.train.exposure_seed+3, stages)

    def _preload_training_rows(self) -> None:
        """Pay token loading/conversion once, before model/GPU initialization.

        Raw-only datasets remain usable: render once at startup. Formal data
        preparation supplies persisted tokens, so no DFS is needed here.
        Neither path draws training RNGs or prepares held-out supervision.
        Online experiments transform these immutable rows after sampling.
        """
        self.prepared_tasks = self.prepared_walks = None
        source = self.store if self.store is not None else self.world
        if self.cfg.train.stage != "A":
            try:
                self.prepared_tasks = PreparedRows(source, self.cfg.protocol.name, self.tok, filler_length(self.cfg))
            except FileNotFoundError:
                pass
            for index in self.train_ids:
                variants = (max(1, self.world.path_count(int(index)))
                            if self.world is not None and self.cfg.protocol.name == "path" else 1)
                for path_index in range(variants):
                    self._render_task(int(index), path_index=path_index)
        if self.world is not None and (self.cfg.train.stage == "A" or self.cfg.train.walk_loss_mode != "none"):
            try:
                self.prepared_walks = PreparedRows(source, "walk", self.tok)
            except FileNotFoundError:
                pass
            for index in range(self.world.n_walks()):
                self._render_walk(index)

    def _load_init(self, path: str) -> None:
        blob = load_checkpoint(path, map_location="cpu")
        self._load_initial_payload(blob, str(path))
        if self.persist:
            note = {"init_from": str(path), "init_sha256": file_sha256(path),
                    "parent_run_id": self.parent_run_id, "parent_step": blob["step"]}
            (self.out / "init_from.json").write_text(json.dumps(note, indent=2) + "\n")

    def _compatible_model(self, blob: dict) -> None:
        keys = ("checkpoint_version", "layers", "heads", "d_model", "d_ff", "ffn", "max_seq_len", "vocab_hash",
                "rope_base", "dropout", "activation", "position_encoding", "tie_embeddings", "knowledge")
        for key in keys:
            if blob["arch"].get(key) != self.snapshot[key]:
                raise TrainError(f"checkpoint mismatch on {key}")

    def _load_initial_payload(self, blob: dict, source: str) -> None:
        self._compatible_model(blob)
        if self.cfg.train.stage != "B" or blob["arch"].get("stage") != "A":
            raise TrainError("stage initialization requires a stage-A checkpoint and a stage-B target")
        for key in ("graph_shape", "graph_seed", "manifest_hash"):
            if blob["arch"].get(key) != self.snapshot[key]:
                raise TrainError(f"stage initialization world/provenance mismatch on {key} ({source})")
        self.raw.load_state_dict(blob["model"])
        self.parent_run_id = blob.get("wandb_run_id")

    def initial_state(self) -> dict:
        """In-memory stage handoff for smoke: no checkpoint is written."""
        return {"model": {k: v.detach().cpu().clone() for k, v in self.raw.state_dict().items()},
                "arch": dict(self.snapshot), "step": self.completed, "wandb_run_id": self.run_id}

    def _load_evaluation(self, path: str) -> None:
        blob = load_checkpoint(path, map_location="cpu")
        self._compatible_model(blob)
        if blob["arch"].get("filler_tokens") != self.snapshot["filler_tokens"]:
            raise TrainError("evaluation filler length does not match checkpoint")
        if blob["arch"]["protocol"] != self.cfg.protocol.name:
            raise TrainError("evaluation protocol does not match checkpoint")
        if self.world is not None and blob["arch"]["manifest_hash"] != self.manifest_hash:
            raise TrainError("evaluation fixed world does not match checkpoint")
        self.raw.load_state_dict(blob["model"])
        self.parent_run_id = blob.get("wandb_run_id")
        self.completed = int(blob["step"])

    def _load_resume(self, path: str) -> None:
        blob = load_checkpoint(path, map_location="cpu")
        notes = _check_resume(dict(blob["arch"]), self.snapshot)
        if blob["step"] > self.cfg.train.max_steps:
            raise TrainError("checkpoint step is already past max_steps")
        self.raw.load_state_dict(blob["model"])
        self.opt.load_state_dict(blob["optimizer"])
        for state in self.opt.state.values():
            for key, value in list(state.items()):
                if torch.is_tensor(value):
                    state[key] = value.to(self.device)
        self.schedule.load_state_dict(blob["schedule"])
        if self.query_sampler is not None:
            self.query_sampler.load_state_dict(blob["query_sampler"])
        if self.walk_sampler is not None:
            self.walk_sampler.load_state_dict(blob["walk_sampler"])
        if self.path_rng is not None:
            self.path_rng.bit_generator.state = blob["path_rng"]
        for name, rng in self.augmentation_rngs.items():
            if name in blob.get("augmentation_rngs", {}):
                rng.bit_generator.state = blob["augmentation_rngs"][name]
        if blob.get("loss_balance_ema") is not None:
            self.loss_balance_ema = [v.to(self.device) for v in blob["loss_balance_ema"]]
        restore_rng(blob["rng"])
        self.completed = int(blob["step"])
        self.run_id = blob["wandb_run_id"]
        self.parent_run_id = blob.get("parent_run_id")
        self.best_score = float(blob.get("best_score", -1.0))
        self.queries_seen = int(blob.get("queries_seen", 0))
        self.walks_seen = int(blob.get("walks_seen", 0))
        self.tokens_seen = int(blob.get("tokens_seen", 0))
        if self.persist:
            (self.out / "resume_notes.json").write_text(json.dumps({"allowed_changes": notes, "from": str(path)}, indent=2) + "\n")

    def fit(self) -> dict:
        cfg = self.cfg
        self.raw.train()
        last_metrics = {}
        while self.completed < cfg.train.max_steps:
            collect = (self.completed + 1) % cfg.train.log_interval == 0 or self.completed + 1 == cfg.train.max_steps
            metrics = self._update(collect_metrics=collect)
            self.completed += 1
            self._touch_rng()
            metrics["step"] = self.completed
            if self.completed % cfg.train.log_interval == 0 or self.completed == cfg.train.max_steps:
                self.tracker.log(self.completed, metrics)
            last_metrics = metrics
            if cfg.train.eval_interval and self.completed % cfg.train.eval_interval == 0:
                self._evaluate(kind="teacher_forced")
                if cfg.train.free_gen_samples > 0 and cfg.train.stage != "A":
                    self._evaluate(kind="free")
            if cfg.train.save_interval and self.completed % cfg.train.save_interval == 0:
                self._save()
        self._save()
        if self.persist and cfg.train.stage != "A" and cfg.evaluation.final_checkpoints:
            self._evaluate_final_checkpoints()
        self.tracker.finish()
        return last_metrics

    def _touch_rng(self) -> None:
        # Advance the global streams once per optimizer update so resume restores them.
        # The values are intentionally unused. Sampler RNGs are separate.
        random.random()
        np.random.random()
        torch.rand((), device="cpu")

    def _training_task(self, index: int):
        """Sample a route within a sampled training query, never across queries."""
        path_index = 0
        if self.path_rng is not None:
            if int(self.world.queries["split"][index]) != 0:
                raise TrainError("held-out queries cannot be sampled for training")
            count = self.world.path_count(index)
            if count:
                path_index = int(self.path_rng.integers(count))
        row = self._render_task(index, path_index=path_index)
        cfg = self.cfg
        edges = self.augmentation_rngs["edges"] if cfg.train.shuffle_context_edges and self.store is not None else None
        search = self.augmentation_rngs["search"] if cfg.protocol.random_search_order else None
        negative = self.augmentation_rngs["negative"] if cfg.protocol.negative_random_path and row.label == 0 else None
        if edges is not None or search is not None or negative is not None:
            adj, nodes, src, dst = self._graph_fields(index)
            row = transform(row, cfg.protocol.name, self.tok, adj, nodes, src, dst,
                            edge_rng=edges, search_rng=search, negative_rng=negative)
            self._check_length(row)
        return row

    def _graph_fields(self, index):
        if self.store is not None:
            arrays = self.store.arrays
            return (self.store.adj(index), self.store.node_vocab_ids(index, self.tok),
                    int(arrays["s_topo"][index]), int(arrays["t_topo"][index]))
        # A fixed world's adjacency and vocabulary mapping never change.
        if not hasattr(self, "_fixed_graph_fields"):
            self._fixed_graph_fields = self.world.adj(), self.world.node_vocab_ids(self.tok)
        adj, nodes = self._fixed_graph_fields
        return adj, nodes, int(self.world.queries["s_topo"][index]), int(self.world.queries["t_topo"][index])

    def _render_task(self, index: int, *, path_index: int = 0):
        # Validation uses path zero and never draws from the training RNG.
        key = ("task", int(index), self.cfg.protocol.name, filler_length(self.cfg), path_index)
        cached = self.render_cache.get(key)
        if cached is not None:
            return cached
        if getattr(self, "prepared_tasks", None) is not None:
            rendered = self.prepared_tasks.row(int(index), path_index)
        elif self.store is not None:
            rendered = self.store.render(int(index), self.cfg.protocol.name, self.tok, filler_tokens=filler_length(self.cfg))
        else:
            rendered = self.world.render_query(int(index), self.cfg.protocol.name, self.tok,
                                               filler_tokens=filler_length(self.cfg), path_index=path_index)
        if self.cfg.protocol.negative_random_path and rendered.label == 0:
            adj, nodes, src, dst = self._graph_fields(index)
            rendered = transform(rendered, "path", self.tok, adj, nodes, src, dst,
                                 negative_rng=np.random.default_rng(self.cfg.evaluation.seed + index))
        self._check_length(rendered)
        self.render_cache[key] = rendered
        return rendered

    def _check_length(self, rendered) -> None:
        multiple = self.cfg.train.bucket_multiple
        padded = ((len(rendered.tokens) + multiple - 1) // multiple) * multiple
        if padded > self.cfg.model.max_seq_len:
            raise TrainError(f"complete example needs {padded} padded tokens, exceeding context "
                             f"{self.cfg.model.max_seq_len}; no samples were truncated")

    def _render_walk(self, index: int):
        key = ("walk", int(index))
        cached = self.render_cache.get(key)
        if cached is not None:
            return cached
        rendered = (self.prepared_walks.row(int(index)) if getattr(self, "prepared_walks", None) is not None
                    else self.world.render_walk(int(index), self.tok))
        self._check_length(rendered)
        self.render_cache[key] = rendered
        return rendered

    def _update(self, collect_metrics: bool = True) -> dict:
        cfg = self.cfg
        data_t0 = time.perf_counter()
        task_rows = []
        walk_rows = []
        n_task = 0
        n_walk = 0
        if self.query_sampler is not None:
            local = self.query_sampler.next()
            global_ids = [int(self.train_ids[i]) for i in local]
            task_rows = [self._training_task(i) for i in global_ids]
            n_task = len(task_rows)
        if self.walk_sampler is not None:
            local = self.walk_sampler.next()
            if cfg.train.stage == "A":
                global_walks = [int(self.train_ids[i]) for i in local]
            else:
                global_walks = local
            walk_rows = [self._render_walk(i) for i in global_walks]
            n_walk = len(walk_rows)
        task_chunks = _chunk(task_rows, cfg.train.microbatch_size, cfg.train.max_tokens_per_microbatch, cfg.train.bucket_multiple)
        walk_chunks = _chunk(walk_rows, cfg.train.microbatch_size, cfg.train.max_tokens_per_microbatch, cfg.train.bucket_multiple)
        data_s = time.perf_counter() - data_t0
        self.opt.zero_grad(set_to_none=True)
        lr = self.schedule.set_lr()
        if collect_metrics and self.device.type == "cuda":
            torch.cuda.synchronize()
        step_t0 = time.perf_counter()
        parts_list = []
        balancing = cfg.train.stage == "B" and cfg.train.walk_loss_mode == "balanced"
        walk_weight = cfg.train.walk_loss_weight if cfg.train.stage == "B" else 1.0
        if balancing and self.loss_balance_ema is not None:
            # Previous-update EMA: one detached coefficient for the whole logical
            # batch. No second forward and no per-microbatch reweighting.
            walk_weight = (self.loss_balance_ema[0] / self.loss_balance_ema[1].clamp_min(1e-12)).clamp(max=100).detach()
        task_value = torch.zeros((), device=self.device)
        walk_value = torch.zeros((), device=self.device)
        real_tokens = 0
        pad_tokens = 0
        for rows in task_chunks + walk_chunks:
            arrays = collate_examples(rows, self.tok.pad, cfg.train.bucket_multiple)
            real = int(arrays["lengths"].sum())
            real_tokens += real
            pad_tokens += int(arrays["tokens"].size - real)
            batch = {k: torch.from_numpy(v) for k, v in arrays.items()}
            if self.device.type == "cuda" and cfg.train.pin_memory:
                batch = {k: v.pin_memory() for k, v in batch.items()}
            batch = {k: v.to(self.device, non_blocking=self.device.type == "cuda") for k, v in batch.items()}
            with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=cfg.train.precision == "bf16"):
                logits = self.model(batch["tokens"])
            parts = combine_losses(
                logits,
                batch["tokens"],
                batch["answer_mask"],
                batch["trace_mask"],
                batch["walk_mask"],
                n_task=n_task,
                n_walk=n_walk,
                walk_weight=walk_weight,
            )
            parts["total_loss"].backward()
            if balancing:
                task_value += (parts["answer"].detach().sum() + parts["trace"].detach().sum()) / max(n_task, 1)
                walk_value += parts["walk"].detach().sum() / max(n_walk, 1)
            if collect_metrics:
                saved = {k: parts[k].detach() for k in ("answer", "trace", "walk", "has_answer", "has_trace", "has_walk")}
                saved["backward_total"] = parts["total_loss"].detach()
                parts_list.append(saved)
        grad_norm = torch.nn.utils.clip_grad_norm_(self.raw.parameters(), cfg.train.grad_clip, error_if_nonfinite=True)
        self.opt.step()
        self.schedule.step()
        if balancing:
            if self.loss_balance_ema is None:
                self.loss_balance_ema = [task_value, walk_value]
            else:
                decay = cfg.train.walk_balance_decay
                for average, value in zip(self.loss_balance_ema, (task_value, walk_value)):
                    average.mul_(decay).add_(value, alpha=1-decay)
        if collect_metrics and self.device.type == "cuda":
            torch.cuda.synchronize()
        step_s = time.perf_counter() - step_t0
        self.queries_seen += n_task
        self.walks_seen += n_walk
        self.tokens_seen += real_tokens
        if not collect_metrics:
            return {"lr": lr}
        metrics = _reduce_logged(parts_list, n_task, n_walk)
        metrics["total_loss"] = float(sum(item["backward_total"] for item in parts_list))
        if n_task:
            metrics["task_loss"] = metrics["answer_loss"] + metrics.get("trace_loss", 0.0)
            distances = (self.store.arrays if self.store is not None else self.world.queries)["distance"]
            for hop in np.unique(distances[global_ids]):
                metrics[f"train_hops/{int(hop)}_fraction"] = float(np.mean(distances[global_ids] == hop))
            if isinstance(self.query_sampler, HopCurriculumSampler):
                # .step already points at the next batch; report the stage used.
                metrics["curriculum_stage"] = int(np.searchsorted(self.query_sampler.ends, self.query_sampler.step-1, side="right"))+1
        if n_walk and n_task:
            metrics["walk_weight"] = float(walk_weight)
            metrics["weighted_walk_loss"] = float(walk_weight) * metrics["walk_loss"]
            metrics["walk_loss_share"] = metrics["weighted_walk_loss"] / max(metrics["task_loss"]+metrics["weighted_walk_loss"], 1e-12)
        total = metrics.get("total_loss")
        if total is None or not math.isfinite(total):
            raise TrainError(f"non-finite loss at step {self.completed + 1}: {metrics}")
        metrics.update(
            {
                "lr": lr,
                "grad_norm": float(grad_norm),
                "step_time": step_s,
                "data_time": data_s,
                "tokens_per_s": real_tokens / max(step_s, 1e-9),
                "padding_fraction": pad_tokens / max(real_tokens + pad_tokens, 1),
                "effective_tokens": real_tokens,
                "microbatches": len(task_chunks) + len(walk_chunks),
            }
        )
        if n_task:
            metrics["queries_seen"] = self.queries_seen
        if n_walk:
            metrics["walks_seen"] = self.walks_seen
        metrics["tokens_seen"] = self.tokens_seen
        if self.device.type == "cuda":
            metrics["gpu_mem_mb"] = torch.cuda.max_memory_allocated(self.device) / (1024 * 1024)
        return metrics

    def _evaluate(self, kind: str) -> None:
        if self.val_ids.size == 0 or self.cfg.train.stage == "A":
            return
        self.raw.eval()
        self.model.eval()
        limit = self.cfg.train.eval_samples if kind == "teacher_forced" else self.cfg.train.free_gen_samples
        chosen = self._selected_ids("val", limit)
        rows = (self._render_task(i) for i in chosen)
        if kind == "teacher_forced":
            packed = self._teacher_forced(rows)
        else:
            packed = self._free(rows)
        record = {"kind": kind, "step": self.completed, **packed}
        if self.persist:
            eval_dir = self.out / "eval"
            eval_dir.mkdir(exist_ok=True)
            (eval_dir / f"{kind}_{self.completed:08d}.json").write_text(json.dumps(record, indent=2) + "\n")
        flat = {f"{kind}/{key}": value for key, value in _flatten_numbers(packed).items()}
        self.tracker.log(self.completed, flat)
        # Gold traces can disclose the answer. Only free-rollout final answers
        # select the main best checkpoint; teacher forcing is diagnostic.
        score = packed["overall"].get(self.cfg.evaluation.selection_metric)
        if kind == "free" and score is not None and score > self.best_score:
            self.best_score = float(score)
            if self.persist:
                save_best(self.out / "checkpoints", self._payload())
        self.raw.train()
        self.model.train()

    def _selected_ids(self, split: str, limit: int, query_view: str | None = None) -> list[int]:
        if self.store is not None:
            ids = self.store.indices(split)
            arrays = self.store.arrays
            if self.cfg.train.query_hops == "one_hop":
                ids = ids[arrays["distance"][ids] == 1]
            shapes = arrays["shape_id"]
        else:
            ids = self.world.indices(split, query_view or self.cfg.evaluation.fixed_query_view)
            arrays = self.world.queries
            shapes = np.zeros(len(arrays["label"]), dtype=np.int8)
        return stratified_ids(ids, shapes, arrays["distance"], arrays["label"], limit, self.cfg.evaluation.seed)

    def evaluate_split(self, split: str, *, query_view: str | None = None,
                       checkpoint_tag: str | None = None, checkpoint_step: int | None = None) -> dict:
        """Free-generation eval. Does not draw the training samplers."""
        if self.cfg.train.stage == "A":
            raise TrainError("stage A has no query evaluation")
        ids = self._selected_ids(split, self.cfg.evaluation.test_samples, query_view)
        rows = (self._render_task(index) for index in ids)
        report = (self._free(rows, record_name=f"{split}_{checkpoint_tag}") if checkpoint_tag
                  else self._free(rows))
        report["unique_queries"] = len(ids)
        report["evaluation_batch_size"] = self.cfg.evaluation.batch_size
        report["query_view"] = query_view or self.cfg.evaluation.fixed_query_view
        report["checkpoint_step"] = self.completed if checkpoint_step is None else checkpoint_step
        report["training_step"] = self.completed
        if checkpoint_tag:
            report["checkpoint_tag"] = checkpoint_tag
        if self.persist:
            suffix = f"_{checkpoint_tag}" if checkpoint_tag else ""
            path = self.out / "eval" / f"free_{split}{suffix}.json"
            path.parent.mkdir(exist_ok=True)
            path.write_text(json.dumps(report, indent=2) + "\n")
        prefix = f"{split}/{checkpoint_tag}" if checkpoint_tag else split
        self.tracker.log(self.completed, {f"{prefix}/{k}": v for k, v in _flatten_numbers(report).items()})
        return report

    def _evaluate_final_checkpoints(self):
        """Select on val, test once, and retain last weights and optimizer state.

        Reports are tagged so best/last scores never overwrite each other.
        Test metrics cannot select a checkpoint or affect training samplers.
        """
        tags = tuple(dict.fromkeys(self.cfg.evaluation.final_checkpoints))
        try:
            for tag in tags:
                path = self.out / "checkpoints" / ("latest.pt" if tag == "last" else "best.pt")
                if not path.exists():
                    raise TrainError(f"missing {tag} checkpoint for final evaluation; enable free validation/save")
                blob = load_checkpoint(path, map_location="cpu")
                self.raw.load_state_dict(blob["model"])
                self.evaluate_split("test", checkpoint_tag=tag, checkpoint_step=int(blob["step"]))
        finally:
            latest = load_checkpoint(self.out / "checkpoints/latest.pt", map_location="cpu")
            self.raw.load_state_dict(latest["model"])

    def _teacher_forced(self, rows) -> dict:
        accumulator = MetricAccumulator()
        iterator = iter(rows)
        while chunk := list(islice(iterator, self.cfg.evaluation.batch_size)):
            arrays = collate_examples(chunk, self.tok.pad, self.cfg.train.bucket_multiple)
            batch = {k: torch.from_numpy(v).to(self.device) for k, v in arrays.items()}
            with torch.no_grad(), torch.autocast(device_type=self.device.type, dtype=torch.bfloat16,
                                               enabled=self.cfg.train.precision == "bf16"):
                logits = self.model(batch["tokens"])
            index = batch["answer_index"]
            picked = logits[torch.arange(logits.shape[0], device=self.device), index - 1].argmax(dim=-1).tolist()
            for token, row in zip(picked, chunk):
                accumulator.update({"correct": token == self.tok.label_id(row.label),
                                    "format_fail": token not in (self.tok.yes, self.tok.no),
                                    "label": row.label, "shape": row.meta.get("shape"),
                                    "distance": row.meta.get("distance"), "filler_tokens": row.filler_tokens})
        return {**accumulator.result(), "source": "teacher_forced"}

    def _free(self, rows, *, record_name: str | None = None) -> dict:
        accumulator = MetricAccumulator()
        preview = []
        predictions = None
        if self.persist and record_name:
            (self.out / "eval").mkdir(exist_ok=True)
            predictions = self.out / "eval" / f"predictions_{record_name}.jsonl"
            predictions.write_text("")
        iterator = iter(rows)
        while chunk := list(islice(iterator, self.cfg.evaluation.batch_size)):
            prompts = [row.tokens[:row.prompt_len] for row in chunk]
            with torch.no_grad():
                generated = generate(self.raw, prompts, self.cfg.protocol.name, self.tok, self.cfg.train.precision,
                                     filler_tokens=filler_length(self.cfg),
                                     max_new_tokens=self.cfg.evaluation.max_new_tokens)
            for row, gen in zip(chunk, generated):
                item = {"correct": gen["pred"] == row.label and not gen["format_fail"],
                        "format_fail": gen["format_fail"], "label": row.label, "shape": row.meta.get("shape"),
                        "distance": row.meta.get("distance"), "filler_tokens": row.filler_tokens, "generation_limit_reached": gen["generation_limit_reached"]}
                if self.cfg.protocol.name in {"dfs_state", "bfs_state"}:
                    reference = row.tokens[row.prompt_len:row.answer_index]
                    item["correct_prefix_tokens"] = prefix_length(gen["trace_tokens"], reference)
                    item["search_complete"] = (not gen["generation_limit_reached"]) and gen["trace_tokens"] == reference
                    parsed = parse_completed_steps(gen["trace_tokens"], self.tok, fifo=self.cfg.protocol.name == "bfs_state")
                    # Search state is diagnostic-only. Offline caches persist
                    # tokens, not Python DFS objects for every training row.
                    meta = row.meta
                    if "reference_steps" not in meta:
                        index = int(meta["index"])
                        full = (self.store.render(index, self.cfg.protocol.name, self.tok) if self.store is not None
                                else self.world.render_query(index, self.cfg.protocol.name, self.tok))
                        meta = full.meta
                    comparison = compare_completed(meta["reference_steps"], parsed, meta["node_token"])
                    malformed = max(0, gen["trace_tokens"].count(self.tok.id(";")) - len(parsed))
                    item.update(completed_records=len(parsed) + malformed,
                                malformed_completed_records=malformed,
                                completed_records_match=comparison["matched_records"],
                                field_correct=comparison["field_correct"])
                accumulator.update(item)
                if predictions is not None:
                    # Full-test rows permit paired comparisons across methods;
                    # token traces are previewed separately, not copied en masse.
                    record = {"query_id": int(row.meta["index"]), "pred": gen["pred"],
                              **{k: v for k, v in item.items() if k in (
                                  "correct", "format_fail", "label", "shape", "distance")}}
                    with predictions.open("a") as handle:
                        handle.write(json.dumps(record) + "\n")
                if self.persist and len(preview) < 4:
                    preview.append(self.tok.decode(gen["tokens"]))
        if self.persist:
            (self.out / "traces").mkdir(exist_ok=True)
            suffix = record_name or f"{self.completed:08d}"
            (self.out / "traces" / f"free_{suffix}.txt").write_text("\n\n".join(preview) + "\n")
        report = accumulator.result()
        report["source"] = "free_generation"
        report["filler_prefill"] = self.cfg.protocol.name == "filler"
        return report

    def _payload(self) -> dict:
        return {
            "model": self.raw.state_dict(),
            "optimizer": self.opt.state_dict(),
            "schedule": self.schedule.state_dict(),
            "step": self.completed,
            "rng": capture_rng(),
            "query_sampler": None if self.query_sampler is None else self.query_sampler.state_dict(),
            "walk_sampler": None if self.walk_sampler is None else self.walk_sampler.state_dict(),
            "path_rng": None if self.path_rng is None else self.path_rng.bit_generator.state,
            "augmentation_rngs": {k: rng.bit_generator.state for k, rng in self.augmentation_rngs.items()},
            "loss_balance_ema": self.loss_balance_ema,
            "arch": self.snapshot,
            "wandb_run_id": self.run_id,
            "parent_run_id": self.parent_run_id,
            "best_score": self.best_score,
            "queries_seen": self.queries_seen,
            "walks_seen": self.walks_seen,
            "tokens_seen": self.tokens_seen,
            "config": json.loads(json.dumps(self.snapshot)),
        }

    def _save(self) -> None:
        if self.persist:
            save_training_checkpoint(self.out / "checkpoints", self._payload(), self.cfg.train.keep_last)


def _reduce_logged(parts_list: list[dict], n_task: int, n_walk: int) -> dict:
    def pooled(field: str, mask: str):
        values = []
        for parts in parts_list:
            take = parts[mask]
            if int(take.sum().item()) == 0:
                continue
            values.append(parts[field][take])
        if not values:
            return None
        cat = torch.cat(values)
        return float(cat.mean().item())

    out = {}
    answer = pooled("answer", "has_answer")
    trace = pooled("trace", "has_trace")
    walk = pooled("walk", "has_walk")
    total = 0.0
    have = False
    if answer is not None and n_task > 0:
        out["answer_loss"] = answer
        total += answer
        have = True
        if trace is not None:
            out["trace_loss"] = trace
            total += trace
    if walk is not None and n_walk > 0:
        out["walk_loss"] = walk
        total += walk
        have = True
    if have:
        out["total_loss"] = total
    return out


def _flatten_numbers(payload: dict, prefix: str = "") -> dict:
    result = {}
    for key, value in payload.items():
        name = f"{prefix}/{key}" if prefix else str(key)
        if isinstance(value, dict):
            result.update(_flatten_numbers(value, name))
        elif isinstance(value, (int, float)):
            result[name] = value
    return result


def _resolve_device(cfg: AppConfig) -> torch.device:
    if cfg.runtime.device == "cpu":
        return torch.device("cpu")
    if cfg.runtime.device == "cuda" or (cfg.runtime.device == "auto" and torch.cuda.is_available()):
        if cfg.runtime.cuda_device is None:
            return torch.device("cuda")
        return torch.device(f"cuda:{cfg.runtime.cuda_device}")
    return torch.device("cpu")
