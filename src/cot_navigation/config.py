"""Typed YAML configuration.

Unknown keys and configurations that cannot fit complete reference traces
fail before training. Experiment numbers live in YAML, not algorithm branches.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml


class ConfigError(ValueError):
    """Raised when a config file or override is illegal."""


ANSWER_TAIL_TOKENS = 2  # YES/NO + EOS; there is no ANS or cutoff interface.


@dataclass
class ShapeSpec:
    layers: int
    width: int


@dataclass
class SplitsSpec:
    train: int
    val: int
    test: int


@dataclass
class DataConfig:
    knowledge: str
    n_nodes: int
    n_edges: int
    shapes: dict[str, ShapeSpec]
    splits: SplitsSpec
    root: str
    node_vocab: int = 256
    data_seed: int = 0
    graph_seeds: tuple[int, ...] = (0, 1, 2)
    max_attempts: int = 2000
    workers: int = 1
    chunk_size: int = 256
    audit_fraction: float = 0.02
    min_edge_coverage: int = 100
    n_walks: int = 50000
    walk_lengths: tuple[int, ...] = (1, 2, 3, 4)
    query_ratios: tuple[float, float, float] = (0.2, 0.1, 0.7)
    # Parametric training must name exactly one world. Generation ignores these.
    graph_shape: str | None = None
    graph_seed: int | None = None


@dataclass
class ModelConfig:
    layers: int = 6
    heads: int = 1
    d_model: int = 128
    d_ff: int = 512
    activation: str = "gelu"
    position_encoding: str = "rope"
    max_seq_len: int = 4096
    tie_embeddings: bool = True
    dropout: float = 0.0
    ffn: str = "standard"
    attn_impl: str = "sdpa"
    compile: bool = False
    compile_mode: str = "default"
    rope_base: float = 10000.0


@dataclass
class ProtocolConfig:
    name: str = "direct"
    # None means K=N. It is fixed for an experiment, never oracle-dependent.
    filler_tokens: int | None = None


@dataclass
class TrainConfig:
    max_steps: int = 10000
    effective_query_batch_size: int = 64
    walk_batch_size: int = 16
    lr: float = 1e-3
    min_lr: float = 1e-4
    warmup_steps: int = 500
    betas: tuple[float, float] = (0.9, 0.99)
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    precision: str = "bf16"
    eval_interval: int = 250
    save_interval: int = 1000
    log_interval: int = 10
    microbatch_size: int = 0
    bucket_multiple: int = 64
    pin_memory: bool = True
    max_tokens_per_microbatch: int = 16384
    persist_outputs: bool = True
    deterministic: bool = False
    stage: str = "context"
    exposure_seed: int = 0
    keep_last: int = 3
    eval_samples: int = 64
    free_gen_samples: int = 16
    model_seed: int = 0


@dataclass
class TrackingConfig:
    mode: str = "disabled"  # online | offline | disabled
    project: str = "cot-graph-reasoning"
    entity: str | None = None
    name: str | None = None
    group: str | None = None
    tags: tuple[str, ...] = ()
    notes: str = ""
    online_failure: str = "error"  # error | offline
    upload_artifacts: bool = False
    resume: str = "allow"


@dataclass
class RuntimeConfig:
    device: str = "auto"
    cuda_device: int | None = None
    probe_kernels: bool = False
    fused_adamw: str = "auto"
    smoke: bool = False


@dataclass
class EvaluationConfig:
    batch_size: int = 8
    seed: int = 1729
    # 0 means the entire test split, independently of short training validation.
    test_samples: int = 0
    # 0 uses remaining context. This is a loop guard, not a search budget:
    # hitting it is a failure and never triggers an oracle/forced answer.
    max_new_tokens: int = 0


@dataclass
class AppConfig:
    output_dir: str
    data: DataConfig
    model: ModelConfig = field(default_factory=ModelConfig)
    protocol: ProtocolConfig = field(default_factory=ProtocolConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    def validate(self) -> None:
        # Smoke is an ephemeral check, never a saved run, even with CLI overrides.
        if self.runtime.smoke:
            self.train.persist_outputs = False
            self.train.save_interval = 0
        if not self.train.persist_outputs:
            self.tracking.mode = "disabled"
        _require(self.data.knowledge in {"context_graph", "parametric_graph"}, "data.knowledge")
        _require(self.data.n_nodes > 1 and self.data.n_edges >= 0, "data.n_nodes/n_edges")
        _require(self.data.node_vocab >= self.data.n_nodes, "node_vocab must cover N nodes")
        _require(self.data.shapes, "at least one shape")
        _require(self.data.max_attempts >= 1, "max_attempts")
        _require(self.data.workers >= 1 and self.data.chunk_size >= 1, "workers/chunk_size")
        for name, shape in self.data.shapes.items():
            product = shape.layers * shape.width
            if product != self.data.n_nodes:
                raise ConfigError(
                    f"shape {name}: layers*width={product} != n_nodes={self.data.n_nodes}"
                )
            if shape.layers < 2 or shape.width < 1:
                raise ConfigError(f"shape {name} needs layers>=2 and width>=1")
            candidates = (shape.layers - 1) * shape.width * shape.width
            if self.data.n_edges > candidates:
                raise ConfigError(
                    f"shape {name}: n_edges={self.data.n_edges} exceeds {candidates} candidate edges"
                )
        n_shapes = len(self.data.shapes)
        for split_name, count in (
            ("train", self.data.splits.train),
            ("val", self.data.splits.val),
            ("test", self.data.splits.test),
        ):
            if count < 0 or count % n_shapes != 0:
                raise ConfigError(f"split {split_name}={count} must be divisible by {n_shapes} shapes")
            per_shape = count // n_shapes
            if per_shape % 2 != 0:
                raise ConfigError(f"split {split_name} per shape ({per_shape}) must be even for 1:1 labels")
        if abs(sum(self.data.query_ratios) - 1.0) > 1e-6:
            raise ConfigError("query_ratios must sum to 1")
        _require(self.protocol.name in {"direct", "path", "dfs_state", "filler"}, "protocol.name")
        _require(self.protocol.filler_tokens is None or self.protocol.filler_tokens > 0, "filler_tokens")
        _require(self.model.ffn in {"standard", "shared_dense"}, "model.ffn")
        if self.model.ffn not in {"standard", "shared_dense"}:
            raise ConfigError("unsupported ffn")
        _require(self.model.attn_impl in {"sdpa", "math"}, "attn_impl")
        _require(self.model.position_encoding == "rope", "only rope is implemented")
        _require(self.model.activation == "gelu", "only gelu is implemented")
        _require(self.model.heads > 0 and self.model.d_model > 0 and self.model.d_ff > 0, "model dimensions")
        if self.model.d_model % self.model.heads != 0:
            raise ConfigError("d_model must be divisible by heads")
        _require((self.model.d_model // self.model.heads) % 2 == 0, "RoPE head dimension must be even")
        if self.model.heads < 1 or self.model.layers < 1:
            raise ConfigError("layers/heads must be positive")
        _require(self.model.tie_embeddings, "this project ties embeddings; set tie_embeddings true")
        _require(self.train.bucket_multiple >= 1, "bucket_multiple")
        worst = _worst_sequence_len(self)
        worst = ((worst + self.train.bucket_multiple - 1) // self.train.bucket_multiple) * self.train.bucket_multiple
        if worst > self.model.max_seq_len:
            raise ConfigError(
                f"complete sequence overflow: conservative bound needs {worst} tokens "
                f"but max_seq_len={self.model.max_seq_len}"
            )
        _require(self.train.precision in {"bf16", "fp32"}, "precision")
        _require(self.train.stage in {"context", "A", "B"}, "train.stage")
        _require(len(self.train.betas) == 2, "betas")
        _require(self.train.effective_query_batch_size >= 1, "batch")
        _require(self.train.max_steps >= 1, "max_steps")
        _require(self.train.microbatch_size >= 0 and self.train.max_tokens_per_microbatch >= 0, "microbatch limits")
        _require(self.evaluation.batch_size >= 1 and self.evaluation.test_samples >= 0
                 and self.evaluation.max_new_tokens >= 0, "evaluation limits")
        if self.train.stage in {"A", "B"}:
            _require(self.train.walk_batch_size > 0, "stages A/B require a positive walk_batch_size")
        _require(self.tracking.resume in {"allow", "must", "never"}, "tracking.resume")
        _require(self.tracking.mode in {"online", "offline", "disabled"}, "tracking.mode")
        _require(self.tracking.online_failure in {"error", "offline"}, "online_failure")
        _require(self.runtime.fused_adamw in {"auto", "enabled", "disabled"}, "fused_adamw")
        _require(self.runtime.device in {"auto", "cpu", "cuda"}, "device")
        if self.train.stage in {"A", "B"} and self.data.knowledge != "parametric_graph":
            raise ConfigError("stages A/B require parametric_graph")
        if self.train.stage == "context" and self.data.knowledge != "context_graph":
            raise ConfigError("stage context requires context_graph")
        if self.data.knowledge == "parametric_graph" and self.train.stage in {"A", "B", "context"}:
            # Training a parametric world must select one graph. Generation CLI does not.
            pass


def _require(cond: bool, message: str) -> None:
    if not cond:
        raise ConfigError(message)


def max_prompt_tokens(data: DataConfig) -> int:
    """M*(EDGE,u,v) + QUERY,s,t,|, or only QUERY,s,t,|."""
    if data.knowledge == "context_graph":
        return 3 * data.n_edges + 4
    return 4


def filler_length(cfg: AppConfig) -> int:
    return cfg.data.n_nodes if cfg.protocol.filler_tokens is None else cfg.protocol.filler_tokens


def dfs_body_bound(data: DataConfig) -> int:
    """Conservative full-trace bound for an adjacent-layer DAG.

    <=2N-1 records, <=N^2 stack tokens, <=N(2N-1) VIS tokens.
    Each out-neighbor is consumed once; candidate-list sum is at most
    sum k(k+1)/2 <= M(W+1)/2. Each record has 3 separators, action, end;
    at most N-1 PUSH arguments. No example is filtered by trace length.
    """
    n = data.n_nodes
    w = max(s.width for s in data.shapes.values())
    return n*n + n*(2*n-1) + (data.n_edges*(w+1)+1)//2 + 5*(2*n-1) + n-1


def _worst_sequence_len(cfg: AppConfig) -> int:
    if cfg.train.stage == "A":
        return max(cfg.data.walk_lengths) + 3  # WALK, start, transitions, EOS
    prompt = max_prompt_tokens(cfg.data)
    if cfg.protocol.name == "direct":
        body = 0
    elif cfg.protocol.name == "filler":
        body = filler_length(cfg)
    elif cfg.protocol.name == "path":
        body = max(s.layers for s in cfg.data.shapes.values())
    else:
        body = dfs_body_bound(cfg.data)
    longest = prompt + body + ANSWER_TAIL_TOKENS
    return max(longest, max(cfg.data.walk_lengths) + 3) if cfg.train.stage == "B" else longest


def load_config(paths: list[str | Path], overrides: list[str] | None = None) -> AppConfig:
    raw: dict[str, Any] = {}
    for path in paths:
        raw = deep_merge(raw, _load_yaml_tree(Path(path)))
    # Materialize typed defaults so valid optional settings are CLI-overridable
    # even when an experiment YAML does not repeat every default field.
    raw = to_plain(_app_from_dict(raw))
    for item in overrides or []:
        _apply_override(raw, item)
    cfg = _app_from_dict(raw)
    cfg.validate()
    return cfg


def dump_config(cfg: AppConfig) -> str:
    return yaml.safe_dump(to_plain(cfg), sort_keys=False, allow_unicode=True)


def to_plain(obj: Any) -> Any:
    if is_dataclass(obj):
        return {f.name: to_plain(getattr(obj, f.name)) for f in fields(obj)}
    if isinstance(obj, dict):
        return {str(k): to_plain(v) for k, v in obj.items()}
    if isinstance(obj, tuple):
        return [to_plain(v) for v in obj]
    if isinstance(obj, list):
        return [to_plain(v) for v in obj]
    return obj


def deep_merge(base: Mapping[str, Any], update: Mapping[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(dict(base))
    for key, value in update.items():
        if key in out and isinstance(out[key], dict) and isinstance(value, dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _load_yaml_tree(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ConfigError(f"config not found: {path}")
    data = yaml.safe_load(path.read_text()) or {}
    if not isinstance(data, dict):
        raise ConfigError(f"{path} must be a mapping")
    includes = data.pop("includes", []) or []
    merged: dict[str, Any] = {}
    for rel in includes:
        inc = (path.parent / rel).resolve()
        merged = deep_merge(merged, _load_yaml_tree(inc))
    return deep_merge(merged, data)


def _apply_override(raw: dict[str, Any], item: str) -> None:
    if "=" not in item:
        raise ConfigError(f"override must be key=value, got {item}")
    dotted, raw_value = item.split("=", 1)
    value = yaml.safe_load(raw_value)
    cursor: Any = raw
    keys = dotted.split(".")
    for key in keys[:-1]:
        if key not in cursor or not isinstance(cursor[key], dict):
            raise ConfigError(f"cannot set unknown path {dotted}")
        cursor = cursor[key]
    if keys[-1] not in cursor and keys[0] not in raw:
        raise ConfigError(f"cannot set unknown path {dotted}")
    # Allow creating a leaf that already exists. Reject brand new top-level names
    # later in the dataclass parser. Nested missing leaves are rejected here if
    # the parent exists but the leaf does not — overrides may change values only.
    if keys[-1] not in cursor:
        raise ConfigError(f"unknown config key in override: {dotted}")
    cursor[keys[-1]] = value


def _reject_unknown(data: Mapping[str, Any], allowed: set[str], where: str) -> None:
    unknown = set(data) - allowed
    if unknown:
        raise ConfigError(f"unknown config keys at {where}: {sorted(unknown)}")


def _need(data: Mapping[str, Any], key: str, where: str) -> Any:
    if key not in data:
        raise ConfigError(f"missing {where}.{key}")
    return data[key]


def _as_int_tuple(value: Any, where: str) -> tuple[int, ...]:
    if not isinstance(value, (list, tuple)) or not value:
        raise ConfigError(f"{where} must be a non-empty list")
    return tuple(int(v) for v in value)


def _as_float_tuple(value: Any, n: int, where: str) -> tuple[float, ...]:
    if not isinstance(value, (list, tuple)) or len(value) != n:
        raise ConfigError(f"{where} must have {n} numbers")
    return tuple(float(v) for v in value)


def _app_from_dict(raw: Mapping[str, Any]) -> AppConfig:
    _reject_unknown(
        raw,
        {"output_dir", "data", "model", "protocol", "train", "tracking", "runtime", "evaluation"},
        "root",
    )
    if "output_dir" not in raw or "data" not in raw:
        raise ConfigError("output_dir and data are required")
    return AppConfig(
        output_dir=str(raw["output_dir"]),
        data=_data_from_dict(raw["data"]),
        model=_model_from_dict(raw.get("model", {})),
        protocol=_protocol_from_dict(raw.get("protocol", {})),
        train=_train_from_dict(raw.get("train", {})),
        tracking=_tracking_from_dict(raw.get("tracking", {})),
        runtime=_runtime_from_dict(raw.get("runtime", {})),
        evaluation=_fill(EvaluationConfig, raw.get("evaluation", {}), "evaluation"),
    )


def _data_from_dict(raw: Mapping[str, Any]) -> DataConfig:
    allowed = {f.name for f in fields(DataConfig)}
    _reject_unknown(raw, allowed, "data")
    shapes_raw = _need(raw, "shapes", "data")
    if not isinstance(shapes_raw, dict) or not shapes_raw:
        raise ConfigError("data.shapes must be a mapping")
    shapes: dict[str, ShapeSpec] = {}
    for name, spec in shapes_raw.items():
        if not isinstance(spec, dict):
            raise ConfigError(f"shape {name} must be a mapping")
        _reject_unknown(spec, {"layers", "width"}, f"data.shapes.{name}")
        shapes[str(name)] = ShapeSpec(layers=int(spec["layers"]), width=int(spec["width"]))
    splits_raw = _need(raw, "splits", "data")
    _reject_unknown(splits_raw, {"train", "val", "test"}, "data.splits")
    splits = SplitsSpec(
        train=int(splits_raw["train"]),
        val=int(splits_raw["val"]),
        test=int(splits_raw["test"]),
    )
    kwargs: dict[str, Any] = {
        "knowledge": str(_need(raw, "knowledge", "data")),
        "n_nodes": int(_need(raw, "n_nodes", "data")),
        "n_edges": int(_need(raw, "n_edges", "data")),
        "shapes": shapes,
        "splits": splits,
        "root": str(_need(raw, "root", "data")),
    }
    if "graph_seeds" in raw:
        kwargs["graph_seeds"] = _as_int_tuple(raw["graph_seeds"], "graph_seeds")
    if "walk_lengths" in raw:
        kwargs["walk_lengths"] = _as_int_tuple(raw["walk_lengths"], "walk_lengths")
    if "query_ratios" in raw:
        kwargs["query_ratios"] = _as_float_tuple(raw["query_ratios"], 3, "query_ratios")
    for key in (
        "node_vocab",
        "data_seed",
        "max_attempts",
        "workers",
        "chunk_size",
        "min_edge_coverage",
        "n_walks",
    ):
        if key in raw:
            kwargs[key] = int(raw[key])
    if "audit_fraction" in raw:
        kwargs["audit_fraction"] = float(raw["audit_fraction"])
    if "graph_shape" in raw and raw["graph_shape"] is not None:
        kwargs["graph_shape"] = str(raw["graph_shape"])
    if "graph_seed" in raw and raw["graph_seed"] is not None:
        kwargs["graph_seed"] = int(raw["graph_seed"])
    return DataConfig(**kwargs)


def _fill(cls: type, raw: Mapping[str, Any], where: str, converters: dict[str, Any] | None = None):
    allowed = {f.name for f in fields(cls)}
    _reject_unknown(raw, allowed, where)
    base = cls()
    values = to_plain(base)
    # to_plain turns tuples into lists; rebuild from the instance instead.
    values = {f.name: getattr(base, f.name) for f in fields(cls)}
    converters = converters or {}
    for key, value in raw.items():
        if key in converters:
            values[key] = converters[key](value)
        else:
            current = values[key]
            if isinstance(current, bool):
                values[key] = bool(value)
            elif isinstance(current, int) and not isinstance(current, bool):
                values[key] = int(value)
            elif isinstance(current, float):
                values[key] = float(value)
            elif isinstance(current, str):
                values[key] = str(value)
            else:
                values[key] = value
    return cls(**values)


def _model_from_dict(raw: Mapping[str, Any]) -> ModelConfig:
    return _fill(ModelConfig, raw, "model")


def _protocol_from_dict(raw: Mapping[str, Any]) -> ProtocolConfig:
    return _fill(
        ProtocolConfig,
        raw,
        "protocol",
        {"filler_tokens": lambda v: None if v is None else int(v), "name": str},
    )


def _train_from_dict(raw: Mapping[str, Any]) -> TrainConfig:
    return _fill(
        TrainConfig,
        raw,
        "train",
        {"betas": lambda v: _as_float_tuple(v, 2, "train.betas")},
    )


def _tracking_from_dict(raw: Mapping[str, Any]) -> TrackingConfig:
    def _opt_str(v: Any) -> str | None:
        return None if v is None else str(v)

    return _fill(
        TrackingConfig,
        raw,
        "tracking",
        {
            "tags": lambda v: tuple(str(x) for x in v),
            "entity": _opt_str,
            "name": _opt_str,
        },
    )


def _runtime_from_dict(raw: Mapping[str, Any]) -> RuntimeConfig:
    def _opt_int(v: Any) -> int | None:
        return None if v is None else int(v)

    return _fill(RuntimeConfig, raw, "runtime", {"cuda_device": _opt_int})
