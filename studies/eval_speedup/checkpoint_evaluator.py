"""Read-only checkpoint evaluation without training caches or an optimizer."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import torch

from cot_navigation.protocols.tokenizer import Tokenizer
from cot_navigation.models.transformer import build_model
from cot_navigation.training.trainer import Trainer, _arch_snapshot, _resolve_device


class CheckpointEvaluator(Trainer):
    """Reuse the trainer's rendering, sampling and scoring, without training.

    The constructor never writes config, checkpoints, metrics or W&B records.
    Callers explicitly provide a tracker when publishing final reports.
    """

    def __init__(self, cfg, checkpoint, *, persist=False, tracker=None):
        cfg.validate()
        self.cfg, self.persist, self.out = cfg, persist, Path(cfg.output_dir)
        self.device = _resolve_device(cfg)
        self.tok = Tokenizer.from_config(cfg)
        self.store = self.world = None
        self.scale_members = []
        self._active_member = 'primary'
        self._fixed_graphs = {}
        self.render_cache = {}
        self.prepared_tasks = self.prepared_walks = None
        self._load_data()
        self.manifest_hash = (self.store if self.store is not None else self.world).fingerprint
        if self.scale_members:
            value = [(m.root, (m.store if m.store is not None else m.world).fingerprint)
                     for m in self.scale_members]
            self.manifest_hash = hashlib.sha256(json.dumps(value).encode()).hexdigest()
            self._activate(self._primary_member())
        if cfg.protocol.trajectory_variants:
            from cot_navigation.data.scale_prepared import ScalePreparedRows
            source = self.store if self.store is not None else self.world
            self.prepared_tasks = ScalePreparedRows(source, cfg, self.tok)
        else:
            from cot_navigation.data.prepared import PreparedRows
            from cot_navigation.config import filler_length
            try:
                source = self.store if self.store is not None else self.world
                self.prepared_tasks = PreparedRows(source, cfg.protocol.name, self.tok, filler_length(cfg))
            except FileNotFoundError:
                pass
        self.snapshot = _arch_snapshot(cfg, self.tok, self.manifest_hash)
        self.raw = build_model(cfg.model, len(self.tok), pad_id=self.tok.pad).to(self.device)
        self.model = self.raw
        self._load_evaluation(str(checkpoint))
        self.raw.eval()
        self.tracker = tracker or SimpleNamespace(log=lambda *args, **kwargs: None)
