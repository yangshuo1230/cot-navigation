"""Ephemeral real-data training checks; no run files, tracking or checkpoints."""
import json
import os
from pathlib import Path
import sys
import time

import torch

ROOT = Path(str(next(_p for _p in __import__('pathlib').Path(__file__).resolve().parents if (_p / 'pyproject.toml').is_file() and (_p / 'studies').is_dir())))
os.chdir(ROOT)
sys.path.insert(0, str(ROOT / 'src'))
from cot_navigation.config import load_config
from cot_navigation.training.trainer import Trainer

torch.set_num_threads(2)
reports = []
for name in ('bfs_state__serial__bidirectional',):
    path = ROOT / 'configs/generated/dual_attention_v1/dual_attention_v1_s0_20261023' / (name + '.yaml')
    cfg = load_config([path], ['runtime.smoke=true'])
    trainer = Trainer(cfg)
    assert not trainer.persist and cfg.tracking.mode == 'disabled'
    for stage, completed in [('n16', 0), ('n100', 100000)]:
        trainer.completed = completed
        for i in range(4):
            start = time.perf_counter()
            metrics = trainer._update(collect_metrics=True)
            torch.cuda.synchronize()
            print(json.dumps({'name': name, 'stage': stage, 'update': i,
                              'elapsed': time.perf_counter() - start, 'metrics': metrics}, default=str), flush=True)
    del trainer
    torch.cuda.empty_cache()
