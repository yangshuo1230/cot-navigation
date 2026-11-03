import json
from types import SimpleNamespace

import pytest
import torch

from cot_navigation.config import EvaluationConfig, TrainConfig
from cot_navigation.training.trainer import Trainer


@pytest.mark.parametrize('free_complete', [False, True])
def test_resume_finishes_missing_validation_without_repeating_updates(tmp_path, free_complete):
    t = object.__new__(Trainer)
    t.cfg = SimpleNamespace(train=TrainConfig(max_steps=1, eval_interval=1,
        free_eval_interval=1, save_interval=1), evaluation=EvaluationConfig(final_checkpoints=()))
    t.raw, t.completed, t._resuming, t.persist, t.out = torch.nn.Linear(1, 1), 1, True, True, tmp_path
    (tmp_path / 'eval').mkdir()
    (tmp_path / 'eval/teacher_forced_00000001.json').write_text(json.dumps({'step': 1}))
    if free_complete:
        (tmp_path / 'eval/free_00000001.json').write_text(json.dumps({'step': 1}))
    calls, saves = [], []
    t._evaluate = lambda kind: calls.append(kind)
    t._save = lambda: saves.append(t.completed)
    t.tracker = SimpleNamespace(finish=lambda: None)
    t.fit()
    assert calls == ([] if free_complete else ['free'])
    assert saves == ([1] if free_complete else [1, 1])
