"""Final reports may be reused only for exactly identical weights."""
import json
from types import SimpleNamespace

import pytest
import torch
from cot_navigation.training.trainer import Trainer


@pytest.mark.parametrize('same_weights', [True, False])
def test_final_checkpoint_reuse_checks_weights_and_preserves_reports(tmp_path, same_weights):
    t = object.__new__(Trainer)
    t.raw = torch.nn.Linear(2, 2)
    best = {k: v.clone() for k, v in t.raw.state_dict().items()}
    last = {k: v.clone() + (0 if same_weights else 1) for k, v in best.items()}
    (tmp_path / 'checkpoints').mkdir()
    (tmp_path / 'eval').mkdir()
    for filename, weights in [('best.pt', best), ('latest.pt', last)]:
        # Equal steps alone do not establish equal weights.
        torch.save({'model': weights, 'step': 20000}, tmp_path / 'checkpoints' / filename)
    t.out, t.persist, t.completed, t.scale_members = tmp_path, True, 20000, []
    t.cfg = SimpleNamespace(evaluation=SimpleNamespace(
        final_checkpoints=('best', 'last'), extra_sets=('wide=ignored',)))
    t.tracker = SimpleNamespace(log=lambda *a, **kw: None)
    calls = []
    def evaluate(split, *, checkpoint_tag, checkpoint_step):
        calls.append(checkpoint_tag)
        report = dict(checkpoint_tag=checkpoint_tag, checkpoint_step=checkpoint_step,
                      training_step=20000, generation_wall_time=2., diagnostics_wall_time=.5,
                      overall={'answer_accuracy': .75})
        (tmp_path / 'eval' / f'free_test_{checkpoint_tag}.json').write_text(json.dumps(report))
        (tmp_path / 'eval' / f'predictions_test_{checkpoint_tag}.jsonl').write_text('{"query_id": 3}\n')
    t.evaluate_split = evaluate
    t._evaluate_extra = lambda tag, step, spec: evaluate('test', checkpoint_tag=tag+'-wide', checkpoint_step=step)
    t._evaluate_final_checkpoints()
    assert calls == (['best', 'best-wide'] if same_weights else ['best', 'best-wide', 'last', 'last-wide'])
    for suffix in ['', '-wide']:
        report = json.loads((tmp_path / 'eval' / f'free_test_last{suffix}.json').read_text())
        assert report['checkpoint_tag'] == 'last' + suffix
        assert report['overall']['answer_accuracy'] == .75
        assert ('reused_from_checkpoint_tag' in report) == same_weights
        assert report['generation_wall_time'] == (0. if same_weights else 2.)
        assert (tmp_path / 'eval' / f'predictions_test_last{suffix}.jsonl').read_text() == '{"query_id": 3}\n'
    for key, value in t.raw.state_dict().items():
        torch.testing.assert_close(value, last[key], atol=0, rtol=0)
