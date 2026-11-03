"""Historical configuration compatibility and explicit policy overrides."""
import copy
from pathlib import Path
import importlib.util
import sys

import pytest
import yaml
from cot_navigation.config import apply_training_profile, load_config, to_plain

ROOT = Path(__file__).resolve().parents[1]


def module(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'scripts' / filename)
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result


def test_legacy_resolved_diagnostics_survive_loading_but_new_profile_wins(tmp_path):
    resolved = tmp_path / 'resolved_config.yaml'
    raw = to_plain(load_config([ROOT / 'configs/experiments/main_context_direct.yaml']))
    raw['evaluation'].pop('trace_diagnostics')
    raw['evaluation'].pop('extra_query_views')
    resolved.write_text(yaml.safe_dump(raw))
    assert load_config([resolved]).evaluation.trace_diagnostics
    assert not load_config([resolved, ROOT / 'configs/train/throughput.yaml']).evaluation.trace_diagnostics
    original = {'train': {'eval_interval': 5000, 'max_steps': 123456},
                'model': {'swa_records': 1, 'compile': True},
                'evaluation': {'extra_sets': ['wide=old'], 'final_checkpoints': ['best', 'last']}}
    saved = copy.deepcopy(original)
    new = apply_training_profile(original)
    assert original == saved
    assert new['train']['max_steps'] == 123456
    assert new['train']['eval_interval'] == 20000
    assert not new['model']['compile'] and not new['evaluation']['compile_generation']
    assert new['evaluation']['final_checkpoints'] == ['last']
    assert not new['evaluation']['extra_sets']


def test_resume_accepts_runtime_policy_changes_but_rejects_loss_change():
    runner = module('policy_formal_runner', 'train_first_round.py')
    args = runner.parse_args(['--run-name', 'policy_unit'])
    _, jobs = runner.build_jobs(args)
    new = runner.plan_record(jobs)
    old = copy.deepcopy(new)
    for entry in old.values():
        entry['config']['train'].update(eval_interval=5000, free_eval_interval=10000,
                                         max_tokens_per_microbatch=16384, log_interval=10)
        entry['config']['evaluation'].update(final_checkpoints=['best', 'last'], trace_diagnostics=True)
        entry['config']['evaluation'].pop('extra_query_views')
    runner.check_resume_plan(old, new)
    bad = copy.deepcopy(new)
    bad['context/direct']['config']['train']['trace_loss_lambda'] = 4.
    with pytest.raises(ValueError, match='differs'):
        runner.check_resume_plan(old, bad)


def test_scale_user_evaluation_overrides_win_over_suite_defaults():
    scale = module('policy_scale_runner', 'scale.py')
    args = scale.parse_args(['--setting', 'context', '--run-name', 'policy_unit', '--dry-run',
        '--set', 'train.eval_interval=40000', '--set', 'train.free_eval_interval=40000',
        '--set', 'evaluation.test_samples=32', '--set', 'evaluation.extra_sets=[wide=data/custom]'])
    _, jobs = scale.build_jobs(args)
    for job in jobs:
        cfg = load_config([job.config], job.overrides)
        assert cfg.train.eval_interval == cfg.train.free_eval_interval == 40000
        assert cfg.evaluation.test_samples == 32
        assert cfg.evaluation.extra_sets == ('wide=data/custom',)
