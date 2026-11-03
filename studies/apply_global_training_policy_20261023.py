from pathlib import Path
import hashlib
import json
import re

ROOT = Path(str(next(_p for _p in __import__('pathlib').Path(__file__).resolve().parents if (_p / 'pyproject.toml').is_file() and (_p / 'studies').is_dir())))
REVIEW = ROOT / 'studies/global_training_policy'
REVIEW.mkdir(parents=True, exist_ok=True)
changed = {}


def write(name, text):
    p = ROOT / name
    before = p.read_bytes() if p.exists() else b''
    if before == text.encode():
        return
    snapshot = REVIEW / 'before' / name
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    assert not snapshot.exists(), name
    snapshot.write_bytes(before)
    changed[name] = hashlib.sha256(before).hexdigest()
    p.write_text(text)


p = ROOT / 'src/cot_navigation/config.py'
s = p.read_text().replace('eval_interval: int = 1000', 'eval_interval: int = 20000', 1)
s = s.replace('log_interval: int = 10', 'log_interval: int = 100', 1)
s = s.replace('free_batch_size: int = 0 ', 'free_batch_size: int = 64 ', 1)
s = s.replace('free_max_cache_mb: int = 0 ', 'free_max_cache_mb: int = 8192 ', 1)
s = s.replace('free_max_prefill_tokens: int = 0 ', 'free_max_prefill_tokens: int = 65536 ', 1)
s = s.replace('compile_generation: bool = False  # GPU opt-in; CPU retains reference eager decode',
              'compile_generation: bool | None = None  # auto: CUDA full attention; CPU/SWA use reference')
s = s.replace('test_samples: int = 0', 'test_samples: int = 256', 1)
s = s.replace('final_checkpoints: tuple[str, ...] = ()', 'final_checkpoints: tuple[str, ...] = ("last",)', 1)
s = s.replace('trace_diagnostics: bool = True', 'trace_diagnostics: bool = False', 1)
s = s.replace('    selection_min_step: int = 0',
              '    extra_query_views: bool = False  # opt-in specialized short-hop/block holdout views\n    selection_min_step: int = 0', 1)
a = '        # Smoke is an ephemeral check, never a saved run, even with CLI overrides.'
s = s.replace(a, '        if self.evaluation.compile_generation is None:\n            self.evaluation.compile_generation = not bool(self.model.swa_records)\n' + a, 1)
a = 'def load_config(paths: list[str | Path], overrides: list[str] | None = None) -> AppConfig:'
helper = '''def apply_training_profile(raw: Mapping[str, Any]) -> dict:
    """Apply current defaults when creating a NEW trial from a historical config.

    Saved/resolved run configurations are never rewritten by the loader.
    Explicit experiment/CLI overrides should be applied after this function.
    """
    profile = Path(__file__).resolve().parents[2] / "configs/train/throughput.yaml"
    out = deep_merge(raw, _load_yaml_tree(profile))
    if out.get("train", {}).get("stage") == "A":
        out["evaluation"]["final_checkpoints"] = []
    if out.get("model", {}).get("swa_records", 0):
        out["evaluation"]["compile_generation"] = False
        out["model"]["compile"] = False
    return out


'''
assert a in s
s = s.replace(a, helper + a, 1)
write('src/cot_navigation/config.py', s)

profile = '''# Shared defaults for new trials; explicit later YAML/CLI values still win.
train:
  max_tokens_per_microbatch: 65536
  max_padding_fraction: 0.30
  compile_loss: true
  log_interval: 100
  eval_interval: 20000
  free_eval_interval: 20000
  save_interval: 10000
evaluation:
  free_batch_size: 64
  free_max_cache_mb: 8192
  free_max_prefill_tokens: 65536
  compile_generation: true
  test_samples: 256
  final_checkpoints: [last]
  extra_sets: []
  extra_query_views: false
  trace_diagnostics: false
'''
write('configs/train/throughput.yaml', profile)

# Production presets must not override the new cadence with legacy 1k/5k.
# Generated configurations and outputs remain immutable historical records.
for p in sorted((ROOT / 'configs').rglob('*.yaml')):
    if 'generated' in p.parts or p.name in {'smoke.yaml', 'throughput.yaml'}:
        continue
    s = p.read_text()
    s = re.sub(r'(?m)^(  eval_interval:) (1000|5000)$', r'\1 20000', s)
    s = re.sub(r'(?m)^(  log_interval:) 10$', r'\1 100', s)
    s = s.replace('max_tokens_per_microbatch: 16384', 'max_tokens_per_microbatch: 65536')
    s = s.replace('final_checkpoints: [best, last]', 'final_checkpoints: [last]')
    s = re.sub(r'(?m)(  final_checkpoints:\n)  - best\n  - last', r'\1  - last', s)
    s = s.replace('  test_samples: 0', '  test_samples: 256')
    if '  eval_interval: 20000' in s and 'free_eval_interval:' not in s:
        s = s.replace('  eval_interval: 20000', '  eval_interval: 20000\n  free_eval_interval: 20000')
    write(p.relative_to(ROOT).as_posix(), s)

# Existing full attention now uses the same aligned decode views as dual.
p = ROOT / 'src/cot_navigation/models/attention.py'
s = p.read_text()
a = '            end = cache.valid_length\n'
assert s.count(a) == 1
s = s.replace(a, '            valid = cache.max_len if cache.valid_length is None else cache.valid_length\n'
                 '            end = min(cache.max_len, ((valid + 7) // 8) * 8)\n', 1)
write(p.relative_to(ROOT).as_posix(), s)

# Launchers that clone old resolved configs must explicitly adopt the profile.
for name in ['train_self_correction.py', 'train_natural_correction.py', 'train_curriculum_loss_swa.py']:
    p = ROOT / 'scripts' / name
    s = p.read_text().replace('from cot_navigation.config import load_config',
                            'from cot_navigation.config import load_config, apply_training_profile', 1)
    if name == 'train_self_correction.py':
        s = s.replace('cfg = yaml.safe_load(original.read_text())', 'cfg = apply_training_profile(yaml.safe_load(original.read_text()))')
    else:
        s = s.replace('cfg=yaml.safe_load(original.read_text())', 'cfg=apply_training_profile(yaml.safe_load(original.read_text()))')
    if name == 'train_curriculum_loss_swa.py':
        s = s.replace("if window:cfg['model']['compile']=False", "if window:\n                    cfg['model']['compile']=False\n                    cfg['evaluation']['compile_generation']=False")
    write(p.relative_to(ROOT).as_posix(), s)

p = ROOT / 'scripts/train_hard_dag.py';s = p.read_text()
s = s.replace("'train.eval_interval=5000'", "'train.eval_interval=20000', 'train.free_eval_interval=20000'")
s = s.replace("'train.save_interval=5000'", "'train.save_interval=10000'")
s = s.replace("'evaluation.final_checkpoints=[best,last]'", "'evaluation.final_checkpoints=[last]'")
s = s.replace("f'evaluation.extra_sets={json.dumps(extras)}'", "'evaluation.extra_sets=[]'")
write(p.relative_to(ROOT).as_posix(), s)

p = ROOT / 'scripts/scale.py';s = p.read_text()
s = s.replace("'train.eval_interval=5000'", "'train.eval_interval=20000', 'train.free_eval_interval=20000'")
s = s.replace("'train.save_interval=5000'", "'train.save_interval=10000'")
s = s.replace("f'evaluation.extra_sets={json.dumps(extra_sets(setting, size, args))}'", "'evaluation.extra_sets=[]'")
s = s.replace("f'evaluation.extra_sets={json.dumps(extra_sets(\"fixed\", 10, args, transfer=True))}'", "'evaluation.extra_sets=[]'")
write(p.relative_to(ROOT).as_posix(), s)

for name in ['block_query_control.py', 'short_hop_control.py']:
    p = ROOT / 'scripts' / name;s = p.read_text()
    s = s.replace('from cot_navigation.config import load_config', 'from cot_navigation.config import load_config, apply_training_profile')
    if name == 'short_hop_control.py':
        s = s.replace("c=yaml.safe_load((old/protocol/'resolved_config.yaml').read_text())", "c=apply_training_profile(yaml.safe_load((old/protocol/'resolved_config.yaml').read_text()))")
        s = s.replace('eval_interval=5000,save_interval=5000', 'eval_interval=20000,free_eval_interval=20000,save_interval=10000')
        s = s.replace("else ['best','last']", "else ['last']")
    else:
        s = s.replace("template=yaml.safe_load((ROOT/PREVIOUS/'configs'/f'short3_{protocol}.yaml').read_text())", "template=apply_training_profile(yaml.safe_load((ROOT/PREVIOUS/'configs'/f'short3_{protocol}.yaml').read_text()))")
        s = s.replace("c['evaluation'].update(test_samples=10000)", "c['evaluation'].update(test_samples=256)")
    s = s.replace('            super()._evaluate_final_checkpoints()\n', '            super()._evaluate_final_checkpoints()\n            if not self.cfg.evaluation.extra_query_views:\n                return\n')
    write(p.relative_to(ROOT).as_posix(), s)

# New dual trials use the shared profile. Resume always uses saved generated YAML.
p = ROOT / 'scripts/train_dual_attention.py';s = p.read_text()
a = '''                cfg = load_config([reference, ROOT / 'configs/train/throughput.yaml'], ['''
b = '''                path = configs / (name.replace('/', '__') + '.yaml')
                if args.resume and path.exists():
                    saved = load_config([path])
                    jobs.append(engine.Job(name, path, [], output / name, saved.train.max_steps))
                    continue
                cfg = load_config([reference, ROOT / 'configs/train/throughput.yaml'], ['''
assert a in s;s = s.replace(a, b, 1)
s = s.replace('# Keep full best/last and all scale/shape tests; only rollout throughput changes.',
              '# Default to last/main; add best or extra sets explicitly for follow-up studies.')
write(p.relative_to(ROOT).as_posix(), s)

REVIEW.joinpath('before_hashes.json').write_text(json.dumps(changed, indent=2) + '\n')
print(json.dumps({'changed_files': len(changed), 'files': list(changed)}, indent=2))
