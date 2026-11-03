"""Versioned dependencies; import existing v5 helpers without changing them."""
import hashlib
import importlib.util
from pathlib import Path
import sys
from prepare import HERE, BASE, digest


def load_v5(repo):
    # Keep this version's launcher/evaluator ahead of same-named v5 modules.
    sys.path[:0] = [str(HERE), str(repo / 'src'), str(BASE)]
    legacy = HERE.parent / 'fixed_navigation_v5'
    sys.path.insert(3, str(legacy))
    spec = importlib.util.spec_from_file_location('frozen_v5_helpers', legacy / 'train.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def source_hash(repo):
    paths = list(HERE.glob('*.py')) + [HERE / 'experiment.json']
    legacy = HERE.parent / 'fixed_navigation_v5'
    paths += [legacy / 'train.py', legacy / 'data.py']
    paths += list((repo / 'src/cot_navigation/models').glob('*.py'))
    paths += list((repo / 'src/cot_navigation/protocols').glob('*.py'))
    paths += [repo / 'src/cot_navigation/evaluation/rollout.py',
              repo / 'studies/fixed_adjacency_suite/train_suite.py']
    paths += [repo / 'src/cot_navigation' / p for p in (
        'training/packed.py', 'training/packed_loss.py', 'data/packed_bank.py',
        'evaluation/packed.py', 'experiments/artifacts.py')]
    return digest({p.relative_to(repo).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)})
