"""One configuration-driven entry point for training and research workflows.

Planning is read-only. Execution uses argv lists, never shell strings.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, asdict
import itertools
import json
import os
from pathlib import Path
import re
import subprocess
import sys

import yaml

ROOT = Path(__file__).resolve().parents[3]


@dataclass
class Step:
    name: str
    argv: list[str]
    depends_on: list[str]


def inside(path, root=ROOT):
    p = Path(path)
    p = (root / p).resolve() if not p.is_absolute() else p.resolve()
    if not p.is_relative_to(root.resolve()):
        raise ValueError(f'Path must stay in the independent project: {p}')
    return p


def expand(text, values):
    for key, value in values.items():
        text = text.replace('${' + key + '}', str(value))
    if '${' in text:
        raise ValueError(f'Unresolved template variable: {text}')
    return text


def plan(path, *, root=ROOT, run_name=None):
    raw = yaml.safe_load(Path(path).read_text())
    if not isinstance(raw, dict) or set(raw) - {'version','run_name','steps'} or raw.get('version') != 1:
        raise ValueError('Workflow requires version: 1, run_name and steps')
    name = run_name or raw.get('run_name')
    if not isinstance(name, str) or not re.fullmatch(r'[A-Za-z0-9_-][A-Za-z0-9_.-]*', name):
        raise ValueError('Invalid run_name')
    if not isinstance(raw.get('steps'), list) or not raw['steps']:
        raise ValueError('steps must be a nonempty list')
    result = []
    for entry in raw['steps']:
        allowed = {'name','kind','configs','set','args','script','module','matrix','depends_on','action'}
        if not isinstance(entry, dict) or set(entry) - allowed:
            raise ValueError(f'Unknown step fields: {entry}')
        matrix = entry.get('matrix', {})
        if not isinstance(matrix, dict) or any(not isinstance(v, list) or not v for v in matrix.values()):
            raise ValueError('matrix values must be nonempty lists')
        for combination in itertools.product(*matrix.values()):
            variables = {**dict(zip(matrix, combination)), 'run': name, 'root': str(root)}
            step_name = expand(entry['name'], variables)
            if not re.fullmatch(r'[A-Za-z0-9_-][A-Za-z0-9_./-]*', step_name) or '..' in step_name.split('/'):
                raise ValueError(f'Invalid step name: {step_name}')
            args = entry.get('args', [])
            if not isinstance(args, list) or any(not isinstance(a, str) for a in args):
                raise ValueError('args must be a list of strings')
            args = [expand(a, variables) for a in args]
            kind = entry['kind']
            argv = [sys.executable]
            if kind in {'train','evaluate','data','benchmark'}:
                argv += ['-m','cot_navigation.cli',kind]
                if kind == 'data':
                    action = entry.get('action', 'prepare')
                    if action not in {'generate','validate','prepare'}:
                        raise ValueError(f'Invalid data action: {action}')
                    argv += [action]
                configs = entry.get('configs', [])
                if not isinstance(configs, list) or not configs:
                    raise ValueError('Typed steps require configs')
                for config in configs:
                    p = inside(expand(config, variables), root)
                    if not p.is_file():
                        raise FileNotFoundError(p)
                    argv += ['--config',str(p)]
                overrides = entry.get('set', {})
                if not isinstance(overrides, dict):
                    raise ValueError('set must map dotted keys to values')
                override_args = []
                for key, value in overrides.items():
                    if isinstance(value, str):
                        value = expand(value, variables)
                    override_args += [f'{key}={json.dumps(value)}']
                # Output is owned by the workflow and isolated per matrix cell.
                output = inside(Path('outputs/workflows')/name/step_name, root)
                override_args += [f'output_dir={json.dumps(str(output))}']
                from cot_navigation.config import load_config
                config_paths = [argv[i+1] for i, a in enumerate(argv) if a == '--config']
                cfg = load_config(config_paths, override_args)
                for p in [cfg.output_dir, cfg.data.root, cfg.protocol.mistake_cache_root,
                          *cfg.train.scale_roots, *cfg.train.world_roots,
                          *[s.split('=',1)[1] for s in cfg.evaluation.extra_sets]]:
                    inside(p, root)
                for override in override_args:
                    argv += ['--set', override]
            elif kind in {'research','legacy','packed'}:
                if 'script' in entry and 'module' in entry:
                    raise ValueError('Choose script or module')
                if 'script' in entry:
                    p = inside(expand(entry['script'], variables), root)
                    if not p.is_file() or p.suffix != '.py':
                        raise ValueError(f'Expected Python script: {p}')
                    argv += [str(p)]
                elif 'module' in entry:
                    module = entry['module']
                    if not module.startswith('cot_navigation.') or not re.fullmatch(r'[A-Za-z0-9_.]+', module):
                        raise ValueError('Module must be in cot_navigation')
                    argv += ['-m', module]
                else:
                    raise ValueError('Research/packed/legacy steps require script or module')
            else:
                raise ValueError(f'Unknown kind: {kind}')
            argv += args
            deps = entry.get('depends_on', [])
            if not isinstance(deps, list):
                raise ValueError('depends_on must be a list')
            result.append(Step(step_name, argv, [expand(d, variables) for d in deps]))
    lookup = {s.name: s for s in result}
    if len(lookup) != len(result):
        raise ValueError('Duplicate expanded step names')
    ordered, pending, done = [], list(result), set()
    while pending:
        ready = [s for s in pending if set(s.depends_on) <= done]
        if not ready:
            raise ValueError('Dependency cycle or unknown dependency')
        for step in ready:
            ordered.append(step)
            pending.remove(step)
            done.add(step.name)
    return name, ordered


def execute(name, steps, *, root=ROOT):
    from cot_navigation.experiments.scheduler import file_lock, write_json
    output = inside(Path('outputs/workflows')/name, root)
    # A workflow is immutable and single-use. Individual backend resume flags
    # remain available, but never silently re-run completed workflow steps.
    output.mkdir(parents=True, exist_ok=False)
    env = {**os.environ, 'PYTHONPATH': str(root/'src'), 'PYTHONDONTWRITEBYTECODE': '1',
           'COT_NAV_PROTECTED_ROOT': str(root.with_name('cot-navigation')),
           'PYTHONUNBUFFERED': '1'}
    with file_lock(output/'.workflow.lock', blocking=False):
        write_json(output/'plan.json', [asdict(s) for s in steps])
        completed = []
        for step in steps:
            log = output/(step.name.replace('/','__')+'.log')
            with log.open('w') as stream:
                proc = subprocess.run(step.argv, cwd=root, env=env, stdout=stream, stderr=subprocess.STDOUT)
            write_json(output/'status.json', {'completed': completed, 'current': step.name, 'exit_code': proc.returncode})
            if proc.returncode:
                return proc.returncode
            completed.append(step.name)
        write_json(output/'status.json', {'completed': completed, 'exit_code': 0})
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('config', type=Path)
    p.add_argument('--run-name')
    p.add_argument('--execute', action='store_true')
    a = p.parse_args(argv)
    name, steps = plan(a.config, run_name=a.run_name)
    print(json.dumps({'run_name':name, 'steps':[asdict(s) for s in steps]}, indent=2))
    return execute(name, steps) if a.execute else 0


if __name__ == '__main__':
    raise SystemExit(main())
