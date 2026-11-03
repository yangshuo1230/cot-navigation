"""Run registered research applications in independent Python processes."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from cot_navigation.runtime.workspace import ROOT

def catalog(root=ROOT):
    return json.loads((root / 'configs/studies.json').read_text())

def command(name, args, root=ROOT):
    entries = catalog(root)
    if name not in entries:
        raise ValueError(f'Unknown study: {name}; use study --list')
    path = (root / entries[name]['path']).resolve()
    if not path.is_relative_to((root / 'studies').resolve()) or not path.is_file():
        raise ValueError(f'Study resource is invalid: {path}')
    if 'module' in entries[name]:
        return [sys.executable, '-m', entries[name]['module'], *args]
    return [sys.executable, str(path), *args]

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--list', action='store_true')
    p.add_argument('name', nargs='?')
    p.add_argument('args', nargs=argparse.REMAINDER)
    a=p.parse_args(argv)
    if a.list:
        print(json.dumps(catalog(),indent=2,ensure_ascii=False));return 0
    if not a.name:p.error('name is required unless --list is used')
    args=a.args[1:] if a.args[:1]==['--'] else a.args
    entry=catalog()[a.name]
    paths=[str(ROOT/entry['pythonpath'])] if 'pythonpath' in entry else []
    env={**os.environ,'PYTHONPATH':':'.join(paths+[str(ROOT/'src'),str(ROOT)]), 'PYTHONDONTWRITEBYTECODE':'1'}
    return subprocess.run(command(a.name,args),cwd=ROOT,env=env).returncode

if __name__=='__main__':
    raise SystemExit(main())
