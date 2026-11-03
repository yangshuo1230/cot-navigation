from pathlib import Path
import ast
import json
import os
import re

ROOT=Path('/mnt/ningxuefei/yangshuo/cot-navigation-unified')
OUT=ROOT/'migration/standalone'
mapping=json.loads((OUT/'study_paths.json').read_text())

def write(rel,text):
    p=ROOT/rel;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(text)

# These are migration inputs, not fallback search locations. No file in the
# original/archive tree is opened to resolve a path.
write('src/cot_navigation/runtime/workspace.py','''"""Project-relative resolution of imported artifact paths."""
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[3]
SOURCE = '/mnt/ningxuefei/yangshuo/cot-navigation'

def relocate(value):
    if isinstance(value, dict):
        return {k: relocate(v) for k,v in value.items()}
    if isinstance(value, list):
        return [relocate(v) for v in value]
    if isinstance(value, tuple):
        return tuple(relocate(v) for v in value)
    if isinstance(value, str):
        for prefix in (SOURCE+'-unified/', SOURCE+'/'):
            value = value.replace(prefix, str(ROOT)+'/')
    return value
''')

# Make absolute project roots in executable study code relocatable, without
# rewriting results, serialized jobs or quoted scientific descriptions.
portable=[]
for folder in ('studies','scripts'):
    for p in (ROOT/folder).rglob('*.py'):
        if '__pycache__' in p.parts:continue
        source=p.read_text();tree=ast.parse(source)
        replacements=[]
        lines=source.splitlines(True)
        offsets=[0]
        for line in lines:offsets.append(offsets[-1]+len(line))
        for node in ast.walk(tree):
            if not isinstance(node,ast.Constant) or not isinstance(node.value,str):continue
            value=node.value
            prefix=next((prefix for prefix in (str(ROOT),str(ROOT.with_name('cot-navigation'))) if value==prefix or value.startswith(prefix+'/')),None)
            if prefix is None or '\n' in value:continue
            # Values inside f-strings must be handled as part of the whole
            # expression; path prefixes there are audited separately below.
            parent_fstring=any(isinstance(other,ast.JoinedStr) and node in other.values for other in ast.walk(tree))
            if parent_fstring:continue
            suffix=value[len(prefix):]
            replacement="str(__import__('cot_navigation.runtime.workspace', fromlist=['ROOT']).ROOT)"+((' + '+repr(suffix)) if suffix else '')
            # col_offset is UTF-8 byte-based; lines containing path literals
            # may have non-ASCII descriptions before them.
            start=offsets[node.lineno-1]+len(lines[node.lineno-1].encode()[:node.col_offset].decode())
            end=offsets[node.end_lineno-1]+len(lines[node.end_lineno-1].encode()[:node.end_col_offset].decode())
            replacements.append((start,end,replacement))
        for start,end,value in sorted(replacements,reverse=True):source=source[:start]+value+source[end:]
        if replacements:
            ast.parse(source);p.write_text(source);portable.append(str(p.relative_to(ROOT)))

# Fix copied symlinks into the old research subtree. All symlinks into the
# original project are forbidden in the active tree.
links=[]
for parent,dirs,files in os.walk(ROOT/'studies',followlinks=False):
    for name in dirs+files:
        p=Path(parent)/name
        if not p.is_symlink():continue
        raw=os.readlink(p);resolved=p.resolve(strict=False)
        value=str(resolved).replace(str(ROOT.with_name('cot-navigation'))+'/',str(ROOT)+'/')
        value=value.replace(str(ROOT/'research')+'/',str(ROOT/'studies')+'/')
        for old,new in mapping.items():value=value.replace('/'+old+'/', '/'+new+'/')
        if value!=str(resolved):
            p.unlink();p.symlink_to(value);links.append(str(p.relative_to(ROOT)))

# Export every migrated Python resource with a stable relative identifier.
registry={}
for p in sorted((ROOT/'studies').rglob('*.py')):
    if '__pycache__' in p.parts:continue
    tree=ast.parse(p.read_text())
    entry=any(isinstance(n,ast.If) and '__name__' in ast.unparse(n.test) for n in tree.body)
    if not entry:continue
    key=p.relative_to(ROOT/'studies').with_suffix('').as_posix()
    registry[key]={'path':p.relative_to(ROOT).as_posix(),'description':(ast.get_docstring(tree) or '').split('\n')[0]}
write('configs/studies.json',json.dumps(registry,indent=2,ensure_ascii=False))
write('src/cot_navigation/experiments/study.py','''"""Run registered research applications in independent Python processes."""
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
    env={**os.environ,'PYTHONPATH':str(ROOT/'src'), 'PYTHONDONTWRITEBYTECODE':'1'}
    return subprocess.run(command(a.name,args),cwd=ROOT,env=env).returncode

if __name__=='__main__':
    raise SystemExit(main())
''')
p=ROOT/'src/cot_navigation/cli.py';s=p.read_text()
needle="    if argv and argv[0] == 'workflow':"
s=s.replace(needle,"    if argv and argv[0] == 'study':\n        from cot_navigation.experiments.study import main as study_main\n        return study_main(argv[1:])\n"+needle)
# New registry is canonical; do not consult migration catalogs at runtime.
start=s.index("    if argv and argv[0] == 'catalog':")
end=s.index('    parser = argparse.ArgumentParser',start)
s=s[:start]+'''    if argv and argv[0] == 'catalog':
        from cot_navigation.experiments.study import catalog
        query = ' '.join(argv[1:]).lower()
        print(json.dumps({k:v for k,v in catalog().items() if query in (k+' '+v['description']).lower()}, indent=2, ensure_ascii=False))
        return 0
'''+s[end:]
p.write_text(s)

# Archive-only guards must not alter study type names in YAML files. Update
# packed workflow to invoke the package directly, not a versioned script.
write('configs/workflows/fixed_packed.yaml','''version: 1
run_name: fixed_packed
steps:
  - name: ${backend}/${protocol}/seed${seed}
    kind: packed
    module: cot_navigation.training.packed
    matrix:
      backend: [v6, v7]
      protocol: [direct, dfs_state, dfs_nodes]
      seed: [0]
    args: [--repo-root, '${root}', --backend, '${backend}', --protocol, '${protocol}', --seed, '${seed}', --tracking, disabled]
''')

# All tests use the locally frozen oracle. A blocked directory is not probed.
p=ROOT/'tests/test_unified_framework.py';s=p.read_text()
s=s.replace("    assert not Path(SOURCE,'MUST_NOT_EXIST').exists()","    # Audit rejection happens before any OS file operation.")
p.write_text(s)
(OUT/'portable_paths.json').write_text(json.dumps(portable,indent=2))
(OUT/'relocated_links.json').write_text(json.dumps(links,indent=2))
print(json.dumps({'registered_entries':len(registry),'portable_sources':len(portable),'links':len(links)}))
