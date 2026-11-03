"""Read-only dependency/configuration and migration-completeness audit."""
from pathlib import Path
import ast
import hashlib
import json
import os
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
mapping=json.loads((ROOT/'configs/study_paths.json').read_text())
report={'missing_migrated_files':[], 'archive_dependencies':[], 'bad_symlinks':[],
        'syntax_errors':[], 'configs':[], 'workflows':[], 'file_mapping':[]}
# Source inventory is a previously captured manifest, not a runtime dependency
# on either old project directory. Files added by the prior phase are audited
# directly via the retained archive outside isolated test execution.
baseline=json.loads((ROOT/'migration/source_baseline.json').read_text())
for rel in baseline:
    if not rel.startswith('research/'):continue
    parts=Path(rel).parts
    target=Path('studies')/mapping.get(parts[1],parts[1])
    for part in parts[2:]:target/=part
    report['file_mapping'].append({'old':rel,'new':str(target)})
    if not (ROOT/target).is_file():report['missing_migrated_files'].append(rel)

for folder in ('src','scripts','studies','tests'):
    for p in (ROOT/folder).rglob('*.py'):
        if '__pycache__' in p.parts:continue
        try:tree=ast.parse(p.read_text())
        except Exception as exc:
            report['syntax_errors'].append({'path':str(p),'error':str(exc)});continue
        for node in ast.walk(tree):
            if not isinstance(node,ast.Constant) or not isinstance(node.value,str):continue
            value=node.value
            # Migration aliases and negative assertions are not file reads.
            if p.name=='workspace.py' or folder=='tests':continue
            if len(value)<500 and ('research/' in value or '/research' in value):
                report['archive_dependencies'].append({'path':str(p.relative_to(ROOT)),'line':node.lineno,'value':value})
for folder in ('src','scripts','studies','data','configs'):
    for parent,dirs,files in os.walk(ROOT/folder,followlinks=False):
        for name in dirs+files:
            p=Path(parent)/name
            if not p.is_symlink():continue
            target=p.resolve(strict=False)
            if target.is_relative_to(ROOT/'research') or target.is_relative_to(ROOT.with_name('cot-navigation')):
                report['bad_symlinks'].append({'path':str(p),'target':str(target)})

from cot_navigation.config import load_config
from cot_navigation.experiments.workflow import plan
for folder in ('configs/experiments','configs/generated'):
    for p in sorted((ROOT/folder).rglob('*.yaml')):
        try:
            cfg=load_config([p]);error=None
        except Exception as exc:error=str(exc)
        report['configs'].append({'path':str(p.relative_to(ROOT)),'error':error})
for p in sorted((ROOT/'configs/workflows').glob('*.yaml')):
    try:
        name,steps=plan(p);error=None;count=len(steps)
    except Exception as exc:error=str(exc);count=0
    report['workflows'].append({'path':str(p.relative_to(ROOT)),'steps':count,'error':error})

# Refresh the one corrected entry, preserving independent evidence for others.
path=ROOT/'migration/standalone/help_audit.json'
help_report=json.loads(path.read_text())
for row in help_report['results']:
    if row['returncode']==0:continue
    entry=json.loads((ROOT/'configs/studies.json').read_text())[row['name']]
    argv=[sys.executable,str(ROOT/entry['path']),'--help']
    result=subprocess.run(argv,cwd=ROOT,capture_output=True,text=True,timeout=30,
        env={**os.environ,'COT_NAV_HELP_CHECK':'1','TORCHINDUCTOR_COMPILE_THREADS':'1','CUDA_VISIBLE_DEVICES':''})
    row.update(returncode=result.returncode,output=(result.stdout+result.stderr)[-4500:])
path.write_text(json.dumps(help_report,indent=2))
report['entrypoints']={'checked':len(help_report['results']),
    'failed':[r['name'] for r in help_report['results'] if r['returncode']!=0]}
(ROOT/'migration/standalone/final_audit.json').write_text(json.dumps(report,indent=2))
print(json.dumps({k:len(v) if isinstance(v,list) else v for k,v in report.items() if k!='file_mapping'},indent=2))
assert not any(report[k] for k in ('missing_migrated_files','archive_dependencies','bad_symlinks','syntax_errors'))
assert not any(r['error'] for r in report['configs']+report['workflows'])
assert not report['entrypoints']['failed']
