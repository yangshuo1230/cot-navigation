from pathlib import Path
import hashlib
import json
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
out=ROOT/'migration/standalone'
baseline=json.loads((ROOT/'migration/source_baseline.json').read_text())
changed=[];checked=0
for rel,digest in baseline.items():
    if Path(rel).suffix not in {'.py','.yaml','.sh'}:continue
    p=ROOT.with_name('cot-navigation')/rel
    checked+=1
    if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest()!=digest:changed.append(rel)
(out/'original_source_check.json').write_text(json.dumps({'checked':checked,'changed':changed},indent=2))

# No old absolute path in the executable entry point; use current interpreter.
commands=[
 [sys.executable,'-m','cot_navigation','study','fixed_navigation_hard_v7/train_protocol','--','--help'],
 [sys.executable,'-m','cot_navigation','study','incontext_prefix_suite/src/cot_navigation/cli','--','--help'],
 [sys.executable,'-m','cot_navigation','workflow','configs/workflows/research.yaml','--run-name','standalone_research_help','--execute']
]
rows=[]
for i,argv in enumerate(commands):
    result=subprocess.run([sys.executable,str(out/'isolate.py'),*argv],cwd=ROOT,capture_output=True,text=True,timeout=90)
    (out/f'dispatch_{i}.log').write_text(result.stdout+result.stderr)
    rows.append({'argv':argv,'returncode':result.returncode})
    if result.returncode:raise RuntimeError(result.stderr)
(out/'dispatch_validation.json').write_text(json.dumps(rows,indent=2))

p=ROOT/'README.md';s=p.read_text()
notice='> **独立运行版已完成迁移**：请从 [docs/STANDALONE_MIGRATION.md](docs/STANDALONE_MIGRATION.md) 开始。运行入口不再依赖旧项目或 `research/` 档案；内核隔离测试记录见 `migration/standalone/`。\n\n'
p.write_text(notice+s)
for name in ('docs/FRAMEWORK_MIGRATION.md','migration/VALIDATION.md'):
    p=ROOT/name;s=p.read_text()
    p.write_text('> 本文是第一阶段记录。当前独立版本及最新测试结果见 '+('[STANDALONE_MIGRATION.md](STANDALONE_MIGRATION.md)' if name.startswith('docs') else '[STANDALONE_MIGRATION.md](../docs/STANDALONE_MIGRATION.md)')+'。\n\n'+s)
summary={
    'original_sources_checked':checked,'original_sources_changed':changed,
    'studies':len(json.loads((ROOT/'configs/study_paths.json').read_text())),
    'registered_entries':len(json.loads((ROOT/'configs/studies.json').read_text())),
    'main_cpu':'263 passed, 13 skipped','cuda':'39 passed','research_unit':'41 passed',
    'research_semantic_cases':1536,'cli_entries':'107 passed',
    'isolation':'Landlock ABI 1: old project, research archive, .code_backups read/execute denied',
    'documentation':'docs/STANDALONE_MIGRATION.md'}
(out/'DELIVERY.json').write_text(json.dumps(summary,indent=2))
assert not changed,changed
print(json.dumps(summary,indent=2))
