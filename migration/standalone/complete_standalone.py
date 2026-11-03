from pathlib import Path
import hashlib
import json
import re

ROOT=Path(__file__).resolve().parents[2]
out=ROOT/'migration/standalone'
p=out/'isolate.py';s=p.read_text()
start=s.index('probes=[');end=s.index('\n\nclass Ruleset',start)
s=s[:start]+'''probes = [p for p in (
    root.with_name('cot-navigation')/'src/cot_navigation/config.py',
    root/'research/fixed_navigation_v5_20261026/train.py',
    *list((root/'.code_backups').glob('*.zip'))[:1],
) if p.is_file()]
'''+s[end:]
p.write_text(s)

cpu=(out/'isolated_cpu.log').read_text()
match=re.search(r'(\d+) passed, (\d+) skipped',cpu)
assert match and 'FAILED ' not in cpu and 'ERROR ' not in cpu
count=int(match.group(1));skipped=int(match.group(2))
assert count>=265
audit=json.loads((out/'final_audit.json').read_text())
assert all(not audit[k] for k in ('missing_migrated_files','archive_dependencies','bad_symlinks','syntax_errors'))
assert not any(r['error'] for r in audit['configs']+audit['workflows'])
assert not audit['entrypoints']['failed']
for report in (out/'research_tests/summary.json',out/'backend_checks/summary.json',out/'dispatch_validation.json'):
    assert all(row['returncode']==0 for row in json.loads(report.read_text())),report

mapping=json.loads((ROOT/'configs/study_paths.json').read_text())
p=ROOT/'migration/interface_map.json';original=p.read_text()
(out/'phase1_interface_map.json').write_text(original)
entries=json.loads(original)
for entry in entries:
    oldpath=Path(entry['old']);parts=oldpath.parts
    if 'research' in parts:
        index=parts.index('research')
        rel=Path('studies')/mapping.get(parts[index+1],parts[index+1])
        for part in parts[index+2:]:rel/=part
        entry['compatible_new']=str(ROOT/rel)
        entry['study_id']=str(rel.relative_to('studies').with_suffix(''))
    assert Path(entry['compatible_new']).is_file(),entry
p.write_text(json.dumps(entries,indent=2))

p=ROOT/'docs/STANDALONE_MIGRATION.md';s=p.read_text().replace('**263 passed，13 skipped**',f'**{count} passed，{skipped} skipped**')
s=s.replace('普通恢复要求 job 身份一致。','普通恢复要求 job 身份一致。标准训练器和启动计划只归一化已知项目根路径，目录迁移不再被误判为数据或目标变化；数据 manifest 哈希及其他科学参数仍严格比较。')
p.write_text(s)
p=out/'DELIVERY.json';receipt=json.loads(p.read_text());receipt['main_cpu']=f'{count} passed, {skipped} skipped'
receipt['interface_mappings']=len(entries)
p.write_text(json.dumps(receipt,indent=2))

# Publish a content manifest for all active code/configs; archived data is not
# part of this executable release manifest.
manifest={}
for folder in ('src','scripts','studies','tests','configs'):
    for p in (ROOT/folder).rglob('*'):
        if not p.is_file() or '__pycache__' in p.parts or p.suffix not in {'.py','.yaml','.yml','.sh'}:continue
        manifest[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
(out/'release_files.json').write_text(json.dumps(manifest,indent=2))
print(json.dumps(receipt,indent=2))
