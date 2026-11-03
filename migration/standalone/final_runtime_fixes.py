from pathlib import Path
import json

root=Path('/mnt/ningxuefei/yangshuo/cot-navigation-unified')
p=root/'studies/scale_error_awareness/reference_sampler_source.py'
s=p.read_text()
for name in ('common','error_detection','error_awareness'):
    s=s.replace('from .'+name+' import','from cot_navigation.analysis.'+name+' import')
p.write_text(s)

names=json.loads((root/'configs/study_paths.json').read_text())
old=json.loads((root/'migration/legacy_source_hashes.json').read_text())
hashes={names.get(k,k):[v] for k,v in old.items()}
for name,folder in [('v6','fixed_navigation_protocols_v6'),('v7','fixed_navigation_hard_v7')]:
    p=root/'migration'/f'{name}_plan.json'
    if p.is_file():
        value=json.loads(p.read_text())['job']['source_hash']
        if value not in hashes[folder]:hashes[folder].append(value)
(root/'configs/checkpoint_migration.json').write_text(json.dumps(hashes,indent=2))

p=root/'src/cot_navigation/training/packed.py';s=p.read_text()
start=s.index("        previous = dict(blob['job'])")
end=s.index('        raw.load_state_dict',start)
s=s[:start]+'''        allowed = json.loads((args.repo_root / 'configs/checkpoint_migration.json').read_text()).get(HERE.name, []) if args.migrate_checkpoint else []
        validate_resume_job(blob['job'], job, allowed_hashes=allowed)
'''+s[end:]
insert='''def validate_resume_job(previous, current, *, allowed_hashes=()):
    """Accept an explicitly recorded code transition, never objective changes."""
    previous = dict(previous)
    if previous.get('source_hash') != current.get('source_hash'):
        if previous.get('source_hash') not in allowed_hashes:
            raise ValueError('Resume data/config/code changed: unknown source hash')
        previous['source_hash'] = current['source_hash']
    if previous != current:
        raise ValueError('Resume data/config/code changed')


'''
s=s.replace('def main(argv=None, *, backend=None):',insert+'def main(argv=None, *, backend=None):')
p.write_text(s)
p=root/'tests/test_standalone.py';s=p.read_text()
s+='''

def test_checkpoint_migration_only_accepts_known_code_and_same_objective():
    from cot_navigation.training.packed import validate_resume_job
    previous={'source_hash':'old','trace_weight':1,'model':{'width':16}}
    current={'source_hash':'new','trace_weight':1,'model':{'width':16}}
    with pytest.raises(ValueError,match='unknown source hash'):
        validate_resume_job(previous,current)
    validate_resume_job(previous,current,allowed_hashes=['old'])
    for changed in ({**current,'trace_weight':2},{**current,'model':{'width':32}}):
        with pytest.raises(ValueError,match='Resume data/config/code changed'):
            validate_resume_job(previous,changed,allowed_hashes=['old'])
    assert previous['source_hash']=='old'
'''
p.write_text(s)
print('Reference import and explicit checkpoint transition validation updated.')
