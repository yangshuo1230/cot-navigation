from pathlib import Path
root=Path('/mnt/ningxuefei/yangshuo/cot-navigation-unified')
p=root/'src/cot_navigation/training/trainer.py';s=p.read_text()
s=s.replace('def _check_resume(saved: dict, current: dict) -> list[str]:\n',
'''def _check_resume(saved: dict, current: dict) -> list[str]:
    # Root relocation is not a dataset/objective change. Dataset hashes and all
    # scientific fields below still have to match; no filesystem fallback.
    from cot_navigation.runtime.workspace import relocate
    saved, current = relocate(saved), relocate(current)
''')
p.write_text(s)
p=root/'src/cot_navigation/experiments/scheduler.py';s=p.read_text()
s=s.replace('    old = json.loads(json.dumps(previous))\n    new = json.loads(json.dumps(current))',
'''    from cot_navigation.runtime.workspace import relocate
    old = relocate(json.loads(json.dumps(previous)))
    new = relocate(json.loads(json.dumps(current)))''')
p.write_text(s)
p=root/'tests/test_standalone.py';s=p.read_text()
s+='''

def test_standard_resume_accepts_only_root_relocation():
    from cot_navigation.runtime.workspace import SOURCE
    from cot_navigation.training.trainer import _arch_snapshot, _check_resume, TrainError
    from cot_navigation.config import load_config
    from cot_navigation.protocols.tokenizer import Tokenizer
    import copy
    cfg=load_config([ROOT/'configs/experiments/smoke_context.yaml'])
    current=_arch_snapshot(cfg,Tokenizer.from_config(cfg),'unchanged_manifest_hash')
    current['scale_roots']=[str(ROOT/'data/test')]
    saved=copy.deepcopy(current);saved['scale_roots']=[SOURCE+'/data/test']
    assert _check_resume(saved,current)==[]
    assert saved['scale_roots']==[SOURCE+'/data/test']
    saved['manifest_hash']='changed'
    with pytest.raises(TrainError,match='manifest_hash'):
        _check_resume(saved,current)


def test_scheduler_resume_relocates_plan_but_rejects_data_changes():
    import copy
    from cot_navigation.experiments.first_round import parse_args,build_jobs
    from cot_navigation.experiments.scheduler import plan_record,check_resume_plan
    from cot_navigation.runtime.workspace import SOURCE
    args=parse_args(['--run-name','migration_resume','--setting','context','--protocols','direct'])
    current=plan_record(build_jobs(args)[1])
    previous=json.loads(json.dumps(current).replace(str(ROOT)+'/',SOURCE+'/'))
    check_resume_plan(previous,current)
    previous['context/direct']['config']['data']['data_seed']+=1
    with pytest.raises(ValueError,match='differs'):
        check_resume_plan(previous,current)
'''
p.write_text(s)
print('Checkpoint and scheduler resume normalize project relocation only.')
