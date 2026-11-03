"""Integration tests intended to run with original/archive reads denied."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest
import torch

from cot_navigation.experiments.artifacts import digest
from cot_navigation.experiments.packed_backend import load_backend
from cot_navigation.experiments.study import catalog,command
from cot_navigation.runtime.workspace import ROOT


def test_registered_resources_never_require_archives():
    entries=catalog()
    assert len(entries)>200
    for name,item in entries.items():
        argv=command(name,['--help'])
        path=ROOT/item['path']
        assert path.is_file()
        assert path.is_relative_to(ROOT/'studies')
        assert '/research/' not in str(path)
        assert argv[1] == '-m' if 'module' in item else Path(argv[1]) == path


@pytest.mark.parametrize('name',['v6','v7'])
def test_backend_can_resolve_without_legacy_imports(name):
    backend=load_backend(name,ROOT)
    assert backend.here.is_relative_to(ROOT/'studies')
    assert not hasattr(backend,'load_v5')
    assert len(backend.source_hash(ROOT))==64


def resources(tmp_path):
    backend=load_backend('v7',ROOT)
    cfg=json.loads((backend.here/'experiment.json').read_text())
    manifest=json.loads((backend.here/'manifest.json').read_text())
    meta=manifest['worlds'][0]
    world=json.loads((backend.base/meta['base_path']).read_text())
    for name,ids in world['panels'].items():
        if name.endswith(('_val','_test')):
            world['panels'][name]=ids[:4]
    target=tmp_path/'resources';target.mkdir()
    worlds=tmp_path/'worlds';worlds.mkdir()
    world_path=worlds/meta['base_path'];world_path.parent.mkdir(parents=True,exist_ok=True)
    world_path.write_text(json.dumps(world))
    bank=meta['banks']['direct']
    bank_path=target/bank['path'];bank_path.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(backend.here/bank['path'],bank_path)
    bank['max_new_tokens']=8
    meta['banks']={'direct':bank};meta['world_hash']=digest(world)
    cfg.update(protocols=['direct'],task_steps=4,warmup_steps=1,query_batch=8,adjacency_rows=2,
               eval_interval=2,adjacency_eval_interval=4,checkpoint_interval=2,log_interval=1,
               validation_per_panel=4,evaluation_batch={'direct':4})
    cfg['model']={'layers':1,'heads':1,'d_model':16,'d_ff':32}
    (target/'experiment.json').write_text(json.dumps(cfg))
    (target/'manifest.json').write_text(json.dumps({'config_sha256':digest(cfg),'worlds':[meta]}))
    return target,worlds,meta['name']


def launch(target,worlds,world,output,*args):
    argv=[sys.executable,'-m','cot_navigation.training.packed','--repo-root',str(ROOT),
          '--resources',str(target),'--world-root',str(worlds),'--world',world,'--protocol','direct',
          '--device','cpu','--eager','--tracking','disabled','--execute','--output',str(output),*args]
    result=subprocess.run(argv,cwd=ROOT,capture_output=True,text=True,timeout=120,
                          env={**os.environ,'PYTHONPATH':str(ROOT/'src'),'OMP_NUM_THREADS':'2'})
    assert result.returncode==0,result.stdout+'\n'+result.stderr


def test_packed_training_resume_exact_and_final_reports(tmp_path):
    target,worlds,world=resources(tmp_path)
    full=tmp_path/'full'; resumed=tmp_path/'resumed'
    launch(target,worlds,world,full)
    launch(target,worlds,world,resumed,'--stop-after','2')
    assert not (resumed/'COMPLETE.json').exists()
    partial=torch.load(resumed/'latest.pt',map_location='cpu',weights_only=False)
    assert partial['step']==2
    launch(target,worlds,world,resumed,'--resume')
    a=torch.load(full/'latest.pt',map_location='cpu',weights_only=False)
    b=torch.load(resumed/'latest.pt',map_location='cpu',weights_only=False)
    assert a['step']==b['step']==4
    assert a['counts']==b['counts']
    assert a['sample_rng']==b['sample_rng']
    assert a['variant_rng']==b['variant_rng']
    assert torch.equal(a['torch_rng'],b['torch_rng'])
    for key in a['model']:
        assert torch.equal(a['model'][key],b['model'][key]),key
    for path in ('final.json','adjacency.json','actual_exposure.json','exposure_counts.json'):
        assert json.loads((full/path).read_text())==json.loads((resumed/path).read_text()),path
    assert json.loads((resumed/'COMPLETE.json').read_text())['step']==4


def test_resume_rejects_changed_objective(tmp_path):
    target,worlds,world=resources(tmp_path)
    output=tmp_path/'resume'
    launch(target,worlds,world,output,'--stop-after','2')
    cfg=json.loads((target/'experiment.json').read_text());cfg['trace_weight']=3
    (target/'experiment.json').write_text(json.dumps(cfg))
    manifest=json.loads((target/'manifest.json').read_text());manifest['config_sha256']=digest(cfg)
    (target/'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(AssertionError,match='Resume data/config/code changed'):
        launch(target,worlds,world,output,'--resume')


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
