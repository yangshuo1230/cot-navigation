"""Actual Trainer adapter continuation tests using in-memory checkpoints only."""
from pathlib import Path
import os,sys,json,copy,gc
HERE=Path(__file__).resolve().parent
os.environ['CUDA_VISIBLE_DEVICES']='3';sys.path.insert(0,str(HERE/'src'))
import torch
from cot_navigation.config import load_config
from runtime import PrefixTrainer,Schedule
torch.set_num_threads(2)
manifest=json.loads((HERE/'manifest.json').read_text())
checks=[]
for protocol,phase in [('direct','bootstrap'),('direct','main_boundary'),('path','main')]:
    job=next(j for j in manifest['jobs'] if j['seed']==0 and j['protocol']==protocol and j['recipe']=='standard' and j['mask']=='graph' and j['order']=='single')
    cfg=load_config([job['config']],['tracking.mode=disabled','train.persist_outputs=false','runtime.smoke=true','runtime.cuda_device=0','model.compile=false'])
    t=PrefixTrainer(cfg,job)
    t._evaluate=lambda *a,**k:None;t._save=lambda:None
    if phase=='main_boundary':t.completed=550000;t.schedule=Schedule(t.opt,t.boot,t.completed)
    for _ in range(2):t._update(False);t.completed+=1;t._touch_rng()
    payload=copy.deepcopy(t._payload())
    t._update(False);t.completed+=1;t._touch_rng();expected={k:v.clone() for k,v in t.raw.state_dict().items()}
    t.tracker.finish();del t;gc.collect();torch.cuda.empty_cache()
    original=torch.load
    try:
        torch.load=lambda *a,**kw:copy.deepcopy(payload)
        resumed=PrefixTrainer(cfg,job,resume='IN_MEMORY_UNIT_TEST')
    finally:torch.load=original
    resumed._evaluate=lambda *a,**k:None;resumed._save=lambda:None
    resumed._update(False);resumed.completed+=1;resumed._touch_rng()
    error=max(float((expected[k]-v).abs().max()) for k,v in resumed.raw.state_dict().items())
    assert error<2e-6,error
    assert resumed.query_sampler.step==(3 if phase in ('main','main_boundary') else 0)
    if phase=='main_boundary':assert resumed.optimizer_reset
    checks.append(dict(protocol=protocol,phase=phase,resume_max_weight_difference=error,optimizer_reset=resumed.optimizer_reset))
    print(json.dumps(checks[-1]),flush=True);resumed.tracker.finish();del resumed,payload,expected;gc.collect();torch.cuda.empty_cache()
(HERE/'runtime_verification.json').write_text(json.dumps(dict(status='PASS',checks=checks,checkpoint_files_created=0,formal_runs_created=0),indent=2))
