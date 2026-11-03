from pathlib import Path
import os,sys,json
HERE=Path(__file__).resolve().parent
os.environ['CUDA_VISIBLE_DEVICES']='4';sys.path.insert(0,str(HERE/'src'))
import torch
from cot_navigation.config import load_config
from runtime import PrefixTrainer,Schedule,BOOT_ENDS
torch.set_num_threads(2)
m=json.loads((HERE/'manifest.json').read_text());j=next(j for j in m['jobs'] if j['seed']==0 and j['protocol']=='direct' and j['recipe']=='stepwise' and j['mask']=='graph')
c=load_config([j['config']],['tracking.mode=disabled','train.persist_outputs=false','runtime.smoke=true','runtime.cuda_device=0','model.compile=false'])
t=PrefixTrainer(c,j);t._evaluate=lambda *a,**k:None;t._save=lambda:None;checks=[]
for step in [0]+list(map(int,BOOT_ENDS[:-1])):
    t.completed=step;t.schedule=Schedule(t.opt,t.boot,step)
    r=t._update(True);assert r['phase']=='bootstrap' and r['answer_loss']>=0
    checks.append(dict(step=step,stage=r['bootstrap_stage'],loss=r['answer_loss']))
    print(json.dumps(checks[-1]),flush=True)
t.tracker.finish()
(HERE/'bootstrap_verification.json').write_text(json.dumps(dict(status='PASS',checks=checks,formal_runs_created=0),indent=2))
