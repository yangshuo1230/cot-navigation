from pathlib import Path
import subprocess,sys,json,os,datetime
ROOT=Path(str(next(_p for _p in __import__('pathlib').Path(__file__).resolve().parents if (_p / 'pyproject.toml').is_file() and (_p / 'studies').is_dir())))
HERE=ROOT/'studies/stepwise_reference';OUT=ROOT/'outputs/stepwise_reference'
sys.path.insert(0,str(ROOT/'src'))
import wandb
from cot_navigation.tracking.logger import _has_wandb_credentials
assert _has_wandb_credentials(wandb),'W&B online credentials missing'
assert json.loads((HERE/'verification.json').read_text())['status']=='PASS'
OUT.mkdir(parents=True,exist_ok=True)
receipt=OUT/'launch_receipt.json'
if receipt.exists():raise RuntimeError('Launch receipt already exists; inspect before relaunch')
with (OUT/'launcher.log').open('a') as log:
    proc=subprocess.Popen([sys.executable,str(HERE/'launch.py')],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
        start_new_session=True,env={**os.environ,'OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2','TORCHINDUCTOR_COMPILE_THREADS':'2'})
result=dict(pid=proc.pid,utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),output=str(OUT),gpu_pool=list(range(6)))
receipt.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
