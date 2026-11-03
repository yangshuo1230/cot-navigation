from pathlib import Path
import os,sys,json,datetime,subprocess,hashlib
ROOT=Path(str(next(_p for _p in __import__('pathlib').Path(__file__).resolve().parents if (_p / 'pyproject.toml').is_file() and (_p / 'studies').is_dir())));HERE=ROOT/'studies/incontext_prefix_suite';OUT=ROOT/'outputs/incontext_prefix_v1'
sys.path.insert(0,str(HERE/'src'))
import wandb
from cot_navigation.tracking.logger import _has_wandb_credentials
assert _has_wandb_credentials(wandb),'Online W&B credentials required'
assert len(json.loads((HERE/'manifest.json').read_text())['jobs'])==39
for filename in ('verification.json','runtime_verification.json','bootstrap_verification.json','benchmark_single.json','benchmark_serial.json'):
    assert json.loads((HERE/filename).read_text())['status']=='PASS',filename
OUT.mkdir(parents=True,exist_ok=True)
receipt=OUT/'launch_receipt.json'
if receipt.exists():raise RuntimeError('Launch receipt exists; inspect instead of duplicate launch')
sources={str(p.relative_to(HERE)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(HERE.rglob('*.py')) if '__pycache__' not in str(p)}
(OUT/'source_hashes.json').write_text(json.dumps(sources,indent=2))
with (OUT/'launcher.log').open('a') as log:
    proc=subprocess.Popen([sys.executable,str(HERE/'launch.py')],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
result=dict(pid=proc.pid,utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),jobs=39,gpus=list(range(8)),output=str(OUT),group='incontext_prefix_v1_20261024')
receipt.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
