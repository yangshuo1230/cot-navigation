from pathlib import Path
import hashlib,json,subprocess,sys,os
root=Path(str(next(_p for _p in __import__('pathlib').Path(__file__).resolve().parents if (_p / 'pyproject.toml').is_file() and (_p / 'studies').is_dir())))
folder=root/'studies/scale_trace_audit'
rows=[json.loads(p.read_text()) for p in (folder/'results').rglob('main.json')]
assert len(rows)==51
for r in rows:
 p=root/'outputs/scale_v1/launches/scale_v1_s0'/r['job']/'checkpoints/latest.pt'
 assert hashlib.sha256(p.read_bytes()).hexdigest()==r['checkpoint_sha256']
 assert r['counts']['old_prediction_matches']==r['counts']['old_prediction_compared']==r['n']
verification={'main_models':51,'main_queries':sum(r['n'] for r in rows),'all_checkpoint_sha256_unchanged':True,'all_main_answers_match_original':True,'optimizer_updates':0}
(folder/'verification.json').write_text(json.dumps(verification,indent=2))
with (folder/'finalize.log').open('a') as log:
 p=subprocess.Popen([sys.executable,str(folder/'finalize.py')],cwd=root,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True,env={**os.environ,'CUDA_VISIBLE_DEVICES':''})
print(json.dumps({'verification':verification,'optional_ood_finalizer_pid':p.pid}))
