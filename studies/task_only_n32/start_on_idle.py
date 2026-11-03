from pathlib import Path
import os,signal,json
root=Path(str(next(_p for _p in __import__('pathlib').Path(__file__).resolve().parents if (_p / 'pyproject.toml').is_file() and (_p / 'studies').is_dir())))
folder=root/'studies/task_only_n32'
assert not (folder/'started_after_audit.json').exists(),'Gate already launched training'
assert not (root/'outputs/task_only_n32_v1/launches/task_only_n32_s0/launch_plan.json').exists(),'Batch already exists'
found=[]
for p in Path('/proc').iterdir():
 if not p.name.isdigit():continue
 try:args=(p/'cmdline').read_bytes().split(b'\0')
 except (PermissionError,FileNotFoundError,ProcessLookupError):continue
 if b'studies/task_only_n32/after_audit.py' in args:
  found.append(int(p.name));os.kill(int(p.name),signal.SIGTERM)
(folder/'gate_replaced.json').write_text(json.dumps({'stopped_only_own_waiting_helper':found,'reason':'Launch authorized N32 batch on idle GPUs while remaining independent frozen audits finish'}))
print('Stopped only waiting gate:',found)
