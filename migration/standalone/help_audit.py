"""Exercise parser entry points without allowing project writes or launches."""
import ast
from concurrent.futures import ThreadPoolExecutor,as_completed
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
registry=json.loads((ROOT/'configs/studies.json').read_text())
candidates=[];without_parser=[]
for name,entry in registry.items():
    p=ROOT/entry['path'];s=p.read_text()
    if 'parse_args(' in s or name.endswith('/train_protocol'):
        candidates.append((name,p))
    else:without_parser.append(name)

def check(item):
    name,path=item
    env={**os.environ,'COT_NAV_HELP_CHECK':'1','CUDA_VISIBLE_DEVICES':'','PYTHONDONTWRITEBYTECODE':'1',
         'PYTHONPATH':str(ROOT/'src')+':'+str(ROOT),'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','TORCHINDUCTOR_COMPILE_THREADS':'1'}
    try:
        entry=registry[name]
        argv=[sys.executable,str(path),'--help']
        if 'module' in entry:
            argv=[sys.executable,'-m',entry['module'],'--help']
            env['PYTHONPATH']=str(ROOT/entry['pythonpath'])+':'+env['PYTHONPATH']
        result=subprocess.run(argv,cwd=ROOT,env=env,
                              capture_output=True,text=True,timeout=25)
        return {'name':name,'returncode':result.returncode,'output':(result.stdout+result.stderr)[-4500:]}
    except subprocess.TimeoutExpired:
        return {'name':name,'returncode':124,'output':'timeout'}

rows=[]
with ThreadPoolExecutor(max_workers=4) as pool:
    for f in as_completed([pool.submit(check,item) for item in candidates]):
        rows.append(f.result())
report={'results':sorted(rows,key=lambda x:x['name']),'non_parser_entries':without_parser}
(ROOT/'migration/standalone/help_audit.json').write_text(json.dumps(report,indent=2))
print(json.dumps({'checked':len(rows),'passed':sum(r['returncode']==0 for r in rows),
                  'failed':[{k:r[k] for k in ('name','returncode','output')} for r in rows if r['returncode']!=0],
                  'non_parser_entries':len(without_parser)},indent=2))
