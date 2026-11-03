from pathlib import Path
import json
import os
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
needle="str(__import__('cot_navigation.runtime.workspace', fromlist=['ROOT']).ROOT)"
replacement="str(next(_p for _p in __import__('pathlib').Path(__file__).resolve().parents if (_p / 'pyproject.toml').is_file() and (_p / 'studies').is_dir()))"
for folder in ('scripts','studies'):
    for p in (ROOT/folder).rglob('*.py'):
        if '__pycache__' in p.parts:continue
        s=p.read_text()
        if needle in s:p.write_text(s.replace(needle,replacement))

output=ROOT/'migration/standalone/backend_checks';output.mkdir(exist_ok=True)
checks=[
 ['.venv/bin/python','studies/fixed_navigation_v5/train.py','--repo-root',str(ROOT),'--world','n80_g0.json','--gpu','2','--eager','--tracking','disabled','--check'],
 ['.venv/bin/python','-m','cot_navigation.training.packed','--backend','v6','--protocol','dfs_state','--gpu','2','--eager','--check'],
 ['.venv/bin/python','-m','cot_navigation.training.packed','--backend','v7','--protocol','path','--gpu','2','--eager','--check'],
]
reports=[]
for index,argv in enumerate(checks):
    result=subprocess.run(argv,cwd=ROOT,env={**os.environ,'TORCHINDUCTOR_COMPILE_THREADS':'1'},capture_output=True,text=True,timeout=180)
    text=result.stdout+result.stderr
    (output/f'backend_{index}.log').write_text(text)
    reports.append({'argv':argv,'returncode':result.returncode,'tail':text[-1500:]})
prefix=ROOT/'studies/incontext_prefix_suite'
for name in ('verify.py','verify_runtime.py','verify_bootstrap.py'):
    path=prefix/name
    code=path.read_text()
    code=code.replace("os.environ['CUDA_VISIBLE_DEVICES']='3'","os.environ.setdefault('CUDA_VISIBLE_DEVICES','2')")
    code=code.replace("os.environ['CUDA_VISIBLE_DEVICES']='4'","os.environ.setdefault('CUDA_VISIBLE_DEVICES','2')")
    for filename in ('verification.json','runtime_verification.json','bootstrap_verification.json'):
        code=code.replace("(HERE/'"+filename+"').write_text", "Path("+repr(str(output/filename))+").write_text")
    # Preserve __file__ and sibling imports, redirect only verification receipts.
    wrapper=output/name
    wrapper.write_text('import sys\nsys.path.insert(0,'+repr(str(prefix))+')\n__file__='+repr(str(path))+'\nexec(compile('+repr(code)+',__file__,"exec"))\n')
    try:
        result=subprocess.run([sys.executable,str(wrapper)],cwd=ROOT,capture_output=True,text=True,timeout=300,
                              env={**os.environ,'CUDA_VISIBLE_DEVICES':'2','TORCHINDUCTOR_COMPILE_THREADS':'1'})
        text=result.stdout+result.stderr;rc=result.returncode
    except subprocess.TimeoutExpired as exc:text=str(exc);rc=124
    (output/(name+'.log')).write_text(text)
    reports.append({'test':name,'returncode':rc,'tail':text[-2500:]})
(output/'summary.json').write_text(json.dumps(reports,indent=2))
print(json.dumps(reports,indent=2))
