from pathlib import Path
import json

root=Path('/mnt/ningxuefei/yangshuo/cot-navigation-unified')
for p in (root/'studies').rglob('*.py'):
    if '__pycache__' in p.parts:continue
    s=p.read_text()
    updated=s.replace('import train_first_round as engine','from cot_navigation.experiments import scheduler as engine')
    if updated!=s:p.write_text(updated)

p=root/'configs/studies.json';entries=json.loads(p.read_text())
for name,entry in entries.items():
    path=Path(entry['path'])
    marker='/src/cot_navigation/'
    if marker in path.as_posix():
        prefix,suffix=path.as_posix().split(marker,1)
        entry['module']='cot_navigation.'+suffix.removesuffix('.py').replace('/','.')
        entry['pythonpath']=prefix+'/src'
p.write_text(json.dumps(entries,indent=2,ensure_ascii=False))
p=root/'src/cot_navigation/experiments/study.py';s=p.read_text()
s=s.replace("    return [sys.executable, str(path), *args]", "    if 'module' in entries[name]:\n        return [sys.executable, '-m', entries[name]['module'], *args]\n    return [sys.executable, str(path), *args]")
s=s.replace("    env={**os.environ,'PYTHONPATH':str(ROOT/'src'), 'PYTHONDONTWRITEBYTECODE':'1'}", "    entry=catalog()[a.name]\n    paths=[str(ROOT/entry['pythonpath'])] if 'pythonpath' in entry else []\n    env={**os.environ,'PYTHONPATH':':'.join(paths+[str(ROOT/'src'),str(ROOT)]), 'PYTHONDONTWRITEBYTECODE':'1'}")
p.write_text(s)
p=root/'migration/standalone/help_audit.py';s=p.read_text()
s=s.replace("'PYTHONPATH':str(ROOT/'src')+':'+str(ROOT),'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'", "'PYTHONPATH':str(ROOT/'src')+':'+str(ROOT),'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','TORCHINDUCTOR_COMPILE_THREADS':'1'")
s=s.replace("        result=subprocess.run([sys.executable,str(path),'--help'],cwd=ROOT,env=env,", "        entry=registry[name]\n        argv=[sys.executable,str(path),'--help']\n        if 'module' in entry:\n            argv=[sys.executable,'-m',entry['module'],'--help']\n            env['PYTHONPATH']=str(ROOT/entry['pythonpath'])+':'+env['PYTHONPATH']\n        result=subprocess.run(argv,cwd=ROOT,env=env,")
p.write_text(s)
p=root/'tests/test_standalone.py';s=p.read_text()
s=s.replace("        assert Path(argv[1]).is_file()\n        assert Path(argv[1]).is_relative_to(ROOT/'studies')\n        assert '/research/' not in argv[1]", "        path=ROOT/item['path']\n        assert path.is_file()\n        assert path.is_relative_to(ROOT/'studies')\n        assert '/research/' not in str(path)\n        assert argv[1] == '-m' if 'module' in item else Path(argv[1]) == path")
p.write_text(s)
print('Study compatibility imports and embedded module entry points updated.')
