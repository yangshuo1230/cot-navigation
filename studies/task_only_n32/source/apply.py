from pathlib import Path
import hashlib,json,shutil
root=Path(str(next(_p for _p in __import__('pathlib').Path(__file__).resolve().parents if (_p / 'pyproject.toml').is_file() and (_p / 'studies').is_dir())));stage=root/'studies/task_only_n32/source'
hashes=json.loads((stage/'original_hashes.json').read_text())
files={'config':root/'src/cot_navigation/config.py','trainer':root/'src/cot_navigation/training/trainer.py'}
for key,path in files.items():
 assert hashlib.sha256(path.read_bytes()).hexdigest()==hashes[key],f'Concurrent edit: {path}'
for key,path in files.items():
 shutil.copyfile(stage/f'{key}_after.py',path)
print('Applied explicit task_only option; legacy default remains False')
