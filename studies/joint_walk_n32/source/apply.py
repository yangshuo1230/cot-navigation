from pathlib import Path
import hashlib,json,shutil
root=Path(str(next(_p for _p in __import__('pathlib').Path(__file__).resolve().parents if (_p / 'pyproject.toml').is_file() and (_p / 'studies').is_dir())));stage=root/'studies/joint_walk_n32/source'
hashes=json.loads((stage/'original_hashes.json').read_text())
files={'config':root/'src/cot_navigation/config.py','trainer':root/'src/cot_navigation/training/trainer.py',
       'engine':root/'scripts/train_first_round.py','runner':root/'scripts/task_only_n32.py'}
for name,path in files.items():
 assert hashlib.sha256(path.read_bytes()).hexdigest()==hashes[name],f'Concurrent edit detected: {path}'
for name,path in files.items():shutil.copyfile(stage/f'{name}_after.py',path)
print('Applied skip_knowledge_stage and joint-run entry; legacy defaults preserved')
