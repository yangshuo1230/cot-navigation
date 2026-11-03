from pathlib import Path
import json
root=Path(str(next(_p for _p in __import__('pathlib').Path(__file__).resolve().parents if (_p / 'pyproject.toml').is_file() and (_p / 'studies').is_dir())) + '/outputs/task_only_n32_v1/launches/task_only_n32_s0/task_only')
for folder in root.iterdir():
 files=sorted((folder/'eval').glob('free_0*.json'))
 if files:
  r=json.loads(files[-1].read_text());print(folder.name,r['step'],r['overall']['n'],r['overall']['answer_accuracy'],r['overall']['balanced_accuracy'])
