from pathlib import Path
import json
root=Path(str(next(_p for _p in __import__('pathlib').Path(__file__).resolve().parents if (_p / 'pyproject.toml').is_file() and (_p / 'studies').is_dir())) + '/outputs/task_only_n32_v1/launches/task_only_n32_s0')
for p in root.rglob('free_test_last.json'):
 r=json.loads(p.read_text());print(p.parent.parent.relative_to(root),r['checkpoint_step'],r['overall'])
print('status',json.loads((root/'status.json').read_text()))
