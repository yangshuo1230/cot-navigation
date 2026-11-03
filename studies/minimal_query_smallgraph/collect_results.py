"""Read-only collection of small study results; excludes weights and secrets."""
import json
from pathlib import Path

root = Path(str(next(_p for _p in __import__('pathlib').Path(__file__).resolve().parents if (_p / 'pyproject.toml').is_file() and (_p / 'studies').is_dir())) + '/studies/minimal_query_smallgraph/runs/minimal_query_smallgraph_v1')
result = {}
for job in sorted(root.iterdir()):
    if not job.is_dir():
        continue
    files = ['experiment.json', 'completed.json', 'direct_diagnostics.json', 'length_summary.json',
             'graph_exposure.json', 'initialization.json', 'mix_counts.json', 'wandb_status.json',
             'eval/free_test_last.json']
    item = {f: json.loads((job/f).read_text()) for f in files if (job/f).exists()}
    for name in ('metrics.jsonl', 'study_metrics.jsonl'):
        path = job/name
        if path.exists():
            item[name] = [json.loads(line) for line in path.read_text().splitlines() if line]
    result[job.name] = item
print(json.dumps(result))
