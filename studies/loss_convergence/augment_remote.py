"""Read existing logs; add numerical checks and decoder metadata to the audit."""
from pathlib import Path
import json
import math

out = Path(__file__).resolve().parent
data = json.loads((out / 'summary.json').read_text())
for run in data['runs']:
    source = Path(run['run'])
    rows = [json.loads(s) for s in (source / 'metrics.jsonl').read_text().splitlines()]
    train = [r for r in rows if 'total_loss' in r]
    run['nonfinite_training_values'] = sum(
        not math.isfinite(float(r[k])) for r in train
        for k in ['total_loss', 'trace_loss', 'answer_loss', 'grad_norm', 'lr'])
    for v in run['validation']:
        report = json.loads((source / 'eval' / f"free_{v['step']:08d}.json").read_text())
        v['decoder_metadata'] = {k: report.get(k) for k in [
            'compiled_generation', 'generation_compile_fallbacks', 'generation_batch_limit']}
(out / 'summary.json').write_text(json.dumps(data, indent=2) + '\n')
print([(r['model'], r['nonfinite_training_values']) for r in data['runs']])
