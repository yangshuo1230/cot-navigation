"""Count marker tokens and repair events in existing compact replay records."""
import json
from pathlib import Path


def main():
    root = Path(__file__).resolve().parent
    output = {}
    for path in sorted((root / 'results').glob('*.json')):
        blob = json.loads(path.read_text())
        for task, case in blob['cases'].items():
            rows = case['native']['rows']
            metrics = {
                'n': len(rows),
                'raw_mistake_tokens': sum(r['raw_mistake_tokens'] for r in rows),
                'repair_events': sum(r['repairs'] for r in rows),
                'true_mark_records': sum(r['true_marks'] for r in rows),
                'false_mark_records': sum(r['false_marks'] for r in rows),
            }
            output.setdefault(task, {}).setdefault(blob['arm'], []).append(
                dict(seed=blob['seed'], **metrics))
    (root / 'marker_counts.json').write_text(
        json.dumps(output, ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
