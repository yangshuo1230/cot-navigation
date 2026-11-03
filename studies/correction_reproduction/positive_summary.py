"""Filter existing final-test predictions to reachable (label=1) queries.

Read-only for training outputs. No generation, checkpoint loading or W&B run.
Semantic legality was saved only as an overall aggregate, so is not inferred.
"""
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RUN = ROOT / 'outputs/correction_reproduction_v1/launches/dfs_correction_repro_20261028'
TASKS = dict(n32='-n32', n64='-n64', main='', deep='-deep', hard='-hard')


def main():
    result = dict(population='label=1, reachable queries only', checkpoint='last 200000',
                  semantic_positive_legality_available=False, tasks={})
    for task, suffix in TASKS.items():
        groups = {}; query_ids = None
        for arm in ['original_clean', 'matched_clean', 'correction']:
            rows = []
            for seed in range(4):
                folder = RUN / f'dfs_state/{arm}/s{seed}/eval'
                metrics = folder / f'free_test_last{suffix}.json'
                predictions = folder / f'predictions_test_last{suffix}.jsonl'
                if not metrics.exists() or not predictions.exists(): continue
                report = json.loads(metrics.read_text())
                if report.get('checkpoint_step') != 200000: continue
                positive = [json.loads(line) for line in predictions.read_text().splitlines()]
                positive = [r for r in positive if r['label'] == 1]
                ids = [r['query_id'] for r in positive]
                if query_ids is None: query_ids = ids
                assert ids == query_ids and len(set(ids)) == len(ids) == 256
                correct = sum(r['correct'] for r in positive)
                accuracy = correct / len(positive)
                assert abs(accuracy - report['overall']['yes_recall']) < 1e-12
                rows.append(dict(seed=seed, n=len(positive), correct=correct,
                    accuracy=accuracy, interrupted=sum(r['cot_interrupted'] for r in positive),
                    format_fail=sum(r['format_fail'] for r in positive)))
            if rows:
                values = [r['accuracy'] for r in rows]
                groups[arm] = dict(seeds=rows, mean=float(np.mean(values)),
                    seed_std=float(np.std(values, ddof=1)) if len(values)>1 else None,
                    all_four_seeds_complete=len(rows)==4)
        if all(a in groups and groups[a]['all_four_seeds_complete'] for a in ['matched_clean','correction']):
            delta = [b['accuracy']-a['accuracy'] for a,b in zip(groups['matched_clean']['seeds'],groups['correction']['seeds'])]
            groups['paired_correction_minus_matched'] = dict(delta_by_seed=delta,
                mean=float(np.mean(delta)),wins=sum(d>0 for d in delta),ties=sum(d==0 for d in delta))
        result['tasks'][task] = groups
    (HERE / 'positive_summary.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
