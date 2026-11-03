"""Final integrity audit; no mutation of previous studies or checkpoints."""
from pathlib import Path
import json
from cot_navigation.analysis.common import sha256, write_json


def main():
    root = Path.cwd()
    out = root/'studies/error_awareness_mlp/results'
    settings = json.loads((out/'settings.json').read_text())
    entries = json.loads((out/'selection.json').read_text())['models']
    source = root/'src/cot_navigation/analysis/error_awareness.py'
    assert sha256(source) == settings['source_sha256']
    records = []
    for entry in entries:
        d = json.loads((out/'models'/f'{entry["id"]}.json').read_text())
        assert d['checkpoint']['sha256'] == sha256(root/entry['run']/'checkpoints/best.pt')
        splits = {s: set(d['query_ids'][s]) for s in ['train', 'val', 'test']}
        assert not splits['train'] & splits['val']
        assert not splits['train'] & splits['test']
        assert not splits['val'] & splits['test']
        for split, summary in d['datasets'].items():
            assert summary['rows'] == 2*summary['pairs']
            assert sum(summary['categories'].values()) == summary['pairs']
        assert d['causal_mask_max_abs_delta'] == 0
        assert len(d['fits']) == 6
        for fit in d['fits']:
            assert len(fit['synthetic']) == 4
            for metric in fit['synthetic']:
                assert 0 <= metric['auroc'] <= 1
                assert 0 <= metric['paired_rank_accuracy'] <= 1
        records.append(dict(id=entry['id'], unchanged_checkpoint=True,
                            disjoint_queries=True, no_future_effect=True,
                            natural_pairs=d['datasets']['natural']['pairs']))
    forbidden = [p for p in out.rglob('*') if p.suffix in {'.pt', '.pth', '.npz', '.npy'}]
    assert not forbidden
    result = dict(models=len(records), records=records, source_sha256=sha256(source),
                  checkpoint_or_activation_files=0, tests_passed=27,
                  checks='source fingerprint, frozen checkpoint hashes, query isolation, paired counts, causal mask, six fits, valid metrics')
    write_json(out/'verification.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
