"""Audit saved evidence, not a second implementation of the experiment."""
from pathlib import Path
import json, math
HERE = Path(__file__).resolve().parent

def load(name):
    value = json.loads((HERE/f'{name}.json').read_text())
    def finite(x):
        if isinstance(x, dict):
            for v in x.values(): finite(v)
        elif isinstance(x, list):
            for v in x: finite(v)
        elif isinstance(x, float): assert math.isfinite(x)
    finite(value)
    return value

for graph in ('simple', 'hard'):
    for protocol in ('direct', 'path'):
        main = load(f'{graph}_{protocol}'); follow = load(f'{graph}_{protocol}_followup')
        assert main['manifest']['checkpoint_sha256'] == follow['manifest']['checkpoint_sha256']
        assert main['checks']['gradient_instrumentation_max_logit_error'] < 1e-4
        assert main['sample_counts']['0'] and main['sample_counts']['2']
        assert set(follow['size_scan']) == {'8', '16', '32', '64', '128'}
        assert len(follow['random_partitions']) == 3
        assert follow['panels']['node_heldout']['node_identity_disjoint']
        for key in ('current_lateral', 'current_depth', 'target_lateral'):
            assert main['regions'][key]['matched_test_n'] > 0
            assert set(main['regions'][key]['scopes']) == {'last', 'all'}
        assert all(k in main['functional']['first_ffn_roles'] for k in ('nodes', 'syntax', 'target', 'separator'))
        if protocol == 'path': assert 'large_controls' in follow
    edit = load(f'{graph}_path_weight_edit')
    assert edit['manifest']['checkpoint_sha256'] == load(f'{graph}_path')['manifest']['checkpoint_sha256']
    assert len(edit['interventions']) == 6
    assert edit['checks']['restored_max_logit_error'] == 0
    for case in edit['interventions']:
        assert set(case['old_successors']).isdisjoint(case['new_successors'])
        assert case['max_other_train_mean_delta'] < 1e-3 and case['recipient_train_mean_error'] < 1e-3
        assert case['baseline']['n'] >= 6 and len(case['controls']) == 3
        assert case['baseline']['final_new_top1'] == 0
        assert 'unexposed_edited' in case and case['unexposed_edited']['n'] > 0
        assert all(c['recipient']['final_new_top1'] == 0 for c in case['controls'].values())

for name in ('REPORT.md', 'functions_and_edits.png', 'regional_controls.png'):
    assert (HERE/name).stat().st_size > 1000
assert '<V' not in (HERE/'REPORT.md').read_text(encoding='utf-8')
assert not any(HERE.glob('*.pt')) and not any(HERE.glob('*.npy')) and not any(HERE.glob('*.npz'))
print('PASS: 4 frozen models, 4 regional-control suites, 12 rank-one edits, two figures; no activation/checkpoint artifacts.')
