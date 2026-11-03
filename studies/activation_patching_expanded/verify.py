"""CPU-only independent reconstruction of every saved legal pair recipe."""
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path


def main():
    import torch
    import yaml
    from cot_navigation.analysis.common import sha256, write_json
    from cot_navigation.analysis.scale_error_awareness import ScaleModel
    from data import Factory
    from engine import FIELDS, different_state, digest
    torch.set_num_threads(4)
    directory = Path(__file__).resolve().parent
    selection = json.loads((directory / 'selection.json').read_text())['models']
    checks = Counter()
    model_checks = {}
    for entry in selection:
        am = ScaleModel(entry, Path.cwd(), device='cpu')
        assert am.checkpoint['sha256'] == entry['sha256']
        with gzip.open(directory / 'datasets' / f"{entry['id']}.json.gz", 'rt') as f:
            dataset = json.load(f)
        assert dataset['settings']['generator_sha256'] == sha256(directory / 'data.py')
        factory = Factory(am)
        cfg = yaml.safe_load((Path.cwd() / entry['run'] / 'resolved_config.yaml').read_text())
        width = cfg['data']['shapes'][cfg['data']['graph_shape']]['width']
        local = Counter()
        for pair in dataset['records']:
            r, d = [factory.restore(pair[arm]['recipe'], pair[arm]['ordinal']) for arm in ('recipient', 'donor')]
            assert digest(r.prefix) == pair['recipient']['sha256']
            assert digest(d.prefix) == pair['donor']['sha256']
            assert r.record[pair['field']] == d.record[pair['field']]
            for s in (r, d):
                assert all(v // width == u // width + 1 for u, vs in enumerate(s.example.adj) for v in vs)
                if am.protocol == 'bfs_state':
                    neighbors = {s.example.nodes[v] for v in s.example.adj[s.example.nodes.index(s.record['frontier'][0])]}
                    assert set(s.record['candidates']) == neighbors - set(s.record['frontier'])
                    local['bfs_queue_determines_candidates'] += 1
            if pair['cohort'] == 'exact_field':
                assert all(different_state(r.record, d.record, f) for f in FIELDS if f != pair['field'])
                local['exact_field'] += 1
            else:
                assert len(r.prefix) == len(d.prefix)
                assert r.prefix[len(r.example.prompt):] == d.prefix[len(d.example.prompt):]
                assert all(r.record[f] == d.record[f] for f in FIELDS + ('action',))
                assert any(different_state(r.following, d.following, f) for f in FIELDS) or r.following['action'] != d.following['action']
                if am.context:
                    assert r.example.nodes[r.example.source] == d.example.nodes[d.example.source]
                    assert r.example.nodes[r.example.target] == d.example.nodes[d.example.target]
                local['shared_history'] += 1
            if am.context:
                assert set(r.example.nodes) == set(d.example.nodes)
            else:
                assert r.example.nodes == d.example.nodes == am.store.node_vocab_ids(am.tok)
                assert r.example.source != r.example.target and d.example.source != d.example.target
                local['fixed_naming_preserved'] += 1
            local['pairs'] += 1
        checks.update(local); model_checks[entry['id']] = dict(local)
        print(json.dumps(dict(model=entry['id'], **local)), flush=True)
        del factory, am
    write_json(directory / 'dataset_verification.json', dict(status='passed', totals=dict(checks), models=model_checks,
        fixed_bfs_frontier_gap='Structurally impossible on adjacent-layer DAGs: CAND = successors(queue[0]) - set(queue).'))


if __name__ == '__main__':
    main()
