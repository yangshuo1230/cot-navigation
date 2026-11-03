"""Meaningful data/supervision regressions; no model files or W&B runs created."""
import argparse
import copy
import json
from pathlib import Path
import random
import sys

from data import HERE, atomic_json, digest, exposure_audit, validate


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo-root', type=Path)
    args = p.parse_args()
    manifest = json.loads((HERE / 'manifest.json').read_text())
    assert manifest['config_sha256'] == digest(json.loads((HERE / 'experiment.json').read_text()))
    results = []
    for meta in manifest['worlds']:
        world = json.loads((HERE / meta['path']).read_text())
        assert digest(world) == meta['sha256']
        verified = validate(world)
        assert digest(exposure_audit(world)) == digest(world['potential_path_exposure'])
        # Ensure the validator catches exactly the dangerous boundary crossing.
        bad = copy.deepcopy(world)
        query = bad['panels']['long_TT_test'].pop()
        bad['panels']['long_train'].append(query)
        try:
            validate(bad)
        except AssertionError:
            pass
        else:
            raise AssertionError('Leaked long test endpoint was not rejected')
        rng = random.Random(18)
        state = rng.getstate()
        first = [rng.randrange(100) for _ in range(32)]
        rng.setstate(state)
        assert first == [rng.randrange(100) for _ in range(32)]
        results.append({'world': meta['path'], **verified})
    if args.repo_root:
        import torch
        sys.path.insert(0, str(args.repo_root / 'src'))
        from train import Bank, losses
        world = json.loads((HERE / manifest['worlds'][0]['path']).read_text())
        cfg = json.loads((HERE / 'experiment.json').read_text())
        for protocol in ('path',):
            bank = Bank(world, protocol, torch.device('cpu'))
            rng = random.Random(55)
            state = rng.getstate()
            ids = bank.sample(rng, cfg)
            rng.setstate(state)
            assert ids == bank.sample(rng, cfg)
            assert sum(int(bank.groups[i]) == 0 for i in ids) == 48
            assert sum(int(bank.groups[i]) == 1 for i in ids) == 16
            assert sum(int(bank.groups[i]) == 2 for i in ids) == cfg['adjacency_rows']
            for qid, r in enumerate(world['records']):
                for i in bank.variants[qid]:
                    tokens, answer, trace, knowledge, group = bank.encoded[i]
                    assert tokens[answer[0]] == bank.tok.label_id(r['y'])
                    assert not knowledge
                    assert all(j >= 4 for j in trace)
                    if protocol == 'path':
                        assert tokens[4] == bank.nodes[r['s']]  # both labels begin source
                        assert trace == list(range(4, answer[0])) + [answer[0] + 1]
                    else:
                        assert answer == [4] and not trace
            tensors = bank.batch(ids)
            logits = torch.randn(*tensors[0].shape, len(bank.tok), requires_grad=True)
            value = losses(logits, *tensors, 1., 1.)
            # Compare independent per-row CE to the vectorized logical objective.
            terms, tasks, knowledge_terms = [], [], []
            for row, i in enumerate(ids):
                tok, ans, trace, km, group = bank.encoded[i]
                ce = torch.nn.functional.cross_entropy(logits[row, :len(tok) - 1],
                    torch.tensor(tok[1:]), reduction='none')
                if group < 2:
                    a = ce[torch.tensor(ans) - 1].mean()
                    t = ce[torch.tensor(trace) - 1].mean() if trace else ce.new_zeros(())
                    tasks.append(a + t)
                else:
                    knowledge_terms.extend(ce[torch.tensor(km) - 1].unbind())
            reference = torch.stack(tasks).mean() + torch.stack(knowledge_terms).mean()
            assert torch.allclose(value[0], reference, atol=1e-6), (value[0], reference)
            # Padding must carry no loss or gradient; next-token alignment matters.
            value[0].backward()
            for row, i in enumerate(ids):
                assert logits.grad[row, bank.lengths[i] - 1:].abs().sum() == 0
    report = {'status': 'PASS', 'manifest_sha256': digest(manifest), 'worlds': results,
              'supervision_checked': bool(args.repo_root),
              'checks': ['bfs_dp_all_pairs', 'node_and_query_holdout', 'trace_legality',
                         'leak_mutation_rejected', 'sampling_replay', 'potential_exposure_replay']}
    if args.repo_root:
        from train import source_hash
        report['source_sha256'] = source_hash(args.repo_root)
        report['checks'] += ['positive_negative_source_prefix', 'loss_matches_independent_reference',
                             'padding_zero_gradient', 'short_long_exact_batch_ratio',
                             'path_only_joint_training_no_parent']
        from launch import jobs_for
        for mode, count in [('pilot', len(cfg['model_seeds'])),
                            ('formal', len(cfg['graph_seeds']) * len(cfg['model_seeds']))]:
            jobs = jobs_for(mode, cfg)
            assert len(jobs) == count and all(j['protocol'] == 'path' and 'parent' not in j for j in jobs)
    atomic_json(HERE / 'verification.json', report)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
