"""Check immutable data, logical loss/gradient scaling and strict DFS semantics."""
import argparse
import json
import random
from pathlib import Path
import numpy as np
import torch
from prepare import HERE, BASE, digest, atomic_json, sha_file
from common import load_v5, source_hash
from semantics import reference, replay_state, replay_nodes


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo-root', type=Path, required=True)
    args = p.parse_args()
    torch.set_num_threads(2)
    v5 = load_v5(args.repo_root)
    from cot_navigation.protocols.tokenizer import Tokenizer
    from packed_bank import PackedBank, logical_loss
    from launch import jobs_for
    cfg = json.loads((HERE / 'experiment.json').read_text())
    manifest = json.loads((HERE / 'manifest.json').read_text())
    assert manifest['config_sha256'] == digest(cfg)
    assert manifest['semantic_replay'] == 'PASS'
    base_manifest = json.loads((BASE / 'manifest.json').read_text())
    assert manifest['base_manifest_sha256'] == digest(base_manifest)
    audits = []
    for w in manifest['worlds']:
        world = json.loads((BASE / w['base_path']).read_text())
        assert digest(world) == w['world_hash']
        for name, info in w['banks'].items():
            assert sha_file(HERE / info['path']) == info['sha256']
            assert info['no_truncation'] and info['max_seq_len'] >= info['max_reference_length'] + cfg['context_headroom']
            audits.append({'world': w['name'], 'protocol': name, 'max_reference_length': info['max_reference_length'],
                           'max_seq_len': info['max_seq_len'], 'no_truncation': True})
    pilot = next(w for w in manifest['worlds'] if w['n_nodes'] == 80)
    schedules = []
    for protocol in cfg['protocols']:
        bank = PackedBank(pilot, protocol, torch.device('cpu'), verify_hash=False)
        ids = bank.sample(random.Random(7), random.Random(8), cfg)
        groups = bank.cpu['groups'][ids]
        assert (sum(groups == 0), sum(groups == 1), sum(groups == 2)) == (48, 16, 16)
        q = [(int(bank.cpu['qids'][i]), int(bank.cpu['variants'][i]) if bank.cpu['groups'][i] == 2 else -1) for i in ids]
        schedules.append(q)
        chunks = bank.microbatches(ids, cfg)
        assert sorted(i for chunk in chunks for i in chunk) == sorted(ids)
        for chunk in chunks:
            assert bank.width(max(bank.lengths[chunk])) * len(chunk) <= cfg['microbatch_token_budget']
        # Small rows in both length strata allow a full FP32 independent loss
        # and gradient reference, with deliberately uneven singleton chunks.
        sample = []
        for group in (0, 1):
            sample += [i for i in range(len(bank.keys)) if bank.cpu['groups'][i] == group and bank.lengths[i] < 128][:2]
        sample += bank.adj_ids[:2]
        assert len(sample) == 6
        tokens, lengths, answers, group = bank.batch(sample)
        denominator = torch.tensor([4, sum(int(bank.lengths[i]) - 3 for i in sample if bank.cpu['groups'][i] == 2), 2, 2]).float()
        logits = torch.randn(*tokens.shape, 270, requires_grad=True)
        terms = []
        knowledge = []
        for b, row in enumerate(sample):
            L = int(bank.lengths[row])
            target = tokens[b, 1:L]
            ce = torch.nn.functional.cross_entropy(logits[b, :L-1], target, reduction='none')
            if bank.cpu['groups'][row] == 2:
                knowledge.extend(ce[2:].unbind())
            else:
                answer = int(bank.cpu['answers'][row])
                value = ce[answer - 1]
                if protocol != 'direct':
                    value = value + torch.cat([ce[3:answer-1], ce[answer:answer+1]]).mean()
                terms.append(value)
        reference_loss = torch.stack(terms).mean() + torch.stack(knowledge).mean()
        reference_loss.backward()
        expected_grad = logits.grad.clone()
        other = logits.detach().clone().requires_grad_(True)
        total = 0
        for b in range(len(sample)):
            v = logical_loss(other[b:b+1], tokens[b:b+1], lengths[b:b+1], answers[b:b+1], group[b:b+1],
                             denominator, 1., 1., protocol != 'direct')
            v[0].backward()
            total += v[0].item()
        assert abs(total - reference_loss.item()) < 1e-5
        assert torch.allclose(other.grad, expected_grad, atol=1e-6)
        print(protocol, 'logical-loss/gradient/packing PASS', flush=True)
        del bank
    assert all(s == schedules[0] for s in schedules), 'QUERY or ADJ schedule changed across protocols'
    original_world = json.loads((BASE / pilot['base_path']).read_text())
    original = v5.Bank(original_world, 'path', torch.device('cpu'))
    old_ids = original.sample(random.Random(7), cfg, random.Random(8))
    expected_schedule = [(int(original.keys[i].split(':')[1]), -1) if original.keys[i].startswith('q:')
                         else (-1, int(original.keys[i].split(':')[1])) for i in old_ids]
    assert schedules[0] == expected_schedule, 'QUERY/ADJ ordering differs from the frozen PATH baseline'
    tok = Tokenizer(256)
    inverse = {tok.node_id(i): i for i in range(6)}
    graph = [[1, 2], [4], [3], [], [], []]
    tokens = [tok.node_id(i) for i in (0, 1, 4, 2, 3)]
    assert replay_nodes(graph, 0, 3, tokens, inverse)['valid']  # 4->2 is NOT an edge, but an implicit POP
    assert not replay_nodes(graph, 0, 3, [tok.node_id(i) for i in (0, 1, 2, 3)], inverse)['valid']
    state = []
    for stack, visited, candidates, action, arg in reference(graph, 0, 3):
        for field in (stack, visited, candidates):
            state.extend(tok.node_id(i) for i in field)
            state.append(tok.id('|'))
        state.append(tok.id(f'<{action}>'))
        if arg is not None:
            state.append(tok.node_id(arg))
        state.append(tok.id(';'))
    assert replay_state(graph, 0, 3, state, tok, inverse)['valid']
    wrong = state[:]
    wrong[0] = tok.node_id(2)
    assert not replay_state(graph, 0, 3, wrong, tok, inverse)['valid']
    assert len(jobs_for('pilot', cfg, cfg['protocols'])) == 16
    report = {'status': 'PASS', 'manifest_sha256': digest(manifest), 'source_sha256': source_hash(args.repo_root),
              'banks': audits, 'checks': ['hard_world_hash_and_role_split_manifest', 'all_reference_variants_semantically_checked',
                'no_reference_truncation', '48_short_16_long_16_adj', 'paired_protocol_query_schedule_including_PATH',
                'logical_microbatch_loss_and_gradient_reference', 'implicit_pop_nodes_not_a_path', 'state_mutation_rejected'],
              'formal_training_started': False}
    atomic_json(HERE / 'verification.json', report)
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
