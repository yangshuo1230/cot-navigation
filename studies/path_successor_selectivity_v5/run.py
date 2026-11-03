"""Frozen PATH successor selectivity: native rollouts, sublayer lenses and ridge.

All model forward and closed-form probe fits use GPU0. Activations and probe
weights stay in RAM. The model is never updated. Query and probe-node holdouts
are distinct: nodes in a probe holdout were seen by the original LM via ADJ.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('MKL_NUM_THREADS', '2')
import numpy as np
import torch
torch.set_num_threads(2)

ROOT = Path.cwd()
HERE = ROOT / 'studies/path_successor_selectivity_v5'
SUITE = ROOT / 'studies/fixed_navigation_v5'
sys.path[:0] = [str(ROOT / 'src'), str(SUITE)]
from data import atomic_json, bfs, digest
from cot_navigation.models.transformer import TransformerLM
from cot_navigation.protocols.tokenizer import Tokenizer
from cot_navigation.evaluation.rollout import generate

STAGES = ['E'] + [name for layer in range(1, 7) for name in (f'L{layer}A', f'L{layer}F')]
TEST = ['short_test', 'long_AA_test', 'long_TA_test', 'long_AT_test', 'long_TT_test']
VAL = ['short_val', 'long_AA_val', 'long_VA_val', 'long_AV_val', 'long_VV_val']


def mean_query(values, qids):
    """Queries, not longer successful trajectories, receive equal metric weight."""
    values = np.asarray(values)
    if not len(values):
        return None
    return float(np.mean([values[qids == q].mean() for q in np.unique(qids)]))


def ranking(labels, scores, eligible, qids):
    auc, recall, hit, random_hit, margin, selected_q = [], [], [], [], [], []
    for y, s, mask, q in zip(labels, scores, eligible, qids):
        y, s = y[mask].astype(bool), s[mask]
        k = int(y.sum())
        if not k or k == len(y):
            continue
        pos, neg = s[y], s[~y]
        differences = pos[:, None] - neg[None, :]
        auc.append(float(((differences > 0) + .5 * (differences == 0)).mean()))
        recall.append(float(y[np.argsort(s, kind='stable')[-k:]].mean()))
        hit.append(float(y[np.argmax(s)]))
        random_hit.append(float(y.mean()))
        margin.append(float(pos.mean() - neg.mean()))
        selected_q.append(q)
    q = np.asarray(selected_q)
    return {'decisions': len(q), 'queries': len(np.unique(q)),
            'auc': mean_query(auc, q), 'recall_at_true_k': mean_query(recall, q),
            'top1_useful': mean_query(hit, q), 'uniform_top1': mean_query(random_hit, q),
            'positive_minus_negative_score': mean_query(margin, q)}


def fit_ridge(features, labels, split, meta):
    """Query-weighted affine ridge, train normalization, validation-only penalty.

    Scores are uncalibrated linear scores, never called probabilities. Labels
    are complete neighbor / useful-neighbor indicator vectors, not one token.
    """
    train, val, test = split
    if min(map(len, split)) == 0:
        return None
    if features.ndim == 2:
        features = features[:, None, :]
    z = torch.as_tensor(features, device='cuda:0').transpose(0, 1)
    y = torch.as_tensor(labels, device='cuda:0')
    def weights(ids):
        counts = Counter(meta['qid'][ids].tolist())
        w = np.array([1. / counts[q] for q in meta['qid'][ids]], dtype=np.float32)
        return torch.tensor(w / w.sum(), device='cuda:0')
    wt, wv = weights(train), weights(val)
    mu = (z[:, train] * wt[None, :, None]).sum(1, keepdim=True)
    sd = ((z[:, train] - mu).square() * wt[None, :, None]).sum(1, keepdim=True).sqrt().clamp_min(.05)
    z = (z - mu) / sd
    z = torch.cat([z, torch.ones((*z.shape[:2], 1), device=z.device)], dim=-1)
    zt = z[:, train]
    gram = zt.transpose(1, 2) @ (zt * wt[None, :, None])
    rhs = zt.transpose(1, 2) @ (y[train] * wt[:, None])
    eye = torch.eye(z.shape[-1], device=z.device)
    eye[-1, -1] = 0
    best = np.full(z.shape[0], np.inf)
    chosen = np.zeros(z.shape[0])
    output = np.zeros((len(test), z.shape[0], labels.shape[-1]), np.float32)
    for penalty in (.001, .01, .1, 1., 10.):
        coefficient = torch.linalg.solve(gram + penalty * eye + 1e-6 * torch.eye(z.shape[-1], device=z.device), rhs)
        prediction = z[:, val] @ coefficient
        error = ((prediction - y[val]).square().mean(-1) * wv).sum(-1).cpu().numpy()
        test_pred = (z[:, test] @ coefficient).transpose(0, 1).cpu().numpy()
        for layer in range(z.shape[0]):
            if error[layer] < best[layer]:
                best[layer], chosen[layer] = error[layer], penalty
                output[:, layer] = test_pred[:, layer]
    return output, chosen.tolist()


@torch.inference_mode()
def collect(seed):
    world = json.loads((SUITE / 'data/n80_g0.json').read_text())
    folder = ROOT / f'outputs/fixed_navigation_v5/pilot/n80/g0_s{seed}/path'
    blob = torch.load(folder / 'latest.pt', map_location='cpu', weights_only=False)
    assert blob['step'] == 50000 and digest(world) == blob['world_hash']
    tok = Tokenizer(256)
    nodes = [tok.node_id(n) for n in world['node_names']]
    inv = {token: u for u, token in enumerate(nodes)}
    n = len(nodes)
    adjacency = np.zeros((n, n), np.float32)
    reach = np.eye(n, dtype=bool)
    for u, vs in enumerate(world['adj']):
        adjacency[u, vs] = 1
        reach[u, list(bfs(world['adj'], u))] = True
    dp = np.eye(n, dtype=bool)
    for u in reversed(range(n)):
        for v in world['adj'][u]:
            dp[u] |= dp[v]
    assert np.array_equal(reach, dp)
    model = TransformerLM(len(tok), **blob['job']['model'], dropout=0.).to('cuda:0').eval().requires_grad_(False)
    model.load_state_dict(blob['model'])
    embedding = model.embed.weight.detach().float().cpu().numpy()[nodes]
    del blob
    panels = ['short_train', 'long_train'] + VAL + TEST
    rows, checks, rollout_examples = [], {}, []
    final = json.loads((folder / 'final.json').read_text())
    for panel in panels:
        ids = world['panels'][panel]
        counter = Counter()
        for start in range(0, len(ids), 128):
            batch = ids[start:start + 128]
            prompts = [[tok.id('<QUERY>'), nodes[world['records'][q]['s']],
                        nodes[world['records'][q]['t']], tok.id('|')] for q in batch]
            outputs = generate(model, prompts, 'path', tok, 'bf16', max_new_tokens=16)
            for qid, prompt, generated in zip(batch, prompts, outputs):
                r = world['records'][qid]
                counter[f"n{r['y']}"] += 1
                counter[f"correct{r['y']}"] += generated['answer_token'] == tok.label_id(r['y'])
                tokens = generated['trace_tokens']
                path = [inv.get(t, -1) for t in tokens]
                valid = bool(path) and path[0] == r['s'] and all(u >= 0 for u in path)
                if valid:
                    valid = all(v in world['adj'][u] for u, v in zip(path, path[1:]))
                counter['positive_valid_goal'] += r['y'] and valid and path[-1] == r['t']
                rows.append({'qid': qid, 'record': r, 'panel': panel,
                             'split': 0 if panel.endswith('_train') else 1 if panel.endswith('_val') else 2,
                             'prompt': prompt, 'trace': tokens, 'generated': generated['generated']})
        if panel in TEST:
            expected = final['panels'][panel]['metrics']['all']
            checks[panel] = {'replayed': dict(counter),
                'all_answer_and_goal_counts_match': all(counter[k] == expected.get(k, 0)
                    for k in ('n0', 'n1', 'correct0', 'correct1', 'positive_valid_goal'))}
    print(json.dumps({'seed': seed, 'native_rollouts': len(rows), 'replay': checks}), flush=True)

    observed, enabled = {}, [True]
    def before(layer):
        def hook(module, args):
            if enabled[0]:
                observed['in', layer] = args[0]
        return hook
    def after(layer):
        def hook(module, args, output):
            if enabled[0]:
                observed['attn', layer] = observed['in', layer] + output[0]
        return hook
    handles = []
    for layer, block in enumerate(model.blocks):
        handles.extend([block.register_forward_pre_hook(before(layer)), block.attn.register_forward_hook(after(layer))])
    feature, metadata, lens, probabilities = [], [], [], []
    observation_error = None
    argmax_disagree = 0
    next_count = 0
    for start in range(0, len(rows), 64):
        part = rows[start:start + 64]
        sequences = [r['prompt'] + r['trace'] for r in part]
        length = max(map(len, sequences))
        tokens = torch.full((len(part), length), tok.pad, dtype=torch.long, device='cuda:0')
        for b, seq in enumerate(sequences):
            tokens[b, :len(seq)] = torch.tensor(seq, device=tokens.device)
        capture = {}
        with torch.autocast('cuda', dtype=torch.bfloat16):
            logits = model(tokens, capture=capture)
            sublayers = [model.embed(tokens)]
            for layer in range(6):
                sublayers.extend([observed['attn', layer], capture['resid'][layer]])
            hidden = torch.stack(sublayers)
            scores = model.lm_head(model.final_norm(hidden))
            if start == 0:
                enabled[0] = False
                untouched = model(tokens)
                enabled[0] = True
                observation_error = float((untouched - logits).abs().max())
                assert observation_error == 0
                assert torch.equal(scores[-1], logits)
        h = hidden.float().cpu().numpy()
        ls = scores[..., nodes].float().cpu().numpy()
        pr = scores.float().softmax(-1)[..., nodes].cpu().numpy()
        native = logits.argmax(-1).cpu().numpy()
        for b, row in enumerate(part):
            r, trace = row['record'], row['trace']
            for j, token in enumerate(trace):
                current = inv.get(token, -1)
                if current < 0 or (j == 0 and current != r['s']):
                    break
                if j and current not in world['adj'][inv[trace[j - 1]]]:
                    break
                if current == r['t'] or not world['adj'][current]:
                    break  # reaching goal predicts answer, not another navigation step
                nxt = row['generated'][j + 1] if j + 1 < len(row['generated']) else -1
                position = 4 + j
                feature.append(h[:, b, position].copy())
                lens.append(ls[:, b, position].copy())
                probabilities.append(pr[:, b, position].copy())
                metadata.append({'qid': row['qid'], 'split': row['split'], 'panel': row['panel'],
                                 'source': r['s'], 'target': r['t'], 'current': current,
                                 'next_token': nxt, 'first': j == 0, 'progress': j, 'gap': r['gap'], 'label': r['y']})
                argmax_disagree += int(native[b, position] != nxt)
                next_count += 1
        if start % 512 == 0:
            print(f'seed{seed} activation batches {start}/{len(rows)}', flush=True)
    for handle in handles:
        handle.remove()
    x, scores, prob = np.asarray(feature), np.asarray(lens), np.asarray(probabilities)
    meta = {key: np.array([r[key] for r in metadata]) for key in metadata[0]}
    ys = adjacency[meta['current']]
    yu = ys * reach[:, meta['target']].T
    mixed = (yu.sum(1) > 0) & (yu.sum(1) < ys.sum(1))
    selection = (meta['split'] == 2) & mixed
    nonlocal_choice = meta['target'] // world['width'] - meta['current'] // world['width'] >= 2
    rows_out = []
    for ix in np.flatnonzero(selection & meta['first']):
        rows_out.append({'index': int(ix), 'qid': int(meta['qid'][ix]), 'source': int(meta['source'][ix]),
                         'target': int(meta['target'][ix]), 'gap': int(meta['gap'][ix]),
                         'useful': np.flatnonzero(yu[ix]).tolist(), 'next': inv.get(int(meta['next_token'][ix]), -1)})
    pairs = []
    for i, a in enumerate(rows_out):
        for b in rows_out[i + 1:]:
            if a['source'] != b['source'] or a['gap'] != b['gap'] or set(a['useful']) & set(b['useful']):
                continue
            both = a['next'] in a['useful'] and b['next'] in b['useful']
            pairs.append({'source': a['source'], 'query_a': a['qid'], 'query_b': b['qid'],
                          'target_a': a['target'], 'target_b': b['target'],
                          'useful_a': a['useful'], 'useful_b': b['useful'],
                          'next_a': a['next'], 'next_b': b['next'], 'both_useful': both,
                          'changes': a['next'] != b['next']})
    out = {'seed': seed, 'checkpoint_step': 50000,
           'checkpoint_sha256': hashlib.sha256((folder / 'latest.pt').read_bytes()).hexdigest(),
           'world_hash': digest(world), 'stage_names': STAGES,
           'native_replay': checks, 'observation_hook_logit_error': observation_error,
           'full_prefix_argmax_disagreement_fraction': argmax_disagree / max(1, next_count),
           'sample_counts': {'queries': len(rows), 'decisions': len(x),
               'split_decisions': {str(s): int((meta['split'] == s).sum()) for s in range(3)},
               'mixed_test_decisions': int(selection.sum()), 'mixed_test_queries': len(np.unique(meta['qid'][selection]))},
           'matched_targets': {'same_source_and_same_target_depth': True, 'pairs': len(pairs),
               'unique_queries': len({p[k] for p in pairs for k in ('query_a', 'query_b')}),
               'unique_sources': len({p['source'] for p in pairs}),
               'both_useful': float(np.mean([p['both_useful'] for p in pairs])) if pairs else None,
               'choice_changes': float(np.mean([p['changes'] for p in pairs])) if pairs else None,
               'examples': pairs[:3]}, 'lenses': {}}
    for group, select in [('all_test', meta['split'] == 2), ('mixed_test', selection),
                          ('nonlocal_mixed_test', selection & nonlocal_choice),
                          ('first_mixed_test', selection & meta['first']),
                          ('first_nonlocal_mixed_test', selection & meta['first'] & nonlocal_choice)] + [
                          (p + '_mixed', selection & (meta['panel'] == p)) for p in TEST]:
        ids = np.flatnonzero(select)
        qids = meta['qid'][ids]
        stages = []
        for layer in range(13):
            p, s = prob[ids, layer], scores[ids, layer]
            mass = (p * ys[ids]).sum(1)
            useful_mass = (p * yu[ids]).sum(1)
            candidate_scores = np.where(ys[ids].astype(bool), s, -np.inf)
            chosen = candidate_scores.argmax(1)
            conditional = np.exp(candidate_scores - candidate_scores.max(1, keepdims=True))
            conditional /= conditional.sum(1, keepdims=True)
            stages.append({'successor_full_vocab_mass': mean_query(mass, qids),
                'successor_node_conditional_mass': mean_query(mass / p.sum(1).clip(1e-30), qids),
                'useful_given_successor_mass': mean_query((conditional * yu[ids]).sum(1), qids),
                'useful_choice_given_successors': mean_query(yu[ids, chosen], qids),
                'uniform_useful_baseline': mean_query(yu[ids].sum(1) / ys[ids].sum(1).clip(1), qids),
                'ranking': ranking(yu[ids], s, ys[ids].astype(bool), qids)})
        out['lenses'][group] = {'decisions': len(ids), 'queries': len(np.unique(qids)), 'stages': stages}
    actual_next = np.array([inv.get(int(t), -1) for t in meta['next_token']])
    behavior = {}
    for name, selected in [('all_mixed', selection), ('first_mixed', selection & meta['first']),
                           ('nonlocal_mixed', selection & nonlocal_choice),
                           ('first_nonlocal_mixed', selection & meta['first'] & nonlocal_choice)] + [
                           (p, selection & (meta['panel'] == p)) for p in TEST]:
        ids = np.flatnonzero(selected)
        hit = np.array([v >= 0 and bool(yu[i, v]) for i, v in zip(ids, actual_next[ids])])
        behavior[name] = {'decisions': len(ids), 'queries': len(np.unique(meta['qid'][ids])),
                          'native_argmax_useful': mean_query(hit, meta['qid'][ids])}
    out['behavior'] = behavior
    far_pairs = [p for p in pairs if world['records'][p['query_a']]['gap'] >= 2]
    out['matched_targets_nonlocal'] = {
        'pairs': len(far_pairs), 'unique_sources': len({p['source'] for p in far_pairs}),
        'unique_queries': len({p[k] for p in far_pairs for k in ('query_a', 'query_b')}),
        'both_useful': float(np.mean([p['both_useful'] for p in far_pairs])) if far_pairs else None,
        'choice_changes': float(np.mean([p['changes'] for p in far_pairs])) if far_pairs else None,
        'examples': far_pairs[:3]}
    del model, hidden, logits, scores, prob
    torch.cuda.empty_cache()
    return out, x, meta, ys, yu, embedding


def analyze(seed):
    out, x, meta, ys, yu, embedding = collect(seed)
    split = [np.flatnonzero(meta['split'] == s) for s in range(3)]
    folds = {}
    for fold_seed in (17, 23, 31):
        rng = np.random.default_rng(fold_seed)
        fold = np.zeros(len(embedding), int)
        for layer in range(10):
            ids = rng.permutation(np.arange(layer * 8, (layer + 1) * 8))
            fold[ids[4:6]] = 1
            fold[ids[6:]] = 2
        for role in ('current', 'target'):
            folds[f'{role}_heldout_{fold_seed}'] = [
                np.flatnonzero((meta['split'] == s) & (fold[meta[role]] == s)) for s in range(3)]
    schemes = {'query_only': split, **folds}
    out['probes'] = {}
    for name, sp in schemes.items():
        test = sp[2]
        output = {'split_decisions': list(map(len, sp)), 'split_queries': [len(np.unique(meta['qid'][v])) for v in sp],
                  'features': {}}
        controls = [('residual', x), ('target_embedding', embedding[meta['target']])]
        # Same-dimensional random node identities test whether ridge simply
        # learns a table. Repeat random controls without altering node folds.
        for control_seed in (101, 202, 303):
            random_nodes = np.random.default_rng(control_seed).normal(0, embedding.std(), embedding.shape).astype(np.float32)
            controls.extend([(f'random_current_{control_seed}', random_nodes[meta['current']]),
                             (f'random_target_{control_seed}', random_nodes[meta['target']])])
        for feature_name, feature in controls:
            tasks = {}
            for task, labels in [('successor', ys), ('useful', yu)]:
                fitted = fit_ridge(feature, labels, sp, meta)
                if fitted is None:
                    tasks[task] = None
                    continue
                prediction, penalty = fitted
                stages, nonlocal_stages = [], []
                for layer in range(prediction.shape[1]):
                    mask = np.ones_like(labels[test], dtype=bool) if task == 'successor' else ys[test].astype(bool)
                    stages.append(ranking(labels[test], prediction[:, layer], mask, meta['qid'][test]))
                    far = meta['target'][test] // 8 - meta['current'][test] // 8 >= 2
                    nonlocal_stages.append(ranking(labels[test][far], prediction[far, layer], mask[far], meta['qid'][test][far]))
                tasks[task] = {'penalty': penalty, 'stages': stages, 'nonlocal_stages': nonlocal_stages}
            output['features'][feature_name] = tasks
        out['probes'][name] = output
        print(f'seed{seed} probes {name} complete', flush=True)
    out['scope_note'] = ('Native legal prefixes including the first failed next choice, never post-illegal-edge states. '
        'Terminal goal positions excluded. Metrics query-equal; mixed means both useful and useless true successors. '
        'Probe train/val/test obey QUERY split; extra node holdout is imposed independently. '
        'No activations or weights saved. Early head lenses are uncalibrated; oracle successor restriction is diagnostic only.')
    HERE.mkdir(parents=True, exist_ok=True)
    atomic_json(HERE / f'seed{seed}.json', out)
    print(f'seed{seed} COMPLETE', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2, 3])
    args = parser.parse_args()
    for seed in args.seeds:
        analyze(seed)
