"""Held-edited-node semantic subspace interventions, separate input/output sides.

Ranks8/16 and coefficient1 are fixed before observing causal effects. Random
orthogonal-complement donor controls and in-subspace random directions match
each semantic edit's norm. All weights/probe coefficients remain in RAM.
"""
from __future__ import annotations
from collections import defaultdict
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

os.environ.setdefault('OMP_NUM_THREADS', '2')
import numpy as np
import torch
torch.set_num_threads(2)
torch.backends.cuda.matmul.allow_tf32 = False
ROOT = Path.cwd()
HERE = ROOT / 'studies/path_semantic_circuit_v5'
BASE = ROOT / 'studies/fixed_navigation_v5'
sys.path[:0] = [str(ROOT / 'src'), str(BASE)]
from data import atomic_json, digest, bfs
from cot_navigation.models.transformer import TransformerLM
from cot_navigation.protocols.tokenizer import Tokenizer
spec = importlib.util.spec_from_file_location('coarse_circuit', ROOT / 'studies/path_causal_circuit_v5/run.py')
coarse = importlib.util.module_from_spec(spec)
spec.loader.exec_module(coarse)

RANKS = (8, 16)
SPLIT_SEEDS = (17, 23, 31)
RANDOM_SEEDS = (101, 202, 303)
RELATIONS = ('ancestor', 'descendant', 'successor')


def partition(anchors, seed, n=80, width=8):
    """48/16/16, all layers represented, both edited nodes in test only."""
    rng = np.random.default_rng(seed + 1709*min(anchors) + 97*max(anchors))
    level = anchors[0] // width
    assert all(i // width == level for i in anchors)
    extra_test = {level, *rng.choice([i for i in range(10) if i != level], 5, replace=False).tolist()}
    extra_val = set(rng.choice(10, 6, replace=False).tolist())
    train, val, test = [], [], []
    for layer in range(10):
        reserved = [i for i in anchors if i // width == layer]
        pool = [i for i in range(layer*width, (layer+1)*width) if i not in reserved]
        rng.shuffle(pool)
        nt, nv = 1 + int(layer in extra_test), 1 + int(layer in extra_val)
        test.extend(reserved + pool[:nt-len(reserved)])
        remaining = pool[nt-len(reserved):]
        val.extend(remaining[:nv])
        train.extend(remaining[nv:])
    assert tuple(map(len, (train, val, test))) == (48, 16, 16)
    assert set(anchors) <= set(test) and not set(test) & (set(train) | set(val))
    return np.array(train), np.array(val), np.array(test)


@torch.inference_mode()
def prototype_features(model, world, tok, nodes):
    """Goal E/A1/F1 states on main-model TRAIN QUERY prompts only."""
    qids = world['panels']['short_train'] + world['panels']['long_train']
    source, target, features = [], [], []
    captured, handles = {}, []
    def pre(module, args):
        captured['input'] = args[0]
    def post(module, args, out):
        captured['attention'] = captured['input'] + out[0]
    handles.extend([model.blocks[0].register_forward_pre_hook(pre), model.blocks[0].attn.register_forward_hook(post)])
    try:
        for start in range(0, len(qids), 128):
            part = qids[start:start+128]
            rows = [world['records'][q] for q in part]
            tokens = torch.tensor([[tok.id('<QUERY>'), nodes[r['s']], nodes[r['t']], tok.id('|')] for r in rows], device='cuda:0')
            cap = {}
            model(tokens, capture=cap)
            h = torch.stack([model.embed(tokens)[:, 2], captured['attention'][:, 2], cap['resid'][0][:, 2]], dim=1)
            features.append(h.cpu().numpy())
            source.extend(r['s'] for r in rows)
            target.extend(r['t'] for r in rows)
    finally:
        for h in handles:
            h.remove()
    return np.asarray(source), np.asarray(target), np.concatenate(features)


class Directions:
    def __init__(self, embedding, prototypes, labels):
        self.embedding, self.prototypes, self.labels, self.cache = embedding, prototypes, labels, {}

    def fit(self, stage, anchors, seed):
        key = (stage, tuple(sorted(anchors)), seed)
        if key in self.cache:
            return self.cache[key]
        tr, va, te = partition(anchors, seed)
        if stage == 0:
            x = self.embedding.copy()
            counts = np.ones(len(x), dtype=int)
        else:
            source, target, h = self.prototypes
            selected = np.isin(source, tr)  # no edited/test/validation source token in fit contexts
            sums, counts = np.zeros_like(self.embedding), np.zeros(len(self.embedding), int)
            np.add.at(sums, target[selected], h[selected, stage])
            np.add.at(counts, target[selected], 1)
            x = sums / counts.clip(1)[:, None]
        tr, va = tr[counts[tr] > 0], va[counts[va] > 0]
        assert not set(anchors) & (set(tr) | set(va)) and counts[list(anchors)].min() > 0
        xx = torch.tensor(x, dtype=torch.float64, device='cuda:0')
        yy = torch.tensor(np.concatenate([self.labels[t] for t in RELATIONS], axis=1), dtype=xx.dtype, device=xx.device)
        mu, sd = xx[tr].mean(0), xx[tr].std(0, unbiased=False).clamp_min(1e-6)
        z = (xx - mu) / sd
        ym = yy[tr].mean(0)
        gram = z[tr] @ z[tr].T / len(tr)
        penalties = (.0001, .001, .01, .1, 1., 10., 100.)
        eye = torch.eye(len(tr), dtype=xx.dtype, device=xx.device)
        matrices = torch.stack([gram + p*eye for p in penalties])
        dual = torch.linalg.solve(matrices, ((yy[tr] - ym)/len(tr)).expand(len(penalties), -1, -1))
        coefficients = z[tr].T[None] @ dual
        prediction = z[va][None] @ coefficients + ym
        result = {'train_n': len(tr), 'val_n': len(va), 'edited_in_train_or_val': False, 'weights': {}, 'basis': {}, 'overlap': {}}
        for i, task in enumerate(RELATIONS):
            sl = slice(i*80, (i+1)*80)
            mse = (prediction[:, :, sl] - yy[va, sl]).square().mean((1, 2))
            choice = int(mse.argmin())
            w = (coefficients[choice, :, sl] / sd[:, None]).cpu().numpy()
            result['weights'][task] = w
            u, _, _ = np.linalg.svd(w, full_matrices=False)
            for rank in RANKS:
                result['basis'][task, rank] = u[:, :rank]
        for rank in RANKS:
            ua, ud = result['basis']['ancestor', rank], result['basis']['descendant', rank]
            result['overlap'][rank] = float(np.linalg.norm(ua.T @ ud, 'fro')**2 / rank)
            for task, other in [('ancestor', 'descendant'), ('descendant', 'ancestor')]:
                w, uother = result['weights'][task], result['basis'][other, rank]
                residual = w - uother @ (uother.T @ w)
                u, _, _ = np.linalg.svd(residual, full_matrices=False)
                result['basis'][task+'_unique', rank] = u[:, :rank]
        self.cache[key] = result
        return result


def edit_variants(delta, u, random_seed):
    semantic = u @ (u.T @ delta)
    norm = np.linalg.norm(semantic)
    rng = np.random.default_rng(random_seed)
    random_matrix = rng.normal(size=(len(delta), u.shape[1]))
    random_matrix -= u @ (u.T @ random_matrix)
    q, _ = np.linalg.qr(random_matrix)
    orth = q @ (q.T @ delta)
    orth *= norm / max(np.linalg.norm(orth), 1e-12)
    direction = u @ rng.normal(size=u.shape[1])
    direction *= norm / max(np.linalg.norm(direction), 1e-12)
    return semantic, orth, direction


def probe_effect(w, delta, edit, labels_delta, special):
    full, partial = delta @ w, edit @ w
    energy = float(partial @ full / max(full @ full, 1e-15))
    truth = float(partial @ labels_delta / max(labels_delta @ labels_delta, 1e-15))
    a, b = special
    return {'probe_fraction_of_full_donor_change': energy,
            'probe_true_direction_correct': float((-full[a] + full[b]) > 0),
            'probe_targeted_edit_alignment': float((-partial[a] + partial[b]) / 2),
            'probe_all_label_alignment': truth,
            'edit_norm_ratio': float(np.linalg.norm(edit) / max(np.linalg.norm(delta), 1e-15))}


def outcome(logit, row, donor, nodes, adj):
    token = int(logit.argmax())
    inverse = {t: i for i, t in enumerate(nodes)}
    choice = inverse.get(token, -1)
    return {'donor_useful': float(choice in donor['useful']), 'recipient_useful': float(choice in row['useful']),
            'legal': float(choice in adj[row['current']])}


def summarize_records(records):
    result = {}
    for key, rows in records.items():
        fields = sorted({k for r in rows for k in r if isinstance(r[k], (float, int, np.number))})
        result[key] = {'evaluations': len(rows), **{f: float(np.mean([r[f] for r in rows if f in r])) for f in fields}}
    return result


@torch.inference_mode()
def goal_edits(model, rows, nodes, tok, states, base_logits, pairs, directions, labels, adj):
    records, controls, examples, edits_by_stage = defaultdict(list), {}, [], defaultdict(list)
    for stage in (0, 1, 2):
        clean_rows = [rows[i] for pair in pairs for i in pair]
        donor_states = states[[i for a, b in pairs for i in (b, a)]]
        full = coarse.patch_forward(model, clean_rows, donor_states, tok, stage, 'target')
        metrics = [outcome(logit, rows[a], rows[b], nodes, adj) for logit, (a, b) in
                   zip(full, [(a, b) for i, j in pairs for a, b in ((i, j), (j, i))])]
        controls[f'stage{stage}_full_swap'] = summarize_records({'x': metrics})['x']
        if stage == 0:
            expected = base_logits[[b for i, j in pairs for a, b in ((i, j), (j, i))]]
            assert np.max(abs(full - expected)) < 1e-3
        for pair_id, (a, b) in enumerate(pairs):
            ra, rb = rows[a], rows[b]
            anchors = [ra['target'], rb['target']]
            delta = states[b, stage, 0].astype(float) - states[a, stage, 0]
            for split_seed in SPLIT_SEEDS:
                fit = directions.fit(stage, anchors, split_seed)
                for relation in ('ancestor', 'descendant', 'successor', 'ancestor_unique', 'descendant_unique'):
                    task = relation.replace('_unique', '')
                    w = fit['weights'][task]
                    truth_delta = labels[task][anchors[1]] - labels[task][anchors[0]]
                    special = (int(nodes.index(int(base_logits[a].argmax()))), int(nodes.index(int(base_logits[b].argmax()))))
                    for rank in RANKS:
                        u = fit['basis'][relation, rank]
                        semantic = u @ (u.T @ delta)
                        variants = [('semantic', semantic, 1), ('complement', delta-semantic, 1)]
                        for random_seed in RANDOM_SEEDS:
                            _, orth, random_direction = edit_variants(delta, u, random_seed + pair_id*71 + split_seed*13)
                            variants.extend([('random_orth_normmatched', orth, 1), ('random_inspace_normmatched', random_direction, 1)])
                        if rank == 8 and relation in ('ancestor', 'descendant'):
                            variants.append(('semantic_half', semantic*.5, 1))
                        for control, edit, _ in variants:
                            key = f'stage{stage}/{relation}/r{rank}/{control}'
                            pe = probe_effect(w, delta, edit, truth_delta, special)
                            pe.update(train_nodes=fit['train_n'], validation_nodes=fit['val_n'],
                                      ancestor_descendant_overlap=fit['overlap'][rank])
                            for recipient, donor, sign in ((a, b, 1), (b, a, -1)):
                                edits_by_stage[stage].append((key, recipient, donor, sign*edit, pe, pair_id))
    for stage, specifications in edits_by_stage.items():
        for start in range(0, len(specifications), 128):
            part = specifications[start:start+128]
            recipient = [rows[r] for _, r, _, _, _, _ in part]
            # Reuse the tested E/A/F positional swap hook, providing edited
            # target vectors. Every other role and the output head stay fixed.
            donor_tensor = np.zeros((len(part), 13, 4, 128), np.float32)
            for i, (_, r, d, edit, _, _) in enumerate(part):
                donor_tensor[i, stage, 0] = states[r, stage, 0] + edit
            logits = coarse.patch_forward(model, recipient, donor_tensor, tok, stage, 'target')
            for logit, (key, r, d, edit, pe, pair_id) in zip(logits, part):
                entry = {**outcome(logit, rows[r], rows[d], nodes, adj), **pe}
                records[key].append(entry)
                if key == 'stage2/ancestor/r8/semantic' and len(examples) < 3:
                    examples.append({'pair': pair_id, 'query': rows[r]['qid'], 'donor_query': rows[d]['qid'], **entry})
        print(f'goal stage{stage} semantic edits complete; {len(specifications)} directed evaluations', flush=True)
    return {'effects': summarize_records(records), 'controls': controls, 'examples': examples}


def head_edits(model, rows, states, base_logits, pairs, nodes, directions, labels, adj, reach):
    normalized = model.final_norm(torch.tensor(states[:, 12, 2], device='cuda:0')).cpu().numpy().astype(float)
    embedding = model.lm_head.weight.detach().cpu().numpy().astype(float)
    records, controls = defaultdict(list), defaultdict(list)
    inverse = {token: i for i, token in enumerate(nodes)}
    for pair_id, (a, b) in enumerate(pairs):
        ra, rb = rows[a], rows[b]
        va, vb = inverse[int(base_logits[a].argmax())], inverse[int(base_logits[b].argmax())]
        assert va in ra['useful'] and vb in rb['useful']
        anchors = [va, vb]
        delta = embedding[nodes[vb]] - embedding[nodes[va]]
        relatives = [i for i, r in enumerate(rows) if i not in (a, b) and
                     r['prefix'][:2] + r['prefix'][3:] == ra['prefix'][:2] + ra['prefix'][3:]
                     and r['target_layer'] == ra['target_layer']
                     and reach[va, r['target']] == reach[vb, r['target']]]
        relatives = relatives[:8]
        def calculate(edit):
            entries = []
            for r, d in ((a, b), (b, a)):
                logit = base_logits[r].copy()
                shift = float(normalized[r] @ edit)
                logit[nodes[va]] += shift
                logit[nodes[vb]] -= shift
                entry = outcome(logit, rows[r], rows[d], nodes, adj)
                entry['directed_pair_margin_shift'] = (-2*shift if r == a else 2*shift)
                entries.append(entry)
            for i in relatives:
                logit = base_logits[i].copy()
                shift = float(normalized[i] @ edit)
                logit[nodes[va]] += shift
                logit[nodes[vb]] -= shift
                baseline_choice = inverse.get(int(base_logits[i].argmax()), -1)
                edited_choice = inverse.get(int(logit.argmax()), -1)
                for entry in entries:
                    entry['unrelated_abs_pair_margin_shift'] = abs(2*shift)
                    entry['unrelated_choice_changed'] = float(edited_choice != baseline_choice)
                    entry['unrelated_useful_after'] = float(edited_choice in rows[i]['useful'])
                    entry['unrelated_useful_before'] = float(baseline_choice in rows[i]['useful'])
                # Each pair averages its available unrelated targets below;
                # collect multiple values separately to avoid silent overwrite.
            if relatives:
                values = []
                for i in relatives:
                    shift = float(normalized[i] @ edit)
                    logit = base_logits[i].copy()
                    logit[nodes[va]] += shift
                    logit[nodes[vb]] -= shift
                    choice = inverse.get(int(logit.argmax()), -1)
                    old = inverse.get(int(base_logits[i].argmax()), -1)
                    values.append([abs(2*shift), choice != old, choice in rows[i]['useful'], old in rows[i]['useful']])
                names = ['unrelated_abs_pair_margin_shift', 'unrelated_choice_changed', 'unrelated_useful_after', 'unrelated_useful_before']
                for entry in entries:
                    entry.update({k: float(v) for k, v in zip(names, np.mean(values, axis=0))})
            for entry in entries:
                entry['unrelated_targets_available'] = float(bool(relatives))
            return entries
        controls['full_row_swap'].extend(calculate(delta))
        # Whole-row swap is a label permutation sanity check, not evidence
        # that descendant semantics are used by the model.
        assert all(x['donor_useful'] == 1 for x in calculate(delta))
        for split_seed in SPLIT_SEEDS:
            fit = directions.fit(0, anchors, split_seed)
            for relation in ('ancestor', 'descendant', 'successor', 'ancestor_unique', 'descendant_unique'):
                task = relation.replace('_unique', '')
                w, truth = fit['weights'][task], labels[task][vb] - labels[task][va]
                for rank in RANKS:
                    u = fit['basis'][relation, rank]
                    semantic = u @ (u.T @ delta)
                    variants = [('semantic', semantic), ('complement', delta-semantic)]
                    for random_seed in RANDOM_SEEDS:
                        _, orth, random_direction = edit_variants(delta, u, random_seed + pair_id*71 + split_seed*13)
                        variants.extend([('random_orth_normmatched', orth), ('random_inspace_normmatched', random_direction)])
                    if rank == 8 and relation in ('ancestor', 'descendant'):
                        variants.append(('semantic_half', .5*semantic))
                    for control, edit in variants:
                        pe = probe_effect(w, delta, edit, truth, (ra['target'], rb['target']))
                        # Raw candidate delta has descendant(tA)<0, descendant(tB)>0;
                        # this task-specific readout check is descriptive only.
                        pe['ancestor_descendant_overlap'] = fit['overlap'][rank]
                        key = f'{relation}/r{rank}/{control}'
                        entries = calculate(edit)
                        records[key].extend({**entry, **pe} for entry in entries)
                        if pair_id == 0 and control == 'semantic' and rank == 8 and split_seed == 17:
                            weight = model.lm_head.weight.detach().clone()
                            weight[nodes[va]] += torch.tensor(edit, dtype=weight.dtype, device=weight.device)
                            weight[nodes[vb]] -= torch.tensor(edit, dtype=weight.dtype, device=weight.device)
                            actual = F_linear(torch.tensor(normalized[a], dtype=weight.dtype, device=weight.device), weight).cpu().numpy()
                            expected = base_logits[a].copy()
                            expected[nodes[va]] += normalized[a] @ edit
                            expected[nodes[vb]] -= normalized[a] @ edit
                            assert np.max(abs(actual - expected)) < 1e-3
    return {'effects': summarize_records(records), 'controls': summarize_records(controls)}


def F_linear(x, weight):
    return torch.nn.functional.linear(x, weight)


@torch.inference_mode()
def run(seed):
    world = json.loads((BASE / 'data/n80_g0.json').read_text())
    path = ROOT / f'outputs/fixed_navigation_v5/pilot/n80/g0_s{seed}/path/latest.pt'
    blob = torch.load(path, map_location='cpu', weights_only=False)
    assert blob['step'] == 50000 and blob['world_hash'] == digest(world)
    tok = Tokenizer(256)
    nodes = [tok.node_id(n) for n in world['node_names']]
    model = TransformerLM(len(tok), **blob['job']['model'], dropout=0.).to('cuda:0').eval().requires_grad_(False)
    model.load_state_dict(blob['model'])
    del blob
    embedding = model.embed.weight.detach().cpu().numpy()[nodes].astype(float)
    reach = np.eye(80, dtype=bool)
    adjacency = np.zeros((80, 80), float)
    for u, successors in enumerate(world['adj']):
        reach[u, list(bfs(world['adj'], u))] = True
        adjacency[u, successors] = 1
    strict = reach.copy().astype(float)
    np.fill_diagonal(strict, 0)
    labels = {'ancestor': strict.T, 'descendant': strict, 'successor': adjacency}
    rows = coarse.collect(model, world, tok, nodes, reach)
    states, baseline = coarse.capture_pair_states(model, rows, tok)
    old_weight = model.lm_head.weight
    model.lm_head.weight = torch.nn.Parameter(old_weight.detach().clone(), requires_grad=False)
    assert model.embed.weight.data_ptr() != model.lm_head.weight.data_ptr()
    after, _ = coarse.ablation_forward(model, rows, tok, ('baseline', 0))
    untie_error = float(abs(after - baseline).max())
    assert untie_error == 0
    bins = defaultdict(list)
    for i, r in enumerate(rows):
        bins[(tuple(r['prefix'][:2] + [-1] + r['prefix'][3:]), r['target_layer'])].append(i)
    inverse = {token: i for i, token in enumerate(nodes)}
    pairs, excluded = [], 0
    for ids in bins.values():
        for j, a in enumerate(ids):
            for b in ids[j+1:]:
                if set(rows[a]['useful']) & set(rows[b]['useful']):
                    continue
                va, vb = inverse.get(int(baseline[a].argmax()), -1), inverse.get(int(baseline[b].argmax()), -1)
                if va not in rows[a]['useful'] or vb not in rows[b]['useful']:
                    excluded += 1
                    continue
                pairs.append((a, b))
    prototypes = prototype_features(model, world, tok, nodes)
    directions = Directions(embedding, prototypes, labels)
    head = head_edits(model, rows, states, baseline, pairs, nodes, directions, labels, world['adj'], reach)
    print(f'seed{seed} head-only interventions complete', flush=True)
    goal = goal_edits(model, rows, nodes, tok, states, baseline, pairs, directions, labels, world['adj'])
    result = {'seed': seed, 'checkpoint_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
              'untied_baseline_max_logit_error': untie_error, 'pairs': len(pairs), 'excluded_incorrect_pairs': excluded,
              'unique_queries': len({rows[i]['qid'] for p in pairs for i in p}),
              'ranks': list(RANKS), 'probe_splits': list(SPLIT_SEEDS), 'random_controls': list(RANDOM_SEEDS),
              'head': head, 'goal': goal, 'fit_cache_entries': len(directions.cache),
              'scope': 'Natural same-prefix/same-depth/disjoint-useful goals; node-heldout probe-defined directions. '
                       'Head row swap sanity is a label permutation. Contextual goal prototypes use main training queries '
                       'whose source is a probe-train node; edited focus nodes never fit/validate directions. '
                       'Original model and all weights remain frozen except temporary output-row experiments.'}
    atomic_json(HERE / f'seed{seed}.json', result)
    print(f'seed{seed} COMPLETE; pairs={len(pairs)}; cache={len(directions.cache)}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2, 3])
    for seed in parser.parse_args().seeds:
        run(seed)
