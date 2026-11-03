"""Answer-readout versus native PATH expansion; frozen last50k checkpoints.

Probe fits use GPU0, with no activation/probe/model checkpoints saved. All
prefixes exclude YES/NO. Relative anchors use final trace length retrospectively;
fixed anchors use an explicit unchanged long-trajectory cohort.
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
HERE = ROOT / 'studies/path_answer_progress_v5'
SUITE = ROOT / 'studies/fixed_navigation_v5'
sys.path[:0] = [str(ROOT / 'src'), str(SUITE)]
from data import atomic_json, digest
from cot_navigation.models.transformer import TransformerLM
from cot_navigation.protocols.tokenizer import Tokenizer
from cot_navigation.evaluation.rollout import generate

TEST = ['short_test', 'long_AA_test', 'long_TA_test', 'long_AT_test', 'long_TT_test']
VAL = ['short_val', 'long_AA_val', 'long_VA_val', 'long_AV_val', 'long_VV_val']
RELATIVE = ['prompt', 'copy', 'p25', 'p50', 'p75', 'end']
FIXED = ['prompt', 'copy', 'node2', 'node3', 'node4', 'node5']


@torch.inference_mode()
def prepare_common_fixed_cohort():
    """Same unique queries for every anchor AND every model initialization.

    This is explicitly a retrospective long-trajectory sensitivity cohort,
    selected by output length, never by whether answer/path was correct.
    """
    path = HERE / 'fixed_cohort.json'
    if path.exists():
        return
    world = json.loads((SUITE / 'data/n80_g0.json').read_text())
    ids = sorted({i for name in ['short_train', 'long_train'] + VAL + TEST for i in world['panels'][name]
                  if world['records'][i]['gap'] >= 4})
    tok = Tokenizer(256)
    nodes = [tok.node_id(i) for i in world['node_names']]
    common, counts = set(ids), {}
    for seed in range(4):
        folder = ROOT / f'outputs/fixed_navigation_v5/pilot/n80/g0_s{seed}/path'
        blob = torch.load(folder / 'latest.pt', map_location='cpu', weights_only=False)
        assert blob['step'] == 50000 and blob['world_hash'] == digest(world)
        model = TransformerLM(len(tok), **blob['job']['model'], dropout=0.).to('cuda:0').eval().requires_grad_(False)
        model.load_state_dict(blob['model'])
        eligible = set()
        for start in range(0, len(ids), 128):
            part = ids[start:start + 128]
            prompts = [[tok.id('<QUERY>'), nodes[world['records'][i]['s']],
                        nodes[world['records'][i]['t']], tok.id('|')] for i in part]
            outputs = generate(model, prompts, 'path', tok, 'bf16', max_new_tokens=16)
            eligible.update(i for i, out in zip(part, outputs) if len(out['trace_tokens']) >= 5)
        common &= eligible
        counts[str(seed)] = len(eligible)
        del model, blob
        torch.cuda.empty_cache()
    atomic_json(path, {'world_hash': digest(world), 'model_seeds': [0, 1, 2, 3],
                       'selection': 'gap>=4 and at least5 trace tokens in every model; not correctness',
                       'eligible_counts': counts, 'query_ids': sorted(common)})


def auc(y, score):
    p, n = score[y == 1], score[y == 0]
    if not len(p) or not len(n):
        return None
    delta = p[:, None] - n[None, :]
    return float(((delta > 0) + .5 * (delta == 0)).mean())


def classification(y, score, threshold):
    prediction = score >= threshold
    pos, neg = y == 1, y == 0
    yes = float(prediction[pos].mean()) if pos.any() else None
    no = float((~prediction[neg]).mean()) if neg.any() else None
    return {'n': len(y), 'n_positive': int(pos.sum()), 'n_negative': int(neg.sum()),
            'yes_recall': yes, 'no_recall': no,
            'ba': (yes + no) / 2 if yes is not None and no is not None else None,
            'accuracy': float((prediction == y).mean()) if len(y) else None,
            'auroc': auc(y, score)}


def threshold_by_validation(y, score):
    unique = np.unique(score)
    thresholds = np.r_[unique, np.nextafter(unique[-1], np.inf)]
    order = np.argsort(score)
    labels, sorted_scores = y[order], score[order]
    cut = np.searchsorted(sorted_scores, thresholds, side='left')
    positives = np.r_[0, np.cumsum(labels)]
    negatives = np.r_[0, np.cumsum(1 - labels)]
    ba = .5 * ((positives[-1] - positives[cut]) / positives[-1] + negatives[cut] / negatives[-1])
    choices = np.flatnonzero(ba == ba.max())
    best = choices[np.argmin(abs(thresholds[choices] - .5))]
    return float(thresholds[best])


def fit(features, y, splits):
    """Each anchor gets a separate balanced affine ridge and validation threshold."""
    tr, va, te = splits
    if any(len(np.unique(y[ix])) < 2 for ix in splits):
        return None
    if features.ndim == 2:
        features = features[:, None, :]
    z = torch.tensor(features, device='cuda:0').transpose(0, 1)
    mu = z[:, tr].mean(1, keepdim=True)
    sd = z[:, tr].std(1, keepdim=True).clamp_min(.05)
    z = (z - mu) / sd
    z = torch.cat([z, torch.ones((*z.shape[:2], 1), device=z.device)], dim=-1)
    labels = torch.tensor(y, dtype=torch.float32, device=z.device)
    weights = torch.tensor([.5 / sum(y[tr] == label) for label in y[tr]], device=z.device, dtype=torch.float32)
    a = z[:, tr]
    gram = a.transpose(1, 2) @ (a * weights[None, :, None])
    rhs = a.transpose(1, 2) @ (labels[tr] * weights)[None, :, None]
    eye = torch.eye(z.shape[-1], device=z.device)
    eye[-1, -1] = 0
    best, penalty, threshold = np.full(z.shape[0], -np.inf), np.zeros(z.shape[0]), np.zeros(z.shape[0])
    scores = np.zeros((len(te), z.shape[0]), np.float32)
    for reg in (.001, .01, .1, 1., 10.):
        w = torch.linalg.solve(gram + reg * eye + 1e-6 * torch.eye(z.shape[-1], device=z.device), rhs)
        vp = (z[:, va] @ w)[..., 0].cpu().numpy()
        tp = (z[:, te] @ w)[..., 0].cpu().numpy()
        for phase in range(z.shape[0]):
            value = auc(y[va], vp[phase])
            if value >= best[phase]:
                best[phase], penalty[phase] = value, reg
                threshold[phase] = threshold_by_validation(y[va], vp[phase])
                scores[:, phase] = tp[phase]
    return scores, {'ridge': penalty.tolist(), 'threshold': threshold.tolist(), 'val_auc': best.tolist()}


@torch.inference_mode()
def collect(seed):
    world = json.loads((SUITE / 'data/n80_g0.json').read_text())
    folder = ROOT / f'outputs/fixed_navigation_v5/pilot/n80/g0_s{seed}/path'
    blob = torch.load(folder / 'latest.pt', map_location='cpu', weights_only=False)
    assert blob['step'] == 50000 and digest(world) == blob['world_hash']
    tok = Tokenizer(256)
    nodes = [tok.node_id(i) for i in world['node_names']]
    model = TransformerLM(len(tok), **blob['job']['model'], dropout=0.).to('cuda:0').eval().requires_grad_(False)
    model.load_state_dict(blob['model'])
    embeddings = model.embed.weight.detach().float().cpu().numpy()[nodes]
    del blob
    rows, checks = [], {}
    final = json.loads((folder / 'final.json').read_text())
    cohort = json.loads((HERE / 'fixed_cohort.json').read_text())
    assert cohort['world_hash'] == digest(world)
    fixed_ids = set(cohort['query_ids'])
    for panel in ['short_train', 'long_train'] + VAL + TEST:
        ids = world['panels'][panel]
        c = Counter()
        for start in range(0, len(ids), 128):
            part = ids[start:start + 128]
            prompts = [[tok.id('<QUERY>'), nodes[world['records'][i]['s']],
                        nodes[world['records'][i]['t']], tok.id('|')] for i in part]
            output = generate(model, prompts, 'path', tok, 'bf16', max_new_tokens=16)
            for qid, prompt, out in zip(part, prompts, output):
                r = world['records'][qid]
                trace = out['trace_tokens']
                assert trace and tok.yes not in trace and tok.no not in trace
                c[f"n{r['y']}"] += 1
                c[f"correct{r['y']}"] += out['answer_token'] == tok.label_id(r['y'])
                rows.append({'qid': qid, 'panel': panel, 'source': r['s'], 'target': r['t'],
                             'label': r['y'], 'gap': r['gap'],
                             'split': 0 if panel.endswith('_train') else 1 if panel.endswith('_val') else 2,
                             'prompt': prompt, 'trace': trace, 'answer': out['answer_token']})
        if panel in TEST:
            expected = final['panels'][panel]['metrics']['all']
            checks[panel] = all(c[k] == expected.get(k, 0) for k in ('n0', 'n1', 'correct0', 'correct1'))
    features, controls, raw_scores, native_ready = [], [], [], []
    fixed_features, fixed_controls, fixed_scores, fixed_ready, fixed_rows = [], [], [], [], []
    causality_error = 0.
    copy_correct = 0
    def extract(part, anchors):
        """Forward only visible prefixes: no future tokens, even behind a mask."""
        all_features, all_scores, all_ready = [], [], []
        controls = []
        for r, ks in zip(part, anchors):
            c = []
            for k in ks:
                visible = r['trace'][:k]
                c.append([k, k*k, bool(visible and visible[-1] == nodes[r['target']]),
                          nodes[r['target']] in visible])
            controls.append(c)
        for phase in range(len(anchors[0])):
            seqs = [r['prompt'] + r['trace'][:ks[phase]] for r, ks in zip(part, anchors)]
            batch = torch.full((len(part), max(map(len, seqs))), tok.pad, dtype=torch.long, device='cuda:0')
            last = torch.tensor([len(seq) - 1 for seq in seqs], device=batch.device)
            for b, seq in enumerate(seqs):
                batch[b, :len(seq)] = torch.tensor(seq, device=batch.device)
            cap = {}
            with torch.autocast('cuda', dtype=torch.bfloat16):
                logits = model(batch, capture=cap)
            ix = torch.arange(len(part), device=batch.device)
            all_features.append(cap['resid'][-1][ix, last].float().cpu().numpy())
            score = logits[ix, last]
            all_scores.append((score[:, tok.yes] - score[:, tok.no]).float().cpu().numpy())
            all_ready.append(torch.isin(score.argmax(-1), torch.tensor([tok.no, tok.yes], device=batch.device)).cpu().numpy())
        return (np.stack(all_features, axis=1), np.asarray(controls, np.float32),
                np.stack(all_scores, axis=1), np.stack(all_ready, axis=1))
    for start in range(0, len(rows), 128):
        part = rows[start:start + 128]
        anchors = [[0, 1] + [max(1, int(np.ceil(len(r['trace']) * p))) for p in (.25, .5, .75)]
                   + [len(r['trace'])] for r in part]
        f, c, s, ready_at = extract(part, anchors)
        features.extend(f)
        controls.extend(c)
        raw_scores.extend(s)
        native_ready.extend(ready_at)
        copy_correct += sum(r['trace'][0] == nodes[r['source']] for r in part)
        eligible = [b for b, r in enumerate(part) if r['qid'] in fixed_ids]
        assert all(len(part[b]['trace']) >= 5 for b in eligible)
        if eligible:
            f, c, s, ready_at = extract([part[b] for b in eligible], [[0, 1, 2, 3, 4, 5]] * len(eligible))
            fixed_features.extend(f)
            fixed_controls.extend(c)
            fixed_scores.extend(s)
            fixed_ready.extend(ready_at)
            fixed_rows.extend(start + b for b in eligible)
        if start == 0:
            sequences = [r['prompt'] + r['trace'] for r in part]
            batch = torch.full((len(part), max(map(len, sequences))), tok.pad, dtype=torch.long, device='cuda:0')
            for b, seq in enumerate(sequences):
                batch[b, :len(seq)] = torch.tensor(seq, device=batch.device)
            # FP32 sanity only. BF16 GEMM shape changes can cause ~0.1 logits
            # even with causal masking, hence all fitted features above use
            # prefix-only forwards rather than the full future trajectory.
            with torch.autocast('cuda', enabled=False):
                full = model(batch)
                prompt = model(batch[:, :4])
            causality_error = float((full[:, 3] - prompt[:, -1]).abs().max())
            assert causality_error < 1e-3
    meta = {k: np.array([r[k] for r in rows]) for k in ('qid', 'panel', 'source', 'target', 'label', 'gap', 'split', 'answer')}
    data = {'relative': {'features': np.asarray(features), 'control': np.asarray(controls),
                          'raw': np.asarray(raw_scores), 'ready': np.asarray(native_ready),
                          'ids': np.arange(len(rows)), 'anchors': RELATIVE},
            'fixed_long_cohort': {'features': np.asarray(fixed_features), 'control': np.asarray(fixed_controls),
                                 'raw': np.asarray(fixed_scores), 'ready': np.asarray(fixed_ready),
                                 'ids': np.asarray(fixed_rows), 'anchors': FIXED}}
    return data, meta, embeddings, {'native_replay_matches_final': checks, 'source_copy_fraction': copy_correct / len(rows),
                                   'features_from_prefix_only_forwards': True,
                                   'FP32_prompt_causality_max_error': causality_error,
                                   'checkpoint_sha256': hashlib.sha256((folder / 'latest.pt').read_bytes()).hexdigest(),
                                   'world_hash': digest(world)}


def run(seed):
    datasets, meta_all, embeddings, checks = collect(seed)
    out = {'seed': seed, 'checkpoint_step': 50000, 'checks': checks, 'experiments': {}}
    for dataset_name, dataset in datasets.items():
        meta = {k: v[dataset['ids']] for k, v in meta_all.items()}
        y, split = meta['label'], meta['split']
        schemes = {'query_only': [np.flatnonzero(split == i) for i in range(3)]}
        for node_seed in (17, 23, 31):
            rng = np.random.default_rng(node_seed)
            node_split = np.zeros(80, int)
            for layer in range(10):
                order = rng.permutation(np.arange(layer * 8, (layer + 1) * 8))
                node_split[order[4:6]], node_split[order[6:]] = 1, 2
            for mode in ('target', 'both'):
                schemes[f'{mode}_node_heldout_{node_seed}'] = [
                    np.flatnonzero((split == i) & (node_split[meta['target']] == i) &
                                   ((node_split[meta['source']] == i) if mode == 'both' else True)) for i in range(3)]
        if dataset_name == 'fixed_long_cohort':
            schemes = {'query_only': schemes['query_only']}
        features = {'state_L6': dataset['features'],
                    'position_only': dataset['control'][:, :, :2],
                    'position_goal': dataset['control'],
                    'endpoint_embeddings': np.concatenate([embeddings[meta['source']], embeddings[meta['target']]], axis=-1)}
        for random_seed in (101, 202, 303):
            vectors = np.random.default_rng(random_seed).normal(size=(80, 128)).astype(np.float32)
            features[f'random_endpoints_{random_seed}'] = np.concatenate([vectors[meta['source']], vectors[meta['target']]], axis=-1)
        output = {'anchors': dataset['anchors'], 'cohort_n': len(y), 'schemes': {}}
        for name, partitions in schemes.items():
            test = partitions[2]
            entry = {'split_n': list(map(len, partitions)),
                     'split_positive_n': [int(y[p].sum()) for p in partitions], 'features': {}}
            for feature_name, feature in features.items():
                if dataset_name == 'fixed_long_cohort' and name != 'query_only':
                    continue  # small fixed-length/node-crossed cohorts are not the main claim
                fitted = fit(feature, y, partitions)
                if fitted is None:
                    entry['features'][feature_name] = {'skipped': 'a partition lacks both labels'}
                    continue
                prediction, rule = fitted
                results = {}
                for panel in ['all'] + TEST:
                    mask = np.ones(len(test), bool) if panel == 'all' else meta['panel'][test] == panel
                    if not mask.any():
                        continue
                    results[panel] = [classification(y[test][mask], prediction[mask, i], rule['threshold'][i])
                                      for i in range(prediction.shape[1])]
                entry['features'][feature_name] = {'fit': rule, 'metrics': results}
            output['schemes'][name] = entry
        partitions = schemes['query_only']
        test = partitions[2]
        native = {}
        for panel in ['all'] + TEST:
            selected = np.ones(len(test), bool) if panel == 'all' else meta['panel'][test] == panel
            if not selected.any():
                continue
            ids = test[selected]
            native[panel] = [{'forced_yesno': classification(y[ids], dataset['raw'][ids, p], 0.),
                              'native_next_token_is_yesno': float(dataset['ready'][ids, p].mean())}
                             for p in range(len(dataset['anchors']))]
        output['native_head_diagnostic'] = native
        native_final = {}
        for panel in ['all'] + TEST:
            selected = np.ones(len(test), bool) if panel == 'all' else meta['panel'][test] == panel
            ids = test[selected]
            if not len(ids):
                continue
            yy = y[ids]
            pred = meta['answer'][ids]
            tok = Tokenizer(256)
            yes = float((pred[yy == 1] == tok.yes).mean()) if np.any(yy == 1) else None
            no = float((pred[yy == 0] == tok.no).mean()) if np.any(yy == 0) else None
            native_final[panel] = {'n': len(ids), 'yes_recall': yes, 'no_recall': no,
                                    'ba': (yes + no) / 2 if yes is not None and no is not None else None}
        output['native_final_answer'] = native_final
        if dataset_name == 'relative':
            shuffled = y.copy()
            rng = np.random.default_rng(181)
            for p in partitions:
                shuffled[p] = rng.permutation(shuffled[p])
            fitted = fit(dataset['features'], shuffled, partitions)
            scores, rule = fitted
            output['shuffled_labels'] = [classification(shuffled[test], scores[:, p], rule['threshold'][p])
                                        for p in range(scores.shape[1])]
        out['experiments'][dataset_name] = output
        print(f'seed{seed} {dataset_name} complete', flush=True)
    HERE.mkdir(parents=True, exist_ok=True)
    atomic_json(HERE / f'seed{seed}.json', out)
    print(f'seed{seed} COMPLETE', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2, 3])
    args = parser.parse_args()
    prepare_common_fixed_cohort()
    for seed in args.seeds:
        run(seed)
