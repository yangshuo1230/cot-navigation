"""L1 graph readout localization: attention vs FFN vs LayerNorm.

Frozen CPU inference, affine probe fitting on GPU0 only. Original query splits
AND focal-node identities are disjoint. No activations, probe/model weights or
W&B runs are saved. Functional interventions patch one L1 token then recompute.
"""
import os, sys, json
from pathlib import Path
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
import numpy as np
import torch
torch.set_num_threads(2)
ROOT = Path.cwd(); OUT = ROOT/'studies/l1_graph_ablation'
sys.path.insert(0, str(ROOT/'studies/direct_lookup_causal'))
import study as d

SEEDS = (17, 31, 47)
BASE_NAMES = ['embedding', 'norm1_embedding', 'attention_delta', 'after_attention',
              'ffn_input_norm2', 'ffn_delta', 'post_ffn', 'no_attention_ffn_input',
              'no_attention_ffn_delta', 'no_attention_post_ffn']


def save(path, obj):
    temp = path.with_suffix('.tmp'); temp.write_text(json.dumps(obj, indent=2, allow_nan=False)); temp.replace(path)


@torch.inference_mode()
def collect(p):
    model = p['model']; block = model.blocks[0]; ids = p['ids']
    e = model.embed(ids); pos = torch.arange(ids.shape[1])[None].expand(len(ids), -1)
    n1 = block.norm1(e); att, _ = block.attn(n1, pos)
    after = e+att; ff_in = block.norm2(after); ff = block.ffn(ff_in)
    without = block.norm2(e); no_att_ff = block.ffn(without)
    states = [e, n1, att, after, ff_in, ff, after+ff, without, no_att_ff, e+no_att_ff]
    names = BASE_NAMES.copy()
    # Trained FFN replaced by independently randomized up/down matrices, each
    # matched to the trained matrix's Frobenius norm and retaining norm2.
    for seed in (201, 303, 407):
        generator = torch.Generator().manual_seed(seed)
        weights = []
        for w in (block.ffn.up.weight, block.ffn.down.weight):
            rand = torch.randn(w.shape, generator=generator); rand *= w.norm()/rand.norm()
            weights.append(rand)
        delta = torch.nn.functional.linear(torch.nn.functional.gelu(torch.nn.functional.linear(ff_in, weights[0]), approximate='none'), weights[1])
        states.extend([delta, after+delta]); names.extend([f'random_ffn_delta_{seed}', f'random_ffn_post_{seed}'])
    features = torch.stack(states, 2)[:, -3:-1].numpy()  # query source, target
    error = float(np.max(np.abs(features[:, :, names.index('post_ffn')]-p['x'][:, :2, 1])))
    assert error < 3e-4, error
    return features, names, error


def evaluate(pred, labels, focal, other, indices, forward):
    """Macro fixed-candidate AUROC in ONE decoder's heldout panel.

    Group repeated queries by focal identity, exclude self/other endpoint and
    topologically impossible candidates. All features share the exact support.
    """
    n = labels.shape[1]; scores = []; truth = []; masks = []
    for u in np.unique(focal[indices]):
        ix = indices[focal[indices] == u]
        mask = np.ones((len(ix), n), bool); mask[:, u] = False
        mask[np.arange(len(ix)), other[ix]] = False
        mask &= ((np.arange(n) > u) if forward else (np.arange(n) < u))[None]
        count = mask.sum(0)
        avg = (pred[ix].astype(np.float64)*mask[:, None, :]).sum(0)/np.maximum(count, 1)[None]
        scores.append(np.round(avg, 8)); truth.append(labels[ix[0]]); masks.append(count > 0)
    scores = np.array(scores); truth = np.array(truth); masks = np.array(masks)
    result = []
    for feature in range(scores.shape[1]):
        aucs = []
        for candidate in range(n):
            valid = masks[:, candidate]
            a = d.linear.auc(truth[valid, candidate], scores[valid, feature, candidate])
            if a is not None: aucs.append(a)
        recall = []; chance = []
        for yy, ss, mm in zip(truth, scores[:, feature], masks):
            yy = yy[mm]; ss = ss[mm]; k = int(yy.sum())
            if k and k < len(yy):
                recall.append(float(yy[np.argsort(ss)[-k:]].mean())); chance.append(k/len(yy))
        result.append(dict(auc=float(np.mean(aucs)) if aucs else None, candidate_count=len(aucs),
            focal_count=len(scores), recall_at_true_k=float(np.mean(recall)) if recall else None,
            random_recall=float(np.mean(chance)) if chance else None))
    return result


def predict(features, w, b):
    with torch.inference_mode():
        x = torch.tensor(features, device='cuda:0'); ww = torch.tensor(w, device='cuda:0'); bb = torch.tensor(b, device='cuda:0')
        return (torch.einsum('nfd,fdk->nfk', x, ww)+bb).cpu().numpy()


def functional(p, x, names):
    ix = np.flatnonzero(p['m']['split'] == 2); label = p['m']['label'][ix]
    result = {}
    for role, position in [('source', -3), ('target', -2)]:
        ri = 0 if role == 'source' else 1
        conditions = {'native': 'post_ffn', 'ffn_off_at_this_position': 'after_attention',
                      'attention_off_at_this_position': 'no_attention_post_ffn'}
        record = {}
        for name, state in conditions.items():
            patched = x[ix, ri, names.index(state)]
            margin, valid = d.forward_patch(p, ix, 1, patched=patched, position=position)
            correct = (margin > 0) == label
            record[name] = dict(n=len(ix), native_accuracy=float((correct&valid).mean()),
                constrained_ba=float(np.mean([correct[label == k].mean() for k in (0, 1)])),
                valid_answer_fraction=float(valid.mean()))
            if name == 'native': assert np.max(np.abs(margin-p['margin'][ix])) < 1e-3
        result[role] = record
    return result


def run(setting):
    p = d.load(setting, device='cpu'); x, names, error = collect(p)
    m = p['m']; n = len(p['nodes']); width = x.shape[-1]
    out = dict(manifest=p['manifest'], features=names, checks={'native_L1_max_error': error},
        roles={}, functional=functional(p, x, names),
        design='Original query AND focal-node split; each seed has one decoder, no pooled cross-fold scores; graph labels strict, exclude other endpoint and invalid topology.',
        training='Frozen CPU model; closed-form affine probe fits GPU0; main model/checkpoint unchanged.')
    matrices = {'ancestors': p['reach'].T.astype(np.float32), 'successors': p['adjacency'], 'descendants': p['reach'].astype(np.float32)}
    for role, ri in [('source', 0), ('target', 1)]:
        focal = m[role]; other = m['target' if role == 'source' else 'source']; features = x[:, ri]
        result = {}
        for relation, matrix in matrices.items():
            labels = matrix[focal]; records = []
            for seed in SEEDS:
                order = np.random.default_rng(seed).permutation(n); fold = np.zeros(n, int)
                fold[order[60:80]] = 1; fold[order[80:]] = 2
                split = [np.flatnonzero((m['split'] == k)&(fold[focal] == k)) for k in range(3)]
                assert not set(focal[split[0]]) & set(focal[split[2]])
                mean = features[split[0], 0].mean(0); std = features[split[0], 0].std(0)
                random = [np.random.default_rng(rs).normal(size=(n, width)).astype(np.float32)[focal]*std+mean for rs in (201, 303, 407)]
                xx = np.concatenate([features, np.stack(random, 1)], axis=1)
                w, b, penalty = d.coefficient_fit(xx, labels, split)
                pred = predict(xx, w, b)
                metrics = evaluate(pred, labels, focal, other, split[2], relation != 'ancestors')
                fixed_w = np.broadcast_to(w[names.index('post_ffn')], w.shape).copy()
                fixed_b = np.broadcast_to(b[names.index('post_ffn')], b.shape).copy()
                same_decoder = evaluate(predict(xx, fixed_w, fixed_b), labels, focal, other, split[2], relation != 'ancestors')
                records.append(dict(seed=seed, query_counts=list(map(len, split)), focal_counts=[len(np.unique(focal[z])) for z in split],
                    fresh_decoder={key: metric for key, metric in zip(names+['random_identity_201', 'random_identity_303', 'random_identity_407'], metrics)},
                    native_post_ffn_decoder_fixed={key: metric for key, metric in zip(names, same_decoder[:len(names)])},
                    ridge=penalty))
            result[relation] = records
            print(setting, role, relation, 'complete', flush=True)
        out['roles'][role] = result
    save(OUT/f'{setting}.json', out)


if __name__ == '__main__':
    OUT.mkdir(parents=True, exist_ok=True)
    for setting in ('simple', 'hard'): run(setting)
