"""L1 source attention: dynamic gates vs fixed mixing vs affine transforms.

Frozen CPU inference; probe fits GPU0. Fixed means and constant-normalization
statistics use ONLY probe-train source nodes/queries for each node split.
Final-answer panel separately uses all original query-train sources.
"""
import os, sys, json, importlib.util
from pathlib import Path
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
import numpy as np
import torch
torch.set_num_threads(2)
ROOT = Path.cwd(); OUT = ROOT/'studies/fixed_attention'
spec = importlib.util.spec_from_file_location('previous_l1', ROOT/'studies/l1_graph_ablation/run.py')
prev = importlib.util.module_from_spec(spec); spec.loader.exec_module(prev)
d = prev.d
CONDITIONS = ['dynamic', 'train_mean_fixed', 'self_only', 'uniform_visible', 'attention_off',
              'mean_fixed_affine_norm', 'self_only_affine_norm']


@torch.inference_mode()
def source_attention(p):
    model = p['model']; block = model.blocks[0]; ids = p['ids']; dim = model.d_model
    e = model.embed(ids); norm = block.norm1(e); pos = torch.arange(ids.shape[1])[None].expand(len(ids), -1)
    qkv = block.attn.qkv(norm).view(len(ids), ids.shape[1], 3, 1, dim)
    q, k, v = qkv.unbind(2)
    q = d.apply_rope(q.transpose(1, 2), pos, block.attn.rope_base)
    k = d.apply_rope(k.transpose(1, 2), pos, block.attn.rope_base)
    s = ids.shape[1]-3
    weights = (q[:, :, s:s+1]@k[:, :, :s+1].transpose(-2, -1)/dim**.5).softmax(-1)[:, 0, 0]
    values = v[:, :s+1, 0]
    native, _ = block.attn(norm, pos)
    reconstructed = block.attn.proj(torch.einsum('nk,nkd->nd', weights, values))
    error = float((reconstructed-native[:, s]).abs().max())
    assert error < 1e-4, error
    return dict(e=e[:, s], weights=weights, values=values, dim=dim, error=error)


def unique_indices(focal, indices):
    return np.array([indices[np.flatnonzero(focal[indices] == u)[0]] for u in np.unique(focal[indices])])


@torch.inference_mode()
def states(p, raw, train):
    block = p['model'].blocks[0]; n = len(p['ids']); dim = raw['dim']
    focal = p['m']['source']; ix = unique_indices(focal, train)
    mean_weights = raw['weights'][ix].mean(0)
    mean_weights /= mean_weights.sum()
    e = raw['e']; std = torch.sqrt(e.var(-1, unbiased=False)+block.norm1.eps)
    mu = e[ix].mean(-1).mean(); sigma = std[ix].mean()
    affine_norm = (e-mu)/sigma*block.norm1.weight+block.norm1.bias
    vv_affine = torch.nn.functional.linear(affine_norm, block.attn.qkv.weight[2*dim:])
    all_states = {}; affine_checks = {}
    for condition in CONDITIONS:
        if condition == 'dynamic': weight = raw['weights']
        elif condition.startswith('self_only'):
            weight = torch.zeros_like(raw['weights']); weight[:, -1] = 1
        elif condition == 'uniform_visible': weight = torch.ones_like(raw['weights'])/raw['weights'].shape[1]
        elif condition == 'attention_off': weight = torch.zeros_like(raw['weights'])
        else: weight = mean_weights[None].expand(n, -1)
        value = raw['values']
        if condition.endswith('affine_norm'):
            value = value.clone(); value[:, -1] = vv_affine
        delta = block.attn.proj(torch.einsum('nk,nkd->nd', weight, value))
        after = e+delta; post = after+block.ffn(block.norm2(after))
        all_states[condition] = {'delta': delta.numpy(), 'after_attention': after.numpy(), 'post_ffn': post.numpy()}
        if condition.endswith('affine_norm'):
            alpha = float(weight[0, -1]); scale = block.norm1.weight/sigma
            a = torch.eye(dim)+alpha*block.attn.proj.weight@block.attn.qkv.weight[2*dim:]@torch.diag(scale)
            # The protocol-token values are constant for all query sources.
            bias = delta[0]-(a-torch.eye(dim))@e[0]
            check = float((after-(e@a.T+bias)).abs().max())
            assert check < 1e-4, check
            singular = torch.linalg.svdvals(a.double())
            affine_checks[condition] = dict(max_reconstruction_error=check,
                rank=int((singular > singular.max()*1e-10).sum()), min_singular=float(singular.min()), max_singular=float(singular.max()))
    return all_states, dict(train_sources=len(ix), mean_weights=mean_weights.tolist(),
                           constant_norm_mu=float(mu), constant_norm_sigma=float(sigma), affine_checks=affine_checks)


def features(raw, condition_states):
    names = ['embedding', 'norm1_embedding']
    # norm1 vector computed separately to retain original learned affine terms.
    values = [raw['e'].numpy(), raw['norm1']]
    for name in CONDITIONS:
        for stage in ('after_attention', 'post_ffn'):
            names.append(name+'/'+stage); values.append(condition_states[name][stage])
    return np.stack(values, 1), names


def functional(p, raw):
    train = np.flatnonzero(p['m']['split'] == 0); ix = np.flatnonzero(p['m']['split'] == 2)
    ss, metadata = states(p, raw, train); y = p['m']['label'][ix]; out = {}
    for name in CONDITIONS:
        margin, valid = d.forward_patch(p, ix, 1, patched=ss[name]['post_ffn'][ix], position=-3)
        correct = (margin > 0) == y
        out[name] = dict(n=len(ix), constrained_ba=float(np.mean([correct[y == k].mean() for k in (0, 1)])),
            native_accuracy=float((correct&valid).mean()), valid_fraction=float(valid.mean()),
            prediction_disagreement=float((((margin > 0)!=(p['margin'][ix] > 0))|(valid!=(p['m']['native_pred'][ix] >= 0))).mean()),
            max_margin_shift=float(np.max(np.abs(margin-p['margin'][ix]))))
        if name == 'dynamic': assert np.max(np.abs(margin-p['margin'][ix])) < 1e-3
    return dict(train_metadata=metadata, results=out)


def run(setting):
    p = d.load(setting, device='cpu'); raw = source_attention(p)
    raw['norm1'] = p['model'].blocks[0].norm1(raw['e']).detach().numpy()
    focal = p['m']['source']; other = p['m']['target']; n = len(p['nodes'])
    matrices = {'descendants': p['reach'].astype(np.float32), 'successors': p['adjacency'], 'ancestors': p['reach'].T.astype(np.float32)}
    out = dict(manifest=p['manifest'], checks={'attention_reconstruction_max_error': raw['error']},
               functional=functional(p, raw), relations={}, splits=[])
    for relation, matrix in matrices.items():
        labels = matrix[focal]; records = []
        for seed in (17, 31, 47):
            order = np.random.default_rng(seed).permutation(n); fold = np.zeros(n, int)
            fold[order[60:80]] = 1; fold[order[80:]] = 2
            split = [np.flatnonzero((p['m']['split'] == k)&(fold[focal] == k)) for k in range(3)]
            ss, metadata = states(p, raw, split[0]); x, names = features(raw, ss)
            w, b, penalties = d.coefficient_fit(x, labels, split)
            pred = prev.predict(x, w, b)
            metrics = prev.evaluate(pred, labels, focal, other, split[2], relation != 'ancestors')
            # Transport the affine-condition decoder back to raw embedding.
            # This proves an actual raw-E linear readout exists with the same
            # predictions; no test labels or new fitted parameters are used.
            ai = names.index('mean_fixed_affine_norm/after_attention')
            block = p['model'].blocks[0]; dim = raw['dim']
            scale = block.norm1.weight.detach().numpy()/metadata['constant_norm_sigma']
            alpha = metadata['mean_weights'][-1]
            a = np.eye(dim)+alpha*block.attn.proj.weight.detach().numpy()@block.attn.qkv.weight.detach().numpy()[2*dim:]@np.diag(scale)
            e = raw['e'].numpy()
            bias = ss['mean_fixed_affine_norm']['after_attention'][0]-e[0]@a.T
            transported = e@(a.T@w[ai])+bias@w[ai]+b[ai]
            transport_error = float(np.max(np.abs(transported-pred[:, ai])))
            assert transport_error < 1e-3, transport_error
            transport_metric = prev.evaluate(transported[:, None], labels, focal, other, split[2], relation != 'ancestors')[0]
            wi = names.index('dynamic/post_ffn')
            fixed = prev.evaluate(prev.predict(x, np.broadcast_to(w[wi], w.shape).copy(), np.broadcast_to(b[wi], b.shape).copy()), labels, focal, other, split[2], relation != 'ancestors')
            records.append(dict(seed=seed, query_counts=list(map(len, split)), focal_counts=[len(np.unique(focal[z])) for z in split],
                train_metadata=metadata, fresh_decoder=dict(zip(names, metrics)),
                fixed_native_post_decoder=dict(zip(names, fixed)), ridge=penalties,
                transported_raw_embedding_probe=transport_metric, transport_max_score_error=transport_error))
        out['relations'][relation] = records
        print(setting, relation, 'complete', flush=True)
    # Distribution is node-balanced rather than weighted by query frequency.
    idx = unique_indices(focal, np.arange(len(focal))); w = raw['weights'][idx].numpy()
    out['native_weights'] = dict(n_sources=len(idx), mean=w.mean(0).tolist(), std=w.std(0).tolist(),
                                minimum=w.min(0).tolist(), maximum=w.max(0).tolist(),
                                self_quantiles=np.quantile(w[:, -1], [0, .05, .25, .5, .75, .95, 1]).tolist())
    prev.save(OUT/f'{setting}.json', out)


if __name__ == '__main__':
    OUT.mkdir(exist_ok=True, parents=True)
    for setting in ('simple', 'hard'): run(setting)
