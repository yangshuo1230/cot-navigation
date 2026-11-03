"""Read-only audit of Cursor's FFN-role claims; originals are not modified."""
import os
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['OPENBLAS_NUM_THREADS'] = '2'
os.environ['MKL_NUM_THREADS'] = '2'
import sys, json
from pathlib import Path
import numpy as np
import torch
torch.set_num_threads(2)
ROOT = Path.cwd(); HERE = ROOT/'studies/cursor_ffn_audit'
sys.path.insert(0, str(ROOT/'studies/ffn_role'))
import study as c


def mean(values): return float(np.mean(values)) if values else None


def within_fold_auc(y, scores, folds, mask):
    columns = []
    for j in range(y.shape[1]):
        weighted = []; weights = []
        for tr, va, te in folds:
            ix = te[mask[te, j]]; a = c.linear.auc(y[ix, j], scores[ix, j])
            if a is not None:
                npos = int(y[ix, j].sum()); nneg = len(ix)-npos
                weighted.append(a); weights.append(npos*nneg)
        if weights: columns.append(float(np.average(weighted, weights=weights)))
    return dict(auc=mean(columns), supported_columns=len(columns))


@torch.inference_mode()
def run(setting):
    p = c.load(setting); model = p['model']; nodes = p['nodes']; n = len(nodes)
    posts, norms, deltas = c.isolated(model, nodes, attn=False)
    e = posts[:, 0]; y = p['ancestors']; mask = c.self_mask(n)
    valid = np.arange(n)[None, :] < np.arange(n)[:, None]
    result = dict(embedding_rank=int(np.linalg.matrix_rank(e.astype(np.float64))),
                  affine_embedding_rank=int(np.linalg.matrix_rank(np.c_[e, np.ones(n)])))
    result['auc'] = []
    for seed in c.SEEDS:
        folds = c.make_folds(n, seed)
        random = np.random.default_rng(1).normal(size=e.shape).astype(np.float32)
        random *= e.std()/random.std()
        for name, x, yy in [('embedding', e, y), ('random_vector', random, y),
                            ('shuffled', e, y[np.random.default_rng(0).permutation(n)])]:
            scores = np.zeros_like(yy); freq = np.zeros_like(yy)
            for train, val, test in folds:
                scores[test] = c.ridge_predict(x, yy, train, val)[test]
                freq[test] = yy[train].mean(0)
            result['auc'].append(dict(seed=seed, feature=name,
                cursor_pooled=c.rank_one(yy, scores, mask)['column_auc'],
                within_fold=within_fold_auc(yy, scores, folds, mask),
                direction_eligible_within_fold=within_fold_auc(yy, scores, folds, valid),
                frequency_only_pooled=c.rank_one(yy, freq, mask)['column_auc'],
                frequency_only_within_fold=within_fold_auc(yy, freq, folds, mask)))
    rank = np.broadcast_to(np.arange(n)[:, None], (n, n))
    result['topological_rank_only'] = dict(all=c.rank_one(y, rank, mask)['column_auc'],
        direction_eligible=c.rank_one(y, rank, valid)['column_auc'])
    train, val, test = c.make_folds(n, 17)[0]
    pred, leftover, r2, penalty = c.residualize(norms[:, 0], deltas[:, 0], train, val, test)
    result['residualization'] = dict(r2=r2, penalty=penalty,
        train_residual_rms=float(np.sqrt(np.mean(leftover[train]**2))),
        test_residual_rms=float(np.sqrt(np.mean(leftover[test]**2))),
        train_input_residual_crosscov_norm=float(np.linalg.norm(norms[train, 0].T@leftover[train]/len(train))))
    # An arbitrary node table is exactly linearly representable on these 100
    # addresses when the address matrix has full row rank. This is not heldout
    # generalization; it audits the claim that linear maps cannot be lookups.
    arbitrary = np.random.default_rng(7).normal(size=(n, 128))
    design = np.c_[e.astype(np.float64), np.ones(n)]
    weights = np.linalg.lstsq(design, arbitrary, rcond=None)[0]
    result['arbitrary_table_affine_reconstruction_rms'] = float(np.sqrt(np.mean((design@weights-arbitrary)**2)))
    # Test whether the query-linear reconstruction criterion is specific to
    # trained FFNs. Keep every original native FFN input unchanged, replace the
    # target function by an independently random standard FFN of the same size.
    toks = p['tokens']; trainq = np.flatnonzero(p['splits'] == 0); valq = np.flatnonzero(p['splits'] == 1)
    length = toks.shape[1]; tri = (trainq[:, None]*length+np.arange(length)).ravel(); vai = (valq[:, None]*length+np.arange(length)).ravel()
    x = model.embed(toks); positions = c.positions(length).expand(len(toks), -1)
    result['random_ffn_affine_val_r2'] = []
    for layer, block in enumerate(model.blocks):
        att, _ = block.attn(block.norm1(x), positions); x = x+att; h = block.norm2(x)
        torch.manual_seed(903+layer)
        w1 = torch.randn(model.d_ff, model.d_model)*.02; w2 = torch.randn(model.d_model, model.d_ff)*.02
        random_delta = torch.nn.functional.linear(torch.nn.functional.gelu(torch.nn.functional.linear(h, w1), approximate='none'), w2)
        xx = h.numpy().reshape(-1, model.d_model); yy = random_delta.numpy().reshape(-1, model.d_model)
        pred, _, _ = c.ridge_one(xx, yy, tri, vai)
        r2 = 1-np.sum((yy[vai]-pred[vai])**2)/np.sum((yy[vai]-yy[tri].mean(0))**2)
        result['random_ffn_affine_val_r2'].append(float(r2))
        x = x+block.ffn(h)
    # Verify reported ablations using hooks on the original model forward,
    # rather than Cursor's manually rewritten forward implementation.
    testq = np.flatnonzero(p['splits'] == 2); labels = p['labels'][testq]
    result['native_hook_checks'] = {}
    for name, pos in [('intact', None), ('source', p['header']+1),
                      ('target', p['header']+2), ('separator', p['header']+3)]:
        handles = []
        if pos is not None:
            def zero(module, args, out):
                changed = out.clone(); changed[:, pos] = 0; return changed
            handles = [block.ffn.register_forward_hook(zero) for block in model.blocks]
        try:
            logits = model(toks[testq])[:, -1].numpy()
            result['native_hook_checks'][name] = c.answer_metrics(logits, labels, p['tok'].yes, p['tok'].no)
        finally:
            for handle in handles: handle.remove()
    maps, r2 = c.fit_query_linear(model, toks, trainq, valq)
    handles = []
    for layer, block in enumerate(model.blocks):
        def replace(module, args, out, index=layer): return c.apply_affine(args[0], maps[index])
        handles.append(block.ffn.register_forward_hook(replace))
    try:
        logits = model(toks[testq])[:, -1].numpy()
        result['native_hook_checks']['affine_ffn'] = c.answer_metrics(logits, labels, p['tok'].yes, p['tok'].no)
    finally:
        for handle in handles: handle.remove()
    return result


HERE.mkdir(exist_ok=True)
out = {g: run(g) for g in ('simple', 'hard')}
(HERE/'audit.json').write_text(json.dumps(out, indent=2, allow_nan=False))
print(json.dumps(out, indent=2))
