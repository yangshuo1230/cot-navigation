"""Does L1 FFN need nonlinear computation on probe-heldout nodes?

Two fitting scopes: source-only; shared node positions in queries whose BOTH
endpoints belong to train/val node sets. Affine maps fit teacher FFN deltas,
never graph labels; graph probes fit separate train labels. CPU frozen inference,
closed-form fitting GPU0. No activation/probe/model weight files.
"""
import os, json, importlib.util
from pathlib import Path
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
import numpy as np
import torch
torch.set_num_threads(2)
ROOT = Path.cwd(); OUT = ROOT/'studies/ffn_affine_holdout'
spec = importlib.util.spec_from_file_location('previous_l1', ROOT/'studies/l1_graph_ablation/run.py')
prev = importlib.util.module_from_spec(spec); spec.loader.exec_module(prev)
d = prev.d
NAMES = ['embedding', 'after_attention', 'norm2_input', 'native_delta', 'native_post',
         'ridge_full_delta', 'ridge_full_post', 'ridge_core_delta', 'ridge_core_post',
         'taylor_full_delta', 'taylor_full_post', 'taylor_core_delta', 'taylor_core_post',
         'unitwise_full_delta', 'unitwise_full_post', 'unitwise_core_delta', 'unitwise_core_post',
         'constant_norm_native_delta', 'constant_norm_native_post']


@torch.inference_mode()
def collect(p):
    b = p['model'].blocks[0]; ids = p['ids']; e = p['model'].embed(ids)
    pos = torch.arange(ids.shape[1])[None].expand(len(ids), -1)
    att, _ = b.attn(b.norm1(e), pos); raw = e+att; norm = b.norm2(raw); delta = b.ffn(norm)
    result = dict(embedding=e[:, -3:-1].numpy(), raw=raw[:, -3:-1].numpy(),
                  norm=norm[:, -3:-1].numpy(), delta=delta[:, -3:-1].numpy())
    err = float(np.abs(result['raw']+result['delta']-p['x'][:, :2, 1]).max())
    assert err < 3e-4
    return result, err


def fit_affine(x, y, split):
    w, b, penalty = d.coefficient_fit(x, y, split)
    pred = prev.predict(x[:, None], w, b)[:, 0]
    return pred, w[0], b[0], penalty[0]


def reconstruction(y, pred, train, test):
    residual = y[test]-pred[test]; center = y[train].mean(0)
    den = float(np.sum((y[test]-center)**2)); rms = float(np.sqrt(np.mean(residual**2)))
    norm = np.linalg.norm(y[test], axis=1); prednorm = np.linalg.norm(pred[test], axis=1)
    cosine = np.sum(y[test]*pred[test], axis=1)/np.maximum(norm*prednorm, 1e-12)
    return dict(n=len(test), r2=1-float(np.sum(residual**2))/den if den else None,
                error_rms=rms, target_rms=float(np.sqrt(np.mean(y[test]**2))), cosine=float(cosine.mean()))


@torch.inference_mode()
def taylor(p, raw, norm, train):
    """One fixed Jacobian at TRAIN mean; no test-dependent linearization."""
    block = p['model'].blocks[0]; up = block.ffn.up.weight.double().numpy(); down = block.ffn.down.weight.double().numpy()
    def core_at(center):
        h = torch.tensor(center, dtype=torch.float64); z = torch.tensor(up)@h
        derivative = .5*(1+torch.erf(z/2**.5))+z*torch.exp(-z*z/2)/(2*np.pi)**.5
        jac = down@(derivative.numpy()[:, None]*up)
        out = down@torch.nn.functional.gelu(z, approximate='none').numpy()
        return jac, out-jac@center
    normmean = norm[train].mean(0).astype(np.float64); ac, bc = core_at(normmean)
    rawmean = raw[train].mean(0).astype(np.float64); mu = rawmean.mean(); centered = rawmean-mu
    sigma = np.sqrt(np.mean(centered**2)+block.norm2.eps)
    gamma = block.norm2.weight.double().numpy(); beta = block.norm2.bias.double().numpy()
    h = centered/sigma*gamma+beta; jcore, biascore = core_at(h)
    dim = len(rawmean)
    jnorm = gamma[:, None]/sigma*(np.eye(dim)-np.ones((dim, dim))/dim-np.outer(centered, centered)/(dim*sigma*sigma))
    af = jcore@jnorm; bf = jcore@h+biascore-af@rawmean
    return ((norm@ac.T+bc).astype(np.float32), (raw@af.T+bf).astype(np.float32))


def answer(p, indices, state, role):
    y = p['m']['label'][indices]; margin, valid = d.forward_patch(p, indices, 1, patched=state[indices], position=-3+role)
    correct = (margin > 0) == y
    return dict(n=len(indices), positives=int(y.sum()),
        constrained_ba=float(np.mean([correct[y == k].mean() for k in (0, 1)])) if len(np.unique(y)) == 2 else None,
        accuracy=float((correct&valid).mean()), valid_fraction=float(valid.mean()),
        changed_fraction=float(((margin > 0)!=(p['margin'][indices] > 0)).mean()))


@torch.inference_mode()
def unitwise(p, raw, norm, train):
    """Retain trained up/down weights; fit 512 scalar GELU lines on train.

    A stronger fixed-weight affine baseline than estimating a fresh dense map
    from a low-rank set of training-node keys. No test targets/statistics used.
    """
    block=p['model'].blocks[0]
    up=block.ffn.up.weight.double().numpy(); down=block.ffn.down.weight.double().numpy()
    z=norm.astype(np.float64)@up.T
    gz=torch.nn.functional.gelu(torch.tensor(z[train]),approximate='none').numpy()
    zm=z[train].mean(0); gm=gz.mean(0); centered=z[train]-zm
    slope=np.mean(centered*(gz-gm),axis=0)/np.maximum(np.mean(centered**2,axis=0),1e-12)
    bias=gm-slope*zm
    core=(z*slope+bias)@down.T
    mu=raw[train].mean(-1).mean()
    sigma=np.sqrt(raw[train].var(-1)+block.norm2.eps).mean()
    approximate_norm=(raw-mu)/sigma*block.norm2.weight.numpy()+block.norm2.bias.numpy()
    zfull=approximate_norm.astype(np.float64)@up.T
    full=(zfull*slope+bias)@down.T
    nonlinear=torch.nn.functional.gelu(torch.tensor(zfull),approximate='none').numpy()@down.T
    return full.astype(np.float32),core.astype(np.float32),nonlinear.astype(np.float32)


@torch.inference_mode()
def joint_answer(p,indices,state):
    """Replace BOTH heldout endpoint positions of L1 in the native model."""
    handles=[]; result=[]; valid=[]; model=p['model']; tok=p['tok']
    for start in range(0,len(indices),128):
        ix=indices[start:start+128]
        value=torch.tensor(state[ix])
        def patch(module,args,output):
            hidden=output[0].clone(); hidden[:,-3:-1]=value
            return (hidden,output[1])
        handle=model.blocks[0].register_forward_hook(patch)
        try:
            logits=model(p['ids'][ix])[:,-1]; result.append((logits[:,tok.yes]-logits[:,tok.no]).numpy())
            top=logits.argmax(-1); valid.append(((top==tok.yes)|(top==tok.no)).numpy())
        finally:handle.remove()
    margin=np.concatenate(result); valid=np.concatenate(valid); y=p['m']['label'][indices]; correct=(margin>0)==y
    return dict(n=len(indices),positives=int(y.sum()),constrained_ba=float(np.mean([correct[y==k].mean() for k in (0,1)])) if len(np.unique(y))==2 else None,
        accuracy=float((correct&valid).mean()),valid_fraction=float(valid.mean()),changed_fraction=float(((margin>0)!=(p['margin'][indices]>0)).mean()))


def fit_scope(p, base, fold, scope):
    m = p['m']; roles = np.stack([m['source'], m['target']], 1)
    raw = base['raw'].reshape(-1, 128); norm = base['norm'].reshape(-1, 128); target = base['delta'].reshape(-1, 128)
    if scope == 'source_only':
        split = [2*np.flatnonzero((m['split'] == k)&(fold[m['source']] == k)) for k in range(3)]
    else:
        queries = [np.flatnonzero((m['split'] == k)&np.all(fold[roles] == k, axis=1)) for k in range(3)]
        split = [(q[:, None]*2+np.arange(2)).ravel() for q in queries]
        for k, ids in enumerate(split):
            assert np.all(fold[roles.reshape(-1)[ids]] == k)
    assert min(map(len, split)) > 1
    pred_full, wf, bf, penalty_full = fit_affine(raw, target, split)
    pred_core, wc, bc, penalty_core = fit_affine(norm, target, split)
    taycore, tayfull = taylor(p, raw, norm, split[0])
    unitfull,unitcore,constnorm=unitwise(p,raw,norm,split[0])
    predictions = dict(ridge_full=pred_full, ridge_core=pred_core, taylor_full=tayfull, taylor_core=taycore,
        unitwise_full=unitfull,unitwise_core=unitcore,constant_norm_native=constnorm)
    states = [base['embedding'], base['raw'], base['norm'], base['delta'], base['raw']+base['delta']]
    for key in predictions:
        delta = predictions[key].reshape(base['raw'].shape); states.extend([delta, base['raw']+delta])
    x = np.stack(states, 2)
    metadata = dict(scope=scope, map_sample_counts=list(map(len, split)),
        map_nodes=[np.unique(roles.reshape(-1)[z]).tolist() for z in split],
        map_input_rank=int(np.linalg.matrix_rank(raw[split[0]].astype(np.float64)-raw[split[0]].mean(0))),
        map_input_effective_rank=int(np.sum(np.linalg.svd(raw[split[0]]-raw[split[0]].mean(0), compute_uv=False)>1e-4)),
        ridge_full=penalty_full, ridge_core=penalty_core, teacher_reconstruction={})
    # A ridge_full post state is an affine function of raw residual. Record its
    # rank; a full-rank transform cannot enlarge the affine function class.
    singular = np.linalg.svd(np.eye(128)+wf, compute_uv=False)
    metadata['ridge_full_post_transform'] = dict(rank=int(np.sum(singular>singular.max()*1e-8)),
        min_singular=float(singular.min()), max_singular=float(singular.max()))
    for key, pred in predictions.items():
        metadata['teacher_reconstruction'][key] = {name: reconstruction(target, pred, split[0], ids) for name, ids in zip(('train','val','strict_test'), split)}
    return x, metadata, predictions, (wf, bf), split[0]


def run(setting):
    p = d.load(setting, device='cpu'); base, error = collect(p); m = p['m']; n = len(p['nodes'])
    matrices = {'descendants': p['reach'].astype(np.float32), 'ancestors': p['reach'].T.astype(np.float32), 'successors': p['adjacency']}
    out = dict(manifest=p['manifest'], checks={'L1_native_max_error':error}, features=NAMES, scopes={})
    for scope in ('source_only', 'shared_strict_queries'):
        records = []
        for seed in (17, 31, 47):
            order = np.random.default_rng(seed).permutation(n); fold = np.zeros(n, int); fold[order[60:80]]=1; fold[order[80:]]=2
            x, metadata, predictions, affine, map_train = fit_scope(p, base, fold, scope)
            record = dict(seed=seed, maps=metadata, roles={})
            if scope=='shared_strict_queries':
                jt=np.flatnonzero((m['split']==2)&(fold[m['source']]==2)&(fold[m['target']]==2))
                record['joint_both_heldout']={}
                for condition,feature in [('native','native_post'),('ffn_off','after_attention')]+[(key,key+'_post') for key in predictions]:
                    record['joint_both_heldout'][condition]=joint_answer(p,jt,x[:,:,NAMES.index(feature)])
            for role in (0,) if scope == 'source_only' else (0,1):
                name = 'source' if role == 0 else 'target'; focal = m[name]; other = m['target' if role == 0 else 'source']
                split = [np.flatnonzero((m['split']==k)&(fold[focal]==k)) for k in range(3)]
                assert not set(focal[split[0]])&set(focal[split[2]])
                te = split[2]; role_result = dict(query_counts=list(map(len,split)),focal_counts=[len(np.unique(focal[z])) for z in split],
                    relations={}, functional={}, heldout_delta_reconstruction={})
                flat = 2*te+role
                for key, pred in predictions.items():
                    role_result['heldout_delta_reconstruction'][key] = reconstruction(base['delta'].reshape(-1,128), pred,
                        map_train, flat)
                for condition, feature in [('native','native_post'),('ffn_off','after_attention')]+[(key,key+'_post') for key in predictions]:
                    state = x[:,role,NAMES.index(feature)]
                    role_result['functional'][condition] = answer(p,te,state,role)
                    if condition=='native': assert role_result['functional'][condition]['changed_fraction']==0
                for relation, matrix in matrices.items():
                    labels = matrix[focal]
                    w,b,penalty = d.coefficient_fit(x[:,role],labels,split)
                    pred = prev.predict(x[:,role],w,b)
                    metrics = prev.evaluate(pred,labels,focal,other,te,relation!='ancestors')
                    ni=NAMES.index('native_post')
                    fixed = prev.evaluate(prev.predict(x[:,role],np.broadcast_to(w[ni],w.shape).copy(),np.broadcast_to(b[ni],b.shape).copy()),labels,focal,other,te,relation!='ancestors')
                    # Fold the full-affine post probe back to raw residual.
                    fi=NAMES.index('ridge_full_post'); wf,bf=affine
                    transported=base['raw'][:,role]@((np.eye(128)+wf)@w[fi])+bf@w[fi]+b[fi]
                    terr=float(np.max(abs(transported-pred[:,fi])))
                    assert terr<1e-3
                    role_result['relations'][relation]=dict(fresh=dict(zip(NAMES,metrics)),native_decoder_fixed=dict(zip(NAMES,fixed)),
                        raw_transported_from_affine_post=prev.evaluate(transported[:,None],labels,focal,other,te,relation!='ancestors')[0],
                        transport_error=terr,ridge=penalty)
                # Original tied output head, without fitting any new decoder.
                # Evaluate at the source/goal position as a diagnostic lens,
                # not as actual YES/NO generation at the separator.
                with torch.inference_mode():
                    head=p['model'].lm_head(p['model'].final_norm(torch.tensor(x[te,role])))
                    nodescores=np.zeros((len(x),len(NAMES),n),np.float32)
                    nodescores[te]=head[...,p['nodes']].numpy()
                role_result['original_head']={}
                for relation,matrix in matrices.items():
                    metrics=prev.evaluate(nodescores,matrix[focal],focal,other,te,relation!='ancestors')
                    role_result['original_head'][relation]=dict(zip(NAMES,metrics))
                record['roles'][name]=role_result
            records.append(record)
            print(setting,scope,seed,'complete',flush=True)
        out['scopes'][scope]=records
    prev.save(OUT/f'{setting}.json',out)


if __name__=='__main__':
    OUT.mkdir(parents=True,exist_ok=True)
    for setting in ('simple','hard'):run(setting)
