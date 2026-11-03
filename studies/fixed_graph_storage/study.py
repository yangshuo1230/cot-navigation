"""Fixed DAG regional/functional localization, without changing training code.

Frozen models only. Discovery uses original query-train; all intervention
scores use original query-test. Samples are native, legal PATH prefixes, not
teacher-forced reference paths. No activation files or fitted probes are saved.
"""
from pathlib import Path
import argparse, gzip, hashlib, json, os, sys, time
os.environ.setdefault('OMP_NUM_THREADS', '2')
import numpy as np
import torch
torch.set_num_threads(2)
ROOT = Path.cwd()
OLD = ROOT/'studies/fixed100_layer_probes'
HERE = ROOT/'studies/fixed_graph_storage'
sys.path[:0] = [str(ROOT/'src'), str(OLD)]
from cot_navigation.config import load_config
from cot_navigation.data.dataset import ParametricWorld
from cot_navigation.models.transformer import build_model
from cot_navigation.protocols.tokenizer import Tokenizer
import linear


def write(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False))
    tmp.replace(path)


def load(setting, protocol, device):
    manifest = json.loads((OLD/f'{setting}_{protocol}.json').read_text())['manifest']
    run = Path(manifest['run']); ck = run/'checkpoints/best.pt'
    assert hashlib.sha256(ck.read_bytes()).hexdigest() == manifest['checkpoint_sha256']
    cfg = load_config([run/'resolved_config.yaml']); tok = Tokenizer.from_config(cfg)
    world = ParametricWorld(Path(cfg.data.root)/'main-seed0')
    adj = world.adj(); nodes = np.array(world.node_vocab_ids(tok)); n = len(nodes)
    a = np.zeros((n, n), bool); r = np.eye(n, dtype=bool); depth = np.zeros(n, int)
    for u in range(n):
        for v in adj[u]:
            assert v > u
            a[u, v] = True; depth[v] = max(depth[v], depth[u]+1)
    for u in reversed(range(n)):
        for v in adj[u]: r[u] |= r[v]
    for u in range(n):
        seen = {u}; queue = [u]
        for v in queue:
            for w in adj[v]:
                if w not in seen: seen.add(w); queue.append(w)
        assert seen == set(np.flatnonzero(r[u]))
    model = build_model(cfg.model, len(tok), tok.pad).to(device).eval().requires_grad_(False)
    model.load_state_dict(torch.load(ck, map_location='cpu', weights_only=False)['model'])
    rows = [json.loads(x) for x in gzip.open(OLD/f'{setting}_{protocol}_queries.jsonl.gz', 'rt')]
    header = [tok.id('<WORLD_0>')] if cfg.train.world_roots else []
    return dict(manifest=manifest, cfg=cfg, tok=tok, model=model, nodes=nodes,
                a=a, reach=r, depth=depth, rows=rows, header=header, device=device)


def partitions(p):
    """Graph-defined spectral lateral regions, balanced within topological deciles.

    The lateral split is NOT a discovered model feature. Within each rank-band,
    half the nodes go to either side by a normalized-Laplacian coordinate. This
    removes the trivial equivalence between region and generator rank-band.
    Depth thirds remain a separate descriptive partition.
    """
    a = p['a'].astype(float); w = a+a.T; d = np.maximum(w.sum(1), 1)
    lap = np.eye(len(a))-w/np.sqrt(d[:, None]*d[None, :])
    eig, vec = np.linalg.eigh(lap)
    coord = vec[:, 1]; side = np.zeros(len(a), int)
    bands = np.arange(len(a))*10//len(a)
    for band in np.unique(bands):
        ix = np.flatnonzero(bands == band); ix = ix[np.argsort(coord[ix])]
        side[ix[len(ix)//2:]] = 1
    depth_region = np.minimum(2, p['depth']*3//(p['depth'].max()+1))
    return dict(lateral=side, depth=depth_region), dict(
        lateral=side.tolist(), depth_region=depth_region.tolist(),
        depth=p['depth'].tolist(), spectral_coordinate=coord.tolist(),
        second_eigenvalue=float(eig[1]),
        lateral_within_edge_fraction=float((side[np.where(a)[0]] == side[np.where(a)[1]]).mean()),
        caveat='Spectral regions balanced within rank bands; actual depth/degree can still differ.')


def samples(p, protocol):
    tok = p['tok']; nodes = p['nodes']; inv = {int(v): i for i, v in enumerate(nodes)}
    out = []
    for row in p['rows']:
        s, t = row['source'], row['target']
        assert bool(p['reach'][s, t]) == bool(row['label'])
        prompt = p['header']+[tok.id('<QUERY>'), int(nodes[s]), int(nodes[t]), tok.id('|')]
        if protocol == 'direct':
            out.append(dict(seq=prompt, query=row['query_ids'], source=s, target=t,
                            current=s, split=row['split'], label=row['label'], progress=0,
                            desired=tok.yes if row['label'] else tok.no))
            continue
        if not row['label']: continue
        legal = True; candidates = []
        for j, token in enumerate(row['trace']):
            u = inv.get(token, -1)
            if u < 0 or (j == 0 and u != s): break
            if j and not p['a'][inv[row['trace'][j-1]], u]: legal = False
            if not legal or u == t: break
            if j+1 >= len(row['trace']): break
            nxt = inv.get(row['trace'][j+1], -1)
            if nxt < 0: break
            candidates.append(dict(seq=prompt+row['trace'][:j+1], query=row['query_ids'],
                source=s, target=t, current=u, split=row['split'], label=1, progress=j,
                desired=int(nodes[nxt])))
        # At most three decisions per query; prevent long traces dominating.
        if candidates:
            ix = np.unique(np.linspace(0, len(candidates)-1, min(3, len(candidates))).astype(int))
            out.extend(candidates[i] for i in ix)
    rng = np.random.default_rng(4102); result = []
    for split, cap in [(0, 1400), (1, 500), (2, 1200)]:
        rows = [r for r in out if r['split'] == split]
        if len(rows) > cap: rows = [rows[i] for i in rng.choice(len(rows), cap, replace=False)]
        result.extend(rows)
    return result


def tensor_batch(p, rows):
    lengths = torch.tensor([len(r['seq']) for r in rows], device=p['device'])
    ids = torch.full((len(rows), int(lengths.max())), p['tok'].pad, dtype=torch.long, device=p['device'])
    for i, row in enumerate(rows): ids[i, :len(row['seq'])] = torch.tensor(row['seq'], device=p['device'])
    return ids, lengths-1


def evaluate(p, rows, masks=None, scope='last', remove=None, capture=False, saliency=False, reference_masks=None):
    """Zero selected FFN neurons, or skip one/all attention/FFN residual branches.

    'last' edits only the current prediction position; 'all' also changes the
    cached-equivalent prefix computation. All runs recompute native SDPA.
    Saliency uses native top-token log-probability, not graph-label gradients.
    """
    model = p['model']; state = {}; handles = []; saved = {}
    if saliency:
        handles.append(model.embed.register_forward_hook(lambda m, a, o: o.detach().requires_grad_(True)))
    for l, block in enumerate(model.blocks):
        def pre(module, args, index=l):
            z = args[0]
            if capture or saliency:
                saved[index] = z
                if saliency: z.retain_grad()
            if masks is not None:
                z = z.clone(); columns = torch.as_tensor(masks[index], device=z.device)
                if reference_masks is None:
                    if scope == 'all': z[..., columns] = 0
                    else: z[state['batch'][:, None], state['position'][:, None], columns[None, :]] = 0
                else:
                    assert scope == 'last'
                    original = args[0][state['batch'], state['position']]
                    ref = torch.as_tensor(reference_masks[index], device=z.device)
                    delta_ref = original[:, ref]@module.weight[:, ref].T
                    delta_random = original[:, columns]@module.weight[:, columns].T
                    scale = delta_ref.norm(dim=-1)/delta_random.norm(dim=-1).clamp_min(1e-12)
                    change = original[:, columns]*scale[:, None]
                    z[state['batch'][:, None], state['position'][:, None], columns[None, :]] -= change
                    error = (delta_random*scale[:, None]).norm(dim=-1)-delta_ref.norm(dim=-1)
                    assert error.abs().max() < 1e-3
                return (z,)
        handles.append(block.ffn.down.register_forward_pre_hook(pre))
        if remove and (remove[1] == 'all' or l == remove[1]):
            module = block.ffn if remove[0] == 'ffn' else block.attn
            def skip(module, args, output):
                if isinstance(output, tuple): delta, weights = output
                else: delta = output
                if scope == 'all': changed = torch.zeros_like(delta)
                elif scope == 'last':
                    changed = delta.clone(); changed[state['batch'], state['position']] = 0
                else:
                    changed = delta.clone(); keep = torch.zeros(delta.shape[:2], dtype=torch.bool, device=delta.device)
                    if scope in ('source', 'target', 'separator'):
                        offset = {'source': 1, 'target': 2, 'separator': 3}[scope]+len(p['header'])
                        keep[:, offset] = True
                    elif scope in ('nodes', 'syntax'):
                        keep = torch.isin(state['ids'], torch.tensor(p['nodes'], device=delta.device))
                        if scope == 'syntax': keep = ~keep
                    else: raise ValueError(scope)
                    changed[keep] = 0
                return (changed, weights) if isinstance(output, tuple) else changed
            handles.append(module.register_forward_hook(skip))
    results = []; activity = []; grads = []; last_logits = []
    context = torch.enable_grad if saliency else torch.inference_mode
    try:
        for start in range(0, len(rows), 64):
            subset = rows[start:start+64]; ids, pos = tensor_batch(p, subset)
            state.update(batch=torch.arange(len(subset), device=p['device']), position=pos, ids=ids)
            with context():
                z = model(ids)[state['batch'], pos]
                if saliency:
                    chosen = z.detach().argmax(-1)
                    z.log_softmax(-1).gather(1, chosen[:, None]).sum().backward()
                if capture or saliency:
                    activity.append(torch.stack([saved[l][state['batch'], pos].detach() for l in range(model.n_layers)], 1).cpu().numpy())
                    if saliency:
                        grads.append(torch.stack([(saved[l]*saved[l].grad)[state['batch'], pos].detach().abs() for l in range(model.n_layers)], 1).cpu().numpy())
            z = z.detach().cpu().numpy(); last_logits.append(z)
            for r, logit in zip(subset, z):
                pred = int(logit.argmax()); u, t = r['current'], r['target']
                node = np.flatnonzero(p['nodes'] == pred)
                successor = p['a'][u]; useful = successor & p['reach'][:, t]
                node_scores = logit[p['nodes']]
                edge = bool(node.size and successor[node[0]])
                good = bool(node.size and useful[node[0]])
                mixed = bool(useful.any() and (successor & ~useful).any())
                top_neighbor = int(np.argmax(np.where(successor, node_scores, -np.inf)))
                k = int(successor.sum())
                ranked = node_scores.copy(); ranked[u] = -np.inf
                results.append(dict(exact=float(pred == r['desired']), edge=float(edge), useful=float(good),
                    answer_valid=float(pred in (p['tok'].yes, p['tok'].no)),
                    constrained_answer=float((logit[p['tok'].yes] > logit[p['tok'].no]) == bool(r['label'])),
                    node_top1_edge=float(successor[int(node_scores.argmax())]),
                    node_top1_useful=float(useful[int(node_scores.argmax())]),
                    within_successor=float(useful[top_neighbor]) if mixed else None,
                    within_successor_random=float(useful.sum()/successor.sum()) if mixed else None,
                    mixed=mixed, successor_recall=float(successor[np.argsort(ranked)[-k:]].mean()) if k else None,
                    desired_logp=float(logit[r['desired']]-np.logaddexp.reduce(logit))))
    finally:
        for h in handles: h.remove()
    return dict(metrics=results, logits=np.concatenate(last_logits),
        activity=np.concatenate(activity) if activity else None,
        saliency=np.concatenate(grads) if grads else None)


def average(metrics, rows, indices):
    """Query-balanced means; all decisions from one query form a cluster."""
    result = dict(n=int(len(indices)), queries=len(set(rows[i]['query'] for i in indices)))
    for key in ('exact', 'edge', 'useful', 'within_successor', 'successor_recall', 'desired_logp',
                'answer_valid', 'constrained_answer', 'node_top1_edge', 'node_top1_useful', 'within_successor_random'):
        by_query = {}
        for i in indices:
            value = metrics[i][key]
            if value is not None: by_query.setdefault(rows[i]['query'], []).append(value)
        result[key] = float(np.mean([np.mean(v) for v in by_query.values()])) if by_query else None
        result[key+'_queries'] = len(by_query)
    for key in ('exact', 'constrained_answer'):
        by_label = []
        for label in (0, 1):
            ix = [i for i in indices if rows[i]['label'] == label]
            by_label.append(float(np.mean([metrics[i][key] for i in ix])) if ix else None)
        result[key+'_balanced'] = float(np.mean(by_label)) if all(v is not None for v in by_label) else None
    return result


def choose_neurons(saliency, rows, groups, key, k):
    """Within-stratum discovery contrast, normalized per sample for scale.

    Match rank-band, degree bin, progress and label when both regions occur.
    Selection is performed independently per layer, fixed k per layer. These
    are operational contribution scores, not evidence of semantic exclusivity.
    """
    x = saliency/(saliency.sum(-1, keepdims=True)+1e-12)
    labels = groups[key]
    strata = [(r['rank_band'] if 'depth' not in key else -1, r['degree_bin'], min(r['progress'], 2), r['label']) for r in rows]
    contrasts = []; counts = []
    for region in np.unique(labels):
        diffs = []; support = 0
        for st in sorted(set(strata)):
            ix = np.array([i for i, s in enumerate(strata) if s == st])
            yes = ix[labels[ix] == region]; no = ix[labels[ix] != region]
            if len(yes) < 3 or len(no) < 3: continue
            diffs.append(x[yes].mean(0)-x[no].mean(0)); support += len(ix)
        if not diffs:
            raise RuntimeError(f'No matched discovery support: {key}/{region}')
        score = np.mean(diffs, 0)
        contrasts.append(np.argsort(score, axis=1)[:, -k:]); counts.append(support)
    return contrasts, counts


def matched_test(rows, labels, key):
    """One fixed heldout pool, shared by all interventions in a partition.

    Match coarse structural/task strata, retain equal decisions per region.
    This does not match source/target identity or all structural properties.
    """
    rng = np.random.default_rng(922)
    strata = [(r['rank_band'] if 'depth' not in key else -1, r['degree_bin'], min(r['progress'], 2), r['label']) for r in rows]
    selected = []
    for st in sorted(set(strata)):
        pools = [np.array([i for i, s in enumerate(strata) if s == st and labels[i] == g]) for g in np.unique(labels)]
        count = min(map(len, pools))
        for pool in pools:
            if count: selected.extend(rng.choice(pool, count, replace=False).tolist())
    return np.asarray(sorted(selected), int)


def regions(p, rows, base, discovery, partitions_):
    tr = np.flatnonzero(np.array([r['split'] for r in rows]) == 0)
    te = np.flatnonzero(np.array([r['split'] for r in rows]) == 2)
    train = [rows[i] for i in tr]; test = [rows[i] for i in te]
    result = {}
    # Lateral current-node regions are primary; depth and target are controls.
    for key, role, partition in [('current_lateral', 'current', 'lateral'),
                                 ('current_depth', 'current', 'depth'),
                                 ('target_lateral', 'target', 'lateral')]:
        gl = partitions_[partition][[r[role] for r in rows]]
        sets, support = choose_neurons(discovery, train, {key: gl[tr]}, key, 32)
        labels = gl[te]; matched = matched_test(test, labels, key)
        if not len(matched): raise RuntimeError(f'No matched heldout samples: {key}')
        variants = {f'region_{g}': mask for g, mask in enumerate(sets)}
        importance = discovery.mean(0)
        variants['global_top'] = np.argsort(importance, axis=1)[:, -32:]
        for seed in (201, 303, 407):
            rng = np.random.default_rng(seed)
            # Match global contribution quantiles to the region-0 set, in count.
            mask = []
            for l in range(importance.shape[0]):
                order = np.argsort(importance[l]); quantile = np.empty(len(order), int)
                quantile[order] = np.arange(len(order))*8//len(order)
                chosen = []
                for q in quantile[sets[0][l]]:
                    candidates = np.setdiff1d(np.flatnonzero(quantile == q), chosen)
                    chosen.append(int(rng.choice(candidates)))
                mask.append(chosen)
            variants[f'importance_matched_random_{seed}'] = np.array(mask)
        entry = dict(discovery_matched_support=support, matched_test_n=len(matched),
            selected_neurons={k: v.tolist() for k, v in variants.items()}, scopes={},
            region_set_overlap=[[float(np.mean([len(set(a)&set(b))/32 for a, b in zip(x, y)])) for y in sets] for x in sets])
        for scope in ('last', 'all'):
            scope_result = {}; outcomes = {}
            for name, mask in variants.items():
                pred = evaluate(p, test, masks=mask, scope=scope)
                outcomes[name] = pred['metrics']
                scope_result[name] = {}
                for region in np.unique(labels):
                    ix = matched[labels[matched] == region]
                    stats = average(pred['metrics'], test, ix)
                    original = average(base['metrics'], rows, te[ix])
                    stats['baseline'] = original
                    for metric in ('exact', 'useful', 'within_successor', 'desired_logp'):
                        stats[metric+'_drop'] = original[metric]-stats[metric] if original[metric] is not None else None
                    scope_result[name][str(region)] = stats
            entry['scopes'][scope] = scope_result
            # Two-region interaction and query-cluster bootstrap, fixed sample
            # pool. CI is conditional on one trained model and discovered masks.
            if len(sets) == 2:
                metric = 'exact' if p['manifest']['protocol'] == 'direct' else 'useful'
                cluster = sorted(set(test[i]['query'] for i in matched)); qmap = {q: i for i, q in enumerate(cluster)}
                vals = np.zeros((2, len(cluster))); count = np.zeros_like(vals)
                for i in matched:
                    g = int(labels[i]); q = qmap[test[i]['query']]
                    own = outcomes[f'region_{g}'][i][metric]
                    other = outcomes[f'region_{1-g}'][i][metric]
                    vals[g, q] += other-own; count[g, q] += 1
                present = count > 0; vals /= np.maximum(count, 1)
                estimate = float(np.mean([vals[g, present[g]].mean() for g in range(2)]))
                rng = np.random.default_rng(718); boots = []
                for _ in range(1000):
                    ix = rng.integers(len(cluster), size=len(cluster)); num = vals[:, ix].sum(1); den = present[:, ix].sum(1)
                    if (den > 0).all(): boots.append(float((num/den).mean()))
                entry.setdefault('interaction', {})[scope] = dict(metric=metric,
                    own_minus_other_ablation_drop=estimate, ci95=np.quantile(boots, [.025, .975]).tolist(),
                    queries=len(cluster), note='Positive means each regional mask harms its own region more, averaged over two regions; conditional query-cluster bootstrap.')
        result[key] = entry
        print('regional', key, 'heldout matched', len(matched), flush=True)
    return result


def functional(p, rows, base):
    te = np.flatnonzero(np.array([r['split'] for r in rows]) == 2)
    test = [rows[i] for i in te]; ix = np.arange(len(test))
    out = dict(baseline=average(base['metrics'], rows, te), ablations={})
    for branch in ('ffn', 'attention'):
        for layer in list(range(p['model'].n_layers))+['all']:
            for scope in ('last', 'all'):
                pred = evaluate(p, test, remove=(branch, layer), scope=scope)
                out['ablations'][f'{branch}_{layer}_{scope}'] = average(pred['metrics'], test, ix)
    out['first_ffn_roles'] = {}
    for scope in ('source', 'target', 'separator', 'nodes', 'syntax'):
        pred = evaluate(p, test, remove=('ffn', 0), scope=scope)
        out['first_ffn_roles'][scope] = average(pred['metrics'], test, ix)
    print('functional ablations done', flush=True)
    return out


@torch.inference_mode()
def pointwise(p):
    """Single-node input; compare embedding-only and FFN-only retrieval.

    FFN-only removes attention residuals at every layer, hence no other tokens
    or query can contribute. A strong result shows a node-wise retrieval map;
    it does not uniquely allocate memory between embeddings and decoder.
    """
    model = p['model']; ids = torch.tensor(p['nodes'][:, None], device=p['device'])
    modes = {}
    for mode in ('native_single_token', 'ffn_only'):
        handles = []
        if mode == 'ffn_only':
            for block in model.blocks:
                handles.append(block.attn.register_forward_hook(lambda m, a, o: (torch.zeros_like(o[0]), o[1])))
        cap = {}; model(ids, capture=cap)
        states = [model.embed(ids)]+cap['resid']
        records = []
        for h in states:
            logits = model.lm_head(model.final_norm(h))[:, 0]
            scores = logits[:, p['nodes']].cpu().numpy()
            prob = logits.softmax(-1)[:, p['nodes']].cpu().numpy()
            rec = []; masses = []
            for u in range(len(ids)):
                k = p['a'][u].sum()
                if not k: continue
                s = scores[u].copy(); s[u] = -np.inf
                rec.append(p['a'][u, np.argsort(s)[-k:]].mean())
                masses.append(prob[u, p['a'][u]].sum())
            records.append(dict(nonempty_nodes=len(rec), recall=float(np.mean(rec)), full_vocab_mass=float(np.mean(masses))))
        modes[mode] = records
        for h in handles: h.remove()
    return modes


def embedding_probes(p):
    """Linear/additive vs bilinear endpoint readouts, fixed-random-node controls.

    Original query split plus probe-node-heldout panels. Graph labels are only
    used as supervised probe labels; this is not original-model performance.
    """
    rows = p['rows']; s = np.array([r['source'] for r in rows]); t = np.array([r['target'] for r in rows])
    split = np.array([r['split'] for r in rows]); y = np.array([r['label'] for r in rows], np.float32)
    e = p['model'].embed.weight.detach().cpu().numpy()[p['nodes']]
    vectors = {'trained_embedding': e}
    for seed in (201, 303, 407): vectors[f'random_identity_{seed}'] = np.random.default_rng(seed).normal(size=e.shape).astype(np.float32)
    result = {}
    for panel in ('query_heldout', 'node_heldout_17', 'node_heldout_31', 'node_heldout_47'):
        fold = np.zeros(len(e), int)
        if panel.startswith('node'):
            order = np.random.default_rng(int(panel.rsplit('_', 1)[1])).permutation(len(e))
            fold[order[60:80]] = 1; fold[order[80:]] = 2
            sp = [np.flatnonzero((split == i)&(fold[s] == i)&(fold[t] == i)) for i in range(3)]
        else: sp = [np.flatnonzero(split == i) for i in range(3)]
        if min(map(len, sp)) < 4 or min(len(np.unique(y[z])) for z in sp) < 2:
            result[panel] = dict(insufficient_support=list(map(len, sp))); continue
        out = dict(counts=list(map(len, sp)), features={})
        for name, embedding in vectors.items():
            train_nodes = np.unique(np.r_[s[sp[0]], t[sp[0]]]); centered = embedding-embedding[train_nodes].mean(0)
            _, _, vt = np.linalg.svd(centered[train_nodes], full_matrices=False)
            z = centered@vt[:16].T; z = z/np.maximum(z[train_nodes].std(0), .05)
            concat = np.c_[embedding[s], embedding[t]]
            product = np.einsum('ni,nj->nij', z[s], z[t]).reshape(len(rows), -1)
            for kind, features in [('linear_concat', concat), ('bilinear16', np.c_[z[s], z[t], product])]:
                pred, info = linear.fit(features, y, sp, binary=True)
                stats = linear.classification(y[sp[2]], pred[sp[2], 0, 0], info['threshold'][0][0])
                out['features'][name+'_'+kind] = dict(test=stats, fit=info)
        result[panel] = out
    return result


def run(setting, protocol, device):
    start = time.time(); p = load(setting, protocol, device)
    parts, graph = partitions(p); rows = samples(p, protocol)
    for r in rows:
        r['rank_band'] = r['current']//10
        r['degree_bin'] = min(int(p['a'][r['current']].sum()), 3)
    tr = np.flatnonzero(np.array([r['split'] for r in rows]) == 0)
    base = evaluate(p, rows)
    train = [rows[i] for i in tr]
    capture = evaluate(p, train, saliency=True)
    # Instrumentation must not change native predictions.
    error = float(np.max(abs(capture['logits']-base['logits'][tr])))
    assert error < 1e-4, error
    out = dict(manifest=p['manifest'], graph=graph, sample_counts={str(i): sum(r['split'] == i for r in rows) for i in range(3)},
        checks=dict(gradient_instrumentation_max_logit_error=error),
        convention='Native FP32, frozen parameters; discovery original train, scores original test; query-balanced means.',
        node_names=[p['tok'].name(int(v)) for v in p['nodes']],
        regions=regions(p, rows, base, capture['saliency'], parts),
        functional=functional(p, rows, base), pointwise=pointwise(p),
        embedding=embedding_probes(p))
    out['seconds'] = time.time()-start
    write(HERE/f'{setting}_{protocol}.json', out)
    print('COMPLETE', setting, protocol, out['seconds'], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--device', default='cuda:1')
    parser.add_argument('--setting', choices=['simple', 'hard', 'all'], default='all')
    parser.add_argument('--protocol', choices=['direct', 'path', 'all'], default='all')
    args = parser.parse_args(); HERE.mkdir(exist_ok=True, parents=True)
    for setting in ('simple', 'hard') if args.setting == 'all' else (args.setting,):
        for protocol in ('direct', 'path') if args.protocol == 'all' else (args.protocol,):
            run(setting, protocol, args.device)
