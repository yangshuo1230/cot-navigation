"""Independent controls for regional neuron localization.

Adds discovery-node-heldout tests, degree/rank/label matched random graph
partitions, a predeclared size scan, per-layer intervention and norm-matched
random-neuron directions. No model/probe checkpoint or activations are saved.
"""
import study as s
from study import np, torch, json, HERE


def setup(setting, protocol, device):
    p = s.load(setting, protocol, device); parts, graph = s.partitions(p)
    rows = s.samples(p, protocol)
    for r in rows:
        r['rank_band'] = r['current']//10
        r['degree_bin'] = min(int(p['a'][r['current']].sum()), 3)
    train = [r for r in rows if r['split'] == 0]; test = [r for r in rows if r['split'] == 2]
    discovery = s.evaluate(p, train, saliency=True)
    base = s.evaluate(p, test)
    return p, parts, graph, train, test, discovery, base


def scores(metrics, base, test, ix):
    value = s.average(metrics, test, ix); original = s.average(base, test, ix)
    value['baseline'] = original
    for key in ('exact', 'edge', 'useful', 'within_successor', 'successor_recall', 'desired_logp'):
        value[key+'_drop'] = original[key]-value[key] if original[key] is not None else None
    return value


def interaction(outcomes, labels, matched, test, metric):
    clusters = sorted(set(test[i]['query'] for i in matched)); lookup = {q: i for i, q in enumerate(clusters)}
    vals = np.zeros((2, len(clusters))); count = np.zeros_like(vals)
    for i in matched:
        g = int(labels[i]); q = lookup[test[i]['query']]
        vals[g, q] += outcomes[1-g][i][metric]-outcomes[g][i][metric]; count[g, q] += 1
    present = count > 0; vals /= np.maximum(count, 1)
    estimate = float(np.mean([vals[g, present[g]].mean() for g in range(2)]))
    rng = np.random.default_rng(718); boot = []
    for _ in range(1000):
        ix = rng.integers(len(clusters), size=len(clusters)); den = present[:, ix].sum(1)
        if (den > 0).all(): boot.append(float((vals[:, ix].sum(1)/den).mean()))
    return dict(own_minus_other_drop=estimate, ci95=np.quantile(boot, [.025, .975]).tolist(),
                matched_decisions=len(matched), queries=len(clusters))


def regional_panel(p, train, test, discovery, base, node_labels, k=32, node_heldout=False,
                   norm_controls=False, per_layer=False):
    tr_labels = node_labels[[r['current'] for r in train]]
    te_labels = node_labels[[r['current'] for r in test]]
    tr_ix = np.arange(len(train)); te_ix = np.arange(len(test))
    nodefold = np.zeros(len(p['nodes']), int)
    if node_heldout:
        rng = np.random.default_rng(581)
        for band in range(10):
            for group in (0, 1):
                nodes = np.flatnonzero((np.arange(len(nodefold))//10 == band)&(node_labels == group))
                rng.shuffle(nodes); nodefold[nodes[len(nodes)//2:]] = 1
        tr_ix = tr_ix[nodefold[[r['current'] for r in train]] == 0]
        te_ix = te_ix[nodefold[[r['current'] for r in test]] == 1]
    selected_train = [train[i] for i in tr_ix]; selected_test = [test[i] for i in te_ix]
    masks, support = s.choose_neurons(discovery['saliency'][tr_ix], selected_train,
                                    {'current_lateral': tr_labels[tr_ix]}, 'current_lateral', k)
    matched = s.matched_test(selected_test, te_labels[te_ix], 'current_lateral')
    if len(matched) == 0: raise RuntimeError('No matched node-heldout test support')
    matched = te_ix[matched]
    out = dict(k=k, train_decisions=len(tr_ix), train_nodes=len(set(r['current'] for r in selected_train)),
        heldout_test_nodes=len(set(test[i]['current'] for i in matched)),
        discovery_support=support, interventions={},
        node_identity_disjoint=node_heldout, node_fold=nodefold.tolist() if node_heldout else None)
    outcomes = []
    for region, mask in enumerate(masks):
        pred = s.evaluate(p, test, masks=mask)
        outcomes.append(pred['metrics'])
        out['interventions'][f'region_{region}'] = {
            str(g): scores(pred['metrics'], base['metrics'], test, matched[te_labels[matched] == g]) for g in (0, 1)}
        if norm_controls:
            for seed in (201, 303, 407):
                rng = np.random.default_rng(seed); random_mask = np.array([rng.choice(p['model'].d_ff, k, replace=False) for _ in masks[0]])
                pred = s.evaluate(p, test, masks=random_mask, reference_masks=mask)
                out['interventions'][f'norm_random_{region}_{seed}'] = {
                    str(g): scores(pred['metrics'], base['metrics'], test, matched[te_labels[matched] == g]) for g in (0, 1)}
    metric = 'exact' if p['manifest']['protocol'] == 'direct' else 'useful'
    out['interaction'] = interaction(outcomes, te_labels, matched, test, metric)
    if per_layer:
        out['layers'] = {}
        for layer in range(p['model'].n_layers):
            layer_outcomes = []
            for region, mask in enumerate(masks):
                one = [np.array([], int) for _ in range(p['model'].n_layers)]; one[layer] = mask[layer]
                pred = s.evaluate(p, test, masks=one); layer_outcomes.append(pred['metrics'])
                out['layers'][f'L{layer+1}_region_{region}'] = {
                    str(g): scores(pred['metrics'], base['metrics'], test, matched[te_labels[matched] == g]) for g in (0, 1)}
            out['layers'][f'L{layer+1}_interaction'] = interaction(layer_outcomes, te_labels, matched, test, metric)
    return out, masks


@torch.inference_mode()
def pointwise_regions(p, masks, node_labels):
    """Apply discovered PATH masks to node-only FFN retrieval, all 100 nodes.

    Out-of-distribution input is diagnostic; use native-query controls above
    for the actual behavior claim. The output head remains unchanged.
    """
    model = p['model']; nodes = p['nodes']; ids = torch.tensor(nodes[:, None], device=p['device'])
    saved = {}; handles = []
    for layer, block in enumerate(model.blocks):
        handles.append(block.attn.register_forward_hook(lambda m, a, o: (torch.zeros_like(o[0]), o[1])))
        def record(m, a, o, index=layer): saved[index] = o[:, 0]
        handles.append(block.register_forward_hook(lambda m, a, o, index=layer: saved.update({index: o[0][:, 0]})))
    out = {}
    for region in (None, 0, 1):
        edit = []
        if region is not None:
            for layer, block in enumerate(model.blocks):
                def ablate(module, args, index=layer):
                    z = args[0].clone(); z[..., masks[region][index]] = 0; return (z,)
                edit.append(block.ffn.down.register_forward_pre_hook(ablate))
        model(ids)
        rec = {}
        for layer in (0, 5):
            logits = model.lm_head(model.final_norm(saved[layer]))[:, nodes].cpu().numpy()
            recalls = np.full(len(nodes), np.nan)
            for u in range(len(nodes)):
                k = int(p['a'][u].sum())
                if k:
                    z = logits[u].copy(); z[u] = -np.inf
                    recalls[u] = p['a'][u, np.argsort(z)[-k:]].mean()
            rec[f'L{layer+1}'] = {str(g): dict(n=int(np.sum((node_labels == g)&np.isfinite(recalls))),
                recall=float(np.nanmean(recalls[node_labels == g]))) for g in (0, 1)}
        out['baseline' if region is None else f'region_{region}'] = rec
        for h in edit: h.remove()
    for h in handles: h.remove()
    return out


def run(setting, protocol, device):
    p, parts, graph, train, test, discovery, base = setup(setting, protocol, device)
    out = dict(manifest=p['manifest'], panels={}, size_scan={}, random_partitions={})
    for k in (8, 16, 32, 64, 128):
        panel, masks = regional_panel(p, train, test, discovery, base, parts['lateral'], k,
                                     norm_controls=k == 32, per_layer=k == 32)
        out['size_scan'][str(k)] = panel
        if k == 32:
            out['pointwise_retrieval'] = pointwise_regions(p, masks, parts['lateral'])
        print(setting, protocol, 'size', k, panel['interaction'], flush=True)
    panel, _ = regional_panel(p, train, test, discovery, base, parts['lateral'], node_heldout=True, norm_controls=True)
    out['panels']['node_heldout'] = panel
    for seed in (201, 303, 407):
        rng = np.random.default_rng(seed); labels = parts['lateral'].copy()
        for band in range(10):
            ix = np.flatnonzero(np.arange(len(labels))//10 == band); labels[ix] = rng.permutation(labels[ix])
        panel, _ = regional_panel(p, train, test, discovery, base, labels)
        out['random_partitions'][str(seed)] = panel
    s.write(HERE/f'{setting}_{protocol}_followup.json', out)
    print('FOLLOWUP COMPLETE', setting, protocol, flush=True)


if __name__ == '__main__':
    parser = s.argparse.ArgumentParser(); parser.add_argument('--device', default='cuda:1')
    parser.add_argument('--setting', choices=['simple', 'hard', 'all'], default='all')
    parser.add_argument('--protocol', choices=['direct', 'path', 'all'], default='all'); args = parser.parse_args()
    for setting in ('simple', 'hard') if args.setting == 'all' else (args.setting,):
        for protocol in ('direct', 'path') if args.protocol == 'all' else (args.protocol,):
            run(setting, protocol, args.device)
