"""Graph-related FFN key/value editing with embeddings/head exactly fixed.

For a graph-selected node pair u,v, a rank-one FFN1 down-weight update sends
the TRAIN mean key for u to v's FFN delta, and annihilates every other observed
node's TRAIN mean key. Independent natural query-test contexts evaluate this
association edit. This is a localization diagnostic, not a fully consistent
counterfactual-graph model and not an embedding-vs-decoder identifiability proof.
"""
import followup as f
import study as s
from study import np, torch, HERE


@torch.inference_mode()
def observe(p, rows, newset):
    result = s.evaluate(p, rows); lens = []
    for start in range(0, len(rows), 64):
        sub = rows[start:start+64]; ids, pos = s.tensor_batch(p, sub); cap = {}
        p['model'](ids, capture=cap)
        h = cap['resid'][0][torch.arange(len(sub), device=p['device']), pos]
        lens.append(p['model'].lm_head(p['model'].final_norm(h)).cpu().numpy())
    scores = np.concatenate(lens); final = result['logits']
    node_scores = scores[:, p['nodes']].copy()
    for i, row in enumerate(rows): node_scores[i, row['current']] = -np.inf
    k = int(newset.sum()); rankings = np.argsort(node_scores, axis=1)[:, -k:]
    result['l1_new_recall'] = newset[rankings].mean(1)
    result['final_new_mass'] = np.exp(final[:, p['nodes'][newset]]-np.logaddexp.reduce(final, axis=1)[:, None]).sum(1)
    pred = final.argmax(1); result['final_new_top1'] = np.isin(pred, p['nodes'][newset])
    return result


def stats(pred, rows, ix):
    value = s.average(pred['metrics'], rows, ix)
    for name in ('l1_new_recall', 'final_new_mass', 'final_new_top1'):
        by_query = {}
        for i in ix: by_query.setdefault(rows[i]['query'], []).append(float(pred[name][i]))
        value[name] = float(np.mean([np.mean(v) for v in by_query.values()])) if by_query else None
    return value


def run(setting, device):
    p, parts, graph, train, test, discovery, base = f.setup(setting, 'path', device)
    current = np.array([r['current'] for r in train]); testcurrent = np.array([r['current'] for r in test])
    nodes = np.unique(current); keys = np.array([discovery['activity'][current == u, 0].mean(0) for u in nodes], np.float64)
    down = p['model'].blocks[0].ffn.down.weight; original = down.detach().clone()
    values = keys@original.cpu().numpy().T
    candidates = []
    for i, u in enumerate(nodes):
        if np.sum(current == u) < 10 or np.sum(testcurrent == u) < 6: continue
        for j, v in enumerate(nodes):
            if i == j or u//10 != v//10 or p['a'][u].sum() != p['a'][v].sum(): continue
            if not p['a'][u].any() or (p['a'][u]&p['a'][v]).any(): continue
            if np.sum(current == v) < 10: continue
            candidates.append((i, j))
    np.random.default_rng(1209).shuffle(candidates)
    selected = []; used = set()
    for pair in candidates:
        u = int(nodes[pair[0]])
        if u not in used: selected.append(pair); used.add(u)
        if len(selected) == 6: break
    out = dict(manifest=p['manifest'], keys_nodes=nodes.tolist(), interventions=[],
        design='Pairs selected by graph and train/test metadata, never by edited outputs. Original embeddings/head and all other weights unchanged.',
        caveat='Mean-key preservation does not guarantee preserving individual-context keys or all other graph functions; first-layer weight edits can affect all token positions.')
    embed_hash = s.hashlib.sha256(p['model'].embed.weight.detach().cpu().numpy().tobytes()).hexdigest()
    for i, j in selected:
        u, v = int(nodes[i]), int(nodes[j]); others = np.delete(keys, i, axis=0)
        _, singular, vt = np.linalg.svd(others, full_matrices=False)
        basis = vt[singular > singular.max()*1e-7]
        key = keys[i]-(keys[i]@basis.T)@basis
        denominator = float(key@keys[i])
        if denominator < 1e-8: continue
        key /= denominator; delta_value = values[j]-values[i]
        delta = np.outer(delta_value, key).astype(np.float32)
        check = keys@delta.T
        error_other = float(np.max(np.linalg.norm(np.delete(check, i, axis=0), axis=1)))
        error_u = float(np.linalg.norm(check[i]-delta_value))
        assert error_other < 1e-3 and error_u < 1e-3
        ix = np.flatnonzero(testcurrent == u); other_ix = np.flatnonzero(testcurrent != u)
        unexposed = np.array([z for z, row in enumerate(test) if int(p['nodes'][u]) not in row['seq']], int)
        oldset = p['a'][u]; newset = p['a'][v]
        before = observe(p, test, newset)
        record = dict(source_node=p['tok'].name(int(p['nodes'][u])), donor_node=p['tok'].name(int(p['nodes'][v])),
            old_successors=[p['tok'].name(int(x)) for x in p['nodes'][oldset]],
            new_successors=[p['tok'].name(int(x)) for x in p['nodes'][newset]],
            update_frobenius=float(np.linalg.norm(delta)), relative_update=float(np.linalg.norm(delta)/original.norm().item()),
            max_other_train_mean_delta=error_other, recipient_train_mean_error=error_u,
            baseline=stats(before, test, ix), other_baseline=stats(before, test, other_ix), controls={})
        record['unexposed_baseline'] = stats(before, test, unexposed)
        try:
            with torch.no_grad(): down.copy_(original+torch.tensor(delta, device=p['device']))
            after = observe(p, test, newset)
            record['edited'] = stats(after, test, ix); record['other_edited'] = stats(after, test, other_ix)
            record['unexposed_edited'] = stats(after, test, unexposed)
            record['examples'] = []
            for z in ix[:6]:
                row = test[z]
                record['examples'].append(dict(query=row['query'], current=record['source_node'], target=p['tok'].name(int(p['nodes'][row['target']])),
                    baseline_next=p['tok'].name(int(before['logits'][z].argmax())),
                    edited_next=p['tok'].name(int(after['logits'][z].argmax())),
                    l1_new_recall_before=float(before['l1_new_recall'][z]), l1_new_recall_after=float(after['l1_new_recall'][z])))
            for seed in (201, 303, 407):
                random = np.random.default_rng(seed).normal(size=len(delta_value)); random *= np.linalg.norm(delta_value)/np.linalg.norm(random)
                random_delta = np.outer(random, key).astype(np.float32)
                assert np.isclose(np.linalg.norm(random_delta), np.linalg.norm(delta), rtol=1e-5)
                with torch.no_grad(): down.copy_(original+torch.tensor(random_delta, device=p['device']))
                control = observe(p, test, newset)
                record['controls'][str(seed)] = dict(recipient=stats(control, test, ix), others=stats(control, test, other_ix))
        finally:
            with torch.no_grad(): down.copy_(original)
        assert s.hashlib.sha256(p['model'].embed.weight.detach().cpu().numpy().tobytes()).hexdigest() == embed_hash
        assert torch.equal(down, original)
        out['interventions'].append(record)
        print(setting, record['source_node'], '=>', record['donor_node'], 'n', len(ix), 'new recall', record['baseline']['l1_new_recall'], record['edited']['l1_new_recall'], flush=True)
    assert out['interventions'], 'No usable graph-defined pairs'
    restored = s.evaluate(p, test)
    out['checks'] = dict(restored_max_logit_error=float(np.max(abs(restored['logits']-base['logits']))),
                         embeddings_and_head_sha256_unchanged=embed_hash)
    assert out['checks']['restored_max_logit_error'] < 1e-5
    s.write(HERE/f'{setting}_path_weight_edit.json', out)


if __name__ == '__main__':
    parser = s.argparse.ArgumentParser(); parser.add_argument('--device', default='cuda:1'); args = parser.parse_args()
    for setting in ('simple', 'hard'): run(setting, args.device)
