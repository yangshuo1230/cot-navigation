"""Query-held-out state probes with legal-history and neighbor-only controls.

All tensors are ephemeral. Only aggregate scores and a few interpretable pairs
are saved. Frozen model forwards and affine head fits run on physical GPU 0.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['WANDB_MODE'] = 'disabled'
import sys
import json
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
import probe_node_search_state as ref
import probe_generated_search_state as gen
# Imported historical modules set CUDA_VISIBLE_DEVICES; restore BEFORE CUDA init.
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from cot_navigation.analysis.scale_followup import make_adapter
from cot_navigation.analysis.probes import fit
from cot_navigation.analysis.set_metrics import scores, set_quality

OUT = Path(__file__).resolve().parent
HEADS = ['L1_after', 'L2_after', 'L3_after', 'L4_after', 'L5_after',
         'L6_after', 'L6_before', 'current_embedding', 'history_bag']
SEED = 20261026


def encode(nodes, members, size, lo):
    y = np.zeros(size, np.float32)
    for v in members:
        y[nodes[int(v)] - lo] = 1
    return y


def collect(am, examples, variants=8):
    """First-expansion states; randomized legal orders; same graphs/queries.

    After-current residual is primary, because current identity is then known.
    Before-current L6 is retained to quantify the position change explicitly.
    """
    groups = []
    meta = []
    n = am.tok.n_node_tokens
    lo = am.tok.node_lo
    for ex in examples:
        for variant in range(variants):
            steps = ref.shuffled_steps(ex.adj, ex.source, ex.target, am.protocol, ex.index, variant)
            first = []
            seen = set()
            for step in steps:
                if step.cur not in seen:
                    seen.add(step.cur)
                    first.append(step)
            order = [step.cur for step in first]
            seq = ex.prompt + [ex.nodes[u] for u in order]
            if len(seq) > am.model.max_seq_len:
                raise AssertionError('context overflow')
            # Use all expansion anchors to preserve honest matched-history support.
            anchors = []
            for k, step in enumerate(first):
                if k == 0 or step.cur == ex.target:
                    continue
                # Independent reconstruction from observable node prefix only.
                prior = gen.decision_state(ex.adj, ex.source, ex.target, order[:k],
                                           am.protocol.startswith('bfs'))
                assert prior is not None and step.cur in prior['legal_next']
                if am.protocol.startswith('dfs'):
                    visited = set(order[:k+1])
                    frontier = prior['frontier'] + [step.cur]
                    assert set(step.visited) == visited
                    assert tuple(step.stack) == tuple(frontier)
                else:
                    visited = set(prior['visited'])
                    frontier = prior['frontier']
                    assert set(step.visited) == visited
                    assert set(step.stack) == set(frontier)
                succ = set(ex.adj[step.cur])
                allowed = succ - visited
                assert allowed == set(step.cands)
                i = len(meta)
                meta.append(dict(query=int(ex.index), variant=variant, current=int(step.cur),
                                 target=int(ex.target), prefix=order[:k+1],
                                 frontier=list(step.stack), visited=sorted(visited),
                                 succ=sorted(succ), allowed=sorted(allowed), nodes=ex.nodes))
                anchors.append((i, ex.row.prompt_len+k))
            groups.append((seq, anchors))
    dim = max(n, am.model.d_model)
    x = np.zeros((len(meta), len(HEADS), dim), np.float32)
    for start in range(0, len(groups), 24):
        chunk = groups[start:start+24]
        _, residual, _ = am.forward([seq for seq, _ in chunk])
        for b, (_seq, anchors) in enumerate(chunk):
            for i, pos in anchors:
                for layer in range(6):
                    x[i, layer, :am.model.d_model] = residual[layer][b, pos].float().cpu().numpy()
                x[i, 6, :am.model.d_model] = residual[5][b, pos-1].float().cpu().numpy()
                token = meta[i]['nodes'][meta[i]['current']]
                x[i, 7, :am.model.d_model] = am.model.embed.weight[token].detach().float().cpu().numpy()
                x[i, 8, :n] = encode(meta[i]['nodes'], meta[i]['prefix'], n, lo)
        del residual
    fields = {}
    for name in ['succ', 'allowed', 'visited', 'frontier']:
        y = np.stack([encode(m['nodes'], m[name], n, lo) for m in meta])
        mask = np.stack([encode(m['nodes'], range(len(m['nodes'])), n, lo) for m in meta])
        fields[name] = dict(x=x, y=y, mask=mask,
                            query=np.asarray([m['query'] for m in meta]))
    return fields, meta


def matched_pairs(meta, exact_stack=False):
    """One pair per query/current/neighbor: same graph, source, target, current.

    No effect-based selection. Compare legal histories where a true successor's
    membership changes. For DFS exact-stack control keep the whole stack equal.
    """
    grouped = defaultdict(list)
    for i, m in enumerate(meta):
        grouped[(m['query'], m['current'])].append(i)
    pairs = []
    for key, ids in grouped.items():
        for node in meta[ids[0]]['succ']:
            a = [i for i in ids if node in meta[i]['allowed']]
            b = [i for i in ids if node not in meta[i]['allowed']]
            found = None
            for ia in a:
                for ib in b:
                    if exact_stack and meta[ia]['frontier'] != meta[ib]['frontier']:
                        continue
                    found = (ia, ib, node)
                    break
                if found:
                    break
            if found:
                pairs.append(found)
    return pairs


def pair_metrics(z, pairs, meta, lo):
    if not pairs:
        return dict(n_pairs=0, n_queries=0, n_current_nodes=0)
    delta = np.stack([z[a, :, meta[a]['nodes'][v]-lo] -
                      z[b, :, meta[b]['nodes'][v]-lo] for a, b, v in pairs])
    return dict(n_pairs=len(pairs), n_queries=len({meta[a]['query'] for a,b,v in pairs}),
                n_current_nodes=len({meta[a]['nodes'][meta[a]['current']] for a,b,v in pairs}),
                directional_accuracy=((delta > 0)+.5*(delta == 0)).mean(0).tolist(),
                mean_logit_delta=delta.mean(0).tolist(),
                fraction_strict_positive=(delta > 0).mean(0).tolist())


def quality(z, bundle, meta):
    result = []
    mixed = (bundle['mask']*bundle['y']).sum(1)>0
    mixed &= (bundle['mask']*(1-bundle['y'])).sum(1)>0
    for h in range(z.shape[1]):
        q = set_quality(z[:,h], bundle['y'], bundle['mask'])
        q['n_queries_mixed'] = len({meta[i]['query'] for i in np.flatnonzero(mixed)})
        q['n_current_nodes_mixed'] = len({meta[i]['nodes'][meta[i]['current']] for i in np.flatnonzero(mixed)})
        result.append(q)
    return result


def main():
    torch.set_num_threads(4)
    doc = dict(heads=HEADS, counts=dict(train=256,val=96,test=128), variants=8,
               protocol='query-disjoint; legal reference prefixes; after-current primary; no node holdout', models={})
    output = OUT / 'results.json'
    if output.exists():
        doc = json.loads(output.read_text())
    for kind, scale, protocol, _ in gen.JOBS:
        name = f'{kind}_{scale}_{protocol}'
        if name in doc['models']:
            continue
        print('MODEL',name,flush=True)
        am = make_adapter(dict(id=name,run=str(gen.run_dir(kind,scale,protocol).relative_to(ROOT))),ROOT,'cuda:0')
        am.layers=list(range(6))
        features, metas, ids = {}, {}, {}
        for split,count in doc['counts'].items():
            examples=am.examples(split,count)
            ids[split]=[e.index for e in examples]
            print('COLLECT',split,len(examples),flush=True)
            features[split],metas[split]=collect(am,examples)
            print('ROWS',split,len(metas[split]),flush=True)
        assert not(set(ids['train'])&set(ids['test']) or set(ids['train'])&set(ids['val']) or set(ids['val'])&set(ids['test']))
        pairs=matched_pairs(metas['test'])
        exact=matched_pairs(metas['test'],True) if protocol.startswith('dfs') else []
        model=dict(checkpoint=am.checkpoint,query_ids=ids,rows={s:len(m) for s,m in metas.items()},tasks={})
        for task in ['succ','allowed','filter','visited','frontier']:
            base='allowed' if task=='filter' else task
            bundles={s:dict(features[s][base]) for s in features}
            if task=='filter':
                for s in bundles:
                    bundles[s]['mask']=features[s]['succ']['y']
            print('FIT',task,flush=True)
            fitted=fit(bundles['train'],bundles['val'],device='cuda:0',seed=SEED,steps=250)
            z=scores(fitted,bundles['test'])
            item=dict(scores=quality(z,bundles['test'],metas['test']),selected_steps=fitted['steps'])
            if task in ['allowed','filter']:
                # Identical outputs; change ONLY evaluation mask for the old full-node head.
                restricted=dict(bundles['test'],mask=features['test']['succ']['y'])
                item['neighbor_only']=quality(z,restricted,metas['test'])
                item['matched']=pair_metrics(z,pairs,metas['test'],am.tok.node_lo)
                item['matched_exact_stack']=pair_metrics(z,exact,metas['test'],am.tok.node_lo)
                if task=='filter':
                    item['examples']=[]
                    for a,b,v in exact[:4] or pairs[:4]:
                        def display(i):
                            m=metas['test'][i]
                            return {key:[am.tok.name(m['nodes'][u]) for u in m[key]]
                                    for key in ['prefix','frontier','visited','allowed']}
                        item['examples'].append(dict(query=metas['test'][a]['query'],
                            current=am.tok.name(metas['test'][a]['nodes'][metas['test'][a]['current']]),
                            focus=am.tok.name(metas['test'][a]['nodes'][v]),unvisited=display(a),visited=display(b),
                            l6_logit_delta=float(z[a,5,metas['test'][a]['nodes'][v]-am.tok.node_lo]-z[b,5,metas['test'][b]['nodes'][v]-am.tok.node_lo])))
            model['tasks'][task]=item
            print('L6',task,item['scores'][5].get('pairwise_auc'),item.get('matched',{}),flush=True)
            del fitted,z
        # A within-neighbor permuted-label control tests the dedicated readout.
        bundles={s:dict(features[s]['allowed'],mask=features[s]['succ']['y']) for s in features}
        fitted=fit(bundles['train'],bundles['val'],device='cuda:0',seed=SEED+17,steps=250,shuffled=True)
        from cot_navigation.analysis.probes import shuffled_labels
        test=dict(bundles['test'],y=shuffled_labels(bundles['test'],SEED+219))
        model['shuffled_filter']=quality(scores(fitted,test),test,metas['test'])
        doc['models'][name]=model
        output.write_text(json.dumps(doc,ensure_ascii=False,indent=2))
        del fitted,am,features,metas
        torch.cuda.empty_cache()
    print('DONE',flush=True)


if __name__=='__main__':
    main()
