"""Independent native-SDPA masking audit; no model source changes.

All interventions use identical full-prefix batches. Global cuts deny edge
keys to EVERY non-graph query in specified layers, including earlier CoT.
Graph tokens retain their ordinary causal attention. No activation dumps.
"""
import runpy, math, json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

P = Path(__file__).resolve().parent
M = runpy.run_path(str(P / 'run.py'))
CONDITIONS = ['native', 'explicit_identity', 'decision_all', 'decision_relevant',
              'decision_matched', 'history_current', 'decision_and_current']
CONDITIONS += ['global_first_' + str(k) for k in range(1, 7)]
CONDITIONS += ['global_last_' + str(k) for k in (1, 2, 3)]


class NativeMask:
    """Replace _attend only: preserve native SDPA values and projections."""
    def __init__(self, model, batch, tokens, lengths, condition):
        self.saved = []
        B, L = tokens.shape
        dev = tokens.device
        qi = torch.arange(L, device=dev)[None, :, None]
        ki = torch.arange(L, device=dev)[None, None, :]
        ends = torch.tensor([c['edge_end'] for c in batch], device=dev)[:, None, None]
        causal = (ki <= qi).expand(B, L, L)
        graph = ki < ends
        decision = qi == (lengths - 1)[:, None, None]
        current = (tokens == torch.tensor([c['cur'] for c in batch], device=dev)[:, None])
        current = current[:, :, None] & (qi >= ends)
        rel = torch.zeros(B, L, dtype=torch.bool, device=dev)
        matched = torch.zeros_like(rel)
        for bi, c in enumerate(batch):
            others = set(range(c['edge_end'] // 3)) - set(c['relevant_edges'])
            for e in c['relevant_edges']:
                rel[bi, 3*e:3*e+3] = True
                j = min(others, key=lambda j: (abs(j-e), j))
                others.remove(j)
                matched[bi, 3*j:3*j+3] = True
        for li, block in enumerate(model.blocks):
            deny = torch.zeros((B, L, L), dtype=torch.bool, device=dev)
            if condition == 'decision_all': deny = decision & graph
            elif condition == 'decision_relevant': deny = decision & rel[:, None, :]
            elif condition == 'decision_matched': deny = decision & matched[:, None, :]
            elif condition == 'history_current': deny = current & graph
            elif condition == 'decision_and_current': deny = (current | decision) & graph
            elif condition.startswith('global_'):
                k = int(condition.rsplit('_', 1)[1])
                active = li < k if 'first' in condition else li >= len(model.blocks)-k
                if active: deny = (qi >= ends) & graph
            elif condition != 'explicit_identity': raise ValueError(condition)
            keep = (causal & ~deny)[:, None]
            assert bool(keep.any(-1).all())
            attn = block.attn
            self.saved.append((attn, attn._attend))
            def attend(q, k, v, original_keep, force_math=False, causal=False, mask=keep):
                assert original_keep is None and causal and not force_math
                return F.scaled_dot_product_attention(q, k, v, attn_mask=mask), None
            attn._attend = attend

    def close(self):
        for attn, original in self.saved: attn._attend = original


def summary(value):
    # Per-query averaging prevents long trajectories dominating the result.
    return np.nanmean(value, axis=0).tolist()


def evaluate(model, tok, cases):
    probes = []
    for c in cases:
        for p in c['probes']:
            if M['ROLES'][p['role']] not in ('candidate_nonempty', 'candidate_empty'): continue
            probes.append(dict(query_id=c['query_id'], prefix=c['tokens'][:p['q']+1],
                edge_end=c['edge_end'], cur=p['cur'], allowed=p['eligible'] or [tok.id('|')],
                empty=not p['eligible'], relevant_edges=np.flatnonzero(c['edges'][:,1] == p['cur']).tolist(),
                case=c, probe=p))
    index = {c['query_id']: i for i, c in enumerate(cases)}
    sums = np.zeros((len(cases), len(CONDITIONS), 3))
    counts = np.zeros_like(sums)
    checks = dict(explicit_identity_argmax_drift=0, explicit_identity_max_logit_diff=0.,
                  old_math_identity_drift=0, old_native_decision_disagreement=0)
    # layer x [top1,top3,top5 CoT mass,current-node CoT mass,total CoT mass]
    concentration = []
    examples = []
    for lo in range(0, len(probes), 16):
        batch = probes[lo:lo+16]
        tokens, lengths = M['pad_rows']([c['prefix'] for c in batch], model.pad_id, 'cuda')
        baseline = None
        native_block = None
        for ci, condition in enumerate(CONDITIONS):
            editor = None if condition == 'native' else NativeMask(model, batch, tokens, lengths, condition)
            try:
                with torch.no_grad(), M['_autocast'](torch.device('cuda'), 'bf16'):
                    full = model(tokens)
                    logits = M['_gather_row'](full, lengths-1).float()
            finally:
                if editor: editor.close()
            if condition == 'native': baseline = logits.clone()
            if condition == 'explicit_identity':
                checks['explicit_identity_argmax_drift'] += int((logits.argmax(-1) != baseline.argmax(-1)).sum())
                checks['explicit_identity_max_logit_diff'] = max(checks['explicit_identity_max_logit_diff'], float((logits-baseline).abs().max()))
            if condition == 'decision_all': native_block = logits.clone()
            pred = logits.argmax(-1).tolist(); prob = logits.softmax(-1)
            for bi, c in enumerate(batch):
                qi = index[c['query_id']]
                v = [float(pred[bi] in c['allowed']), float(prob[bi,c['allowed']].sum()), float(c['empty'])]
                sums[qi,ci] += v; counts[qi,ci] += 1
        # Independent old hook calculation versus native mask, same batch.
        old = M['EndQueryLesion'](model, tok); old.prepare(batch)
        try:
            identity = old.forward('identity'); blocked = old.forward('graph_all')
        finally: old.close()
        checks['old_math_identity_drift'] += int((identity.argmax(-1) != baseline.argmax(-1)).sum())
        checks['old_native_decision_disagreement'] += int((blocked.argmax(-1) != native_block.argmax(-1)).sum())
        # Read-only native QKV extraction at the exact decision positions.
        reader = M['SelectedRows'](model)
        readcases = [dict(c['case'],tokens=c['prefix'],probes=[c['probe']]) for c in batch]
        try: weights = reader.forward(readcases)
        finally: reader.close()
        for bi, c in enumerate(batch):
            t = np.asarray(c['prefix']); keys = np.arange(len(t))
            cot = keys >= c['case']['prompt_len']; current = cot & (t == c['cur'])
            stats = []
            for li in range(len(model.blocks)):
                w = weights[li][bi,:,0,:len(t)].mean(0)
                ranked = np.sort(w[cot])[::-1]
                stats.append([ranked[:k].sum() for k in (1,3,5)] + [w[current].sum(),w[cot].sum()])
                if c['query_id'] == cases[0]['query_id'] and not c['empty'] and len(examples)<12:
                    order = np.argsort(w)[-5:][::-1]
                    examples.append(dict(query_id=c['query_id'],layer=li+1,cur=int(c['cur']),
                        top=[dict(position=int(k),token=tok.decode([int(t[k])]),weight=float(w[k]),
                                  region='graph' if k<c['edge_end'] else 'query' if k<c['case']['prompt_len'] else 'cot') for k in order]))
            concentration.append((index[c['query_id']], np.asarray(stats)))
        print('batch',lo,'/',len(probes),flush=True)
    values = np.divide(sums,counts,out=np.full_like(sums,np.nan),where=counts>0)
    conc=np.zeros((len(cases),len(model.blocks),5)); n=np.zeros((len(cases),1,1))
    for qi,v in concentration: conc[qi]+=v; n[qi]+=1
    conc=np.divide(conc,n,out=np.full_like(conc,np.nan),where=n>0)
    # Store only aggregate paired confidence intervals, no intermediate matrices.
    rng=np.random.default_rng(20261024)
    dif=values-values[:,:1]
    boot=np.stack([np.nanmean(dif[rng.integers(0,len(cases),len(cases))],axis=0) for _ in range(500)])
    return dict(n_fields=len(probes),checks=checks,metrics=['legal_argmax','legal_probability','empty_fraction'],
                means=summary(values),delta_ci_low=np.nanquantile(boot,.025,axis=0).tolist(),
                delta_ci_high=np.nanquantile(boot,.975,axis=0).tolist(),concentration=summary(conc),examples=examples)


def main():
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(.15)
    result=dict(conditions=CONDITIONS,results={})
    for case in ('n16','main'):
        for arm in ('clean','error'):
            cfg,tok,model=M['load_model'](arm)
            extra=dict(s.split('=',1) for s in cfg.evaluation.extra_sets)
            source=M['D']['ContextStore'](cfg.data.root if case=='main' else extra[case])
            cache=M['D']['ScalePreparedRows'](source,cfg,tok)
            folder=M['AUDIT']/'strict' if case=='main' else M['AUDIT']/'small'
            rows=json.loads((folder/('dfs_clean_'+case+'.json')).read_text())['rows']
            cases=M['build_cases'](source,cache,tok,[r['query_id'] for r in rows])
            result['results'][case+'_'+arm]=evaluate(model,tok,cases)
            M['write'](P/'recheck.json',result)
            print('DONE',case,arm,flush=True)
            del model;torch.cuda.empty_cache()
    result['finished']=True
    M['write'](P/'recheck.json',result)


if __name__=='__main__': main()
