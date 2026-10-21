"""Detect whether the last consumed CoT token was wrong, before sampling the next.

Frozen navigation models; synthetic, query-disjoint supervision; natural errors
are an external test. No future/dummy token, feature cache or probe checkpoint.
Run as: python -m cot_navigation.analysis.error_awareness --help
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import copy
import json
import math
import os
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .common import AnalysisModel, write_json, sha256
from .error_detection import semantic_slots, events
from .patching import free_rollouts
from .set_metrics import auc_ap

FIELDS = ['frontier', 'visited', 'candidates', 'action', 'argument',
          'semicolon', 'path_node', 'bfs_expansion']
SEED = 20261019


def body_tokens(ex, tok, tokens):
    p = ex.row.prompt_len
    stop = next((i for i in range(p, len(tokens))
                 if tokens[i] in {tok.yes, tok.no, tok.id('<EOS>')}), len(tokens))
    return list(tokens[p:stop])


def synthetic_events(examples, protocol, tok, count, seed):
    """Many distinct interventions on valid prefixes, never on downstream damage.

    Sampling covers in-graph wrong nodes alongside omitted/excess set members
    and wrong actions, balancing fields rather than individual token positions.
    Alternatives are sampled from legal sets, rather than minimum-ID repairs.
    Fixed-world events share queries and are not independent graph examples.
    """
    rng = np.random.default_rng(seed)
    quota = min(64, max(2, math.ceil(count / max(1, len(examples)))))
    out = []
    for ex in examples:
        body = body_tokens(ex, tok, ex.row.tokens)
        slots = list(semantic_slots(ex, protocol, tok, body))
        groups = defaultdict(list)
        for at, good, field in slots:
            if body[at] not in good:
                raise ValueError(f'Invalid reference: {ex.index}, {at}, {field}')
            good = set(good)
            for bad in sorted(set(ex.nodes) - good):
                kind = 'in_graph_node' if body[at] in ex.nodes else 'excess_node'
                groups[(kind, field)].append((at, good, bad, field, kind))
            # Premature delimiter = missing frontier/visited/candidate node.
            if field in {'frontier', 'visited', 'candidates'} and tok.id('|') not in good:
                groups[('premature_delimiter', field)].append(
                    (at, good, tok.id('|'), field, 'premature_delimiter'))
            if field == 'action':
                for name in ['PUSH', 'POP', 'FOUND', 'EXHAUST']:
                    bad = tok.id(f'<{name}>')
                    if bad not in good:
                        groups[('wrong_action', field)].append((at, good, bad, field, 'wrong_action'))
        seen = set()
        keys = list(groups)
        rng.shuffle(keys)
        # Round-robin categories/fields prevents long visited lists dominating.
        for key in keys:
            rng.shuffle(groups[key])
        picked = 0
        while keys and picked < quota:
            for key in list(keys):
                if picked >= quota:
                    break
                if not groups[key]:
                    keys.remove(key)
                    continue
                at, good, bad, field, kind = groups[key].pop()
                correct = int(rng.choice(sorted(good)))
                signature = (at, bad, correct)
                if signature in seen:
                    continue
                seen.add(signature)
                prefix = list(ex.prompt) + body[:at]
                out.append(dict(query=ex.index, position=len(prefix), field=field,
                                kind=kind, source='synthetic', correct=prefix+[correct],
                                wrong=prefix+[int(bad)], wrong_token=int(bad),
                                correct_token=correct))
                picked += 1
    rng.shuffle(out)
    return out[:count]


def extract(am, examples, event_list, batch_size):
    """Last-input residual only: x[t] is absent from BOTH paired sequences."""
    if not event_list:
        return None
    lookup = {ex.index: ex for ex in examples}
    rows = [(i, label, event, event[key]) for i, event in enumerate(event_list)
            for label, key in [(0, 'correct'), (1, 'wrong')]]
    # Bucketing limits padding overhead. Restore event IDs rather than order.
    rows.sort(key=lambda r: len(r[3]))
    xs, ys, queries, ids, kinds, fields = [], [], [], [], [], []
    for start in range(0, len(rows), batch_size):
        chunk = rows[start:start+batch_size]
        _, residual, _ = am.forward([r[3] for r in chunk])
        for b, (event_id, label, event, seq) in enumerate(chunk):
            assert len(seq) == event['position'] + 1
            ex, pos = lookup[event['query']], len(seq)-1
            # The control sees token identity, field and position, but no graph
            # or prior context. It has exactly the same probe architecture.
            embedding = am.model.embed.weight[seq[-1]].detach().float()
            obs = embedding.new_zeros(4+len(FIELDS))
            obs[:4] = obs.new_tensor([pos/4096, (pos-ex.row.prompt_len)/4096,
                                      float(seq[-1] in ex.nodes), float(seq[-1] == ex.nodes[ex.target])])
            obs[4+FIELDS.index(event['field'])] = 1
            deep = torch.stack([residual[layer][b, pos].float() for layer in am.layers])
            deep = F.pad(deep, (0, len(obs)))
            xs.append(torch.cat([deep, torch.cat([embedding, obs])[None]], 0).cpu().numpy())
            ys.append(label); queries.append(ex.index); ids.append(event_id)
            kinds.append(event['kind']); fields.append(event['field'])
        del residual
    return dict(x=np.asarray(xs), y=np.asarray(ys, np.float32),
                query=np.asarray(queries), event=np.asarray(ids),
                kind=np.asarray(kinds), field=np.asarray(fields))


class ParallelProbe(nn.Module):
    """Independent heads for each layer and the matched identity control."""
    def __init__(self, heads, dim, hidden=0, dropout=.1):
        super().__init__()
        self.hidden = hidden
        self.w = nn.Parameter(torch.empty(heads, dim, hidden or 1))
        self.b = nn.Parameter(torch.zeros(heads, hidden or 1))
        nn.init.normal_(self.w, std=1/math.sqrt(dim))
        if hidden:
            self.v = nn.Parameter(torch.randn(heads, hidden)/math.sqrt(hidden))
            self.c = nn.Parameter(torch.zeros(heads))
            self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        h = torch.einsum('nhd,hdk->nhk', x, self.w) + self.b
        if not self.hidden:
            return h[..., 0]
        return torch.einsum('nhk,hk->nh', self.dropout(F.gelu(h)), self.v) + self.c


def train_probe(train, val, hidden, seed, steps, batch_size, device):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    x = torch.tensor(train['x'], device=device)
    vx = torch.tensor(val['x'], device=device)
    mean, scale = x.mean(0), x.std(0).clamp_min(.05)
    x, vx = (x-mean)/scale, (vx-mean)/scale
    y = torch.tensor(train['y'], device=device)
    vy = torch.tensor(val['y'], device=device)
    # Query weighting avoids treating many edits of one fixed-world query as
    # independent supervision. Each event has both labels and equal weight.
    counts = Counter(train['query'].tolist())
    weight = torch.tensor([1/counts[int(q)] for q in train['query']], device=device)
    weight /= weight.mean()
    probe = ParallelProbe(x.shape[1], x.shape[2], hidden).to(device)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=.001 if hidden else .01, weight_decay=.01)
    best = np.full(x.shape[1], np.inf)
    best_state = copy.deepcopy(probe.state_dict())
    selected = [0]*len(best)
    history = []
    for step in range(1, steps+1):
        probe.train()
        idx = torch.tensor(rng.integers(len(x), size=min(batch_size, len(x))), device=device)
        logits = probe(x[idx])
        loss = (F.binary_cross_entropy_with_logits(logits, y[idx, None].expand_as(logits), reduction='none')
                * weight[idx, None]).mean()
        optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
        if step % 50 == 0 or step == steps:
            probe.eval()
            with torch.no_grad():
                vl = F.binary_cross_entropy_with_logits(probe(vx), vy[:, None].expand(-1, x.shape[1]),
                                                        reduction='none').mean(0).cpu().numpy()
            current = probe.state_dict()
            for head in range(len(best)):
                if vl[head] < best[head]:
                    best[head] = vl[head]; selected[head] = step
                    for key in best_state:
                        best_state[key][head] = current[key][head]
            history.append(dict(step=step, train_bce=float(loss.detach()), val_bce=vl.tolist()))
    probe.load_state_dict(best_state); probe.eval()
    return probe, mean, scale, dict(selected_steps=selected, best_val_bce=best.tolist(), history=history)


def predict(fitted, data):
    if data is None:
        return None
    probe, mean, scale, _ = fitted
    with torch.no_grad():
        x = torch.tensor(data['x'], device=mean.device)
        return probe((x-mean)/scale).cpu().numpy()


def evaluate(data, logits):
    if data is None:
        return None
    result = []
    for head in range(logits.shape[1]):
        z, y = logits[:, head], data['y'].astype(bool)
        p = 1/(1+np.exp(-np.clip(z, -60, 60)))
        row = dict(n=len(y), errors=int(y.sum()), queries=len(set(data['query'].tolist())),
                   **auc_ap(y, z), balanced_accuracy=float(((z[y]>=0).mean()+(z[~y]<0).mean())/2),
                   wrong_probability=float(p[y].mean()), correct_probability=float(p[~y].mean()))
        by_pair = defaultdict(dict)
        for i, pair in enumerate(data['event']):
            by_pair[int(pair)][int(y[i])] = float(z[i])
        row['paired_rank_accuracy'] = float(np.mean([
            (r[1] > r[0])+.5*(r[1] == r[0]) for r in by_pair.values() if 0 in r and 1 in r]))
        row['by_kind'] = {}
        row['by_field'] = {}
        for key in ['kind', 'field']:
            for value in sorted(set(data[key])):
                mask = data[key] == value
                row['by_'+key][str(value)] = dict(n=int(mask.sum()), **auc_ap(y[mask], z[mask]))
        result.append(row)
    return result


def dataset_summary(data, event_list):
    return dict(pairs=len(event_list), rows=0 if data is None else len(data['y']),
                unique_queries=0 if data is None else len(set(data['query'].tolist())),
                categories=dict(Counter(e['kind'] for e in event_list)),
                fields=dict(Counter(e['field'] for e in event_list)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--previous', type=Path, default=Path('research/mechanistic_followup_20261019/results'))
    parser.add_argument('--output', type=Path, default=Path('research/error_awareness_mlp_20261019/results'))
    parser.add_argument('--models', nargs='*')
    parser.add_argument('--train-pairs', type=int, default=8192)
    parser.add_argument('--val-pairs', type=int, default=2048)
    parser.add_argument('--test-pairs', type=int, default=4096)
    parser.add_argument('--train-queries', type=int, default=4096)
    parser.add_argument('--val-queries', type=int, default=1024)
    parser.add_argument('--test-queries', type=int, default=2048)
    parser.add_argument('--natural-queries', type=int, default=256)
    parser.add_argument('--steps', type=int, default=1000)
    parser.add_argument('--hidden', type=int, default=64)
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    parser.add_argument('--forward-batch', type=int, default=64)
    parser.add_argument('--probe-batch', type=int, default=1024)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'; os.environ['WANDB_MODE'] = 'disabled'
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    root = Path.cwd()
    selection = json.loads((args.previous/'selection.json').read_text())
    entries = [e for e in selection['models'] if e['family'] in {'path', 'dfs', 'bfs'}]
    if args.models:
        entries = [e for e in entries if e['id'] in args.models]
    args.output.mkdir(parents=True, exist_ok=True)
    settings = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items() if k != 'resume'}
    settings.update(source_sha256=sha256(Path(__file__)), physical_gpu=0,
                    readout='h[layer,t-1] after consuming x[t-1], before sampling x[t]; no x[t] input',
                    supervision='synthetic only; natural first errors reserved for external test',
                    excluded='final answer/EOS and downstream tokens after first error')
    path = args.output/'settings.json'
    if path.exists() and json.loads(path.read_text()) != settings:
        raise ValueError('Settings changed; use a fresh output directory')
    write_json(path, settings); write_json(args.output/'selection.json', dict(models=entries))
    for entry in entries:
        out = args.output/'models'/f'{entry["id"]}.json'
        if args.resume and out.exists():
            print('SKIP', entry['id'], flush=True); continue
        started = time.time()
        print('MODEL', entry['id'], flush=True)
        am = AnalysisModel(entry, root)
        old = json.loads((args.previous/'models'/f'{entry["id"]}.json').read_text())
        if am.checkpoint['sha256'] != old['checkpoint']['sha256']:
            raise ValueError('Navigation checkpoint changed')
        examples = {split: am.examples(split, getattr(args, split+'_queries'))
                    for split in ['train', 'val', 'test']}
        # The storage split is authoritative and reused across every mutation.
        ids = {s: {e.index for e in exs} for s, exs in examples.items()}
        assert not ids['train'] & ids['val'] and not ids['train'] & ids['test'] and not ids['val'] & ids['test']
        data, summary = {}, {}
        for offset, split in enumerate(['train', 'val', 'test']):
            ev = synthetic_events(examples[split], am.protocol, am.tok,
                                  getattr(args, split+'_pairs'), SEED+offset)
            data[split] = extract(am, examples[split], ev, args.forward_batch)
            summary[split] = dataset_summary(data[split], ev)
            print('DATA', split, summary[split], flush=True)
        natural_examples = am.examples('test', args.natural_queries)
        free, rollout = free_rollouts(am, natural_examples, max_new=0)
        natural_events = []
        for ex in natural_examples:
            ev = events(ex, am.protocol, am.tok, free[ex.index])
            if ev is not None:
                natural_events.append(ev)
        natural = extract(am, natural_examples, natural_events, args.forward_batch)
        summary['natural'] = dataset_summary(natural, natural_events)
        print('NATURAL', summary['natural'], flush=True)
        # Readout causality audit: adding a future token must not change h[t-1].
        ex = examples['train'][0]
        seq = list(ex.row.tokens[:min(len(ex.row.tokens)-1, ex.row.prompt_len+4)])
        _, a, _ = am.forward([seq]); _, b, _ = am.forward([seq+[am.tok.yes]])
        future_delta = max(float((a[l][0, -1]-b[l][0, len(seq)-1]).abs().max()) for l in am.layers)
        del a, b
        result = dict(entry=entry, checkpoint=am.checkpoint, datasets=summary,
                      query_ids={s: sorted(v) for s, v in ids.items()},
                      heads=[f'L{l+1}' for l in am.layers]+['token_control'],
                      readout=settings['readout'], causal_mask_max_abs_delta=future_delta,
                      natural_rollout=dict(queries=len(rollout), errors=len(natural_events),
                                           limits=sum(r['limit'] for r in rollout)), fits=[])
        interesting = []
        for arch, hidden in [('linear', 0), ('mlp', args.hidden)]:
            for seed in args.seeds:
                fitted = train_probe(data['train'], data['val'], hidden, SEED+seed,
                                     args.steps, args.probe_batch, am.device)
                test_logits, natural_logits = predict(fitted, data['test']), predict(fitted, natural)
                fit = dict(architecture=arch, seed=seed, hidden=hidden,
                           training=fitted[3], synthetic=evaluate(data['test'], test_logits),
                           natural=evaluate(natural, natural_logits))
                result['fits'].append(fit)
                print('FIT', entry['id'], arch, seed,
                      'synthetic L6', fit['synthetic'][2]['auroc'],
                      'natural L6', None if fit['natural'] is None else fit['natural'][2]['auroc'], flush=True)
                # Persist a few confidently wrong paired readings for inspection,
                # not all traces or activations. Selection is diagnostic only.
                if arch == 'mlp' and seed == args.seeds[0] and natural is not None:
                    for i in np.argsort(natural_logits[:, 2])[:3]:
                        if natural['y'][i] != 1:
                            continue
                        event = natural_events[int(natural['event'][i])]
                        interesting.append(dict(query=event['query'], field=event['field'],
                            kind=event['kind'], wrong_token=event['wrong_token'],
                            legal_repair=event['correct_token'], error_logit=float(natural_logits[i, 2]),
                            prefix_tail=event['wrong'][-32:]))
                del fitted
        result.update(seconds=time.time()-started, interesting_cases=interesting)
        write_json(out, result)
        print('DONE', entry['id'], result['seconds'], flush=True)
        del am, data, natural
        torch.cuda.empty_cache()
    print('COMPLETE', flush=True)


if __name__ == '__main__':
    main()
