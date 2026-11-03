"""Same-token residual swaps on frozen search checkpoints.

Run from the server project root. Hooks are scoped to prefill and removed
before decode; subsequent layers rebuild the intervened positions' KV cache.
The training model source and all checkpoints remain untouched.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, replace
import hashlib
import json
import os
from pathlib import Path
import time

FIELDS = ('frontier', 'visited', 'candidates')
GROUPS = ('nodes', 'bar', 'operator', 'semicolon', 'nodes_bar', 'all')


def different_state(a, b, field):
    """Frontier order matters; mere visited/candidate permutation does not."""
    return a[field] != b[field] if field == 'frontier' else set(a[field]) != set(b[field])


def digest(tokens):
    return hashlib.sha256(json.dumps(tokens, separators=(',', ':')).encode()).hexdigest()


@dataclass
class Snapshot:
    example: object
    tokens: list
    record: dict
    following: dict
    ordinal: int
    cut_at: int | None = None

    @property
    def prefix(self):
        return self.tokens[:self.cut_at if self.cut_at is not None else self.record['end'] + 1]


def snapshots(am, split, count, max_prefix):
    """Keep complete, independently replayed, legal prepared-data prefixes.

    A donor is another query, never an altered/inconsistent synthetic state.
    Target/answer tokens are absent from every prefix and activation source.
    """
    from cot_navigation.analysis.common import records
    result = []
    audit = Counter()
    for ex in am.examples(split, count):
        rows = records(ex.row.tokens, len(ex.prompt), am.tok)
        ff, seen = [ex.nodes[ex.source]], {ex.nodes[ex.source]}
        adj = {ex.nodes[u]: {ex.nodes[v] for v in vs} for u, vs in enumerate(ex.adj)}
        for i, row in enumerate(rows):
            cur = ff[0 if am.protocol == 'bfs_state' else -1]
            cc = adj[cur] - seen
            action = ('FOUND' if cur == ex.nodes[ex.target] else 'PUSH' if cc else
                      'POP' if len(ff) > 1 else 'EXHAUST')
            aa = row['action']
            valid = (row['frontier'] == ff and len(row['visited']) == len(seen)
                     and set(row['visited']) == seen and len(row['candidates']) == len(cc)
                     and set(row['candidates']) == cc and aa[0] == am.tok.id(f'<{action}>')
                     and (len(aa) == 2 and aa[1] in cc if action == 'PUSH' else len(aa) == 1))
            if not valid:
                raise AssertionError(f'Illegal prepared prefix {split}/{ex.index}/{i}')
            audit['legal_records'] += 1
            if i + 1 < len(rows) and row['end'] + 1 <= max_prefix:
                result.append(Snapshot(ex, ex.row.tokens, row, rows[i + 1], i))
            if action == 'PUSH':
                ff = ff + [aa[1]]
                seen.add(aa[1])
            elif action == 'POP':
                ff = ff[1:] if am.protocol == 'bfs_state' else ff[:-1]
            else:
                break
        audit['queries'] += 1
    audit['eligible_records'] = len(result)
    return result, dict(audit)


def pair_cases(recipients, donors, per_field):
    """Exact serialized field match; BOTH other fields must differ.

    The bounded index keeps distinct other-field states rather than flooding
    it with duplicate initial states. Selection uses no model predictions.
    Prefer equal operators, but never sacrifice valid field pairs to force
    this. Operator/all groups are omitted when operator tokens differ.
    PUSH arguments are never patched.
    """
    indices = {f: defaultdict(list) for f in FIELDS}
    for donor in donors:
        for f in FIELDS:
            key = tuple(donor.record[f])
            bucket = indices[f][key]
            signature = tuple(tuple(donor.record[g]) for g in FIELDS if g != f)
            if len(bucket) < 64 and not any(
                signature == tuple(tuple(d.record[g]) for g in FIELDS if g != f) for d in bucket
            ):
                bucket.append(donor)
    result, coverage = [], {}
    for f in FIELDS:
        eligible, used = [], set()
        for recipient in recipients:
            if recipient.example.index in used:
                continue
            key = tuple(recipient.record[f])
            candidates = [d for d in indices[f].get(key, [])
                          if all(different_state(d.record, recipient.record, g) for g in FIELDS if g != f)]
            if not candidates:
                continue
            # Match prefix length as closely as available; retain actual gaps.
            donor = min(candidates, key=lambda d: (d.record['action'][0] != recipient.record['action'][0], abs(len(d.prefix) - len(recipient.prefix))))
            eligible.append((recipient, donor))
            used.add(recipient.example.index)
        # Spread over short and long histories without cherry-picking outputs.
        eligible.sort(key=lambda pair: (len(pair[0].record[f]) == 0, pair[0].ordinal, pair[0].example.index))
        coverage[f] = dict(eligible_recipient_queries=len(eligible), selected=min(per_field, len(eligible)),
                           native_empty_field_cases=sum(not r.record[f] for r, _ in eligible),
                           selected_operator_identity_pairs=sum(r.record['action'][0] == d.record['action'][0] for r, d in eligible[:per_field]))
        result.extend((f, r, d) for r, d in eligible[:per_field])
    return result, coverage


def sites(snapshot, field, group):
    """Return node/closing-bar/operator/record-end sites; exclude PUSH arg."""
    r = snapshot.record
    a, b, c = r['separators']
    bounds = {'frontier': (r['start'], a), 'visited': (a + 1, b), 'candidates': (b + 1, c)}
    start, end = bounds[field]
    pieces = {'nodes': list(range(start, end)), 'bar': [end],
              'operator': [c + 1], 'semicolon': [r['end']]}
    groups = {'nodes_bar': ('nodes', 'bar'), 'all': ('nodes', 'bar', 'operator', 'semicolon')}
    names = groups.get(group, (group,))
    return [pos for name in names for pos in pieces[name]]


@contextmanager
def residual_patch(model, layer, replacements):
    """Replace post-block residuals ONLY in this prefill invocation.

    replacements maps batch-row to (positions, vectors). Replacement is full
    residual, not an additive delta. Lower-layer KV is preserved, higher-layer
    KV is rebuilt. Last-layer past-site patches are a negative control.
    """
    def hook(_module, _args, output):
        x, weights = output
        x = x.clone()
        for row, (positions, vectors) in replacements.items():
            x[row, positions] = vectors.to(device=x.device, dtype=x.dtype)
        return x, weights
    handle = model.blocks[layer].register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


def field_similarity(actual, expected, field):
    if actual is None:
        return 0.0
    a, b = actual[field], expected[field]
    if field == 'frontier':
        # Exact order matters for both stack and queue.
        return float(a == b)
    aa, bb = set(a), set(b)
    return len(aa & bb) / len(aa | bb) if aa | bb else 1.0


def score_continuation(am, recipient, donor, generated):
    from cot_navigation.analysis.common import records
    # Restore the clamped part for parsing, never as generated evidence.
    fixed = recipient.tokens[recipient.record['start']:len(recipient.prefix)] if recipient.cut_at is not None else []
    parsed = records(fixed + generated, 0, am.tok)
    row = parsed[0] if parsed else None
    result = {'parseable': row is not None, 'generated_count': len(generated),
              'hit_cap': not parsed and not any(t in generated for t in (am.tok.yes, am.tok.no, am.tok.id('<EOS>'), am.tok.id('<STOP>')))}
    for f in FIELDS:
        result[f + '_recipient_similarity'] = field_similarity(row, recipient.following, f)
        result[f + '_donor_similarity'] = field_similarity(row, donor.following, f)
        result[f + '_exact'] = bool(row is not None and row[f] == recipient.following[f])
        starts = [recipient.record['start']] + [p + 1 for p in recipient.record['separators'][:2]]
        result[f + '_is_generated'] = recipient.cut_at is None or starts[FIELDS.index(f)] >= recipient.cut_at
    result['action_exact'] = bool(row is not None and row['action'] == recipient.following['action'])
    result['action_donor_exact'] = bool(row is not None and row['action'] == donor.following['action'])
    # Legal alternatives are distinct from the deterministic reference action.
    if row is not None:
        expected = recipient.following
        result['action_legal'] = bool(row['action'] and row['action'][0] == expected['action'][0]
            and (len(row['action']) == 2 and row['action'][1] in expected['candidates']
                 if expected['action'][0] == am.tok.id('<PUSH>') else row['action'] == expected['action']))
    else:
        result['action_legal'] = False
    return result


def parsed_readout(am, snapshot, generated):
    from cot_navigation.analysis.common import records
    fixed = snapshot.tokens[snapshot.record['start']:len(snapshot.prefix)] if snapshot.cut_at is not None else []
    parsed = records(fixed + generated, 0, am.tok)
    return parsed[0] if parsed else None


def extract_sources(am, cases, layers, groups, batch):
    """Cache only selected vectors in RAM. Never write activation tensors.

    Each no-history token is evaluated independently as prompt + that token.
    Its RoPE position equals the recipient position; the prompt retains its
    normal positions. This is not prompt + the entire matched field.
    """
    import torch
    from cot_navigation.evaluation.rollout import pad_rows
    cross, bare, tasks = {}, {}, []
    for i, (f, r, d) in enumerate(cases):
        union = [p for p in sites(r, f, 'all') if p < len(r.prefix)]
        for p in union:
            tasks.append((i, p, r.example.prompt + [r.tokens[p]]))
    # Donor forward is unpatched and uses its natural positions.
    for at in range(0, len(cases), batch):
        chunk = cases[at:at + batch]
        _, activations, _ = am.forward([d.prefix for _, _, d in chunk])
        for j, (f, r, d) in enumerate(chunk):
            rp = [p for p in sites(r, f, 'all') if p < len(r.prefix)]
            dp = [p for p in sites(d, f, 'all') if p < len(d.prefix)]
            assert r.record[f] == d.record[f]
            cross[at + j] = {layer: activations[layer][j, dp].detach().clone() for layer in layers}
        del activations
    for at in range(0, len(tasks), batch):
        chunk = tasks[at:at + batch]
        tokens, lengths = pad_rows([t[2] for t in chunk], am.tok.pad, am.device)
        positions = torch.arange(tokens.shape[1], device=am.device).expand(len(chunk), -1).clone()
        for j, (_, p, seq) in enumerate(chunk):
            positions[j, len(seq) - 1] = p
        capture = {}
        with torch.autocast(am.device.type, dtype=torch.bfloat16, enabled=am.device.type == 'cuda'):
            am.model(tokens, positions=positions, capture=capture)
        for j, (i, p, seq) in enumerate(chunk):
            bare[i, p] = {layer: capture['resid'][layer][j, len(seq) - 1].detach().clone() for layer in layers}
    return cross, bare


def execute(am, cases, layers, groups, batch, max_new, control_stride=16):
    """Paired batched prefill + unpatched native cached greedy continuation."""
    import numpy as np
    import torch
    from cot_navigation.evaluation.rollout import pad_rows, _decode_step
    # Full grids at L2/L4; sparse L6 and self controls avoid spending most
    # computation repeating already verified, structurally disconnected sites.
    source_layers = sorted(set(layers) | {am.model.n_layers - 1})
    cross, bare = extract_sources(am, cases, source_layers, groups, batch)
    jobs = []
    for i, (field, r, d) in enumerate(cases):
        jobs.append((i, -1, 'baseline', 'none'))
        jobs.append((i, -1, 'baseline', 'donor_control'))
        case_layers = sorted(set(layers) | ({am.model.n_layers - 1} if i % control_stride == 0 else set()))
        for layer in case_layers:
            for group in groups:
                if not sites(r, field, group):
                    continue
                if any(p >= len(r.prefix) for p in sites(r, field, group)):
                    continue
                if [r.tokens[p] for p in sites(r, field, group)] != [d.tokens[p] for p in sites(d, field, group)]:
                    continue  # Never swap an operator represented by a different token.
                sources = ('no_history', 'cross_example') + (('self',) if i % control_stride == 0 else ())
                for source in sources:
                    jobs.append((i, layer, group, source))
    # Batch a single intervention layer at a time. Other rows are unaffected.
    jobs.sort(key=lambda j: (j[1], len(cases[j[0]][1].prefix)))
    outputs, observations = {}, []
    start = 0
    while start < len(jobs):
        end = start + 1
        while end < min(start + max(1, batch // 2), len(jobs)) and jobs[end][1] == jobs[start][1]:
            end += 1
        # Each patch has an unpatched twin in the SAME padded batch. This
        # controls BF16/kernel-layout differences without sharing interventions.
        chunk = [item for job in jobs[start:end] for item in
                 (job, (job[0], -1, 'paired_clean', 'none'))]
        prefixes = [cases[j[0]][2 if chunk[(row // 2) * 2][3] == 'donor_control' else 1].prefix for row, j in enumerate(chunk)]
        tokens, lengths = pad_rows(prefixes, am.tok.pad, am.device)
        cache = am.model.new_cache(len(chunk), am.device, torch.bfloat16 if am.device.type == 'cuda' else torch.float32)
        layer = chunk[0][1]
        # Self activations use precisely the same padded batch/kernel layout.
        capture = {}
        with torch.autocast(am.device.type, dtype=torch.bfloat16, enabled=am.device.type == 'cuda'):
            if any(job[3] == 'self' for job in chunk):
                am.model(tokens, capture=capture)
            replacements = {}
            for row, (i, ll, group, source) in enumerate(chunk):
                if source in ('none', 'donor_control'):
                    continue
                field, r, d = cases[i]
                selected = sites(r, field, group)
                union = [p for p in sites(r, field, 'all') if p < len(r.prefix)]
                if source == 'self':
                    vectors = capture['resid'][ll][row, selected]
                elif source == 'cross_example':
                    vectors = cross[i][ll][[union.index(p) for p in selected]]
                else:
                    vectors = torch.stack([bare[i, p][ll] for p in selected])
                replacements[row] = (selected, vectors)
            with residual_patch(am.model, layer, replacements) if layer >= 0 else empty_context():
                logits = am.model(tokens, cache=cache, cache_mode='prefill')
        cache.lengths.copy_(lengths)
        last = logits[torch.arange(len(chunk), device=am.device), lengths - 1].float()
        probabilities = last.softmax(-1).cpu().numpy()
        generated = [[] for _ in chunk]
        active = [True] * len(chunk)
        for step in range(max_new):
            picked = last.argmax(-1).tolist()
            feed = [am.tok.pad] * len(chunk)
            for j, token in enumerate(picked):
                if not active[j]:
                    continue
                generated[j].append(token)
                if token in (am.tok.id(';'), am.tok.yes, am.tok.no, am.tok.id('<EOS>'), am.tok.id('<STOP>')) or len(prefixes[j]) + len(generated[j]) >= am.model.max_seq_len - 1:
                    active[j] = False
                else:
                    feed[j] = token
            if not any(active):
                break
            last = _decode_step(am.model, cache, torch.tensor(feed, device=am.device).unsqueeze(1),
                                'bf16' if am.device.type == 'cuda' else 'fp32',
                                torch.tensor(active, device=am.device)).float()
        for j in range(0, len(chunk), 2):
            job = chunk[j]
            i, ll, group, source = job
            field, r, d = cases[i]
            if source == 'donor_control':
                r, d = d, r
            key = f'{i}/{ll}/{group}/{source}'
            bp, bg = probabilities[j + 1], generated[j + 1]
            p = probabilities[j]
            m = (p + bp) / 2
            js = float(0.5 * (np.sum(p * np.log(np.maximum(p, 1e-30) / np.maximum(m, 1e-30)))
                              + np.sum(bp * np.log(np.maximum(bp, 1e-30) / np.maximum(m, 1e-30)))))
            row = dict(case=i, field=field, layer=ll + 1, group=group, source=source,
                       js=max(js, 0.0), next_token_changed=int(p.argmax() != bp.argmax()),
                       continuation_changed=generated[j] != bg,
                       probability_max_abs_delta=float(np.max(np.abs(p - bp))),
                       **score_continuation(am, r, d, generated[j]))
            # Both expected next tokens are explicit, pre-specified references.
            rn, dn = r.tokens[len(r.prefix)], d.tokens[len(d.prefix)]
            row.update(next_reference_tokens_differ=rn != dn,
                       donor_next_margin_shift=float((p[dn] - p[rn]) - (bp[dn] - bp[rn])))
            control = score_continuation(am, r, d, bg)
            row['control_action_legal'] = control['action_legal']
            row['control_parseable'] = control['parseable']
            for f in FIELDS:
                row[f + '_donor_similarity_shift'] = row[f + '_donor_similarity'] - control[f + '_donor_similarity']
                row[f + '_recipient_similarity_shift'] = row[f + '_recipient_similarity'] - control[f + '_recipient_similarity']
                row[f + '_control_exact'] = control[f + '_exact']
                row[f + '_target_differs'] = different_state(r.following, d.following, f)
            row['action_target_differs'] = r.following['action'] != d.following['action']
            parsed, original = parsed_readout(am, r, generated[j]), parsed_readout(am, r, bg)
            row['action_changed'] = bool(parsed is not None and original is not None and parsed['action'] != original['action'])
            row['action_control_exact'] = control['action_exact']
            row['action_control_donor_exact'] = control['action_donor_exact']
            for f in FIELDS:
                row[f + '_changed'] = bool(parsed is not None and original is not None and different_state(parsed, original, f))
            observations.append(row)
            if source in ('no_history', 'cross_example') and generated[j] != bg:
                score = max((row[f + '_donor_similarity_shift'] for f in FIELDS if row[f + '_is_generated'] and row[f + '_target_differs']), default=0)
                detail = dict(generated=am.tok.decode(generated[j]), control=am.tok.decode(bg),
                    recipient_reference={f: am.tok.decode(r.following[f]) for f in FIELDS + ('action',)},
                    donor_reference={f: am.tok.decode(d.following[f]) for f in FIELDS + ('action',)},
                    interesting_rank=score + int(row['action_donor_exact'] and row['action_target_differs']), **row)
                if len(outputs) < 8 or detail['interesting_rank'] > min(v['interesting_rank'] for v in outputs.values()):
                    outputs[key] = detail
                    if len(outputs) > 8:
                        outputs.pop(min(outputs, key=lambda k: outputs[k]['interesting_rank']))
        start = end
        if start % 120 < 12 or start == len(jobs):
            print(json.dumps(dict(done=start, total=len(jobs))), flush=True)
    return observations, outputs


@contextmanager
def empty_context():
    yield


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models', nargs='+', default=['context_l10_dfs', 'context_l10_bfs', 'fixed_l10_dfs', 'fixed_l10_bfs'])
    parser.add_argument('--gpu', type=int, default=1)
    parser.add_argument('--recipient-queries', type=int, default=256)
    parser.add_argument('--donor-queries', type=int, default=2048)
    parser.add_argument('--cases', type=int, default=4, help='Per matched field, distinct recipient queries')
    parser.add_argument('--max-prefix', type=int, default=1024)
    parser.add_argument('--max-new', type=int, default=320)
    parser.add_argument('--batch', type=int, default=24)
    parser.add_argument('--layers', type=int, nargs='+', default=[2, 4, 6], help='One-based post-block boundaries')
    parser.add_argument('--groups', nargs='+', choices=GROUPS, default=list(GROUPS))
    parser.add_argument('--audit-only', action='store_true', help='CPU-only pair coverage; use a separate --out directory')
    parser.add_argument('--readout', choices=['next_record', 'current_record'], default='next_record',
                        help='Common record-end cut, or cut immediately after the matched field closing bar')
    parser.add_argument('--out', type=Path, default=Path('studies/activation_patching/results'))
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    import torch
    from cot_navigation.analysis.common import write_json, sha256
    from cot_navigation.analysis.scale_error_awareness import ScaleModel
    torch.set_num_threads(4)
    root = Path.cwd()
    selection = json.loads(Path(__file__).with_name('selection.json').read_text())['models']
    entries = {e['id']: e for e in selection}
    args.out.mkdir(parents=True, exist_ok=True)
    for model_id in args.models:
        destination = args.out / f'{model_id}.json'
        if destination.exists():
            raise FileExistsError(f'Refusing stale result reuse: {destination}; use --out for a new design')
        started = time.time()
        entry = entries[model_id]
        am = ScaleModel(entry, root, device='cpu' if args.audit_only else 'cuda:0')
        assert am.checkpoint['sha256'] == entry['sha256'], 'Checkpoint changed since validation selection'
        recipients, ra = snapshots(am, 'test', args.recipient_queries, args.max_prefix)
        donors, da = snapshots(am, 'train', args.donor_queries, args.max_prefix)
        cases, coverage = pair_cases(recipients, donors, args.cases)
        if args.readout == 'current_record':
            cases = [(f, replace(r, cut_at=r.record['separators'][FIELDS.index(f)] + 1, following=r.record),
                      replace(d, cut_at=d.record['separators'][FIELDS.index(f)] + 1, following=d.record)) for f, r, d in cases]
        print(json.dumps(dict(model=model_id, coverage=coverage)), flush=True)
        metadata = []
        for f, r, d in cases:
            assert all(different_state(r.record, d.record, g) for g in FIELDS if g != f)
            assert r.record[f] == d.record[f]
            assert r.example.split != d.example.split
            metadata.append(dict(field=f, recipient_query=r.example.index, donor_query=d.example.index,
                recipient_split=r.example.split, donor_split=d.example.split,
                recipient_record=r.ordinal, donor_record=d.ordinal,
                matched_tokens=am.tok.decode(r.record[f]), matched_length=len(r.record[f]),
                recipient_length=len(r.prefix), donor_length=len(d.prefix),
                recipient_prefix_sha256=digest(r.prefix), donor_prefix_sha256=digest(d.prefix),
                recipient_state={g: am.tok.decode(r.record[g]) for g in FIELDS + ('action',)},
                donor_state={g: am.tok.decode(d.record[g]) for g in FIELDS + ('action',)},
                recipient_following={g: am.tok.decode(r.following[g]) for g in FIELDS + ('action',)},
                donor_following={g: am.tok.decode(d.following[g]) for g in FIELDS + ('action',)}))
        with torch.inference_mode():
            rows, examples = execute(am, cases, [n - 1 for n in args.layers], args.groups, args.batch, args.max_new) if cases and not args.audit_only else ([], {})
        assert sha256(root / entry['run'] / 'checkpoints/best.pt') == entry['sha256']
        write_json(destination, dict(model=entry, checkpoint=am.checkpoint, args={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            status='pair_audit' if args.audit_only else 'complete' if cases else 'no_strict_native_pairs', recipients=ra, donors=da, coverage=coverage,
            cases=metadata, observations=rows, examples=examples, wall_seconds=time.time() - started))
        del am
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
