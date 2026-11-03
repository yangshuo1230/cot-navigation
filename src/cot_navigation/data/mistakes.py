"""Audited, compact corruption plans over immutable canonical search traces.

Preparation parses and validates graph transitions once. Training performs a
plan lookup and list splices only: no graph search, record parsing or disk IO.
Each event inserts one erroneous state + MISTAKE before its canonical record;
the erroneous record never mutates the underlying search machine.
"""
from dataclasses import replace
from pathlib import Path
import time

import numpy as np

from cot_navigation.data.provenance import digest
from cot_navigation.data.shards import (atomic_json, atomic_savez, file_sha256,
                                       is_complete, load_json, load_npz, mark_complete)
from cot_navigation.data.scale_prepared import identity as clean_identity
from cot_navigation.protocols.render import truncate_example

FIELDS = ('cands', 'visited')
OPS = ('add', 'drop', 'replace')
VERSION = 1


def records(row, tok):
    """Absolute token spans, including the closing semicolon."""
    start = row.prompt_len
    result = []
    for end in range(start, row.answer_index):
        if row.tokens[end] != tok.id(';'):
            continue
        bars = [p for p in range(start, end) if row.tokens[p] == tok.id('|')]
        if len(bars) != 3:
            raise ValueError('canonical state record must contain three field separators')
        b0, b1, b2 = bars
        result.append(dict(start=start, end=end+1, action_start=b2+1,
                           stack=row.tokens[start:b0], visited=row.tokens[b0+1:b1],
                           cands=row.tokens[b1+1:b2],
                           spans={'visited': (b0+1, b1), 'cands': (b1+1, b2)},
                           action=row.tokens[b2+1:end]))
        start = end+1
    tail = row.tokens[start:row.answer_index]
    if tail not in ([], [tok.id('<STOP>')]):
        raise ValueError('canonical trace has an incomplete record')
    return result


def audit_reference(row, recs, tok, *, fifo):
    """Independent replay: arbitrary legal neighbor order, exact membership."""
    prompt = row.tokens[:row.prompt_len]
    query = prompt[-4:]
    if query[0] != tok.id('<QUERY>') or query[-1] != tok.id('|'):
        raise ValueError('expected an in-context EDGE list and QUERY')
    src, dst = query[1:3]
    adj = {}
    for i in range(0, len(prompt)-4, 3):
        edge, u, v = prompt[i:i+3]
        if edge != tok.id('<EDGE>'):
            raise ValueError('invalid edge-list grammar')
        adj.setdefault(u, set()).add(v)
    stack, visited = [src], [src]
    terminal = False
    for rec in recs:
        if terminal or rec['stack'] != stack or rec['visited'] != visited:
            raise ValueError('invalid canonical queue/stack or visited transition')
        cur = stack[0] if fifo else stack[-1]
        expected = adj.get(cur, set()) - set(visited)
        if len(rec['cands']) != len(set(rec['cands'])) or set(rec['cands']) != expected:
            raise ValueError('canonical candidates disagree with graph and visited')
        action = rec['action']
        if cur == dst:
            wanted = [tok.id('<FOUND>')]
        elif rec['cands']:
            wanted = [tok.id('<PUSH>'), rec['cands'][0]]
        elif len(stack) > 1:
            wanted = [tok.id('<POP>')]
        else:
            wanted = [tok.id('<EXHAUST>')]
        if action != wanted:
            raise ValueError('invalid canonical search action')
        if action[0] == tok.id('<PUSH>'):
            stack.append(action[1]); visited.append(action[1])
        elif action[0] == tok.id('<POP>'):
            stack.pop(0 if fifo else -1)
        else:
            terminal = True
            if (action[0] == tok.id('<FOUND>')) != bool(row.label):
                raise ValueError('terminal search action disagrees with label')


def _edit(rec, field, nodes, src, rng):
    values = rec[field]
    pool = [n for n in nodes if n not in values]
    # Avoid the easy cue of inserting already-visited candidate nodes.
    if field == 'cands':
        pool = [n for n in pool if n not in rec['visited']]
        removable = list(range(len(values)))
    else:
        # Preserve the source and current node; detect history errors rather
        # than an obviously missing current node. Both endpoints of the queue
        # remain represented even before fifo/lifo selection.
        protected = {src, rec['stack'][0], rec['stack'][-1]}
        removable = [i for i, n in enumerate(values) if n not in protected]
    choices = ([0] if pool else []) + ([1] if removable else []) + ([2] if pool and removable else [])
    if not choices:
        return None
    op = int(rng.choice(choices))
    offset = int(rng.integers(len(values)+1)) if op == 0 else int(rng.choice(removable))
    node = int(rng.choice(pool)) if op != 1 else -1
    return op, offset, node


def choose_plan(row, recs, tok, field, rng, cfg):
    """Select one genuine membership error whose repair cannot be truncated."""
    nodes = sorted({n for n in row.tokens[:row.prompt_len]
                    if tok.node_lo <= n < tok.node_lo+tok.n_node_tokens})
    src = row.tokens[row.prompt_len-3]
    reserve = cfg.protocol.cot_margin + 3  # possible STOP + YES/NO + EOS
    for i in rng.permutation(len(recs)):
        rec = recs[int(i)]
        edit = _edit(rec, field, nodes, src, rng)
        if edit is None:
            continue
        op, offset, node = edit
        delta = (1, -1, 0)[op]
        bad_len = rec['action_start']-rec['start'] + delta + 2
        if rec['end']+bad_len > cfg.model.max_seq_len-reserve:
            continue
        lo, hi = rec['spans'][field]
        return np.asarray([rec['start'], rec['end'], lo, hi, rec['action_start'],
                           op, offset, node], dtype=np.int32)
    raise ValueError(f'no complete correctable {field} event fits query {row.meta.get("index")}')


def apply_plan(row, plan, tok, cfg, *, mask_errors=None):
    """Pure in-memory splice; canonical input is never modified."""
    start, end, lo, hi, action, op, offset, node = map(int, plan)
    field = row.tokens[lo:hi]
    if op == 0:
        field = field[:offset]+[node]+field[offset:]
    elif op == 1:
        field = field[:offset]+field[offset+1:]
    elif op == 2:
        field = field.copy(); field[offset] = node
    else:
        raise ValueError('unknown corruption edit')
    bad = row.tokens[start:lo]+field+row.tokens[hi:action]+[tok.id('<MISTAKE>'), tok.id(';')]
    extra = len(bad)
    answer = row.answer_index+extra
    # The closing delimiter is also masked: after deletion its prediction
    # would otherwise teach the model to terminate the wrong field early.
    mask_lo, mask_hi = lo, lo+len(field)+1
    updated = replace(row, tokens=row.tokens[:start]+bad+row.tokens[start:],
                      answer_index=answer,
                      trace_indices=list(range(row.prompt_len, answer))+[answer+1],
                      meta={**row.meta, 'mistake': True,
                            'mistake_field': 'cands' if action == hi+1 else 'visited',
                            'mistake_edit': OPS[op], 'mistake_mask_span': (mask_lo, mask_hi),
                            'mistake_action_index': start+extra-2,
                            'repair_start': start+extra, 'repair_end': end+extra})
    updated = truncate_example(updated, tok, cfg.model.max_seq_len, cfg.protocol.cot_margin)
    if updated.meta['repair_end'] > updated.answer_index:
        raise ValueError('corruption repair was cut off')
    if cfg.protocol.mistake_mask_errors if mask_errors is None else mask_errors:
        updated.trace_indices = (list(range(row.prompt_len, mask_lo))
                                 + list(range(mask_hi, updated.answer_index))
                                 + [updated.answer_index+1])
    return updated


def catalog_identity(source, cfg, tok):
    return dict(version=VERSION, source=source.fingerprint,
                canonical=clean_identity(source, cfg, tok), seed=cfg.protocol.mistake_seed,
                fields=list(FIELDS), single_event=True, protect_repair=True)


def catalog_path(source, cfg, tok):
    return Path(cfg.protocol.mistake_cache_root)/f'{cfg.protocol.name}_{digest(catalog_identity(source,cfg,tok))[:24]}.npz'


def prepare_catalog(source, prepared, cfg, tok):
    """One compact eight-int plan per canonical variant and error field."""
    spec = catalog_identity(source, cfg, tok)
    path = catalog_path(source, cfg, tok)
    if is_complete(path) and path.with_suffix('.json').exists() and load_json(path.with_suffix('.json'))['identity'] == spec:
        return path
    start_time = time.perf_counter()
    count = len(prepared.arrays['fields'])
    plans = np.full((count, len(FIELDS), 8), -1, dtype=np.int32)
    counts = {field: {op: 0 for op in OPS} for field in FIELDS}
    truncated, variants = 0, 0
    for index in source.indices('train'):
        index = int(index)
        for variant in range(prepared.variants(index)):
            row = prepared.row(index, variant)
            recs = records(row, tok)
            audit_reference(row, recs, tok, fifo=cfg.protocol.name == 'bfs_state')
            row_index = int(prepared.arrays['query_offsets'][index])+variant
            for f, field in enumerate(FIELDS):
                rng = np.random.default_rng(cfg.protocol.mistake_seed+index*104729+variant*1009+f*7919)
                plan = choose_plan(row, recs, tok, field, rng, cfg)
                plans[row_index, f] = plan
                # Verify the actual splice, masks, complete repair and label.
                wrong = apply_plan(row, plan, tok, cfg)
                a, b = wrong.meta['repair_start'], wrong.meta['repair_end']
                if wrong.tokens[a:b] != row.tokens[int(plan[0]):int(plan[1])]:
                    raise ValueError('repair record differs from canonical state/action')
                lo, hi = int(plan[2]), int(plan[3])
                bad_field = wrong.tokens[lo:wrong.meta['mistake_mask_span'][1]-1]
                if set(bad_field) == set(row.tokens[lo:hi]) or len(bad_field) != len(set(bad_field)):
                    raise ValueError('injected event is not a unique-node membership error')
                if wrong.tokens[wrong.answer_index] != tok.label_id(row.label):
                    raise ValueError('corruption changed the answer')
                counts[field][OPS[int(plan[5])]] += 1
                truncated += int(wrong.meta.get('truncated', False))
            variants += 1
        if (index+1) % 10000 == 0:
            print(f'PREP {cfg.protocol.name} {source.root.name} {index+1} queries', flush=True)
    atomic_savez(path, plans=plans)
    atomic_json(path.with_suffix('.json'), dict(identity=spec, train_variants=variants,
                plans=int(variants*2), edit_counts=counts, truncated_rows=truncated,
                seconds=time.perf_counter()-start_time, bytes=path.stat().st_size))
    mark_complete(path, file_sha256(path))
    return path


class CorrectionCatalog:
    def __init__(self, source, prepared, cfg, tok):
        path = catalog_path(source, cfg, tok)
        if not is_complete(path) or load_json(path.with_suffix('.json'))['identity'] != catalog_identity(source,cfg,tok):
            raise FileNotFoundError(f'missing audited correction plans: {path}')
        self.plans = load_npz(path)['plans']
        self.offsets = prepared.arrays['query_offsets']

    def plan(self, index, variant, field):
        value = self.plans[int(self.offsets[index])+variant, field]
        if value[0] < 0:
            raise ValueError('held-out query or absent correction plan sampled for training')
        return value
