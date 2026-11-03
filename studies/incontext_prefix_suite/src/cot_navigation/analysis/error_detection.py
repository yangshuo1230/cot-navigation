"""First semantic CoT error and query-disjoint paired linear detection.

The oracle accepts any legal path, set serialization order and PUSH choice.
It stops at the FIRST error, avoiding labels derived from a corrupted state.
Final YES/NO/EOS are excluded. FOUND/EXHAUST are CoT actions and are checked
semantically: a premature FOUND is a genuine detectable trajectory error.
"""
from __future__ import annotations

import numpy as np
import torch


def semantic_slots(ex, protocol, tok, body):
    """Yield (offset, legal next-token set, field) under the valid prefix."""
    nodes = set(ex.nodes); inv = {v: i for i, v in enumerate(ex.nodes)}
    bar, semi = tok.id('|'), tok.id(';')
    if protocol == 'path':
        previous = None
        for at, token in enumerate(body):
            allowed = {ex.nodes[ex.source]} if previous is None else {ex.nodes[v] for v in ex.adj[inv[previous]]}
            yield at, allowed, 'path_node'
            if token not in allowed: return
            previous = token
        return
    if protocol.endswith('_nodes'):
        # Weak semantic BFS check: all reachable shallower levels must be
        # expanded first, but ties can be serialized in ANY order.
        dist = {ex.source: 0}; todo = [ex.source]
        for u in todo:
            for v in ex.adj[u]:
                if v not in dist: dist[v] = dist[u]+1; todo.append(v)
        left = {ex.nodes[v]: d for v, d in dist.items()}
        for at, token in enumerate(body):
            level = min(left.values()) if left else None
            allowed = {v for v, d in left.items() if d == level}
            yield at, allowed, 'bfs_expansion'
            if token not in allowed: return
            del left[token]
        return
    if protocol not in {'dfs_state', 'bfs_state'}: return
    frontier, visited = [ex.nodes[ex.source]], {ex.nodes[ex.source]}
    field, remaining, pointer, at, action, argument = 'frontier', None, 0, 0, None, None
    while at < len(body):
        current = frontier[0 if protocol == 'bfs_state' else -1]
        candidates = {ex.nodes[v] for v in ex.adj[inv[current]]}-visited
        if field == 'frontier':
            allowed = {frontier[pointer]} if pointer < len(frontier) else {bar}
        elif field in {'visited', 'candidates'}:
            if remaining is None: remaining = set(visited if field == 'visited' else candidates)
            allowed = set(remaining) if remaining else {bar}
        elif field == 'action':
            name = 'FOUND' if current == ex.nodes[ex.target] else 'PUSH' if candidates else 'POP' if len(frontier)>1 else 'EXHAUST'
            allowed = {tok.id(f'<{name}>')}
        elif field == 'argument': allowed = candidates
        else: allowed = {semi}
        yield at, allowed, field
        token = body[at]
        if token not in allowed: return
        if field == 'frontier':
            if token == bar: field, remaining = 'visited', None
            else: pointer += 1
        elif field in {'visited', 'candidates'}:
            if token == bar:
                field, remaining = ('candidates' if field == 'visited' else 'action'), None
            else: remaining.remove(token)
        elif field == 'action':
            action = token
            field = 'argument' if token == tok.id('<PUSH>') else 'semicolon'
        elif field == 'argument': argument, field = token, 'semicolon'
        else:
            if action in {tok.id('<FOUND>'), tok.id('<EXHAUST>')}: return
            if action == tok.id('<PUSH>'):
                frontier.append(argument); visited.add(argument)
            elif protocol == 'bfs_state': frontier.pop(0)
            else: frontier.pop()
            field, pointer, remaining = 'frontier', 0, None
        at += 1


def events(ex, protocol, tok, tokens, synthetic=False, seed=0):
    """One event/query: natural first error, or a valid-prefix node corruption."""
    p = ex.row.prompt_len
    terminal = {tok.yes, tok.no, tok.id('<EOS>')}
    stop = next((i for i in range(p, len(tokens)) if tokens[i] in terminal), len(tokens))
    body = tokens[p:stop]
    slots = list(semantic_slots(ex, protocol, tok, body))
    if not slots: return None
    if synthetic:
        # Only corrupt nodes into other IN-GRAPH nodes; avoid a trivial
        # out-of-graph/format detector. Prefer the middle of valid reasoning.
        candidates = [(i, good, f) for i, good, f in slots
                      if body[i] in good and body[i] in ex.nodes and set(ex.nodes)-good]
        if not candidates: return None
        at, good, field = candidates[len(candidates)//2]
        options = sorted(set(ex.nodes)-good)
        bad = options[np.random.default_rng(seed+ex.index).integers(len(options))]
        clean, wrong = tokens[:p+at+1], tokens[:p+at]+[bad]
        original = body[at]
    else:
        bad_slots = [(i, good, f) for i, good, f in slots if body[i] not in good]
        if not bad_slots: return None
        at, good, field = bad_slots[0]
        if not good: return None  # no alternative legal continuation to repair
        original = sorted(good)[0]
        wrong, clean = tokens[:p+at+1], tokens[:p+at]+[original]
        bad = body[at]
    position = p+at
    nxt = tokens[position+1] if position+1 < stop else None
    # A repair may require a different following token (e.g. repairing an
    # omitted candidate needs another '|' before PUSH). Keep the full paired
    # previous-event task, but ALSO score the stricter subset where the repaired
    # prefix remains entirely legal after appending the same actual next token.
    repair_next_valid = False
    if nxt is not None:
        repaired_body=(clean+[nxt])[p:]
        checked=list(semantic_slots(ex,protocol,tok,repaired_body))
        repair_next_valid=(len(checked)==len(repaired_body)
                           and all(repaired_body[i] in good for i,good,_ in checked))
    return {'query': ex.index, 'position': position, 'field': field,
            'kind': 'in_graph_node' if bad in ex.nodes and original in ex.nodes else 'format_or_other',
            'source': 'synthetic' if synthetic else 'natural',
            'wrong': wrong, 'correct': clean, 'next_token': nxt,
            'wrong_token': bad, 'correct_token': original,
            'repair_next_valid':repair_next_valid,
            'next_control': 'same actual next token appended to both prefixes; label refers to prior event'}


def extract_events(am, examples, event_list, offset):
    """offset=1 holds the CURRENT token identical across paired prefixes."""
    by_id = {ex.index: ex for ex in examples}
    rows = []
    for event in event_list:
        if offset and event['next_token'] is None: continue
        for label, key in [(1, 'wrong'), (0, 'correct')]:
            seq = event[key]+([event['next_token']] if offset else [])
            rows.append((event, label, seq))
    if not rows: return None
    output = {k: [] for k in ['x','y','mask','query','stage','goal_seen','event','kind','field','repair_next_valid']}
    fields = ['frontier','visited','candidates','action','argument','semicolon','path_node','bfs_expansion']
    for start in range(0, len(rows), 16):
        batch = rows[start:start+16]
        _, residual, _ = am.forward([r[2] for r in batch])
        for b, (event, label, seq) in enumerate(batch):
            ex, pos = by_id[event['query']], len(seq)-1
            # Both controls use current token embedding. At +1 it is identical
            # within pairs, making token-identity-only discrimination impossible.
            base = am.model.embed.weight[seq[-1]].float()
            obs = base.new_zeros(4+len(fields))
            obs[:4] = base.new_tensor([pos/4096, (pos-ex.row.prompt_len)/4096,
                                      float(seq[-1] in ex.nodes), float(seq[-1]==ex.nodes[ex.target])])
            obs[4+fields.index(event['field'])] = 1
            control = torch.cat([base, obs])
            h = torch.stack([residual[l][b,pos].float() for l in am.layers])
            h = torch.nn.functional.pad(h, (0,len(obs)))
            x = torch.cat([h, control[None]],0).cpu().numpy()
            item = dict(x=x,y=[label],mask=[1.],query=ex.index,stage='error' if not offset else 'next',
                        goal_seen=False,event=ex.index,kind=event['kind'],field=event['field'],
                        repair_next_valid=event['repair_next_valid'])
            for key in output: output[key].append(item[key])
        del residual
    return {k: np.asarray(v) for k,v in output.items()}
