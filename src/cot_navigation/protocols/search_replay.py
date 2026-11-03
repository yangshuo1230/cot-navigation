"""Reference DFS and independent semantic replay for state/node protocols.

No target reachability hints affect neighbor order. These routines are offline
data preparation / evaluation only; the optimizer never searches a graph.
"""
from __future__ import annotations


def reference(adj, source, target):
    stack, visited = [source], [source]
    seen, rows = {source}, []
    while True:
        current = stack[-1]
        candidates = [v for v in adj[current] if v not in seen]
        if current == target:
            action, arg = 'FOUND', None
        elif candidates:
            action, arg = 'PUSH', candidates[0]
        elif len(stack) > 1:
            action, arg = 'POP', None
        else:
            action, arg = 'EXHAUST', None
        rows.append((stack[:], visited[:], candidates, action, arg))
        if action in ('FOUND', 'EXHAUST'):
            return rows
        if action == 'PUSH':
            stack.append(arg)
            visited.append(arg)
            seen.add(arg)
        else:
            stack.pop()


def replay_state(adj, source, target, tokens, tok, inverse):
    """Validate every field/action, accepting any legal candidate permutation."""
    stack, visited, seen = [source], [source], {source}
    cursor, records, pushes, terminal = 0, 0, [], None
    separator, semi = tok.id('|'), tok.id(';')
    actions = {tok.id(f'<{a}>'): a for a in ('PUSH', 'POP', 'FOUND', 'EXHAUST')}
    def field():
        nonlocal cursor
        result = []
        while cursor < len(tokens) and tokens[cursor] != separator:
            if tokens[cursor] not in inverse:
                raise ValueError('non_node_in_state')
            result.append(inverse[tokens[cursor]])
            cursor += 1
        if cursor >= len(tokens):
            raise ValueError('incomplete_state')
        cursor += 1
        return result
    try:
        while cursor < len(tokens):
            if terminal:
                raise ValueError('tokens_after_terminal')
            actual_stack, actual_visited, candidates = field(), field(), field()
            if actual_stack != stack:
                raise ValueError('stack_mismatch')
            if actual_visited != visited:
                raise ValueError('visited_mismatch')
            expected = {v for v in adj[stack[-1]] if v not in seen}
            if len(candidates) != len(set(candidates)) or set(candidates) != expected:
                raise ValueError('candidate_mismatch')
            if cursor >= len(tokens) or tokens[cursor] not in actions:
                raise ValueError('bad_action')
            action = actions[tokens[cursor]]
            cursor += 1
            arg = None
            if action == 'PUSH':
                if cursor >= len(tokens) or tokens[cursor] not in inverse:
                    raise ValueError('missing_push_node')
                arg = inverse[tokens[cursor]]
                cursor += 1
            if cursor >= len(tokens) or tokens[cursor] != semi:
                raise ValueError('missing_record_end')
            cursor += 1
            if stack[-1] == target:
                expected_action = 'FOUND'
            elif expected:
                expected_action = 'PUSH'
            elif len(stack) > 1:
                expected_action = 'POP'
            else:
                expected_action = 'EXHAUST'
            if action != expected_action:
                raise ValueError('illegal_action')
            records += 1
            if action == 'PUSH':
                if not candidates or arg != candidates[0]:
                    raise ValueError('push_not_first_candidate')
                stack.append(arg)
                visited.append(arg)
                seen.add(arg)
                pushes.append((stack[-2], arg))
            elif action == 'POP':
                stack.pop()
            else:
                terminal = action
        if terminal is None:
            raise ValueError('missing_terminal')
        return {'valid': True, 'error': None, 'records': records, 'terminal': terminal,
                'visited': visited, 'push_edges': pushes}
    except ValueError as error:
        return {'valid': False, 'error': str(error), 'records': records, 'terminal': terminal,
                'visited': visited, 'push_edges': pushes}


def replay_nodes(adj, source, target, tokens, inverse):
    """Expansion order is NOT a contiguous path; reconstruct implicit POPs."""
    order = []
    for token in tokens:
        if token not in inverse:
            return {'valid': False, 'error': 'non_node', 'terminal': None, 'visited': order, 'push_edges': []}
        order.append(inverse[token])
    if not order or order[0] != source:
        return {'valid': False, 'error': 'wrong_start', 'terminal': None, 'visited': [], 'push_edges': []}
    stack, seen, pushes = [source], {source}, []
    for node in order[1:]:
        if stack[-1] == target:
            return {'valid': False, 'error': 'continued_after_goal', 'terminal': 'FOUND', 'visited': order, 'push_edges': pushes}
        while stack and not any(v not in seen for v in adj[stack[-1]]):
            stack.pop()
        if not stack or node in seen or node not in adj[stack[-1]]:
            return {'valid': False, 'error': 'illegal_expansion_or_skipped_frontier', 'terminal': None,
                    'visited': order, 'push_edges': pushes}
        pushes.append((stack[-1], node))
        stack.append(node)
        seen.add(node)
    if stack[-1] == target:
        terminal = 'FOUND'
    else:
        terminal = 'EXHAUST' if all(all(v in seen for v in adj[u]) for u in stack) else None
    return {'valid': terminal is not None, 'error': None if terminal else 'premature_stop',
            'terminal': terminal, 'visited': order, 'push_edges': pushes}
