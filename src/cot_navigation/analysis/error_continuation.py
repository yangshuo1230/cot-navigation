"""Audit later DFS records after natural annotation errors.

The reference machine follows graph-legal emitted actions, never printed state
fields. It stops after the first illegal action. Cancellation does not advance
the machine. Counts use erroneous records, so long lists do not inflate history.
"""
from collections import Counter


def count_bin(n):
    return str(n) if n < 3 else '3-4' if n < 5 else '5+'


def audit_continuation(trace, adj, src, dst, tok):
    bar, semi = tok.id('|'), tok.id(';')
    marker = tok.stoi.get('<MISTAKE>', -1)
    nodes = set(adj) | {v for vs in adj.values() for v in vs} | {src, dst}
    stack, visited = [src], {src}
    records, start = [], 0
    history, unresolved, first, last_bad = 0, 0, None, None
    previous_bad_nodes, all_bad_nodes = set(), set()
    dead = False
    for end, token in enumerate(trace):
        if token != semi:
            continue
        if dead:
            break
        body = trace[start:end]
        groups, field = [], []
        for t in body:
            if t == bar:
                groups.append(field); field = []
            else:
                field.append(t)
        groups.append(field)
        ri = len(records)
        errors, bad_nodes = Counter(), set()
        missing_useful = False
        current = stack[-1]
        eligible = set(adj.get(current, [])) - visited
        def node(t):
            return tok.node_lo <= t < tok.node_lo + tok.n_node_tokens
        def err(kind, value=None):
            errors[kind] += 1
            if value is not None and node(value): bad_nodes.add(value)
        if len(groups) != 4:
            err('format_fields')
        for fi, name in enumerate(['frontier', 'visited', 'candidate']):
            if fi >= len(groups):
                continue
            values = groups[fi]; seen = set()
            expected = stack if fi == 0 else visited if fi == 1 else eligible
            for j, t in enumerate(values):
                if not node(t): err(name + '_format'); continue
                if fi == 0:
                    if j >= len(stack): err('frontier_extra', t)
                    elif t != stack[j]: err('frontier_wrong_order' if t in stack else 'frontier_wrong_node', t)
                elif t in seen: err(name + '_duplicate', t)
                elif t not in expected:
                    kind = ('out_of_graph' if t not in nodes else
                            'visited' if fi == 2 and t in visited else
                            'nonedge' if fi == 2 else 'extra')
                    err(name + '_' + kind, t)
                seen.add(t)
            # A missing closing field is a format error, not an observed omission.
            if fi < len(groups) - 1:
                if (len(values) < len(stack)) if fi == 0 else bool(set(expected) - set(values)):
                    err(name + '_omission')
        candidates = groups[2] if len(groups) > 2 else []
        action = groups[3] if len(groups) == 4 else []
        cancel = action == [marker]
        wanted = '<FOUND>' if current == dst else '<PUSH>' if eligible else '<POP>' if len(stack) > 1 else '<EXHAUST>'
        action_legal = False
        if len(groups) == 4 and not cancel:
            op = tok.id(wanted)
            if not action: err('action_missing')
            elif action[0] != op: err('action_wrong_operator')
            elif len(action) != (2 if wanted == '<PUSH>' else 1): err('action_arity')
            elif wanted == '<PUSH>' and action[1] not in eligible:
                t = action[1]
                err('action_push_out_of_graph' if t not in nodes else 'action_push_visited' if t in visited else 'action_push_nonedge', t)
            else:
                action_legal = True
                if wanted == '<PUSH>' and (not candidates or action[1] != candidates[0]):
                    err('action_candidate_priority')
        bad = bool(errors)
        if first is None and bad:
            # Stable field/token-order primary type, not alphabetic order.
            first = next(iter(errors))
        copied = bad_nodes & previous_bad_nodes
        inherited = bad_nodes & all_bad_nodes
        printed_illegal = set(candidates) - eligible
        illegal_push = (len(action) == 2 and action[0] == tok.id('<PUSH>')
                        and action[1] not in eligible)
        records.append(dict(record=ri + 1, token_start=start, token_end=end + 1,
            search_depth=len(stack), visited_size=len(visited), eligible=len(eligible),
            at_goal=current == dst, prior_bad_records=history, prior_unresolved=unresolved,
            prior_bin=count_bin(history), unresolved_bin=count_bin(unresolved),
            since_last_bad=None if last_bad is None else ri - last_bad,
            first_type=first if history else 'none', bad=bad, errors=dict(errors),
            bad_token_events=sum(errors.values()), annotation_bad=any(not k.startswith('action_') for k in errors),
            action_bad=any(k.startswith('action_') for k in errors), action_legal=action_legal,
            cancel=cancel, true_cancel=cancel and bad, false_cancel=cancel and not bad,
            copied_bad_nodes=len(copied), inherited_bad_nodes=len(inherited),
            bad_node_count=len(bad_nodes), illegal_push=illegal_push,
            illegal_push_from_current_candidates=bool(illegal_push and action[1] in printed_illegal),
            recovered_after_error=bool(history and not bad and action_legal),
            unmarked_recovery=bool(unresolved and not bad and action_legal)))
        if bad:
            history += 1; last_bad = ri
            if not cancel: unresolved += 1
        previous_bad_nodes = bad_nodes
        all_bad_nodes.update(bad_nodes)
        start = end + 1
        if cancel:
            continue
        if not action_legal:
            dead = True
        elif wanted == '<PUSH>':
            stack.append(action[1]); visited.add(action[1])
        elif wanted == '<POP>':
            stack.pop()
        else:
            dead = True
    return dict(records=records, stopped_on_illegal_action=bool(records and not records[-1]['cancel'] and not records[-1]['action_legal']),
        trailing_partial=bool(start < len(trace) and not dead),
        unscored_suffix_tokens=len(trace)-start if dead else 0)
