"""Native free generation and protocol-aware strict DFS replay (no repairs)."""
from collections import Counter
import numpy as np
from cot_navigation.protocols.search_replay import replay_state, replay_nodes


def evaluate(model, bank, ids, batch_size, max_new_tokens=None):
    from cot_navigation.evaluation.rollout import generate
    from cot_navigation.protocols.tokenizer import Tokenizer
    tok = Tokenizer(256)
    world, protocol = bank.world, bank.protocol
    nodes = [tok.node_id(i) for i in world['node_names']]
    inverse = {token: u for u, token in enumerate(nodes)}
    counters, errors, examples = {}, Counter(), []
    cap = max_new_tokens or bank.info['max_new_tokens']
    for start in range(0, len(ids), batch_size):
        part = ids[start:start + batch_size]
        prompts = [[tok.id('<QUERY>'), nodes[world['records'][i]['s']],
                    nodes[world['records'][i]['t']], tok.id('|')] for i in part]
        outputs = generate(model, prompts, protocol, tok, 'bf16' if bank.device.type == 'cuda' else 'fp32',
                           max_new_tokens=cap)
        for qid, out in zip(part, outputs):
            r = world['records'][qid]
            correct = out['answer_token'] == tok.label_id(r['y'])
            semantic = None
            if protocol == 'dfs_state':
                semantic = replay_state(world['adj'], r['s'], r['t'], out['trace_tokens'], tok, inverse)
            elif protocol == 'dfs_nodes':
                semantic = replay_nodes(world['adj'], r['s'], r['t'], out['trace_tokens'], inverse)
            elif protocol == 'path':
                path = [inverse.get(token, -1) for token in out['trace_tokens']]
                legal = bool(path) and path[0] == r['s'] and all(u >= 0 for u in path)
                if legal:
                    legal = all(v in world['adj'][u] for u, v in zip(path, path[1:]))
                goal = legal and path[-1] == r['t']
                valid = legal and r['t'] not in path[:-1] and (goal if r['y'] else not goal)
                semantic = {'valid': valid, 'terminal': 'FOUND' if goal else 'WALK_END',
                            'error': None if valid else 'illegal_path_or_wrong_endpoint'}
            for bucket in ('all', f"gap{r['gap']}"):
                c = counters.setdefault(bucket, Counter())
                c['n'] += 1
                c[f"n{r['y']}"] += 1
                c[f"correct{r['y']}"] += correct
                c['correct'] += correct
                c['generated_tokens'] += len(out['generated'])
                c['truncated'] += out['generation_limit_reached']
                if semantic:
                    valid = semantic['valid'] and not out['generation_limit_reached']
                    c['trace_valid'] += valid
                    c[f"trace_valid{r['y']}"] += valid
                    terminal_token = tok.yes if semantic['terminal'] == 'FOUND' else tok.no
                    c['trace_answer_consistent'] += valid and terminal_token == out['answer_token']
            if semantic:
                errors[f"y{r['y']}/{semantic['error'] or 'valid'}"] += 1
            if (not correct or semantic and not semantic['valid']) and len(examples) < 2:
                # Compact failure IDs and a short tail; no huge state dumps.
                examples.append({'query_id': qid, 'label': r['y'], 'answer': out['answer_token'],
                                 'error': semantic['error'] if semantic else None,
                                 'generated_length': len(out['generated']), 'tail': out['generated'][-20:]})
    metrics = {}
    for name, c in counters.items():
        yes = c['correct1'] / c['n1'] if c['n1'] else None
        no = c['correct0'] / c['n0'] if c['n0'] else None
        metrics[name] = {**dict(c), 'yes_recall': yes, 'no_recall': no,
            'ba': (yes + no) / 2 if yes is not None and no is not None else None,
            'accuracy': c['correct'] / c['n'],
            'trace_valid_rate': c['trace_valid'] / c['n'] if protocol != 'direct' else None,
            'positive_trace_valid_rate': c['trace_valid1'] / c['n1'] if protocol != 'direct' and c['n1'] else None,
            'negative_trace_valid_rate': c['trace_valid0'] / c['n0'] if protocol != 'direct' and c['n0'] else None}
    model.train()
    return {'metrics': metrics, 'errors': dict(errors), 'examples': examples,
            'semantics': 'Every state field/action replayed' if protocol == 'dfs_state' else
                         'Implicit DFS stack reconstructed; expansion order is not a path' if protocol == 'dfs_nodes' else
                         'Positive: legal complete goal path; negative: legal random walk, not exhaustive search' if protocol == 'path' else 'answer_only'}


def actual_exposure(bank, counts):
    """Protocol-semantic discovery-tree relations, never adjacency of VIS tokens.

    DFS Nodes relations require the known graph to replay implicit POPs; they
    are tagged semantic relations, not explicit contiguous token witnesses.
    """
    world, n = bank.world, bank.world['n_nodes']
    edges = {(u, v) for u, vs in enumerate(world['adj']) for v in vs}
    tree_facts, label_facts = set(), set()
    for row, key in enumerate(bank.keys):
        if not key.startswith('q:') or not counts.get(key):
            continue
        qid = int(bank.cpu['qids'][row])
        r = world['records'][qid]
        if r['y']:
            label_facts.add((r['s'], r['t']))
        a, b = bank.cpu['fact_offsets'][row:row + 2]
        for fact in bank.cpu['facts'][a:b]:
            tree_facts.add((int(fact) // n, int(fact) % n))
    h = [[] for _ in range(n)]
    for u, v in edges | tree_facts | label_facts:
        h[u].append(v)
    def distances(source):
        found, todo = {source: 0}, [source]
        for u in todo:
            for v in h[u]:
                if v not in found:
                    found[v] = found[u] + 1
                    todo.append(v)
        return found
    dist = [distances(s) for s in range(n)]
    panels = {}
    for name, ids in world['panels'].items():
        if name.endswith('_test'):
            positives = [world['records'][q] for q in ids if world['records'][q]['y']]
            panels[name] = {'positive_n': len(positives),
                'semantic_tree_relation_seen': sum((r['s'], r['t']) in tree_facts for r in positives),
                'direct_query_label_seen': sum((r['s'], r['t']) in label_facts for r in positives),
                'macro_distance_counts': dict(Counter(dist[r['s']][r['t']] for r in positives))}
    return {'protocol': bank.protocol, 'panels': panels,
            'note': 'Complete discovery-tree relations per trace; not adjacency of consecutive VIS or expansion tokens'}
