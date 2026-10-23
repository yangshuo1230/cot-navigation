"""Held-out recognition and autonomous one-record repair diagnostics.

No oracle MISTAKE is supplied: generation starts immediately before the action
in an erroneous record. The correct full record must follow MISTAKE ;.
The cap is exactly the expected two-record suffix length, so this diagnostic
measures local repair, not final-answer accuracy or full search completion.
"""
from collections import defaultdict

import numpy as np
import torch

from cot_navigation.data.mistakes import FIELDS, apply_plan, choose_plan, records
from cot_navigation.evaluation.rollout import _autocast, pad_rows, generate
from cot_navigation.protocols.validate_trace import _parse_record


@torch.no_grad()
def evaluate_mistakes(model, rows, cfg, tok):
    cases = []
    for i, row in enumerate(rows):
        field = FIELDS[i % len(FIELDS)]
        rng = np.random.default_rng(cfg.evaluation.seed+int(row.meta['index'])*104729+13007)
        plan = choose_plan(row, records(row, tok), tok, field, rng, cfg)
        wrong = apply_plan(row, plan, tok, cfg)
        cases.append(dict(field=field, bad=wrong.tokens[:wrong.meta['mistake_action_index']],
                          clean=row.tokens[:int(plan[4])],
                          expected=row.tokens[int(plan[0]):int(plan[1])]))
    return evaluate_cases(model, cases, cfg, tok)


@torch.no_grad()
def evaluate_cases(model, cases, cfg, tok):
    """Shared scorer for synthetic diagnostics and held-out observed errors."""
    model.eval()
    device = next(model.parameters()).device
    counts = defaultdict(lambda: defaultdict(int))
    for start in range(0, len(cases), cfg.evaluation.batch_size):
        chunk = cases[start:start+cfg.evaluation.batch_size]
        # Clean state prefixes provide a matched false-positive control.
        tokens, lengths = pad_rows([c['clean'] for c in chunk], tok.pad, device)
        with _autocast(device, cfg.train.precision):
            logits = model(tokens)
        clean_picks = logits[torch.arange(len(chunk),device=device),lengths-1].argmax(-1).tolist()
        # Batched local repair; longer cases determine the loop cap, while
        # every case is assessed only up to its own expected repair suffix.
        cap = max(len(c['expected'])+2 for c in chunk)
        generated = generate(model, [c['bad'] for c in chunk], cfg.protocol.name,
                             tok, cfg.train.precision, max_new_tokens=cap, cot_margin=0)
        for case, clean, gen in zip(chunk, clean_picks, generated):
            output = gen['generated']
            detect = bool(output and output[0] == tok.id('<MISTAKE>'))
            prefix = output[:2] == [tok.id('<MISTAKE>'), tok.id(';')]
            expected = case['expected']
            exact = prefix and output[2:2+len(expected)] == expected
            membership = False
            if prefix and tok.id(';') in output[2:]:
                end = output.index(tok.id(';'), 2)
                got = _parse_record(output[2:end], tok)
                exp = _parse_record(expected[:-1], tok)
                if got and exp:
                    node_fields = got['stack']+got['visited']+got['cands']
                    membership = (all(tok.node_lo <= n < tok.node_lo+tok.n_node_tokens for n in node_fields)
                                  and got['stack'] == exp['stack'] and got['visited'] == exp['visited']
                                  and len(got['cands']) == len(set(got['cands']))
                                  and set(got['cands']) == set(exp['cands'])
                                  and got['action'] == exp['action']
                                  and (got['arg'] == (got['cands'][0] if got['cands'] else None)
                                       if got['action'] == 'PUSH' else got['arg'] is None))
            for name in ['overall', case['field']]:
                c = counts[name]
                c['n'] += 1
                c['detected'] += int(detect)
                c['false_alarm'] += int(clean == tok.id('<MISTAKE>'))
                c['exact_repair'] += int(exact)
                c['state_repair'] += int(membership)
    result = {}
    for name, c in counts.items():
        result[name] = {**dict(c), 'detection_rate': c['detected']/c['n'],
                        'false_alarm_rate': c['false_alarm']/c['n'],
                        'exact_repair_rate': c['exact_repair']/c['n'],
                        'state_repair_rate': c['state_repair']/c['n'],
                        'repair_given_detection': c['exact_repair']/c['detected'] if c['detected'] else None}
    return result
