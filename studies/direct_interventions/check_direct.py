"""Frozen Direct diagnostics. Inference only: no optimizer or training loop.

The main graph edit is a bipartite 2x2 switch: two edges exchange destinations.
It flips the queried edge while preserving every node's in/out degree, the
complete token histogram, prompt length, graph size, DAG and edge positions.
An unrelated 2x2 switch is the matched label-preserving control.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from cot_navigation.config import load_config
from cot_navigation.data.reachability import bfs_reaches, build_reach, reaches
from cot_navigation.models.transformer import build_model
from cot_navigation.protocols.render import context_prompt
from cot_navigation.protocols.tokenizer import Tokenizer

ROOT = Path(str(next(_p for _p in __import__('pathlib').Path(__file__).resolve().parents if (_p / 'pyproject.toml').is_file() and (_p / 'studies').is_dir())))
RUNS = ROOT / 'studies/minimal_query_smallgraph/runs/minimal_query_smallgraph_v1'
SEED = 20261018
TOK = Tokenizer()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def adj(rec):
    result = [[] for _ in rec['nodes']]
    for u, v in rec['edges']:
        result[u].append(v)
    return result


def label(rec):
    a = adj(rec)
    result = int(bfs_reaches(a, rec['s'], rec['t']))
    assert result == int(reaches(build_reach(a), rec['s'], rec['t']))
    assert result == int([rec['s'], rec['t']] in rec['edges'])
    return result


def prompt(rec):
    return context_prompt(TOK, rec['nodes'], rec['edges'], rec['s'], rec['t'])


def degrees(rec):
    incoming, outgoing = [0] * 16, [0] * 16
    for u, v in rec['edges']:
        outgoing[u] += 1
        incoming[v] += 1
    return incoming, outgoing


def switches(rec, critical):
    """Enumerate swaps without consulting model predictions.

    For a positive query, (s,t),(u,v) -> (s,v),(u,t) removes (s,t).
    For a negative query, (s,v),(u,t) -> (s,t),(u,v) adds (s,t).
    Both newly formed edges must have been absent.
    """
    edges = rec['edges']
    present = set(map(tuple, edges))
    options = []
    for i, (u, v) in enumerate(edges):
        for j in range(i + 1, len(edges)):
            x, y = edges[j]
            if u == x or v == y or u // 4 != x // 4:
                continue
            if (u, y) in present or (x, v) in present:
                continue
            changed = dict(rec, edges=[e[:] for e in edges])
            changed['edges'][i][1], changed['edges'][j][1] = y, v
            if critical:
                if label(changed) == rec['label']:
                    continue
            else:
                # Do not touch any occurrence of either query endpoint.
                if {u, v, x, y} & {rec['s'], rec['t']}:
                    continue
                assert label(changed) == rec['label']
            options.append((i, j))
    return options


def apply_switch(rec, positions):
    result = dict(rec, edges=[e[:] for e in rec['edges']])
    i, j = positions
    result['edges'][i][1], result['edges'][j][1] = result['edges'][j][1], result['edges'][i][1]
    result['label'] = label(result)
    return result


def balanced(records, count):
    result = []
    for y in (0, 1):
        selected = [r for r in records if r['label'] == y][:count // 2]
        if len(selected) != count // 2:
            raise ValueError('insufficient balanced graphs')
        result.extend(selected)
    return result


def construct(data, count):
    rng = np.random.default_rng(SEED)
    all_cases, stats = [], {}
    for split, records in (('train', data['graphs']), ('new_graph', data['test'])):
        chosen = balanced(records, count)
        counts = Counter()
        for gi, raw in enumerate(chosen):
            base = {k: deepcopy(raw[k]) for k in ('nodes', 'edges', 's', 't', 'label')}
            base['edges'] = [base['edges'][i] for i in rng.permutation(24)]
            assert label(base) == base['label']
            variants = {'base': base}
            for name, critical in (('critical_edge', True), ('irrelevant_edges', False)):
                choices = switches(base, critical)
                if choices:
                    variants[name] = apply_switch(base, choices[int(rng.integers(len(choices)))])
            incoming, outgoing = degrees(base)
            for field, title in (('t', 'target'), ('s', 'source')):
                original = base[field]
                layer = original // 4
                for flip, suffix in ((True, 'opposite'), (False, 'same_label')):
                    choices = []
                    for node in range(layer * 4, layer * 4 + 4):
                        if node == original:
                            continue
                        # Avoid trivial absent-endpoint / empty-in/out tests.
                        if field == 't' and incoming[node] == 0:
                            continue
                        if field == 's' and outgoing[node] == 0:
                            continue
                        candidate = dict(base, **{field: node})
                        y = label(candidate)
                        if (y != base['label']) == flip:
                            candidate['label'] = y
                            choices.append(candidate)
                    if choices:
                        variants[f'query_{title}_{suffix}'] = choices[int(rng.integers(len(choices)))]
            for j in range(4):
                variants[f'edge_order_{j}'] = dict(base, edges=[base['edges'][i][:] for i in rng.permutation(24)])
            base_tokens = prompt(base)
            for name, rec in variants.items():
                ids = prompt(rec)
                assert len(ids) == 76
                assert len(rec['edges']) == len(set(map(tuple, rec['edges']))) == 24
                assert all(v // 4 == u // 4 + 1 for u, v in rec['edges'])
                assert rec['nodes'] == base['nodes'] and label(rec) == rec['label']
                if name in ('critical_edge', 'irrelevant_edges'):
                    assert rec['s'] == base['s'] and rec['t'] == base['t']
                    assert Counter(ids) == Counter(base_tokens)
                    assert degrees(rec) == degrees(base)
                    assert sum(a != b for a, b in zip(ids, base_tokens)) == 2
                if name.startswith('edge_order') or name == 'irrelevant_edges':
                    assert rec['label'] == base['label']
                counts[name] += 1
            all_cases.append(dict(split=split, graph_id=gi, original_index=raw.get('index'), variants=variants))
        stats[split] = dict(counts)
    return all_cases, stats


@torch.inference_mode()
def score(model, records, batch=256, bf16=False):
    result = []
    for start in range(0, len(records), batch):
        chunk = records[start:start + batch]
        tokens = torch.tensor([prompt(rec) for rec in chunk], dtype=torch.long, device='cuda')
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=bf16):
            logits = model(tokens)[:, -1].float()
        logit_pair = logits[:, [TOK.no, TOK.yes]]
        conditional = logit_pair.softmax(-1)[:, 1]
        full_probs = logits.softmax(-1)
        raw_yes = full_probs[:, TOK.yes]
        valid_mass = raw_yes + full_probs[:, TOK.no]
        prediction = logits.argmax(-1)
        margin = logits[:, TOK.yes] - logits[:, TOK.no]
        values = torch.stack([conditional, raw_yes, valid_mass, margin, prediction.float()], -1).cpu().tolist()
        for rec, (p, raw, mass, m, pred) in zip(chunk, values):
            pred = int(pred)
            result.append(dict(p_yes=p, raw_p_yes=raw, answer_mass=mass, margin=m, pred_token=pred,
                               pred=1 if pred == TOK.yes else 0 if pred == TOK.no else -1,
                               label=rec['label']))
    return result


def bootstrap_mean(values, seed=SEED):
    x = np.asarray(values, dtype=float)
    if not len(x):
        return None
    rng = np.random.default_rng(seed)
    means = x[rng.integers(len(x), size=(2000, len(x)))].mean(-1)
    return dict(mean=float(x.mean()), ci95=np.quantile(means, [.025, .975]).tolist())


def summarize(rows):
    report = {}
    for split in ('train', 'new_graph'):
        groups = [r for r in rows if r['split'] == split]
        baseline = [r['scores']['base'] for r in groups]
        section = dict(baseline=dict(n=len(baseline),
                         accuracy=float(np.mean([r['pred'] == r['label'] for r in baseline])),
                         yes_rate=float(np.mean([r['pred'] == 1 for r in baseline])),
                         p_yes_mean=float(np.mean([r['p_yes'] for r in baseline])),
                         p_yes_std=float(np.std([r['p_yes'] for r in baseline])),
                         p_yes_range=[min(r['p_yes'] for r in baseline), max(r['p_yes'] for r in baseline)],
                         answer_mass_min=min(r['answer_mass'] for r in baseline)), tests={})
        names = sorted(set(k for r in groups for k in r['scores'] if k != 'base' and not k.startswith('edge_order')))
        for name in names:
            pairs = [(r['scores']['base'], r['scores'][name]) for r in groups if name in r['scores']]
            delta = np.array([b['p_yes'] - a['p_yes'] for a, b in pairs])
            dm = np.array([b['margin'] - a['margin'] for a, b in pairs])
            opposite = 'opposite' in name or name == 'critical_edge'
            item = dict(n=len(pairs), mean_abs_delta_p=float(np.abs(delta).mean()),
                        abs_delta_p_median=float(np.median(np.abs(delta))),
                        mean_abs_delta_margin=float(np.abs(dm).mean()),
                        prediction_flip_rate=float(np.mean([a['pred'] != b['pred'] for a, b in pairs])),
                        base_accuracy=float(np.mean([a['pred'] == a['label'] for a, b in pairs])),
                        modified_accuracy=float(np.mean([b['pred'] == b['label'] for a, b in pairs])),
                        both_correct=float(np.mean([a['pred'] == a['label'] and b['pred'] == b['label'] for a, b in pairs])))
            if opposite:
                direction = np.array([2 * b['label'] - 1 for a, b in pairs])
                signed = direction * delta
                item.update(signed_delta_p=bootstrap_mean(signed), signed_delta_margin=bootstrap_mean(direction * dm),
                            correct_direction_fraction=float(np.mean(signed > 0)), ties=float(np.mean(signed == 0)))
                for y in (0, 1):
                    selected = [i for i, (a, b) in enumerate(pairs) if a['label'] == y]
                    item[f'original_label_{y}'] = dict(n=len(selected),
                        signed_delta_p=bootstrap_mean(signed[selected]) if selected else None)
            section['tests'][name] = item
        # Four order repetitions are averaged within graph before aggregation.
        deltas, flips, spans = [], [], []
        for row in groups:
            base = row['scores']['base']
            orders = [r for k, r in row['scores'].items() if k.startswith('edge_order')]
            deltas.append(np.mean([abs(r['p_yes'] - base['p_yes']) for r in orders]))
            flips.append(np.mean([r['pred'] != base['pred'] for r in orders]))
            spans.append(np.ptp([base['p_yes']] + [r['p_yes'] for r in orders]))
        section['tests']['edge_order'] = dict(n=len(groups), repeats_per_graph=4,
            mean_abs_delta_p=float(np.mean(deltas)), prediction_flip_rate=float(np.mean(flips)),
            mean_p_yes_range=float(np.mean(spans)))
        # On identical eligibility subsets, compare critical vs unrelated edit.
        paired = []
        for r in groups:
            s = r['scores']
            if 'critical_edge' in s and 'irrelevant_edges' in s:
                paired.append(abs(s['critical_edge']['p_yes'] - s['base']['p_yes'])
                              - abs(s['irrelevant_edges']['p_yes'] - s['base']['p_yes']))
        section['critical_minus_irrelevant_abs_delta'] = dict(n=len(paired), effect=bootstrap_mean(paired))
        report[split] = section
    return report


def examples(rows):
    result = []
    for split in ('train', 'new_graph'):
        selected = [r for r in rows if r['split'] == split and 'critical_edge' in r['scores']]
        # Fixed input-order selection, not selected for model effect.
        for y in (0, 1):
            rec = next(r for r in selected if r['scores']['base']['label'] == y)
            result.append(rec)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--per-split', type=int, default=512)
    parser.add_argument('--output', type=Path, default=ROOT / 'studies/direct_interventions/results')
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    if args.per_split < 2 or args.per_split % 2:
        parser.error('--per-split must be positive/even')
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    data = json.loads((RUNS / 'direct_data.json').read_text())
    cases, eligibility = construct(data, args.per_split)
    print('CASES VALIDATED', len(cases), eligibility, flush=True)
    if args.check_only:
        return
    if args.output.exists():
        raise FileExistsError(f'refusing to overwrite {args.output}')
    args.output.mkdir(parents=True)
    write_json(args.output / 'inputs.json', cases)
    started = time.time()
    summary = dict(seed=SEED, inference_only=True, precision='fp32', per_split=args.per_split,
                   eligibility=eligibility, models={})
    flat = [(i, name, rec) for i, case in enumerate(cases) for name, rec in case['variants'].items()]
    for job in ('direct_single', 'direct_multi'):
        path = RUNS / job / 'checkpoints/latest.pt'
        before = hashlib.sha256(path.read_bytes()).hexdigest()
        cfg = load_config([RUNS / job / 'resolved_config.yaml'])
        model = build_model(cfg.model, len(TOK), TOK.pad).cuda().eval()
        blob = torch.load(path, map_location='cpu', weights_only=False)
        assert blob['step'] == 10000
        model.load_state_dict(blob['model'])
        del blob
        predictions = score(model, [rec for _, _, rec in flat])
        rows = [dict(split=c['split'], graph_id=c['graph_id'], original_index=c['original_index'],
                     scores={}) for c in cases]
        for (i, name, rec), pred in zip(flat, predictions):
            rows[i]['scores'][name] = pred
        report = summarize(rows)
        # Check whether tiny probabilities / output bias depend on BF16 rounding.
        original = [case['variants']['base'] for case in cases]
        repeated = score(model, original, batch=64)
        alternate = score(model, original, bf16=True)
        baseline = [row['scores']['base'] for row in rows]
        repeat_delta = [abs(a['p_yes'] - b['p_yes']) for a, b in zip(baseline, repeated)]
        report['fp32_batch_check'] = dict(n=len(original),
            prediction_disagreements=sum(a['pred'] != b['pred'] for a, b in zip(baseline, repeated)),
            mean_abs_delta_p=float(np.mean(repeat_delta)), max_abs_delta_p=float(max(repeat_delta)))
        report['bf16_check'] = dict(n=len(original),
            prediction_disagreements=sum(a['pred'] != b['pred'] for a, b in zip(baseline, alternate)),
            mean_abs_delta_p=float(np.mean([abs(a['p_yes'] - b['p_yes']) for a, b in zip(baseline, alternate)])),
            bf16_accuracy=float(np.mean([a['pred'] == a['label'] for a in alternate])),
            bf16_yes_rate=float(np.mean([a['pred'] == 1 for a in alternate])))
        # Existing training files must remain byte-identical.
        assert before == hashlib.sha256(path.read_bytes()).hexdigest()
        report.update(checkpoint=str(path), checkpoint_sha256=before, step=10000)
        summary['models'][job] = report
        with (args.output / f'{job}_predictions.jsonl').open('w') as f:
            for row in rows:
                f.write(json.dumps(row) + '\n')
        selected_examples = examples(rows)
        case_lookup = {(c['split'], c['graph_id']): c for c in cases}
        for item in selected_examples:
            item['inputs'] = case_lookup[item['split'], item['graph_id']]['variants']
        write_json(args.output / f'{job}_examples.json', selected_examples)
        print('MODEL DONE', job, 'seconds', time.time() - started, flush=True)
        del model
        torch.cuda.empty_cache()
    summary['seconds'] = time.time() - started
    write_json(args.output / 'summary.json', summary)
    write_json(args.output / 'completed.json', dict(seconds=summary['seconds'], cases=len(cases),
               scored_prompts_per_model=len(flat), training_steps=0))
    print('ALL DONE', summary['seconds'], flush=True)


if __name__ == '__main__':
    main()
