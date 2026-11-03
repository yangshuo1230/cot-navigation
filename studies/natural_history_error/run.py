"""Frozen-model natural first-error readout; paired, query-disjoint evaluation.

The wrong branch is genuinely greedy-generated, including its continuation.
The control repairs ONE token and replays that same continuation. At each
anchor keep only pairs whose repaired prefix is semantically valid through
that anchor. Thus long-delay conclusions cannot use a newly broken control.
No navigation training, weights, activation caches, or W&B runs are written.
"""
import argparse
from collections import Counter
import gzip
import importlib.util
import json
import os
from pathlib import Path
import time


def history_module(root):
    spec = importlib.util.spec_from_file_location('history', root/'studies/history_error/run.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def natural_event(am, ex, sequence, hist):
    """Uniform legal repair chosen without looking at future continuation."""
    import numpy as np
    from cot_navigation.analysis.error_detection import semantic_slots
    p = len(ex.prompt)
    terminal = {am.tok.yes, am.tok.no, am.tok.id('<EOS>'),
                am.tok.id('<FOUND>'), am.tok.id('<EXHAUST>')}
    if '<STOP>' in am.tok.stoi:
        terminal.add(am.tok.id('<STOP>'))
    end = next((i for i in range(p, len(sequence)) if sequence[i] in terminal), len(sequence))
    wrong = list(sequence[:end])
    slots = list(semantic_slots(ex, am.protocol, am.tok, wrong[p:]))
    bad = next(((i, good, field) for i, good, field in slots if wrong[p+i] not in good), None)
    if bad is None:
        return None
    at, good, field = bad
    if not good:
        return None
    tau = p + at
    rng = np.random.default_rng(20261022031 + ex.index)
    repair = int(rng.choice(sorted(good)))
    correct = list(wrong)
    correct[tau] = repair
    first_bad = hist.token_legality(ex, am.protocol, am.tok, correct)
    legal_end = len(correct) if first_bad is None else first_bad
    # Locate later VIS endings by record delimiters, only retaining those
    # reached by the fully legal repaired parse. No oracle after a bad state.
    semi, bar = am.tok.id(';'), am.tok.id('|')
    record_start = p
    for i in range(p, tau):
        if wrong[i] == semi:
            record_start = i + 1
    frontier = []
    for token in wrong[record_start:tau+1]:
        if token == bar or token not in ex.nodes:
            break
        frontier.append(token)
    current = frontier[0 if am.protocol == 'bfs_state' else -1] if frontier else ex.nodes[ex.source]
    positions = {'pre_error': tau-1, 'error': tau,
                 **{f'token_{k}': tau+k for k in (1, 4, 16, 64)}}
    later = [i+1 for i in range(tau, len(correct)) if correct[i] == semi]
    for k in (1, 2, 4):
        pos = len(correct)
        if len(later) >= k:
            start = later[k-1]
            bars = [i for i in range(start, len(correct))
                    if correct[i] == bar]
            next_semi = next((i for i in range(start, len(correct)) if correct[i] == semi), len(correct))
            if len(bars) >= 2 and bars[1] < next_semi:
                pos = bars[1]
        positions[f'record_{k}_vis'] = pos
    valid = {a: pos < legal_end and pos < len(correct) for a, pos in positions.items()}
    positions = {a: min(pos, len(correct)-1) for a, pos in positions.items()}
    assert hist.token_legality(ex, am.protocol, am.tok, wrong) == tau
    assert valid['pre_error'] and valid['error']
    assert [i for i, (a,b) in enumerate(zip(correct,wrong)) if a != b] == [tau]
    return dict(query=ex.index, original_label=ex.label, correct=correct, wrong=wrong,
                error_position=tau, positions=positions, valid_anchors=valid,
                current_token=current, prompt_length=p, graph_control=False,
                field=field, kind='in_graph_node' if repair in ex.nodes and wrong[tau] in ex.nodes else 'format_or_other',
                repair_legal_until=legal_end, repair_token=repair,
                wrong_token=wrong[tau], generated_tokens=len(sequence)-p)


def filtered(data, anchor, hist, kind=None, field=None):
    import numpy as np
    if data is None:
        return None
    ai = hist.ANCHORS.index(anchor)
    mask = data['valid'][:, ai].copy()
    if kind:
        mask &= data['kind'] == kind
    if field:
        mask &= data['field'] == field
    if not mask.any():
        return None
    out = {k: v[mask] for k,v in data.items() if isinstance(v, np.ndarray)}
    begin = ai * len(hist.HEADS)
    out['x'] = np.ascontiguousarray(out['x'][:, begin:begin+len(hist.HEADS)])
    return out


def score(data, logits, anchor, hist):
    if data is None:
        return []
    from cot_navigation.analysis.error_awareness import evaluate
    results = evaluate(data, logits)
    for layer, row in zip(hist.HEADS, results):
        row.update(anchor=anchor, layer=layer)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu', type=int, default=7)
    parser.add_argument('--models', nargs='+', default=['context_l10_dfs_state','context_l10_bfs_state','fixed_l10_dfs_state','fixed_l10_bfs_state'])
    parser.add_argument('--train-queries', type=int, default=192)
    parser.add_argument('--val-queries', type=int, default=96)
    parser.add_argument('--test-queries', type=int, default=256)
    parser.add_argument('--max-new', type=int, default=1024)
    parser.add_argument('--steps', type=int, default=600)
    parser.add_argument('--seeds', nargs='+', type=int, default=[0,1,2])
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    os.environ['WANDB_MODE'] = 'disabled'
    import numpy as np
    import torch
    from cot_navigation.analysis.scale_followup import make_adapter, generate
    from cot_navigation.analysis.error_awareness import train_probe, predict
    from cot_navigation.analysis.common import write_json, sha256
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    root = Path.cwd()
    folder = root/'studies/natural_history_error'
    hist = history_module(root)
    entries = json.loads((root/'studies/scale_mechanistic/results/selection.json').read_text())['models']
    for entry in entries:
        if entry['id'] not in args.models:
            continue
        output = folder/'results/models'/f"{entry['id']}.json"
        if output.exists() and json.loads(output.read_text()).get('complete'):
            continue
        started = time.time()
        print('MODEL', entry['id'], flush=True)
        am = make_adapter(entry, root, 'cuda:0')
        assert am.checkpoint['sha256'] == entry['checkpoint_sha256']
        datasets, summaries, saved, query_sets = {}, {}, {}, []
        for split in ('train','val','test'):
            examples = am.examples(split, getattr(args, split+'_queries'))
            query_sets.append({ex.index for ex in examples})
            cache = folder/'rollouts'/f"{entry['id']}_{split}.json.gz"
            if cache.exists():
                with gzip.open(cache, 'rt') as f:
                    generated = json.load(f)
                assert generated['checkpoint_sha256'] == entry['checkpoint_sha256']
                assert generated['query_ids'] == [ex.index for ex in examples]
                assert generated['max_new'] == args.max_new
                outs = generated['outputs']
            else:
                outs = []
                for i in range(0,len(examples),32):
                    chunk = examples[i:i+32]
                    outs.extend(generate(am, [ex.prompt for ex in chunk], cap=args.max_new, batch=32))
                    print(' GENERATE', split, len(outs), '/', len(examples), flush=True)
                cache.parent.mkdir(parents=True, exist_ok=True)
                with gzip.open(cache, 'wt') as f:
                    json.dump(dict(checkpoint_sha256=entry['checkpoint_sha256'],
                        query_ids=[ex.index for ex in examples], max_new=args.max_new, outputs=outs), f)
            events = []
            for ex, out in zip(examples, outs):
                ev = natural_event(am, ex, ex.prompt+out['trace_tokens'], hist)
                if ev:
                    events.append(ev)
            summary = dict(queries=len(examples), pairs=len(events),
                query_ids=[ex.index for ex in examples], event_query_ids=[e['query'] for e in events],
                answer_correct=sum(out['pred']==ex.label for ex,out in zip(examples,outs)),
                generation_limits=sum(out['generation_limit_reached'] for out in outs),
                fields=dict(Counter(e['field'] for e in events)),
                kinds=dict(Counter(e['kind'] for e in events)),
                valid_pairs={a:sum(e['valid_anchors'][a] for e in events) for a in hist.ANCHORS},
                mean_delays={a:float(np.mean([e['positions'][a]-e['error_position'] for e in events if e['valid_anchors'][a]]))
                             if any(e['valid_anchors'][a] for e in events) else None for a in hist.ANCHORS})
            print(' DATA', split, summary, flush=True)
            data = hist.extract(am, events, 8192) if events else None
            if data is not None:
                lookup = {e['query']:e for e in events}
                data['kind'] = np.array([lookup[int(q)]['kind'] for q in data['query']])
                data['field'] = np.array([lookup[int(q)]['field'] for q in data['query']])
                summary['pre_error_max_difference'] = data['pre_error_max_difference']
            datasets[split], summaries[split], saved[split] = data, summary, events
        assert not any(query_sets[i]&query_sets[j] for i in range(3) for j in range(i))
        fits = []
        # Strict natural-event training uses only legally repaired pairs at
        # EACH anchor. Long-delay scarcity is reported, never filled in.
        for anchor in hist.ANCHORS:
            sets = {s:filtered(d,anchor,hist) for s,d in datasets.items()}
            sizes = {s:0 if d is None else len(d['y'])//2 for s,d in sets.items()}
            if sizes['train'] < 20 or sizes['val'] < 8 or sizes['test'] < 8:
                print(' SKIP NATURAL FIT', anchor, sizes, flush=True)
                continue
            for hidden, architecture in ((0,'linear'),(64,'mlp64')):
                for seed in args.seeds:
                    fitted = train_probe(sets['train'],sets['val'],hidden,seed,args.steps,512,'cuda:0')
                    fits.append(dict(training='natural_repair',architecture=architecture,seed=seed,
                        anchor=anchor, sizes=sizes,
                        test=score(sets['test'],predict(fitted,sets['test']),anchor,hist)))
                    del fitted
            print(' FIT NATURAL', anchor, sizes, flush=True)
        # Recreate prior CAND-only training; identical source/splits/hyperparams.
        synth = {}
        for split, count in (('train',768),('val',192)):
            events, _, _ = hist.build_events(am,split,count,max(count*4,512))
            synth[split] = hist.extract(am,events,8192)
        for hidden,architecture in ((0,'linear'),(64,'mlp64')):
            for seed in args.seeds:
                fitted = train_probe(synth['train'],synth['val'],hidden,seed,args.steps,512,'cuda:0')
                data = datasets['test']
                if data is not None:
                    logits = predict(fitted,data)
                    for anchor in hist.ANCHORS:
                        ai = hist.ANCHORS.index(anchor)
                        for subset in ('all','in_graph_node','candidates'):
                            mask = data['valid'][:,ai].copy()
                            if subset=='in_graph_node': mask &= data['kind']=='in_graph_node'
                            if subset=='candidates': mask &= data['field']=='candidates'
                            selected = filtered(data,anchor,hist,
                                kind='in_graph_node' if subset=='in_graph_node' else None,
                                field='candidates' if subset=='candidates' else None)
                            if selected is not None:
                                fits.append(dict(training='synthetic_transfer',architecture=architecture,
                                    seed=seed,anchor=anchor,subset=subset,
                                    test=score(selected,logits[mask,ai*len(hist.HEADS):(ai+1)*len(hist.HEADS)],anchor,hist)))
                del fitted
        assert sha256(am.run/'checkpoints/best.pt') == entry['checkpoint_sha256']
        write_json(output,dict(complete=True,entry=entry,settings=vars(args),dataset=summaries,
            fits=fits,examples={s:ev[:3] for s,ev in saved.items()},
            source_sha256=sha256(Path(__file__)),history_source_sha256=sha256(root/'studies/history_error/run.py'),
            checkpoint_unchanged=True,elapsed_seconds=time.time()-started))
        print('DONE',entry['id'],round(time.time()-started,1),flush=True)
        del am,datasets,synth
        torch.cuda.empty_cache()


if __name__=='__main__':
    main()
