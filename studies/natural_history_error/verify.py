"""CPU-only full reconstruction with an independent state-machine checker."""
from collections import Counter
import gzip
import importlib.util
import json
from pathlib import Path


def first_bad(ex, protocol, tok, sequence):
    """Independent token checker, accepting all legal set orders/PUSH choices."""
    frontier, visited = [ex.nodes[ex.source]], {ex.nodes[ex.source]}
    inv = {v:i for i,v in enumerate(ex.nodes)}
    i, n = len(ex.prompt), len(sequence)
    bar, semi = tok.id('|'), tok.id(';')
    while i < n:
        if not frontier:
            return i
        current = frontier[0 if protocol=='bfs_state' else -1]
        for token in frontier + [bar]:
            if i >= n: return None
            if sequence[i] != token: return i
            i += 1
        candidates = {ex.nodes[v] for v in ex.adj[inv[current]]} - visited
        for members in (visited,candidates):
            left = set(members)
            while left:
                if i >= n: return None
                if sequence[i] not in left: return i
                left.remove(sequence[i]); i += 1
            if i >= n: return None
            if sequence[i] != bar: return i
            i += 1
        name = 'FOUND' if current==ex.nodes[ex.target] else 'PUSH' if candidates else 'POP' if len(frontier)>1 else 'EXHAUST'
        if i >= n: return None
        if sequence[i] != tok.id(f'<{name}>'): return i
        i += 1
        arg = None
        if name=='PUSH':
            if i >= n: return None
            arg = sequence[i]
            if arg not in candidates: return i
            i += 1
        if i >= n: return None
        if sequence[i] != semi: return i
        i += 1
        if name in ('FOUND','EXHAUST'):
            return i if i < n else None
        if name=='PUSH':
            frontier.append(arg); visited.add(arg)
        else:
            frontier.pop(0 if protocol=='bfs_state' else -1)
    return None


def main():
    from cot_navigation.analysis.scale_followup import make_adapter
    from cot_navigation.analysis.common import write_json,sha256
    root=Path.cwd(); folder=root/'studies/natural_history_error'
    spec=importlib.util.spec_from_file_location('study',folder/'run.py')
    study=importlib.util.module_from_spec(spec); spec.loader.exec_module(study)
    hist=study.history_module(root)
    reports=[]
    for path in sorted((folder/'results/models').glob('*.json')):
        result=json.loads(path.read_text()); entry=result['entry']
        assert result['complete'] and result['checkpoint_unchanged']
        assert result['source_sha256']==sha256(folder/'run.py')
        am=make_adapter(entry,root,'cpu')
        counts=Counter(); sets=[]; coverage={}
        for split in ('train','val','test'):
            with gzip.open(folder/'rollouts'/f"{entry['id']}_{split}.json.gz",'rt') as f:
                cache=json.load(f)
            events=[]; query_ids=cache['query_ids']; sets.append(set(query_ids)); categories=Counter()
            for query,out in zip(query_ids,cache['outputs']):
                ex=am.example(query,split)
                full=ex.prompt+out['trace_tokens']
                stop_tokens={am.tok.yes,am.tok.no,am.tok.id('<EOS>')}
                if '<STOP>' in am.tok.stoi: stop_tokens.add(am.tok.id('<STOP>'))
                end=next((i for i in range(len(ex.prompt),len(full)) if full[i] in stop_tokens),len(full))
                first=first_bad(ex,am.protocol,am.tok,full[:end])
                if first is None:
                    categories['no_semantic_error_in_generated_prefix']+=1
                elif full[first] in {am.tok.id('<FOUND>'),am.tok.id('<EXHAUST>')}:
                    categories['first_error_is_excluded_terminal_action']+=1
                else:
                    categories['first_error_is_nonterminal']+=1
                event=study.natural_event(am,ex,ex.prompt+out['trace_tokens'],hist)
                if event is None: continue
                events.append(event)
                assert first_bad(ex,am.protocol,am.tok,event['wrong'])==event['error_position']
                bad=first_bad(ex,am.protocol,am.tok,event['correct'])
                assert (len(event['correct']) if bad is None else bad)==event['repair_legal_until']
                for anchor,valid in event['valid_anchors'].items():
                    if valid:
                        pos=event['positions'][anchor]
                        assert first_bad(ex,am.protocol,am.tok,event['correct'][:pos+1]) is None
                        counts[anchor]+=1
            summary=result['dataset'][split]
            assert len(events)==summary['pairs']
            assert summary['pre_error_max_difference']==0 if events else True
            assert [e['query'] for e in events]==summary['event_query_ids']
            assert {a:sum(e['valid_anchors'][a] for e in events) for a in hist.ANCHORS}==summary['valid_pairs']
            coverage[split]=dict(categories)
        assert not any(sets[i]&sets[j] for i in range(3) for j in range(i))
        for fit in result['fits']:
            for row in fit['test']:
                assert 0 <= row['auroc'] <= 1
                if row['anchor']=='pre_error':
                    assert abs(row['auroc']-.5)<1e-12
                if row['layer']=='current_token' and row['anchor']!='error':
                    assert abs(row['auroc']-.5)<1e-12
        reports.append(dict(model=entry['id'],pairs=sum(result['dataset'][s]['pairs'] for s in ('train','val','test')),
                            valid_anchors_all_splits=dict(counts),checkpoint_unchanged=True,
                            generation_coverage=coverage))
        print('VERIFIED',entry['id'],flush=True)
        del am
    assert len(reports)==4
    write_json(folder/'verification.json',dict(status='PASS',models=reports,
        independent_oracle='full state-machine token reconstruction',
        source_sha256=sha256(folder/'run.py')))


if __name__=='__main__':
    main()
