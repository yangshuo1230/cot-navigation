"""Independent replay of saved legal pairs, plus checkpoint/split provenance."""
import json
import os
from pathlib import Path
import sys


def replay_prefix(ex, prefix, tok, fifo):
    from cot_navigation.analysis.common import records
    body = prefix[len(ex.prompt):]
    adj = {ex.nodes[u]: {ex.nodes[v] for v in vs} for u, vs in enumerate(ex.adj)}
    frontier = [ex.nodes[ex.source]]
    visited = set(frontier)
    end = len(ex.prompt)
    for r in records(prefix, len(ex.prompt), tok):
        assert r['frontier'] == frontier
        assert set(r['visited']) == visited and len(r['visited']) == len(visited)
        current = frontier[0 if fifo else -1]
        eligible = adj[current] - visited
        assert set(r['candidates']) == eligible and len(r['candidates']) == len(eligible)
        # Selected prefixes never include a terminal action.
        if eligible:
            assert r['action'][0] == tok.id('<PUSH>') and len(r['action']) == 2
            node = r['action'][1]
            assert node in eligible
            frontier.append(node); visited.add(node)
        else:
            assert len(frontier) > 1 and r['action'] == [tok.id('<POP>')]
            frontier.pop(0 if fifo else -1)
        end = r['end'] + 1
    tail = prefix[end:]
    bars = [i for i, v in enumerate(tail) if v == tok.id('|')]
    assert len(bars) == 2 and bars[-1] == len(tail)-1
    a, b = bars
    assert tail[:a] == frontier and set(tail[a+1:b]) == visited and b-a-1 == len(visited)
    return frontier, visited, adj[frontier[0 if fifo else -1]]-visited


def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
    os.environ['WANDB_MODE'] = 'disabled'
    root = Path.cwd()
    sys.path.insert(0,str(root/'src'))
    from cot_navigation.analysis.scale_followup import make_adapter
    from cot_navigation.analysis.common import sha256, write_json
    import torch
    torch.set_num_threads(2)
    result_root = root/'studies/valid_history/results'
    previous = root/'studies/scale_mechanistic/results'
    nmodels = nprefix = nchanged = ncontrol = 0
    for path in sorted((result_root/'models').glob('*.json')):
        d = json.loads(path.read_text()); e = d['entry']
        assert d['complete'] and d['checkpoint_unchanged']
        assert sha256(root/e['run']/'checkpoints/best.pt') == e['checkpoint_sha256']
        old = json.loads((previous/'models'/path.name).read_text())
        assert d['query_ids'] == old['query_ids']['test']
        nmodels += 1
        if not e['protocol'].endswith('_state'):continue
        am = make_adapter(e,root,'cuda:1')
        for task in ['changed','same_state']:
            for c in d['pairs'][task].get('examples',[]):
                ex = am.example(c['query'],'test')
                states = {}
                for arm, saved in c['arms'].items():
                    states[arm] = replay_prefix(ex,saved['prefix'],am.tok,am.protocol=='bfs_state')
                    f,v,eligible = states[arm]
                    assert f == saved['frontier'] and v == set(saved['visited'])
                    assert eligible == set(saved['expected_candidates'])
                    nprefix += 1
                fa,va,ca = states['a'];fb,vb,cb = states['b']
                assert fa[0 if am.protocol=='bfs_state' else -1] == fb[0 if am.protocol=='bfs_state' else -1]
                if c['match'] in ['exact_frontier','exact_state']:assert fa == fb
                if c['equal_length']:assert len(c['arms']['a']['prefix']) == len(c['arms']['b']['prefix'])
                if task=='changed':
                    focus = am.tok.id(c['focus'])
                    assert focus in ca and focus in vb and focus not in cb
                    nchanged += 1
                else:
                    assert va == vb and ca == cb
                    ncontrol += 1
        del am;torch.cuda.empty_cache()
    assert nmodels == 60
    report = dict(models=nmodels,checkpoint_hashes_unchanged=True,query_splits_unchanged=True,
        independently_replayed_saved_prefixes=nprefix,changed_pairs=nchanged,same_state_pairs=ncontrol,
        all_generated_prefixes='also certified at construction by independent semantic_slots oracle',
        status='PASS')
    write_json(result_root/'verification.json',report)
    print(report)


if __name__=='__main__':main()
