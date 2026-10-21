"""Semantic labels, many events/query and strictly last-input readout."""
from types import SimpleNamespace
import numpy as np
import torch

from cot_navigation.analysis.error_awareness import synthetic_events, extract, ParallelProbe
from cot_navigation.analysis.error_detection import semantic_slots


class Tok:
    yes, no, pad = 20, 21, 0
    def id(self, name):
        return {'|': 10, ';': 11, '<PUSH>': 12, '<POP>': 13,
                '<FOUND>': 14, '<EXHAUST>': 15, '<EOS>': 22}[name]


def example(index=1):
    # Valid DFS: source 1, target 2, one PUSH and terminal FOUND.
    tokens = [30, 1, 2, 31, 1, 10, 1, 10, 2, 10, 12, 2, 11,
              1, 2, 10, 1, 2, 10, 10, 14, 11, 20, 22]
    return SimpleNamespace(index=index, adj=[[1], [], []], nodes=[1, 2, 3],
        source=0, target=1, prompt=tokens[:4], row=SimpleNamespace(tokens=tokens, prompt_len=4))


def test_synthetic_prefix_labels_and_diversity():
    ex, tok = example(), Tok()
    evs = synthetic_events([ex], 'dfs_state', tok, 40, 42)
    assert len(evs) > 15
    assert {'in_graph_node', 'excess_node', 'premature_delimiter', 'wrong_action'} <= {e['kind'] for e in evs}
    assert len({(e['position'], e['wrong_token'], e['correct_token']) for e in evs}) == len(evs)
    for e in evs:
        assert e['wrong'][:-1] == e['correct'][:-1]
        assert len(e['wrong']) == len(e['correct']) == e['position']+1
        for key, legal in [('correct', True), ('wrong', False)]:
            body = e[key][4:]
            slots = list(semantic_slots(ex, 'dfs_state', tok, body))
            assert len(slots) == len(body)
            assert all(body[i] in good for i, good, _ in slots[:-1])
            assert (body[-1] in slots[-1][1]) == legal


def test_deterministic_query_ids():
    a = synthetic_events([example(1), example(2)], 'dfs_state', Tok(), 30, 8)
    b = synthetic_events([example(1), example(2)], 'dfs_state', Tok(), 30, 8)
    assert a == b
    assert {e['query'] for e in a} == {1, 2}


def test_readout_does_not_append_next_token():
    ex, tok = example(), Tok()
    evs = synthetic_events([ex], 'dfs_state', tok, 12, 9)
    seen = []
    class Fake:
        layers = [3, 4, 5]
        model = SimpleNamespace(embed=torch.nn.Embedding(64, 128))
        def forward(self, seqs):
            seen.extend(seqs)
            h = torch.zeros(len(seqs), max(map(len, seqs)), 128)
            for b, seq in enumerate(seqs):
                for p in range(len(seq)):
                    h[b, p] = sum(seq[:p+1])
            return None, [h]*6, None
    data = extract(Fake(), [ex], evs, 4)
    assert len(seen) == len(evs)*2
    expected = sorted(tuple(e[k]) for e in evs for k in ['correct', 'wrong'])
    assert sorted(map(tuple, seen)) == expected
    assert len(set(data['event'])) == len(evs)
    for event_id in range(len(evs)):
        mask = data['event'] == event_id
        assert sorted(data['y'][mask]) == [0, 1]
    assert np.isfinite(data['x']).all()


def test_mlp_has_nonlinear_independent_heads():
    torch.manual_seed(1)
    probe = ParallelProbe(4, 140, hidden=64, dropout=0).eval()
    x, y = torch.randn(5, 4, 140), torch.randn(5, 4, 140)
    assert probe(x).shape == (5, 4)
    assert not torch.allclose(probe(x+y), probe(x)+probe(y)-probe(torch.zeros_like(x)))
    linear = ParallelProbe(4, 140)
    assert torch.allclose(linear(x+y), linear(x)+linear(y)-linear(torch.zeros_like(x)), atol=1e-5)
