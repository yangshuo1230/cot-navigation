"""Semantic correctness checks; runs in memory, writes no smoke artifacts."""
import unittest
from types import SimpleNamespace

import numpy as np

from cot_navigation.analysis.common import Example
from cot_navigation.protocols.render import context_prompt, parametric_prompt, render_task
from cot_navigation.protocols.tokenizer import Tokenizer
from data import Factory, history_donor, rename_to_match, validate, reachable
from engine import FIELDS, different_state


class ConstructionTests(unittest.TestCase):
    def adapter(self, context=True, protocol='dfs_state'):
        tok = Tokenizer(32, enable_stop=True)
        adj = [[1, 2], [3], [4], [5], [5], [6], [7], []]
        nodes = [tok.node_id(i) for i in range(8)]
        edges = [(u, v) for u, vs in enumerate(adj) for v in vs]
        prompt = context_prompt(tok, nodes, edges, 0, 7) if context else parametric_prompt(tok, nodes, 0, 7)
        row = render_task(tok, protocol, 1, prompt, adj, 0, 7, nodes)
        ex = Example(0, 'train', 1, adj, nodes, 0, 7, row)
        return SimpleNamespace(tok=tok, protocol=protocol, context=context, world_prefix=[], example=lambda i, s: ex)

    def recipe(self, **overrides):
        return dict(base_split='train', base_index=0, source=0, target=7, priority_seed=3, **overrides)

    def test_suffix_renaming_preserves_full_history_and_query_names(self):
        factory = Factory(self.adapter())
        r = factory.render(self.recipe())[0]
        d = history_donor(r, 'frontier', factory, np.random.default_rng(1))
        self.assertIsNotNone(d)
        self.assertEqual(r.prefix[len(r.example.prompt):], d.prefix[len(d.example.prompt):])
        self.assertEqual(len(r.prefix), len(d.prefix))
        self.assertEqual(r.example.nodes[r.example.source], d.example.nodes[d.example.source])
        self.assertEqual(r.example.nodes[r.example.target], d.example.nodes[d.example.target])
        self.assertEqual(set(r.example.nodes), set(d.example.nodes))
        self.assertNotEqual(set(r.following['candidates']), set(d.following['candidates']))
        self.assertNotEqual(r.example.prompt, d.example.prompt)

    def test_fixed_world_query_counterfactual_is_legal(self):
        factory = Factory(self.adapter(context=False))
        r = factory.render(self.recipe())[0]
        d = history_donor(r, 'visited', factory, np.random.default_rng(1))
        self.assertIsNotNone(d)
        self.assertEqual(r.example.nodes, d.example.nodes)
        self.assertEqual(r.example.adj, d.example.adj)
        self.assertEqual(r.prefix[len(r.example.prompt):], d.prefix[len(d.example.prompt):])
        self.assertEqual(d.following['action'], [factory.am.tok.id('<FOUND>')])

    def test_fixed_world_renaming_is_rejected(self):
        factory = Factory(self.adapter(context=False))
        ids = [factory.am.tok.node_id(i) for i in reversed(range(8))]
        with self.assertRaisesRegex(AssertionError, 'Never rename'):
            factory.render(self.recipe(node_ids=ids))

    def test_inconsistent_displayed_candidate_is_rejected(self):
        factory = Factory(self.adapter())
        rows = factory.render(self.recipe())
        states = [dict(s.record) for s in rows] + [dict(rows[-1].following)]
        states[0]['candidates'] = []
        with self.assertRaises(AssertionError):
            validate(rows[0].example, states, factory.am.tok, False)

    def test_bfs_reference_replay_and_counterfactual(self):
        factory = Factory(self.adapter(context=False, protocol='bfs_state'))
        found = []
        for r in factory.render(self.recipe()):
            d = history_donor(r, 'candidates', factory, np.random.default_rng(2))
            if d is not None:
                found.append((r, d))
                self.assertEqual(d.following['action'], [factory.am.tok.id('<FOUND>')])
        self.assertTrue(found)

    def test_constrained_matching_relabels_the_whole_example(self):
        factory = Factory(self.adapter())
        pool = []
        for seed in range(12):
            # A side-branch target creates legal backtracking states, unlike
            # the straight successful path used by the other fixtures.
            recipe = self.recipe(); recipe['priority_seed'] = seed; recipe['target'] = 2
            pool.extend(factory.render(recipe))
        found = None
        for r in pool:
            for d in pool:
                if len(r.record['visited']) != len(d.record['visited']):
                    continue
                result = rename_to_match(r, d, 'visited', factory, np.random.default_rng(5))
                if result is not None:
                    found = (r, result)
                    break
            if found:
                break
        self.assertIsNotNone(found)
        r, d = found
        self.assertEqual(r.record['visited'], d.record['visited'])
        self.assertTrue(all(different_state(r.record, d.record, f) for f in FIELDS if f != 'visited'))
        self.assertEqual(set(r.example.nodes), set(d.example.nodes))
        self.assertEqual(reachable(r.example.adj, r.example.source, r.example.target), bool(r.example.label))

    def test_engine_including_donor_control_executes_without_files(self):
        import torch
        from cot_navigation.models.transformer import TransformerLM
        from cot_navigation.evaluation.rollout import pad_rows
        from engine import execute
        torch.manual_seed(4); torch.set_num_threads(1)
        am = self.adapter()
        am.device = torch.device('cpu')
        am.model = TransformerLM(len(am.tok), 3, 1, 16, 32, 256).eval().requires_grad_(False)
        def forward(sequences):
            tokens, _ = pad_rows(sequences, am.tok.pad, am.device)
            capture = {}
            logits = am.model(tokens, capture=capture)
            return logits, capture['resid'], None
        am.forward = forward
        factory = Factory(am)
        r = factory.render(self.recipe())[0]
        d = history_donor(r, 'frontier', factory, np.random.default_rng(1))
        with torch.inference_mode():
            rows, _ = execute(am, [('frontier', r, d)], [0, 1], ['nodes', 'bar', 'operator', 'semicolon', 'all'], 8, 8)
        self.assertEqual(sum(row['source'] == 'donor_control' for row in rows), 1)
        controls = [row for row in rows if row['source'] == 'self']
        self.assertTrue(controls)
        self.assertTrue(all(row['probability_max_abs_delta'] == 0 and not row['continuation_changed'] for row in controls))


if __name__ == '__main__':
    unittest.main()
