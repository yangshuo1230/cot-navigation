"""CPU checks of actual causal routing, not just tensor assignment."""
import unittest
from types import SimpleNamespace
import torch

from engine import Snapshot, sites, pair_cases, residual_patch, different_state, score_continuation
from cot_navigation.models.transformer import TransformerLM
from cot_navigation.evaluation.rollout import cached_logits


class PatchTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(19)
        torch.set_num_threads(1)
        self.model = TransformerLM(40, 3, 1, 16, 32, 32).eval()
        self.tokens = torch.tensor([[1, 2, 3, 4, 5]])

    def capture(self, tokens):
        result = {}
        with torch.no_grad():
            logits = self.model(tokens, capture=result)
        return logits, result['resid']

    def test_same_token_and_self_identity(self):
        clean, activations = self.capture(self.tokens)
        with torch.no_grad(), residual_patch(self.model, 1, {0: ([1, 3], activations[1][0, [1, 3]])}):
            patched = self.model(self.tokens)
        torch.testing.assert_close(clean, patched, rtol=0, atol=0)
        self.assertEqual(len(self.model.blocks[1]._forward_hooks), 0)

    def test_past_final_layer_is_causally_disconnected(self):
        clean, acts = self.capture(self.tokens)
        with torch.no_grad(), residual_patch(self.model, 2, {0: ([1], acts[2][0, [1]] + 5)}):
            patched = self.model(self.tokens)
        torch.testing.assert_close(clean[:, -1], patched[:, -1], rtol=0, atol=0)

    def test_earlier_layer_past_patch_reaches_later_predictions(self):
        clean, acts = self.capture(self.tokens)
        replacement = torch.randn_like(acts[0][0, [1]]) * 3
        with torch.no_grad(), residual_patch(self.model, 0, {0: ([1], replacement)}):
            patched = self.model(self.tokens)
        self.assertGreater((clean[:, -1] - patched[:, -1]).abs().max().item(), 1e-5)
        # Causality: an intervention cannot affect earlier positions.
        torch.testing.assert_close(clean[:, 0], patched[:, 0], rtol=0, atol=0)

    def test_rebuilt_cache_matches_patched_full_forward(self):
        _, acts = self.capture(self.tokens)
        replacement = torch.randn_like(acts[0][0, [1]])
        extended = torch.cat([self.tokens, torch.tensor([[6]])], 1)
        with torch.no_grad(), residual_patch(self.model, 0, {0: ([1], replacement)}):
            full = self.model(extended)[:, -1]
        cache = self.model.new_cache(1, 'cpu', torch.float32)
        with torch.no_grad():
            with residual_patch(self.model, 0, {0: ([1], replacement)}):
                self.model(self.tokens, cache=cache, cache_mode='prefill')
            cached = self.model(torch.tensor([[6]]), positions=torch.tensor([[5]]), cache=cache, cache_mode='decode')[:, -1]
        torch.testing.assert_close(full, cached, rtol=1e-5, atol=1e-6)

    def test_hook_cleanup_after_error(self):
        with self.assertRaises(RuntimeError):
            with residual_patch(self.model, 0, {}):
                raise RuntimeError('expected')
        self.assertEqual(len(self.model.blocks[0]._forward_hooks), 0)

    def test_sites_exclude_action_argument_and_groups_separate(self):
        r = dict(start=5, end=15, separators=[7, 10, 12], frontier=[1, 2], visited=[3, 4], candidates=[6], action=[8, 9])
        s = Snapshot(SimpleNamespace(index=1), list(range(20)), r, {}, 0)
        self.assertEqual(sites(s, 'visited', 'nodes'), [8, 9])
        self.assertEqual(sites(s, 'visited', 'operator'), [13])
        self.assertNotIn(14, sites(s, 'visited', 'all'))
        self.assertEqual(sites(s, 'visited', 'all'), [8, 9, 10, 13, 15])

    def test_pair_requires_both_other_fields_to_differ(self):
        def make(index, ff, vv, cc):
            r = dict(frontier=ff, visited=vv, candidates=cc, action=[8], end=4)
            return Snapshot(SimpleNamespace(index=index), [0] * 8, r, {}, 0)
        recipient = make(1, [1], [1, 2], [3])
        invalid = make(2, [1], [1, 4], [3])
        good = make(3, [1], [1, 5], [4])
        pairs, _ = pair_cases([recipient], [invalid, good], 2)
        frontier = [p for p in pairs if p[0] == 'frontier']
        self.assertEqual(len(frontier), 1)
        self.assertEqual(frontier[0][2].example.index, 3)

    def test_permutation_is_not_a_semantically_different_visited_state(self):
        self.assertFalse(different_state({'visited': [1, 2]}, {'visited': [2, 1]}, 'visited'))
        self.assertTrue(different_state({'frontier': [1, 2]}, {'frontier': [2, 1]}, 'frontier'))

    def test_within_record_readout_excludes_clamped_fields(self):
        ids = {'|': 4, ';': 5, '<PUSH>': 8, '<EOS>': 2, '<STOP>': 3}
        am = SimpleNamespace(tok=SimpleNamespace(id=ids.__getitem__, yes=6, no=7))
        tokens = [0, 10, 4, 10, 4, 11, 4, 8, 11, 5]
        r = dict(start=1, end=9, separators=[2, 4, 6], frontier=[10], visited=[10], candidates=[11], action=[8, 11])
        s = Snapshot(SimpleNamespace(index=1), tokens, r, r, 0, cut_at=5)
        scores = score_continuation(am, s, s, [11, 4, 8, 11, 5])
        self.assertTrue(scores['action_legal'])
        self.assertFalse(scores['visited_is_generated'])
        self.assertFalse(scores['frontier_is_generated'])
        self.assertTrue(scores['candidates_is_generated'])


if __name__ == '__main__':
    unittest.main()
