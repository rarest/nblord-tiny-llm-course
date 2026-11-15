import copy
import math
import random
import subprocess
import sys
import unittest

from experiments.multihead_attention import model as baseline
from experiments.transformer_depth import model as depth
from experiments.transformer_depth import probe


class DepthTests(unittest.TestCase):
    def test_reader_probe_can_use_bundled_checkpoint_without_extra_arguments(self):
        import json
        result = subprocess.run([sys.executable, '-m', 'experiments.transformer_depth.probe'],
                                check=True, capture_output=True, text=True)
        self.assertTrue(json.loads(result.stdout)['causal_check']['passed'])

    def test_reader_probe_reports_actual_causal_comparison(self):
        self.assertTrue(callable(getattr(probe, 'inspect', None)))
        if not hasattr(probe, 'inspect'):
            return
        model = depth.from_single(baseline.init_model(12), layers=2)
        result = probe.inspect(model, [1, 2, 3], 2, 5)
        self.assertEqual(result['input_ids'], [1, 2, 3])
        self.assertEqual(result['changed_input_ids'], [1, 2, 5])
        self.assertEqual(len(result['layers']), 2)
        self.assertTrue(result['causal_check']['passed'])

    def test_depth_model_exposes_trainable_interface(self):
        for name in ('from_single', 'forward', 'loss_and_gradients', 'train_step'):
            with self.subTest(name=name):
                self.assertTrue(callable(getattr(depth, name, None)), name)

    def test_one_layer_matches_the_existing_model(self):
        if not hasattr(depth, 'from_single'):
            self.skipTest('stack interface not implemented')
        single = baseline.init_model(12)
        stacked = depth.from_single(single, layers=1)
        tokens, targets = [1, 2, 3], [2, 3, 4]
        self.assertEqual(baseline.forward(single, tokens)['logits'], depth.forward(stacked, tokens)['logits'])
        old_loss, old_grad = baseline.loss_and_gradients(single, [(tokens, targets)])
        new_loss, new_grad = depth.loss_and_gradients(stacked, [(tokens, targets)])
        self.assertAlmostEqual(old_loss, new_loss, places=13)
        for name in baseline.PARAMETERS:
            actual = new_grad['blocks'][0][name] if name in depth.BLOCK_PARAMETERS else new_grad[name]
            self.assertEqual(old_grad[name], actual, name)

    def test_two_independent_layers_preserve_causality_and_update(self):
        if not hasattr(depth, 'from_single'):
            self.skipTest('stack interface not implemented')
        model = depth.from_single(baseline.init_model(12), layers=2)
        self.assertIsNot(model['blocks'][0]['wq'], model['blocks'][1]['wq'])
        first = depth.forward(model, [1, 2, 3])
        changed = depth.forward(model, [1, 2, 5])
        for a, b in zip(first['blocks'], changed['blocks']):
            self.assertEqual(a['after_feedforward'][:2], b['after_feedforward'][:2])
        self.assertEqual(first['logits'][:2], changed['logits'][:2])
        before = copy.deepcopy(model)
        loss, grads = depth.train_step(model, [([1, 2, 3], [2, 3, 4])], 0.1)
        self.assertTrue(math.isfinite(loss))
        for i in range(2):
            self.assertNotEqual(before['blocks'][i]['wq'], model['blocks'][i]['wq'])
            self.assertTrue(all(math.isfinite(v) for row in grads['blocks'][i]['wq'] for v in row))

    def test_extra_layer_continues_initial_random_sequence(self):
        single = baseline.init_model(12, seed=23)
        stacked = depth.from_single(single, layers=2, seed=23)
        rng = random.Random(23)
        for name in baseline.MATRICES:
            for row in single[name]:
                for _ in row:
                    rng.uniform(-0.25, 0.25)
        self.assertEqual(stacked['blocks'][1]['wq'][0][0], rng.uniform(-0.25, 0.25))
        self.assertNotEqual(stacked['embedding'][0], stacked['blocks'][1]['wq'][0])

    def test_both_layers_gradients_match_finite_difference(self):
        if not hasattr(depth, 'from_single'):
            self.skipTest('stack interface not implemented')
        model = depth.from_single(baseline.init_model(12), layers=2)
        examples = [([1, 2, 3], [2, 3, 4])]
        _, gradients = depth.loss_and_gradients(model, examples)
        epsilon = 1e-5
        for layer in range(2):
            for name in ('wq', 'wk', 'wv', 'wa', 'w1', 'w2', 'ln1_gain', 'ln2_gain'):
                weight = model['blocks'][layer][name][0] if name in baseline.MATRICES else model['blocks'][layer][name]
                analytic = gradients['blocks'][layer][name][0][0] if name in baseline.MATRICES else gradients['blocks'][layer][name][0]
                initial = weight[0]
                weight[0] = initial + epsilon
                plus = depth.loss_and_gradients(model, examples)[0]
                weight[0] = initial - epsilon
                minus = depth.loss_and_gradients(model, examples)[0]
                weight[0] = initial
                with self.subTest(layer=layer, parameter=name):
                    self.assertAlmostEqual(analytic, (plus-minus)/(2*epsilon), places=7)

    def test_shared_block_accumulates_both_uses_and_updates_once(self):
        tied = depth.from_single(baseline.init_model(12), layers=1)
        tied['blocks'].append(tied['blocks'][0])
        independent = copy.deepcopy(tied)
        independent['blocks'][1] = copy.deepcopy(independent['blocks'][0])
        examples = [([1, 2, 3], [2, 3, 4])]
        _, independent_grad = depth.loss_and_gradients(independent, examples)
        _, shared_grad = depth.loss_and_gradients(tied, examples)
        for name in depth.BLOCK_PARAMETERS:
            groups = [g[name] if name in baseline.MATRICES else [g[name]]
                      for g in (shared_grad['blocks'][0], *independent_grad['blocks'])]
            for shared_row, first_row, second_row in zip(*groups):
                for shared, first, second in zip(shared_row, first_row, second_row):
                    self.assertTrue(math.isclose(shared, first+second, rel_tol=1e-8, abs_tol=1e-10), name)
        expected = (independent_grad['blocks'][0]['wq'][0][0]
                    + independent_grad['blocks'][1]['wq'][0][0])
        self.assertAlmostEqual(shared_grad['blocks'][0]['wq'][0][0], expected, places=12)
        weight = tied['blocks'][0]['wq'][0][0]
        depth.train_step(tied, examples, 0.1)
        self.assertAlmostEqual(tied['blocks'][0]['wq'][0][0], weight - 0.1*expected, places=12)


if __name__ == '__main__':
    unittest.main()
