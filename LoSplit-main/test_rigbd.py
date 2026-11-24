"""Regression checks for concrete RIGBD failure modes."""
import unittest
from types import SimpleNamespace
import torch
import torch.nn.functional as F
from rigbd_utils import sample_edges, select_candidates, split_training_nodes, recovery_loss
from experiment_defenses import rigbd_validation_key


class RigbdTests(unittest.TestCase):
    def setUp(self):
        # Undirected 0--1--2; remove just the 2 -> 1 message.
        self.edges = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
        self.weights = torch.tensor([.25, .25, 2.5, 2.5])
        self.keep = torch.tensor([True, True, True, False])

    def test_receiver_restoration_does_not_protect_leaf_message(self):
        e, w = sample_edges(self.edges, self.weights, 3, .5, keep=self.keep)
        self.assertFalse(((e[0] == 2) & (e[1] == 1)).any())
        self.assertTrue(torch.bincount(e[1], minlength=3).gt(0).all())
        legacy, _ = sample_edges(self.edges, self.weights, 3, .5,
                                 restore_axis=0, keep=self.keep)
        self.assertTrue(((legacy[0] == 2) & (legacy[1] == 1)).any())

    def test_restoration_preserves_weights_and_never_revives_zero_edges(self):
        edges = torch.cat([self.edges, torch.tensor([[0], [2]])], 1)
        weights = torch.cat([self.weights, torch.zeros(1)])
        e, w = sample_edges(edges, weights, 4, 1, keep=torch.zeros(4, dtype=torch.bool))
        self.assertTrue(torch.equal(e, self.edges))
        self.assertTrue(torch.equal(w, self.weights))

    def test_sampling_stream_is_independent_of_global_rng(self):
        a = torch.Generator().manual_seed(10)
        b = torch.Generator().manual_seed(10)
        expected = [sample_edges(self.edges, self.weights, 3, .5, a) for _ in range(4)]
        torch.rand(200)
        actual = [sample_edges(self.edges, self.weights, 3, .5, b) for _ in range(4)]
        for pair, other in zip(expected, actual):
            self.assertTrue(all(torch.equal(x, y) for x, y in zip(pair, other)))

    def test_integer_endpoint_probabilities_and_empty_graph(self):
        for ratio in (0, 1):
            e, w = sample_edges(self.edges, self.weights, 3, ratio)
            self.assertTrue(torch.equal(e, self.edges))
            self.assertTrue(torch.equal(w, self.weights))
        e, w = sample_edges(torch.empty((2,0), dtype=torch.long), torch.empty(0), 3, .5)
        self.assertEqual(e.shape, (2,0))
        self.assertEqual(w.numel(), 0)

    def test_label_boundary_has_no_minus_one_error(self):
        labels = torch.tensor([0, 0, 1, 0, 2, 1])
        ids, info = select_candidates(torch.arange(6., 0., -1), labels, torch.arange(6))
        self.assertEqual(ids.tolist(), [0, 1, 2, 3])
        self.assertEqual(info['InferredTarget'], 0)
        self.assertFalse(info['UsesOracleCount'])

    def test_first_boundary_does_not_select_everything(self):
        ids, info = select_candidates(torch.tensor([3., 2., 1.]), torch.tensor([1, 2, 0]),
                                     torch.arange(3), rule='known_label', target_class=0)
        self.assertEqual(ids.numel(), 0)

    def test_ties_respect_strict_threshold(self):
        ids, _ = select_candidates(torch.tensor([5., 4., 3., 3.]), torch.tensor([0, 0, 1, 2]), torch.arange(4))
        self.assertEqual(ids.tolist(), [0, 1])

    def test_constant_scores_do_not_select_arbitrary_labels(self):
        ids, _ = select_candidates(torch.zeros(4), torch.zeros(4, dtype=torch.long), torch.arange(4))
        self.assertEqual(ids.numel(), 0)

    def test_tensor_indices_are_partitioned_by_value(self):
        retained, selected = split_training_nodes(torch.tensor([1, 2, 3]), torch.tensor([2]), 'cpu')
        self.assertEqual(retained.tolist(), [1, 3])
        self.assertEqual(selected.tolist(), [2])
        self.assertEqual(retained.dtype, torch.long)

    def test_empty_groups_have_finite_loss_and_gradients(self):
        for ids in ([], [0, 1, 2]):
            logits = torch.tensor([[1., 0.], [0., 1.], [2., 0.]], requires_grad=True)
            retained, selected = split_training_nodes(torch.arange(3), ids, 'cpu')
            loss = recovery_loss(F.log_softmax(logits, 1), torch.tensor([0, 1, 0]), retained, selected)
            loss.backward()
            self.assertTrue(torch.isfinite(loss))
            self.assertTrue(torch.isfinite(logits.grad).all())

    def test_recovery_matches_losplit_objective(self):
        output = F.log_softmax(torch.tensor([[1., 0.], [0., 1.], [2., 0.]]), 1)
        labels = torch.tensor([0, 1, 0])
        retained, selected = split_training_nodes(torch.arange(3), [2], 'cpu')
        expected = F.nll_loss(output[:2], labels[:2]) + F.softmax(output[2:], 1)[0, 0]
        self.assertTrue(torch.equal(expected, recovery_loss(output, labels, retained, selected)))

    def test_grid_choice_does_not_depend_on_test_outcomes(self):
        rows = [dict(ValidationAccuracy=.8, ASR=.99, K=20, b=.5),
                dict(ValidationAccuracy=.7, ASR=0., K=10, b=.3)]
        self.assertIs(max(rows, key=rigbd_validation_key), rows[0])

    def test_leaf_feature_is_invisible_at_two_steps_without_self_loops(self):
        from torch_geometric.nn import GCNConv
        torch.manual_seed(10)
        x = torch.rand(3, 4)
        scrubbed = x.clone()
        scrubbed[2] = 0
        for loops in (False, True):
            first = GCNConv(4, 4, add_self_loops=loops, bias=False)
            second = GCNConv(4, 2, add_self_loops=loops, bias=False)
            a = second(first(x, self.edges), self.edges)[1]
            b = second(first(scrubbed, self.edges), self.edges)[1]
            self.assertEqual(torch.equal(a, b), not loops)


if __name__ == '__main__':
    unittest.main()
