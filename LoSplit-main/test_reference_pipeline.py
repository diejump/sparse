"""Checks of label access, five metrics, graph operators and portable bundles."""
from dataclasses import replace
import math
from pathlib import Path
import tempfile
import unittest

import torch

from experiment_defenses import confusion
from models.ABP_Local import graph_views
from models.ABP_Reference import (build_reference_model, fit_reference_threshold,
    load_reference_bundle, log_output, threshold_prediction)
from models.ABP_Sparse import SparseBranchDetector
from reference_experiment import classification_counts
from reference_training import ReferenceConfig, fit_reference_method, split_reference


def undirected(pairs):
    return torch.tensor([(a,b) for u,v in pairs for a,b in ((u,v),(v,u))], dtype=torch.long).t()


def fixture():
    x = torch.zeros(11, 16)
    x[:10, :2] = .5
    x[10] = 1/16
    e = undirected([(i,(i+1)%10) for i in range(10)] + [(2,10)])
    y = torch.tensor([0,1,0,1,0,1,0,1,-1,-1,-1])
    return x, e, torch.ones(e.size(1)), y, torch.arange(6), torch.tensor([6,7])


class ReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)

    def test_detection_metrics_share_the_same_population(self):
        row = confusion(torch.tensor([1,2,99]), torch.tensor([2,3,88]), torch.arange(5))
        self.assertEqual([row[k] for k in ('TP','FP','FN','TN')], [1,1,1,2])
        self.assertEqual(row['Precision'], .5)
        self.assertEqual(row['Recall'], .5)
        self.assertEqual(row['FPR'], 1/3)
        empty = confusion(torch.tensor([], dtype=torch.long), torch.arange(2), torch.arange(2))
        self.assertIsNone(empty['Precision'])
        self.assertIsNone(empty['FPR'])

    def test_structural_rule_matches_vertex_removal_definition(self):
        import networkx as nx
        from models.ABP_Sparse import project_graph
        for seed in range(40):
            graph = nx.gnp_random_graph(9, .3, seed=seed)
            pairs = list(graph.edges())
            if not pairs:
                continue
            e = undirected(pairs)
            anchors = torch.tensor([0,3,6])
            _,_,blocked = project_graph(e, None, 9, anchors)
            expected = set()
            for u,v in e.t().tolist():
                reduced = graph.copy(); reduced.remove_node(v)
                component = nx.node_connected_component(reduced,u)
                if not component.intersection(anchors.tolist()):
                    expected.add((u,v))
            self.assertEqual(set(map(tuple,blocked.t().tolist())),expected)

    def test_ca_asr_exact_counts_and_population(self):
        truth = torch.tensor([0,1,2,0,1,2])
        row = classification_counts(torch.tensor([0,2,2,0,0,0]), torch.tensor([0,0,0,0,0,2]),
                                    truth, torch.tensor([0,1,2]), torch.tensor([3,4,5]), 0)
        self.assertEqual(row, dict(CA=2/3, ASR=.5, CleanCorrect=2, CleanCount=3, ASRHits=1, ASRCount=2))

    def test_small_reference_uses_infinite_order_statistic(self):
        logits = torch.tensor([[3.,0.], [2.,0.]])
        threshold, count = fit_reference_threshold(logits, torch.ones(2,dtype=torch.long),
            torch.arange(2), 0, .005, small_sample='infinite')
        self.assertTrue(math.isinf(threshold))
        self.assertEqual(count, 2)
        self.assertEqual(threshold_prediction(logits, 0, threshold).tolist(), [1,1])

    def test_rank_with_ties_has_expected_leave_one_out_bound(self):
        values = torch.tensor([0.,0.,1.,2.,2.,5.])
        accepted = 0
        for j in range(len(values)):
            logits = torch.stack([values, torch.zeros_like(values)], 1)
            ref = torch.tensor([k for k in range(len(values)) if k != j])
            threshold, _ = fit_reference_threshold(logits, torch.ones(6,dtype=torch.long), ref, 0, .34)
            accepted += int(values[j] > threshold)
        self.assertLessEqual(accepted/6, .34)

    def test_ratio_scales_with_retained_training_and_nested_sampling(self):
        a,b = split_reference(torch.arange(541), .2, None, 7)
        self.assertEqual((len(a),len(b)), (109,432))
        c,_ = split_reference(torch.arange(541), .2, 120, 7)
        self.assertTrue(torch.isin(a,c).all())
        self.assertFalse(torch.isin(a,b).any())

    def test_unobserved_labels_have_no_effect(self):
        inputs = fixture()
        cfg = ReferenceConfig(epochs=3, hidden=4, members=1, count=2)
        _, first, metadata = fit_reference_method(*inputs, cfg)
        changed = inputs[3].clone(); changed[8:] = 999
        _, second, _ = fit_reference_method(*inputs[:3], changed, *inputs[4:], cfg)
        self.assertEqual(first['nclass'], 2)
        for role in ('normal','recovery'):
            for k,v in first['members'][0][role].items():
                self.assertTrue(torch.equal(v,second['members'][0][role][k]))
        self.assertEqual(metadata['SelectedNodes'], [2])

    def test_reference_labels_are_excluded_from_recovery_fit(self):
        inputs = fixture()
        cfg = ReferenceConfig(epochs=3, hidden=4, members=1, count=2)
        _, first, metadata = fit_reference_method(*inputs, cfg)
        changed = inputs[3].clone()
        refs = torch.tensor(metadata['ReferenceNodes'])
        changed[refs] = 1-changed[refs]
        _, second, _ = fit_reference_method(*inputs[:3], changed, *inputs[4:], cfg)
        self.assertFalse(set(metadata['RecoveryTrainingNodes']) & set(metadata['ReferenceNodes']))
        for k,v in first['members'][0]['recovery'].items():
            self.assertTrue(torch.equal(v,second['members'][0]['recovery'][k]))

    def test_three_backbones_fit_and_reload(self):
        inputs = fixture()
        for architecture in ('GCN','GAT','GraphSage'):
            with self.subTest(architecture=architecture), tempfile.TemporaryDirectory() as folder:
                cfg = ReferenceConfig(epochs=2, hidden=4, members=1, count=2, architecture=architecture)
                model,bundle,_ = fit_reference_method(*inputs,cfg)
                path = Path(folder)/'model.pt'; torch.save(bundle,path)
                restored = load_reference_bundle(path)
                self.assertTrue(torch.equal(model.predict(*inputs[:3]), restored.predict(*inputs[:3])))

    def test_no_candidates_uses_ordinary_ensemble(self):
        inputs = list(fixture()); inputs[0][10] = inputs[0][0]
        cfg = ReferenceConfig(epochs=2, hidden=4, members=1, count=2)
        model,bundle,meta = fit_reference_method(*inputs,cfg)
        self.assertEqual(meta['InferredClass'], -1)
        self.assertEqual(meta['ModelCount'], 1)
        self.assertEqual(model.predict(*inputs[:3]).shape, (11,))

    def test_original_outputs_preserved_for_three_graph_operators(self):
        base = undirected([(0,1),(1,2),(2,0),(2,3)])
        extended = torch.cat([base, undirected([(1,4),(4,5)])],1)
        x = torch.zeros(6,16); x[:4,:2]=.5; x[4:]=1/16
        anchors = torch.tensor([0,1,2])
        detector = SparseBranchDetector(2).fit(x,anchors)
        e,w = graph_views(detector,x,extended,None,anchors)['selective']
        for architecture in ('GCN','GAT','GraphSage'):
            cfg = ReferenceConfig(hidden=4, architecture=architecture)
            model = build_reference_model(cfg,architecture,16,3,'cpu').eval()
            with torch.no_grad():
                expected = log_output(model(x[:4],base))
                actual = log_output(model(x,e,w))[:4]
            torch.testing.assert_close(expected,actual,atol=1e-6,rtol=1e-6)


if __name__ == '__main__':
    unittest.main(verbosity=2)
