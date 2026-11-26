"""SPEAR checkpoint round-trip and LoSplit fitting-contract regression tests."""
import argparse
import contextlib
import copy
import io
import importlib.util
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import torch

from experiment_defenses import add_arguments, run_comparison
from models.backdoor_SPEAR import Backdoor, GraphTrojanNet
from models.GCN import GCN
from spear_snapshot import generator_metadata, load_spear_generator, validate_snapshot


class SpearSnapshotTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(5)
        self.args = SimpleNamespace(dataset='Cora', hidden=8, alpha_int=4, thrd=.5)
        self.device = torch.device('cpu')
        self.x = torch.randn(8, 1433)
        self.edges = torch.tensor([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]])
        self.weights = torch.ones(6)
        self.labels = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1])
        self.generator = Backdoor(self.args, self.device)
        self.generator.shadow_model = GCN(self.args, 1433, 8, 2, dropout=0., device=self.device)
        self.generator.trojan = GraphTrojanNet(self.device, 8, 4)
        self.snapshot = dict(snapshot_version=2, attack='SPEAR', dataset='Cora',
            num_original_nodes=8, poison_x=self.x, poison_edge_index=self.edges,
            poison_edge_weights=self.weights, poison_labels=self.labels,
            original_labels=self.labels, idx_train=torch.tensor([0, 1]),
            idx_attach=torch.tensor([2]), idx_val=torch.tensor([3]),
            idx_clean_test=torch.tensor([4, 5]), idx_atk=torch.tensor([6, 7]),
            mask_edge_index=self.edges[:, :0], target_class=0,
            **generator_metadata(self.generator, self.args))

    def test_roundtrip_preserves_injection_and_topology(self):
        restored = load_spear_generator(self.snapshot, self.device)
        nodes = self.snapshot['idx_atk']
        with torch.no_grad():
            expected = self.generator.inject_trigger(nodes, self.x, self.edges, self.weights, self.device)
            actual = restored.inject_trigger(nodes, self.x, self.edges, self.weights, self.device)
        for left, right in zip(expected, actual):
            self.assertTrue(torch.equal(left, right))
        self.assertTrue(torch.equal(actual[1], self.edges))
        self.assertTrue(torch.equal(actual[2], self.weights))
        self.assertTrue(torch.equal(actual[0][:6], self.x[:6]))
        self.assertFalse(torch.equal(actual[0][nodes], self.x[nodes]))

    def test_rejects_incomplete_shadow_or_changed_feature_dimensions(self):
        incomplete = dict(self.snapshot)
        del incomplete['shadow_model_state']
        with self.assertRaisesRegex(ValueError, 'shadow_model_state'):
            load_spear_generator(incomplete, self.device)
        changed = dict(self.snapshot, selected_feature_indices=torch.arange(4))
        with self.assertRaisesRegex(ValueError, 'feature selection differs'):
            load_spear_generator(changed, self.device)

    def test_generator_pairing_rejects_a_different_state(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'generator.pth'
            torch.save(self.snapshot['trigger_generator_state'], path)
            validate_snapshot(self.snapshot, 'Cora', 8, path)
            changed = copy.deepcopy(self.snapshot['trigger_generator_state'])
            changed[next(iter(changed))] += 1
            torch.save(changed, path)
            with self.assertRaisesRegex(ValueError, 'Generator differs'):
                validate_snapshot(self.snapshot, 'Cora', 8, path)


class LoSplitContractTests(unittest.TestCase):
    def test_recovery_accepts_gat_and_sage_on_cpu_and_available_cuda(self):
        root = Path(__file__).resolve().parent
        devices = ['cpu'] + (['cuda:0'] if torch.cuda.is_available() else [])
        for filename, class_name in [('models/GAT.py', 'GAT'), ('models/SAGE.py', 'GraphSage'),
                ('DPGBA/models/GAT.py', 'GAT'), ('DPGBA/models/SAGE.py', 'GraphSage')]:
            spec = importlib.util.spec_from_file_location('recovery_' + filename.replace('/', '_'), root / filename)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            for device_name in devices:
                with self.subTest(file=filename, device=device_name):
                    device = torch.device(device_name)
                    kwargs = dict(nfeat=4, nhid=8, nclass=2, device=device)
                    if class_name == 'GAT':
                        kwargs['heads'] = 2
                    model = getattr(module, class_name)(**kwargs).to(device)
                    x = torch.randn(6, 4, device=device)
                    e = torch.tensor([[0,1,1,2,2,3,3,4], [1,0,2,1,3,2,4,3]], device=device)
                    y = torch.tensor([0,1,0,1,0,1], device=device)
                    train = torch.tensor([0,1,2,3], device=device)
                    model.fit(x, e, x.new_ones(e.size(1)), y.clone(), train,
                        torch.tensor([4], device=device), train_iters=2, finetune2=True,
                        attach=train[:1], clean=train[1:], target_label=0, gamma=.05)
                    model.eval()
                    with torch.no_grad():
                        output = model(x, e)
                    self.assertEqual(tuple(output.shape), (6,2))
                    self.assertTrue(torch.isfinite(output).all())

    def test_shared_parser_accepts_existing_dpgba_split_options(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--split_lr', type=float, default=.03)
        parser.add_argument('--split_epoch', type=int, default=10)
        add_arguments(parser)
        self.assertIsNone(parser.parse_args([]).split_lr)
        self.assertEqual(parser.parse_args(['--split_lr=.008']).split_lr, .008)

    def test_original_graph_and_training_order_and_snapshot_label_isolation(self):
        device = torch.device('cpu')
        x = torch.ones(7, 2)
        edges = torch.tensor([[0, 1, 2], [1, 2, 0]])
        weights = torch.tensor([1., 0., 1.])
        labels = torch.tensor([0, 1, 0, 1, 1, 0, 1])
        snapshot = dict(attack='SPEAR', poison_x=x, poison_edge_index=edges,
            poison_edge_weights=weights, poison_labels=labels, idx_train=torch.tensor([2, 0]),
            idx_attach=torch.tensor([1]), idx_val=torch.tensor([3]),
            idx_clean_test=torch.tensor([4]), idx_atk=torch.tensor([6]),
            mask_edge_index=edges[:, :0])
        observed = {}

        class Splitter(torch.nn.Module):
            def fit(self, features, e, w, y, anchors, val, **kwargs):
                observed.update(edges=e.clone(), weights=w.clone(), anchors=anchors.clone())
                return ([anchors[:1], anchors[2:], anchors[:1]],
                        [anchors[1:], anchors[:2], anchors[1:]], [0., 4., 1.], [1, 0, 1])

        class Classifier(torch.nn.Module):
            def fit(self, features, e, w, y, anchors, val, **kwargs):
                observed['fit'] = kwargs
                y[kwargs['attach']] = 0  # The author recovery mutates its labels.
            def forward(self, features, e, w):
                return torch.log_softmax(torch.zeros(features.size(0), 2), dim=1)

        class Generator:
            def inject_trigger(self, nodes, features, e, w, device):
                return features, e, w

        def construct(args, name, data, device, **kwargs):
            return Splitter() if name == 'LoSplit' else Classifier()

        args = SimpleNamespace(defense_mode='losplit', skip_rigbd=False, classifier_seed=None,
            seed=10, cuda=False, dataset='Cora', test_model='GCN', epochs=2,
            split_lr=.008, split_epoch=3, gamma=.05, target_class=0, model_save_path='')
        with contextlib.redirect_stdout(io.StringIO()):
            result = run_comparison(args, SimpleNamespace(y=labels.clone()), device, snapshot,
                Generator(), construct, None)
        self.assertTrue(torch.equal(observed['edges'], edges))
        self.assertTrue(torch.equal(observed['weights'], weights))
        self.assertEqual(observed['anchors'].tolist(), [2, 0, 1])
        self.assertEqual(labels.tolist(), [0, 1, 0, 1, 1, 0, 1])
        self.assertTrue(observed['fit']['finetune2'])
        self.assertEqual(observed['fit']['target_label'], 0)
        self.assertEqual(result['BestSplitEpoch'], 1)
        self.assertEqual(result['SelectedNodes'], [1])


if __name__ == '__main__':
    unittest.main()
