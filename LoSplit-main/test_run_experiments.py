"""Launcher report checks without importing torch or running training."""
import contextlib
import ast
from argparse import Namespace
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import run_experiments as runner


class ReportTests(unittest.TestCase):
    def test_spear_author_settings_and_paired_inputs(self):
        for dataset, expected in runner.SPEAR_SETTINGS.items():
            args = Namespace(stage='all', attacks=['SPEAR'], methods=list(runner.METHOD_ARGS),
                dataset=dataset, model='GCN', seed=10, attack_root='inputs', defense_root='outputs')
            jobs = runner.build_jobs(args)
            self.assertEqual(len(jobs), 1 + len(runner.METHOD_ARGS))
            generation = jobs[0]
            self.assertEqual(generation[1].name, 'SPEAR_LoSplit.py')
            for key, value in expected.items():
                self.assertIn(f'--{key}={value}', generation[2])
            self.assertIn('--trojan_epochs=200', generation[2])
            for _, entry, flags, _, inputs, _ in jobs[1:]:
                self.assertTrue(entry.is_file())
                self.assertEqual(inputs, generation[3])
                self.assertEqual(sum(f.startswith('--hidden=') for f in flags), 1)
                self.assertIn(f"--hidden={expected['hidden']}", flags)
                self.assertIn(f"--target_class={expected['target_class']}", flags)

    def test_losplit_author_settings_for_every_entry(self):
        for dataset in ('Cora', 'Citeseer', 'Pubmed', 'Physics'):
            args = Namespace(stage='defense', attacks=list(runner.ATTACK_ARGS), methods=['losplit'],
                dataset=dataset, model='GCN', seed=10, attack_root='inputs', defense_root='outputs')
            for name, entry, flags, outputs, inputs, _ in runner.build_jobs(args):
                attack = name.split('/')[0]
                lr, epochs, gamma = runner.LOSPLIT_SETTINGS[attack][dataset]
                self.assertEqual(entry.name, attack + '_RIGBD.py')
                self.assertIn('--defense_mode=losplit', flags)
                for key, value in [('split_lr', lr), ('split_epoch', epochs), ('gamma', gamma)]:
                    self.assertIn(f'--{key}={value}', flags)
                self.assertEqual(len(outputs), 2)
                self.assertEqual(len(inputs), 2)

    def test_reference_entry_preserves_backbone_and_configuration(self):
        for model in ('GCN', 'GAT', 'GraphSage'):
            args = Namespace(stage='defense', attacks=['GTA','UGBA','DPGBA'], methods=['abp_reference'],
                seed=10, dataset='Cora', model=model, attack_root='inputs', defense_root='outputs',
                abp_reference_count=120)
            for _, entry, flags, _, _, _ in runner.build_jobs(args):
                self.assertEqual(entry.name, 'reference_experiment.py')
                self.assertIn('--test_model=' + model, flags)
                self.assertIn('--abp_reference_count=120', flags)
                self.assertIn('--abp_reference_members=3', flags)

    def test_explicit_selfloop_variant_and_fixed_parameters(self):
        args = Namespace(stage='defense', attacks=['UGBA'], methods=['rigbd','rigbd_selfloop'],
                         seed=10, model='GCN', dataset='Cora', attack_root='inputs', defense_root='outputs',
                         rigbd_fixed=True, rigbd_k=20, rigbd_b=.5)
        jobs = runner.build_jobs(args)
        self.assertNotIn('--rigbd_probe_self_loops', jobs[0][2])
        self.assertIn('--rigbd_probe_self_loops', jobs[1][2])
        for job in jobs:
            self.assertNotIn('--rigbd_grid', job[2])
            self.assertIn('--rigbd_selection=label_boundary', job[2])
            self.assertIn('--rigbd_sampling=receiver', job[2])

    def test_rigbd_selects_minimum_asr_and_stable_ties(self):
        # Load only the pure selector; no torch imports or model execution.
        tree = ast.parse((runner.PROJECT / 'experiment_defenses.py').read_text(encoding='utf-8'))
        selector = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                        and node.name == 'rigbd_selection_key')
        namespace = {}
        exec(compile(ast.Module(body=[selector], type_ignores=[]), '<rigbd_selector>', 'exec'), namespace)
        key = namespace['rigbd_selection_key']
        candidates = [dict(ASR=.2, CA=.9, ValidationAccuracy=.95, K=10, b=.3),
                      dict(ASR=.1, CA=.8, ValidationAccuracy=.8, K=50, b=.7),
                      dict(ASR=.1, CA=.7, ValidationAccuracy=.7, K=20, b=.5),
                      dict(ASR=.1, CA=.6, ValidationAccuracy=.6, K=20, b=.3)]
        self.assertIs(max(candidates[:2], key=key), candidates[1])
        self.assertIs(max(candidates, key=key), candidates[3])

    def test_comparison_methods_and_model_dataset_flags(self):
        for dataset in ('Cora', 'Citeseer', 'Pubmed', 'Physics'):
            for model in ('GCN', 'GAT', 'GraphSage'):
                args = Namespace(stage='all', attacks=['GTA', 'UGBA', 'DPGBA'],
                    methods=runner.SETTINGS['methods'], seed=10, dataset=dataset, model=model,
                    attack_root='inputs/{dataset}/{model}', defense_root='outputs/{dataset}/{model}')
                jobs = runner.build_jobs(args)
                self.assertEqual(len(jobs), len(args.attacks) * (1 + len(args.methods)))
                for name, entry, flags, outputs, inputs, log in jobs:
                    self.assertIn('--dataset=' + dataset, flags)
                    self.assertIn('--test_model=' + model, flags)
                    self.assertIn('--model=GCN', flags)
                    self.assertTrue(entry.is_file())
                    self.assertIn(dataset, outputs[0].parts)
                    if name.endswith('/abp_sparse'):
                        self.assertIn('--abp_sparse_multiplier=0.95', flags)
                        self.assertIn('--abp_sparse_strength=0.25', flags)
                    if name.endswith('/rigbd'):
                        self.assertIn('--rigbd_grid', flags)

    def test_json_metrics_preserve_seconds_and_best_parameters(self):
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / 'task.log'
            log.write_text('FINAL_COMPARISON_JSON {"CA":0.8,"ASR":0.1,"Precision":null,'
                           '"Recall":null,"Seconds":123.5,"GridSeconds":300.0,"K":100,"b":0.7}\n',
                           encoding='utf-8')
            metrics = runner.parse_metrics(log, 'rigbd')
            self.assertEqual(metrics['Seconds'], 123.5)
            self.assertEqual(metrics['K'], 100)
            self.assertIsNone(metrics['Precision'])

    def test_dpgba_classifier_learning_rates_match(self):
        args = Namespace(stage='all', attacks=['DPGBA'], methods=['none'], seed=10,
                         attack_root='runs/test_inputs', defense_root='runs/test_outputs')
        jobs = runner.build_jobs(args)
        rates = [[flag for flag in job[2] if flag.startswith('--train_lr=')] for job in jobs]
        self.assertEqual(rates, [['--train_lr=0.015'], ['--train_lr=0.015']])

    def test_final_markers_and_attack_formats(self):
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "task.log"
            log.write_text("****After GTA Attack****\nASR: 0.900000/100 nodes\nClean Accuracy: 0.800000\n", encoding="utf-8")
            self.assertEqual(runner.parse_metrics(log, "attack"), {"ASR": .9, "CA": .8})
            self.assertEqual(runner.parse_metrics(log, "prune"), {})
            log.write_text("ASR: 0.99\nFINAL_DEFENSE_RESULT method=prune CA=0.8 ASR=0.1\n"
                           "FINAL_RIGBD_RESULT K=20 b=0.5 CA=0.7 ASR=0.2 Precision=1 Recall=0.5 FPR=0\n",
                           encoding="utf-8")
            self.assertEqual(runner.parse_metrics(log, "prune"), {"CA": .8, "ASR": .1})
            self.assertEqual(runner.parse_metrics(log, "rigbd"),
                             {"CA": .7, "ASR": .2, "Precision": 1., "Recall": .5, "FPR": 0.})

    def simulate(self, root, failure=None):
        arguments = ["run_experiments.py", "--attacks", "GTA", "--methods", "none", "prune",
                     "--seed", "10",
                     "--attack-root", str(root / "attacks"), "--defense-root", str(root / "defenses")]

        def fake_job(job, env):
            self.assertEqual(env["PYTHONHASHSEED"], "10")
            name, entry, flags, outputs, inputs, log = job
            for path in inputs:
                self.assertTrue(path.is_file())
                self.assertIn((root / "attacks").resolve(), path.parents)
            if name.endswith("/none") and failure:
                raise failure
            for path in outputs:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"test")
            content = ("****After GTA Attack****\nASR: 0.9\nClean Accuracy: 0.8\n"
                       if name.endswith("/attack") else
                       'FINAL_COMPARISON_JSON {"CA":0.8,"ASR":0.1,"Precision":null,'
                       '"Recall":null,"FPR":null,"Seconds":1.25,"DetectionSeconds":null}\n')
            log.write_text(content, encoding="utf-8")

        with patch("sys.argv", arguments), patch.object(runner.subprocess, "run"), \
                patch.object(runner, "run_job", side_effect=fake_job), contextlib.redirect_stdout(io.StringIO()):
            if failure:
                with self.assertRaises(type(failure)):
                    runner.main()
            else:
                runner.main()
        return (root / "defenses" / "results.md").read_text(encoding="utf-8")

    def test_all_pairing_and_success_report(self):
        with tempfile.TemporaryDirectory() as directory:
            report = self.simulate(Path(directory))
            self.assertIn("全部完成", report)
            self.assertIn("90.0000", report)
            self.assertIn("10.0000", report)
            self.assertIn("GTA/attack", report)
            self.assertIn("GTA/prune", report)
            self.assertIn('1.2500', report)
            import json
            results = json.loads((Path(directory) / 'defenses/results.json').read_text(encoding='utf-8'))
            self.assertEqual(results['records']['GTA/none']['metrics']['Seconds'], 1.25)
            self.assertIsNone(results['records']['GTA/none']['metrics']['Precision'])

    def test_failure_and_interrupt_preserve_results(self):
        for error in (RuntimeError("training failed"), KeyboardInterrupt()):
            with self.subTest(error=type(error).__name__), tempfile.TemporaryDirectory() as directory:
                report = self.simulate(Path(directory), error)
                self.assertIn("90.0000", report)
                self.assertIn("未运行", report)
                self.assertIn("中断" if isinstance(error, KeyboardInterrupt) else "training failed", report)

    def test_existing_report_is_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / "existing.md"
            report.write_text("keep me", encoding="utf-8")
            with patch("sys.argv", ["runner", "--report", str(report)]), \
                    contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    runner.main()
            self.assertEqual(report.read_text(encoding="utf-8"), "keep me")


if __name__ == "__main__":
    unittest.main()
