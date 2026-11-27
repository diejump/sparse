"""Launcher report checks without importing torch or running training."""
import contextlib
import ast
from argparse import Namespace
import io
from pathlib import Path
import tempfile
import unittest
import argparse
import sys
from unittest.mock import patch

import run_experiments as runner


class ReportTests(unittest.TestCase):

    def test_complete_entry_parsing_matrix_in_actual_source_order(self):
        from check_experiment_parameters import check_matrix
        result = check_matrix()
        self.assertEqual(result['parsed_jobs'], 472)
        self.assertEqual(result['model_imports'], 0)
        self.assertEqual(result['training_runs'], 0)

    def test_gta_ugba_pubmed_losplit_regression(self):
        from check_experiment_parameters import parse_entry
        args = Namespace(**dict(runner.SETTINGS, stage='defense', dataset='Pubmed',
                                attacks=['GTA', 'UGBA'], methods=['losplit']))
        for name, entry, flags, _, _, _ in runner.build_jobs(args):
            parsed = parse_entry(entry, flags)
            algorithm = name.split('/')[0]
            self.assertEqual((parsed.split_lr, parsed.split_epoch, parsed.gamma),
                             runner.LOSPLIT_SETTINGS[algorithm]['Pubmed'])
            self.assertEqual(parsed.vs_number, runner.ATTACK_SETTINGS[algorithm]['Pubmed']['vs_number'])

    def test_entry_checker_catches_early_full_parse(self):
        from check_experiment_parameters import parse_entry
        source = (runner.PROJECT / 'GTA_RIGBD.py').read_text(encoding='utf-8')
        broken = source.replace('add_arguments(parser)', 'args = parser.parse_args()\nadd_arguments(parser)', 1)
        with tempfile.TemporaryDirectory() as directory:
            entry = Path(directory) / 'early_parse.py'
            entry.write_text(broken, encoding='utf-8')
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parse_entry(entry, ['--dataset=Pubmed', '--defense_mode=losplit',
                                    '--split_lr=.005', '--split_epoch=15', '--gamma=.55'])

    def test_author_budgets_seeds_and_training_parameters(self):
        budgets = {'GTA': [48, 43, 201, 200], 'UGBA': [44, 42, 162, 394],
                   'DPGBA': [50, 45, 194, 200], 'SPEAR': [50, 47, 222, 207]}
        for algorithm, expected in budgets.items():
            for dataset, budget in zip(('Cora', 'Citeseer', 'Pubmed', 'Physics'), expected):
                args = Namespace(stage='all', attacks=[algorithm], methods=['none', 'prune', 'losplit'],
                                 dataset=dataset, model='GCN', seed=None, epochs=None,
                                 attack_root='inputs/seed{seed}', defense_root='outputs/seed{seed}')
                jobs = runner.build_jobs(args)
                config = runner.ATTACK_SETTINGS[algorithm][dataset]
                self.assertEqual(config['vs_number'], budget)
                for name, _, flags, outputs, _, _ in jobs:
                    values = dict(flag[2:].split('=', 1) for flag in flags if '=' in flag)
                    for key in ('seed', 'hidden', 'train_lr', 'epochs', 'target_class'):
                        self.assertEqual(float(values[key]), float(config[key]), (name, dataset, key))
                    self.assertIn('seed' + str(config['seed']), outputs[0].parts)
                    self.assertEqual(len(flags), len(set(flag.split('=')[0] for flag in flags)))
                self.assertEqual(config['seed'], 12 if algorithm == 'DPGBA' else 10)
        physics = runner.ATTACK_SETTINGS['DPGBA']['Physics']
        self.assertEqual((physics['hidden'], physics['lr'], physics['weight_ood']), (128, .002, 50))
        self.assertEqual(runner.ATTACK_SETTINGS['DPGBA']['Cora']['range'], .01)

    def test_direct_parsers_resolve_central_dataset_defaults_without_models(self):
        for algorithm, datasets in runner.ATTACK_SETTINGS.items():
            path = runner.PROJECT / ('DPGBA' if algorithm == 'DPGBA' else '') / (algorithm + '_LoSplit.py')
            tree = ast.parse(path.read_text(encoding='utf-8-sig'))
            # Execute argument declarations only. All data/model statements stay outside this module.
            declarations = [node for node in tree.body if (
                isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'parser'
                                                     for t in node.targets)) or (
                isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Attribute)
                and isinstance(node.value.func.value, ast.Name)
                and node.value.func.value.id == 'parser' and node.value.func.attr == 'add_argument')]
            module = compile(ast.Module(body=declarations, type_ignores=[]), '<parser declarations>', 'exec')
            for dataset, expected in datasets.items():
                namespace = dict(argparse=argparse, ALGORITHM_DEFAULTS=runner.ALGORITHM_DEFAULTS)
                exec(module, namespace)
                with patch.object(sys, 'argv', [path.name, '--dataset=' + dataset]):
                    runner.configure_algorithm_parser(namespace['parser'], algorithm)
                    actual = vars(namespace['parser'].parse_args())
                for key, value in expected.items():
                    self.assertEqual(actual[key], value, (algorithm, dataset, key))

    def test_explicit_seed_and_epoch_overrides_follow_paired_jobs(self):
        args = Namespace(stage='all', attacks=['DPGBA'], methods=['none'], dataset='Physics',
                         model='GCN', seed=41, epochs=17,
                         attack_root='inputs/seed{seed}', defense_root='outputs/seed{seed}')
        for _, _, flags, outputs, _, _ in runner.build_jobs(args):
            self.assertIn('--seed=41', flags)
            self.assertIn('--epochs=17', flags)
            self.assertIn('seed41', outputs[0].parts)

    def test_range_entry_preserves_backbone_and_three_controls(self):
        for model in ('GCN','GAT','GraphSage'):
            args=Namespace(stage='defense',attacks=['GTA','UGBA','DPGBA','SPEAR'],methods=['range_reference'],
                dataset='Citeseer',model=model,seed=41,attack_root='inputs',defense_root='outputs')
            for _,entry,flags,_,_,_ in runner.build_jobs(args):
                self.assertEqual(entry.name,'range_reference_launcher.py')
                for flag in ('--test_model='+model,'--hidden=32','--train_lr=0.01','--range_gamma=2.0','--range_rho=0.2','--range_alpha=0.005','--device=cpu'):
                    self.assertIn(flag,flags)
                self.assertFalse(any(flag.startswith(('--dropout=', '--weight_decay=')) for flag in flags))
    def test_feature_method_uses_one_frozen_configuration(self):
        args=Namespace(stage='defense',attacks=['GTA','UGBA','DPGBA','SPEAR'],methods=['abp_feature_reference'],
            seed=31,dataset='Cora',model='GCN',attack_root='inputs',defense_root='outputs',feature_repair='none')
        jobs=runner.build_jobs(args)
        self.assertEqual(len(jobs),4)
        for _,entry,flags,_,_,_ in jobs:
            self.assertEqual(entry.name,'feature_reference_launcher.py')
            self.assertIn('--hidden=32',flags)
            self.assertIn('--train_lr=0.01',flags)
            self.assertIn('--members=1',flags)
            self.assertIn('--feature-repair=none',flags)
            self.assertFalse(any(flag.startswith(('--dropout=', '--weight_decay=')) for flag in flags))

    def test_spear_author_settings_and_paired_inputs(self):
        for dataset, expected in runner.SPEAR_SETTINGS.items():
            args = Namespace(stage='all', attacks=['SPEAR'], methods=[m for m in runner.METHOD_ARGS if m not in ('abp_feature_reference','range_reference')],
                dataset=dataset, model='GCN', seed=10, attack_root='inputs', defense_root='outputs')
            jobs = runner.build_jobs(args)
            self.assertEqual(len(jobs), 1 + len(args.methods))
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
