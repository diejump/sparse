"""Replay actual argument-parsing order without loading data or model modules."""
import argparse
import ast
from functools import lru_cache
import json
from pathlib import Path
import shlex
import sys
from unittest.mock import patch
import zipfile

import run_experiments as runner


@lru_cache(maxsize=None)
def parser_program(path):
    tree = ast.parse(Path(path).read_text(encoding='utf-8-sig'))
    main = next((n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main'), None)
    body = main.body if main else tree.body
    statements, parser_name, result_name = [], None, None
    for node in body:
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
            call = node.value
            if isinstance(call.func, ast.Attribute) and call.func.attr == 'ArgumentParser':
                parser_name = node.targets[0].id
                statements.append(node)
                continue
        if parser_name is None:
            continue
        # Keep the real order of declarations, helper calls and the first full parse.
        if isinstance(node, ast.Assign) and any(
                isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                and isinstance(n.func.value, ast.Name) and n.func.value.id == parser_name
                and n.func.attr in ('parse_args', 'parse_known_args') for n in ast.walk(node)):
            statements.append(node)
            result_name = node.targets[0].id
            break
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            func = node.value.func
            if (isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name)
                    and func.value.id == parser_name and func.attr == 'add_argument') or (
                    isinstance(func, ast.Name) and func.id in
                    ('add_arguments', 'add_reference_arguments', 'configure_algorithm_parser')):
                statements.append(node)
        if isinstance(node, ast.For) and any(
                isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                and isinstance(n.func.value, ast.Name) and n.func.value.id == parser_name
                and n.func.attr == 'add_argument' for n in ast.walk(node)):
            statements.append(node)
    if parser_name is None or result_name is None:
        raise ValueError(f'Argument parser not found: {path}')
    return compile(ast.Module(body=statements, type_ignores=[]), str(path), 'exec'), parser_name, result_name


def pure_helper(filename, name):
    tree = ast.parse((runner.PROJECT / filename).read_text(encoding='utf-8-sig'))
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    namespace = dict(SETTINGS=runner.SETTINGS)
    exec(compile(ast.Module(body=[function], type_ignores=[]), filename, 'exec'), namespace)
    return namespace[name]


ADD_ARGUMENTS = pure_helper('experiment_defenses.py', 'add_arguments')
ADD_REFERENCE_ARGUMENTS = pure_helper('reference_experiment.py', 'add_reference_arguments')


def parse_entry(path, flags):
    code, parser_name, result_name = parser_program(str(path))
    namespace = dict(argparse=argparse, __doc__='', ALGORITHM_DEFAULTS=runner.ALGORITHM_DEFAULTS,
                     PRUNE_SETTINGS=runner.PRUNE_SETTINGS, SETTINGS=runner.SETTINGS,
                     configure_algorithm_parser=runner.configure_algorithm_parser,
                     add_arguments=ADD_ARGUMENTS, add_reference_arguments=ADD_REFERENCE_ARGUMENTS)
    with patch.object(sys, 'argv', [str(path), *flags]):
        exec(code, namespace)
        # Check unknown flags even for legacy entries using parse_known_args.
        strict = namespace[parser_name].parse_args()
    parsed = namespace[result_name]
    if vars(parsed) != vars(strict):
        raise AssertionError(f'Parser results differ: {path}')
    return parsed


def check_matrix():
    count = 0
    for dataset in ('Cora', 'Citeseer', 'Pubmed', 'Physics'):
        for model in ('GCN', 'GAT', 'GraphSage'):
            methods = [method for method in runner.METHOD_ARGS
                       if (method != 'abp_feature_reference' or dataset in ('Cora', 'Citeseer'))
                       and (method != 'rigbd_selfloop' or model == 'GCN')]
            args = argparse.Namespace(**dict(runner.SETTINGS, dataset=dataset, model=model,
                                              stage='all', methods=methods))
            for name, entry, flags, _, _, _ in runner.build_jobs(args):
                algorithm, method = name.split('/')
                parsed = parse_entry(entry, flags)
                expected = runner.ATTACK_SETTINGS[algorithm][dataset]
                for key in ('seed', 'epochs', 'target_class'):
                    assert getattr(parsed, key) == expected[key], (name, dataset, key)
                model_config = (runner.REFERENCE_MODEL_SETTINGS if method in
                                ('abp_feature_reference', 'range_reference') else expected)
                for key in ('hidden', 'train_lr'):
                    assert getattr(parsed, key) == model_config[key], (name, dataset, key)
                if method == 'attack':
                    for key, value in expected.items():
                        assert getattr(parsed, key) == value, (name, dataset, key)
                elif method == 'losplit':
                    assert (parsed.split_lr, parsed.split_epoch, parsed.gamma) == runner.LOSPLIT_SETTINGS[algorithm][dataset]
                elif method == 'prune':
                    assert parsed.prune_thr == runner.PRUNE_SETTINGS[algorithm]
                count += 1
    return dict(parsed_jobs=count, datasets=4, algorithms=4, backbones=3,
                data_loads=0, model_imports=0, training_runs=0)


def source_defaults(source):
    values = {}
    for node in ast.walk(ast.parse(source)):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == 'add_argument' and node.args
                and isinstance(node.args[0], ast.Constant)):
            for keyword in node.keywords:
                if keyword.arg == 'default':
                    try:
                        values[node.args[0].value.lstrip('-')] = ast.literal_eval(keyword.value)
                    except (ValueError, TypeError):
                        pass
    return values


def check_author_source(path):
    checked = set()
    parameter_count = 0
    with zipfile.ZipFile(path) as archive:
        script_path = next(n for n in archive.namelist() if n.endswith('/defense.sh'))
        prefix = script_path[:-len('defense.sh')]
        for line in archive.read(script_path).decode('utf-8-sig').splitlines():
            if not line.startswith('python '):
                continue
            tokens = shlex.split(line)
            flags = dict(t[2:].split('=', 1) for t in tokens[2:] if t.startswith('--') and '=' in t)
            algorithm = Path(tokens[1]).name.split('_')[0]
            dataset = flags.get('dataset')
            if algorithm not in runner.ATTACK_SETTINGS or dataset not in runner.ATTACK_SETTINGS[algorithm]:
                continue
            entry = tokens[1][2:] if tokens[1].startswith('./') else tokens[1]
            expected = source_defaults(archive.read(prefix + entry).decode('utf-8-sig'))
            for key, value in flags.items():
                try:
                    expected[key] = ast.literal_eval(value)
                except (ValueError, SyntaxError):
                    expected[key] = value
            for key, value in runner.ATTACK_SETTINGS[algorithm][dataset].items():
                if expected[key] != value:
                    raise AssertionError((algorithm, dataset, key, value, expected[key]))
                parameter_count += 1
            expected_split = tuple(expected[key] for key in ('split_lr', 'split_epoch', 'gamma'))
            assert runner.LOSPLIT_SETTINGS[algorithm][dataset] == expected_split, (algorithm, dataset)
            prune_path = ('DPGBA/' if algorithm == 'DPGBA' else '') + algorithm + '_RIGBD.py'
            prune = source_defaults(archive.read(prefix + prune_path).decode('utf-8-sig'))['prune_thr']
            assert runner.PRUNE_SETTINGS[algorithm] == prune, algorithm
            checked.add((algorithm, dataset))
    if len(checked) != 16:
        raise AssertionError(f'Expected 16 author configurations; found {len(checked)}')
    return dict(author_configurations=len(checked), author_algorithm_parameters=parameter_count,
                author_split_configurations=16, author_prune_configurations=16)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-zip', type=Path, help='Optional original LoSplit archive for source comparison')
    arguments = parser.parse_args()
    result = check_matrix()
    if arguments.source_zip:
        result.update(check_author_source(arguments.source_zip))
    print(json.dumps(result, ensure_ascii=False, indent=2))
