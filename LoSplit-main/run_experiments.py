"""Cross-platform launcher. Edit SETTINGS, then run this file with Python 3.10."""
import argparse
from datetime import datetime
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time


# IDE users: edit this block and click Run. Paths are relative to this file.
name = "GCN4"
SETTINGS = {
    "stage": "all",                 # attack / defense / all
    "attacks": ["GTA", "UGBA", "DPGBA", "SPEAR"],
    "methods": ["none", "losplit", "abp", "abp_sparse", "abp_reference", "abp_feature_reference","prune", "rigbd","rigbd_selfloop"],
    "dataset": "Cora",              # Cora / Citeseer / Pubmed / Physics
    "model": "GCN",                 # GCN / GAT / GraphSage
    "deterministic": True,
    "abp_sparse_multiplier": 0.95,
    "abp_sparse_strength": 0.25,
    "abp_reference_multiplier": 2.0,
    "abp_reference_ratio": 0.2,       # fraction of retained training nodes
    "abp_reference_count": None,     # explicit count overrides the ratio
    "abp_reference_risk": 0.005,
    "abp_reference_members": 3,      # total fitted models = 2 * members
    "abp_reference_seed": 20261124,
    "abp_reference_variant": "full",
    "feature_members": 1,
    "feature_repair": "unique_mode",
    "feature_tolerance": 1e-6,
    "seed": 12,
    "gpu": "0",                     # physical GPU; child uses visible device 0
    "attack_root": f"runs/comparison_inputs_{name}/{{dataset}}/{{model}}/seed{{seed}}",
    "defense_root": f"runs/comparison_outputs_{name}/{{dataset}}/{{model}}/seed{{seed}}",
    "report": None,                 # None: <defense_root>/results.md (attack-only: attack_root)
}
PROJECT = Path(__file__).resolve().parent
ATTACK_ARGS = {
    "GTA": ["--train_lr=0.01", "--lr=0.01", "--trojan_epochs=400",
            "--vs_number=48", "--split_lr=0.01", "--split_epoch=10", "--gamma=0.3"],
    "UGBA": ["--train_lr=0.01", "--lr=0.01", "--trojan_epochs=400",
             "--vs_number=44", "--homo_loss_weight=50", "--homo_boost_thrd=0.5",
             "--split_lr=0.01", "--split_epoch=15", "--gamma=0.2"],
    "DPGBA": ["--train_lr=0.015", "--lr=0.015", "--trojan_epochs=300",
              "--vs_number=50", "--k=50", "--outter_size=4096",
              "--selection_method=none", "--weight_target=1", "--weight_ood=1",
              "--weight_targetclass=2", "--split_lr=0.01", "--split_epoch=15", "--gamma=1.0"],
    "SPEAR": [],  # Dataset-specific author settings below.
}
# Values transcribed from the original defense.sh, including SPEAR's width.
SPEAR_SETTINGS = {
    "Cora": dict(hidden=80, vs_number=50, homo_loss_weight=.1, target_loss_weight=1,
                 alpha_int=30, shadow_lr=.0002, trojan_lr=.0002, target_class=0),
    "Citeseer": dict(hidden=64, vs_number=47, homo_loss_weight=.05, target_loss_weight=1,
                     alpha_int=10, shadow_lr=.0002, trojan_lr=.0002, target_class=0),
    "Pubmed": dict(hidden=64, vs_number=222, homo_loss_weight=.15, target_loss_weight=2,
                   alpha_int=10, shadow_lr=.005, trojan_lr=.005, target_class=2),
    "Physics": dict(hidden=64, vs_number=207, homo_loss_weight=.15, target_loss_weight=2,
                    alpha_int=15, shadow_lr=.0005, trojan_lr=.0005, target_class=0),
}
LOSPLIT_SETTINGS = {
    "GTA": {"Cora": (.01, 10, .3), "Citeseer": (.05, 10, .45),
            "Pubmed": (.005, 15, .55), "Physics": (.00001, 20, .55)},
    "UGBA": {"Cora": (.01, 15, .2), "Citeseer": (.01, 10, .45),
             "Pubmed": (.03, 15, 0.), "Physics": (.001, 10, 0.)},
    "DPGBA": {"Cora": (.01, 15, 1.), "Citeseer": (.01, 10, .85),
              "Pubmed": (.01, 40, 1.), "Physics": (.0001, 20, .4)},
    "SPEAR": {"Cora": (.008, 10, .05), "Citeseer": (.08, 20, .8),
              "Pubmed": (.03, 15, .8), "Physics": (.001, 20, .7)},
}
METHOD_ARGS = {
    "none": ["--defense_mode=none", "--skip_rigbd"],
    "losplit": ["--defense_mode=losplit"],
    "prune": ["--defense_mode=prune", "--skip_rigbd"],
    "rigbd": ["--defense_mode=none", "--rigbd_grid"],
    "rigbd_selfloop": ["--defense_mode=none", "--rigbd_probe_self_loops"],
    "abp": ["--defense_mode=abp"],
    "abp_sparse": ["--defense_mode=abp_sparse"],
    "abp_reference": ["--defense_mode=abp_reference"],
    "abp_feature_reference": ["--defense_mode=abp_feature_reference"],
}


def project_path(value):
    path = Path(value).expanduser()
    return (PROJECT / path).resolve()


def build_jobs(args):
    dataset, model = getattr(args, 'dataset', 'Cora'), getattr(args, 'model', 'GCN')
    substitutions = dict(dataset=dataset, model=model, seed=args.seed)
    attack_root = project_path(args.attack_root.format(**substitutions))
    defense_root = project_path(args.defense_root.format(**substitutions))
    common = (["--deterministic"] if getattr(args, 'deterministic', True) else []) + [f"--dataset={dataset}", "--model=GCN", f"--test_model={model}",
              "--device_id=0", f"--seed={args.seed}", "--hidden=32",
              f"--epochs={getattr(args, 'epochs', 200)}", "--target_class=0"]
    jobs = []
    for attack in args.attacks:
        attack_common = common.copy()
        attack_flags = ATTACK_ARGS[attack].copy()
        if attack == "SPEAR":
            settings = SPEAR_SETTINGS[dataset]
            attack_common = [flag for flag in common
                             if not flag.startswith(('--hidden=', '--target_class='))]
            attack_common += [f"--hidden={settings['hidden']}", f"--target_class={settings['target_class']}"]
            attack_flags = [f"--{key}={value}" for key, value in settings.items()
                            if key not in ('hidden', 'target_class')]
            attack_flags += ["--train_lr=0.01", "--trojan_epochs=200", "--outter_size=512"]
        prefix = "DPGBA/" if attack == "DPGBA" else ""
        generator = attack_root / attack / "generator.pth"
        snapshot = attack_root / attack / "poison.pt"
        inputs = [f"--trigger_generator_address={generator}", f"--pre_train_param={snapshot}"]
        if args.stage in ("attack", "all"):
            jobs.append((f"{attack}/attack", PROJECT / f"{prefix}{attack}_LoSplit.py",
                         attack_common + attack_flags + inputs,
                         [generator, snapshot], [], attack_root / attack / "attack.log"))
        if args.stage in ("defense", "all"):
            for method in args.methods:
                output = defense_root / attack / f"{method}.pth"
                extra = METHOD_ARGS[method].copy()
                if method == 'losplit':
                    lr, epochs, gamma = LOSPLIT_SETTINGS[attack][dataset]
                    extra += [f'--split_lr={lr}', f'--split_epoch={epochs}', f'--gamma={gamma}']
                if method in ('rigbd', 'rigbd_selfloop'):
                    if getattr(args, 'rigbd_fixed', False):
                        extra = [flag for flag in extra if flag != '--rigbd_grid']
                    extra += [f"--rigbd_k={getattr(args, 'rigbd_k', 20)}",
                              f"--rigbd_b={getattr(args, 'rigbd_b', .5)}",
                              f"--rigbd_selection={getattr(args, 'rigbd_selection', 'label_boundary')}",
                              f"--rigbd_sampling={getattr(args, 'rigbd_sampling', 'receiver')}",
                              f"--rigbd_grid_selection={getattr(args, 'rigbd_grid_selection', 'validation')}"]
                if method == "abp_sparse":
                    extra += [f"--abp_sparse_multiplier={getattr(args, 'abp_sparse_multiplier', .95)}",
                              f"--abp_sparse_strength={getattr(args, 'abp_sparse_strength', .25)}"]
                if method == "abp_reference":
                    for key in ('multiplier','ratio','count','risk','members','seed','variant'):
                        value = getattr(args, 'abp_reference_' + key, SETTINGS['abp_reference_' + key])
                        if value is not None:
                            extra.append(f'--abp_reference_{key}={value}')
                    if getattr(args, 'reference_device', None):
                        extra.append('--device=' + args.reference_device)
                if method == 'abp_feature_reference':
                    for key in ('multiplier','ratio','count','risk','seed','variant'):
                        value=getattr(args,'abp_reference_'+key,SETTINGS['abp_reference_'+key])
                        if value is not None:extra.append(f'--abp_reference_{key}={value}')
                    extra += [f"--members={getattr(args,'feature_members',1)}",
                              f"--feature-repair={getattr(args,'feature_repair','unique_mode')}",
                              f"--feature-tolerance={getattr(args,'feature_tolerance',1e-6)}"]
                    if getattr(args,'reference_device',None):extra.append('--device='+args.reference_device)
                if method == "prune":
                    extra.append(f"--prune_thr={0.1 if attack == 'SPEAR' else 0.2 if attack == 'UGBA' else 0.8}")
                entry = PROJECT / ('feature_reference_launcher.py' if method == 'abp_feature_reference' else
                    'reference_experiment.py' if method == 'abp_reference' else f"{prefix}{attack}_RIGBD.py")
                method_common=attack_common
                if method=='abp_feature_reference':
                    method_common=[flag for flag in attack_common if not flag.startswith('--hidden=')]+['--hidden=32']
                method_lr='--train_lr=0.015' if attack=='DPGBA' and method!='abp_feature_reference' else '--train_lr=0.01'
                jobs.append((f"{attack}/{method}", entry,
                             method_common + [method_lr] + inputs + extra
                             + [f"--model_save_path={output}"],
                             [output, Path(str(output) + '.meta.json')],
                             [generator, snapshot], output.with_suffix(".log")))
    return jobs


def parse_metrics(log, method):
    """Only use final markers for defenses; intermediate ASRs may be misleading."""
    if not log.is_file():
        return {}
    content = log.read_text(encoding="utf-8-sig", errors="replace")
    number = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
    metrics = {}
    if method == "attack":
        matches = re.findall(r"\*{4}After \w+ Attack\*{4}\s*ASR:\s*(" + number
                             + r")[^\r\n]*\r?\nClean Accuracy:\s*(" + number + r")", content)
        if matches:
            metrics.update(ASR=float(matches[-1][0]), CA=float(matches[-1][1]))
    else:
        final_json = [line.partition(' ')[2] for line in content.splitlines()
                      if line.startswith('FINAL_COMPARISON_JSON ')]
        if final_json:
            metrics.update(json.loads(final_json[-1]))
        marker = "FINAL_RIGBD_RESULT" if method == "rigbd" else "FINAL_DEFENSE_RESULT"
        lines = [line for line in content.splitlines() if line.startswith(marker + " ")]
        if lines:
            metrics.update({key: float(value) for key, value in re.findall(
                r"\b(CA|ASR|Precision|Recall|FPR)=(" + number + r")", lines[-1])})
    for key in ('CA', 'ASR', 'Precision', 'Recall', 'FPR'):
        if key in metrics and metrics[key] is not None and not (
                math.isfinite(metrics[key]) and 0 <= metrics[key] <= 1):
            del metrics[key]
    sha = re.findall(r"^SNAPSHOT_SHA256:\s*([a-fA-F0-9]{64})\s*$", content, re.MULTILINE)
    if sha:
        metrics["snapshot_sha256"] = sha[-1]
    return metrics


def write_report(path, args, jobs, records, started, overall):
    def cell(value):
        return str(value).replace("|", "\\|").replace("\r", " ").replace("\n", " ")

    def link(target):
        from urllib.parse import quote
        try:
            relative = os.path.relpath(target, path.parent).replace(os.sep, "/")
            url = quote(relative, safe='/.:')
        except ValueError:  # Windows paths on different drives
            url = target.resolve().as_uri()
        return f"[{cell(target.name)}]({url})"

    def percent(metrics, key):
        return f"{metrics[key] * 100:.4f}" if metrics.get(key) is not None else "—"

    lines = ["# 攻击与防御实验结果", "", f"- 开始时间：{started}",
             f"- 报告更新时间：{datetime.now().astimezone().isoformat(timespec='seconds')}",
             f"- 运行状态：{cell(overall)}", f"- 阶段：{args.stage}；数据集：{getattr(args, 'dataset', 'Cora')}；模型：{getattr(args, 'model', 'GCN')}；seed：{args.seed}",
             f"- Python：`{sys.executable}`；物理 GPU：`{args.gpu}`",
             f"- 攻击目录：`{project_path(args.attack_root)}`",
             f"- 防御目录：`{project_path(args.defense_root)}`", "",
             "CA 为干净测试准确率，ASR 为非目标类测试节点上的指定类别预测率；五项比例指标单位为 %，耗时单位为秒。",
             "攻击入口与 none 分别训练分类器；仅当分类器构造前随机状态、训练输入和超参数均对齐时，才应得到一致结果。缺失指标显示 —，不视为零。", "",
             "方法耗时为预处理、建模、训练和评估的同步计时，不含数据加载和文件写入。检测耗时截止检测/图过滤完成；RIGBD 含初始模型训练，不含检测后的重训。网格总耗时包含全部候选。",
             "Precision、Recall、FPR 在同一训练节点总体上计算；none、abp、prune 的三项检测指标记为 —。RIGBD 默认根据训练标签推断类别与分界，网格按最高验证准确率选择，平局依次选更小 K、b。",
             "rigbd 为无自环检测器；rigbd_selfloop 为显式自环探测变体，默认固定 K=20、b=0.5。两者均按消息接收端恢复孤立节点，重训沿用 LoSplit 概率均值目标。", "",
             "| 攻击 | 阶段/方法 | 状态 | CA (%) | ASR (%) | Precision (%) | Recall (%) | FPR (%) | 方法耗时(s) | 检测耗时(s) | 网格总耗时(s) | 进程耗时(s) | 参数 | 日志 |",
             "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|"]
    for job in jobs:
        name, entry, arguments, outputs, inputs, log = job
        attack, method = name.split("/")
        record = records.get(name, {"status": "未运行"})
        metrics = record.get("metrics", {})
        elapsed = f"{metrics['Seconds']:.4f}" if 'Seconds' in metrics else "—"
        detection = f"{metrics['DetectionSeconds']:.4f}" if metrics.get('DetectionSeconds') is not None else '—'
        grid = f"{metrics['GridSeconds']:.4f}" if 'GridSeconds' in metrics else "—"
        wall = f"{record['seconds']:.2f}" if 'seconds' in record else "—"
        parameters = f"K={metrics['K']}, b={metrics['b']}" if method in ('rigbd', 'rigbd_selfloop') and 'K' in metrics else (
            f"multiplier={metrics['Multiplier']}, strength={metrics['Strength']}" if 'Multiplier' in metrics else '—')
        if method == 'losplit' and 'SplitLR' in metrics:
            parameters = (f"split_lr={metrics['SplitLR']}, split_epoch={metrics['SplitEpochs']}, "
                          f"gamma={metrics['Gamma']}, selected_epoch={metrics['BestSplitEpoch']}")
        if method in ('abp_reference','abp_feature_reference') and 'Configuration' in metrics:
            cfg = metrics['Configuration']
            parameters = (f"multiplier={cfg['multiplier']}, rho={cfg['ratio']}, m={metrics['ReferenceCount']}, "
                          f"alpha={cfg['risk']}, M={cfg['members']}, variant={cfg['variant']}")
            if method=='abp_feature_reference':parameters+=f", repair={cfg['feature_repair']}"
        lines.append("| " + " | ".join([attack, method, cell(record["status"]),
                     *[percent(metrics, key) for key in ("CA", "ASR", "Precision", "Recall", "FPR")],
                     elapsed, detection, grid, wall, parameters, link(log) if log.exists() else "—"]) + " |")
    lines += ["", "## 文件与执行记录", "",
              "all 模式按攻击逐个执行：先生成 generator.pth / poison.pt，再将同一对文件传给该攻击的所有防御。",
              "defense 模式仅汇总本次选择的防御任务；历史攻击指标可在其 attack.log 查看。", ""]
    for name, entry, arguments, outputs, inputs, log in jobs:
        record = records.get(name, {})
        lines += [f"### {name}", ""]
        if record.get("error"):
            lines += [f"错误：{cell(record['error'])}", ""]
        metrics = record.get("metrics", {})
        if metrics.get("snapshot_sha256"):
            lines += [f"投毒快照 SHA256：`{metrics['snapshot_sha256']}`", ""]
        for label, paths in (("输入", inputs), ("输出", outputs)):
            for target in paths:
                lines.append(f"- {label}：{link(target)}（{'存在' if target.is_file() else '不存在'}）")
        # JSON preserves argument boundaries on both Windows and Linux.
        lines += ["", "执行参数（JSON 数组；未运行任务为计划参数）：", "", "```json",
                  json.dumps([sys.executable, "-u", str(entry), *arguments], ensure_ascii=False), "```", ""]
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("\n".join(lines), encoding="utf-8")
    temporary.replace(path)
    result_path = path.with_suffix('.json')
    temporary_json = result_path.with_suffix('.json.tmp')
    temporary_json.write_text(json.dumps(dict(status=overall, started=started, records=records),
                                          ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temporary_json.replace(result_path)


def run_job(job, env):
    name, entry, arguments, outputs, inputs, log = job
    missing = [str(path) for path in inputs if not path.is_file()]
    if missing:
        raise RuntimeError("Missing paired attack files: " + ", ".join(missing))
    for path in outputs:
        path.parent.mkdir(parents=True, exist_ok=True)
    print(f"\nRunning {name}; log: {log}", flush=True)
    with log.open("w", encoding="utf-8") as stream:
        process = subprocess.Popen([sys.executable, "-u", str(entry), *arguments],
                                   cwd=str(PROJECT), env=env, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True,
                                   encoding="utf-8", errors="replace")
        try:
            for line in process.stdout:
                print(line, end="", flush=True)
                stream.write(line)
                stream.flush()
            code = process.wait()
        except BaseException:
            if process.poll() is None:
                process.terminate()
                process.wait()
            raise
    if code:
        raise RuntimeError(f"{name} failed (exit {code}); see {log}")
    missing = [str(path) for path in outputs if not path.is_file()]
    if missing:
        raise RuntimeError("Process succeeded but outputs are missing: " + ", ".join(missing))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["attack", "defense", "all"], default=SETTINGS["stage"])
    parser.add_argument("--attacks", nargs="+", choices=list(ATTACK_ARGS), default=SETTINGS["attacks"])
    parser.add_argument("--methods", nargs="+", choices=list(METHOD_ARGS), default=SETTINGS["methods"])
    parser.add_argument('--rigbd-fixed', action='store_true', help='Use fixed K/b instead of the legacy grid')
    parser.add_argument('--rigbd-k', type=int, default=20)
    parser.add_argument('--rigbd-b', type=float, default=.5)
    parser.add_argument('--rigbd-selection', choices=('label_boundary','losplit','fixed_count'), default='label_boundary')
    parser.add_argument('--rigbd-sampling', choices=('receiver','legacy_source'), default='receiver')
    parser.add_argument('--rigbd-grid-selection', choices=('validation','min_asr'), default='validation')
    parser.add_argument('--dataset', type=lambda s: {k.lower(): k for k in ('Cora', 'Citeseer', 'Pubmed', 'Physics')}.get(s.lower(), s),
                        choices=['Cora', 'Citeseer', 'Pubmed', 'Physics'], default=SETTINGS['dataset'])
    parser.add_argument('--model', type=lambda s: {k.lower(): k for k in ('GCN', 'GAT', 'GraphSage')}.get(s.lower(), s),
                        choices=['GCN', 'GAT', 'GraphSage'], default=SETTINGS['model'])
    parser.add_argument('--deterministic', action=argparse.BooleanOptionalAction, default=SETTINGS['deterministic'])
    parser.add_argument('--abp-sparse-multiplier', type=float, default=SETTINGS['abp_sparse_multiplier'])
    parser.add_argument('--abp-sparse-strength', type=float, default=SETTINGS['abp_sparse_strength'])
    for key, kind in [('multiplier',float),('ratio',float),('count',int),('risk',float),('members',int),('seed',int)]:
        parser.add_argument('--abp-reference-' + key, type=kind, default=SETTINGS['abp_reference_' + key])
    parser.add_argument('--abp-reference-variant', choices=['full','no_reference','full_projection','keep_detected_labels'],
                        default=SETTINGS['abp_reference_variant'])
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--reference-device', choices=['cpu','cuda:0'], default=None)
    parser.add_argument('--feature-members',type=int,default=SETTINGS['feature_members'])
    parser.add_argument('--feature-repair',choices=['unique_mode','none','clip_normalize'],default=SETTINGS['feature_repair'])
    parser.add_argument('--feature-tolerance',type=float,default=SETTINGS['feature_tolerance'])
    parser.add_argument("--seed", type=int, default=SETTINGS["seed"])
    parser.add_argument("--gpu", default=SETTINGS["gpu"])
    parser.add_argument("--attack-root", default=SETTINGS["attack_root"])
    parser.add_argument("--defense-root", default=SETTINGS["defense_root"])
    parser.add_argument("--report", default=SETTINGS["report"], help="Markdown report path")
    parser.add_argument("--dry-run", action="store_true", help="Print jobs without importing torch or writing files")
    args = parser.parse_args()
    if 'abp_feature_reference' in args.methods and (args.dataset!='Cora' or args.model!='GCN'):
        parser.error('The feature-consistency protocol currently requires Cora and GCN')
    if args.feature_members<1 or not math.isfinite(args.feature_tolerance) or args.feature_tolerance<=0:
        parser.error('Feature members and tolerance must be positive and finite')
    if not math.isfinite(args.abp_sparse_multiplier) or args.abp_sparse_multiplier <= 0:
        parser.error('Sparse multiplier must be finite and positive')
    if not math.isfinite(args.abp_sparse_strength) or args.abp_sparse_strength < 0:
        parser.error('Sparse strength must be finite and nonnegative')
    if args.epochs < 1 or args.abp_reference_members < 1:
        parser.error('Epochs and reference members must be positive')
    if not math.isfinite(args.abp_reference_multiplier) or args.abp_reference_multiplier < 1:
        parser.error('Reference multiplier must be finite and at least one')
    if not math.isfinite(args.abp_reference_ratio) or not 0 < args.abp_reference_ratio < 1:
        parser.error('Reference ratio must lie strictly between zero and one')
    if not math.isfinite(args.abp_reference_risk) or not 0 < args.abp_reference_risk < 1:
        parser.error('Reference risk must lie strictly between zero and one')
    if args.abp_reference_count is not None and args.abp_reference_count < 1:
        parser.error('Reference count must be positive')
    for key in ('attack_root', 'defense_root'):
        setattr(args, key, getattr(args, key).format(dataset=args.dataset, model=args.model, seed=args.seed))
    jobs = build_jobs(args)
    report = project_path(args.report) if args.report else project_path(
        args.attack_root if args.stage == "attack" else args.defense_root) / "results.md"
    env = os.environ.copy()
    # Keep DPGBA's local modules first while making shared root modules importable.
    env['PYTHONPATH'] = str(PROJECT) + (os.pathsep + env['PYTHONPATH'] if env.get('PYTHONPATH') else '')
    env.update(CUDA_VISIBLE_DEVICES=str(args.gpu), PYTHONHASHSEED=str(args.seed),
               CUBLAS_WORKSPACE_CONFIG=":4096:8", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
               MPLBACKEND="Agg", PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
    print(f"Python: {sys.executable}\nProject: {PROJECT}\nSeed: {args.seed}; physical GPU: {args.gpu}")
    print(f"Markdown report: {report}")
    for name, entry, arguments, outputs, inputs, log in jobs:
        print(f"\n{name}: {[sys.executable, '-u', str(entry), *arguments]}")
    if args.dry_run:
        return
    if report.exists():
        parser.error(f"Report already exists: {report}. Choose a new --report path.")
    if report.suffix.lower() != '.md':
        parser.error('Report path must end with .md (a companion .json is also written)')
    if report.with_suffix('.json').exists():
        parser.error(f"Result already exists: {report.with_suffix('.json')}")
    protected = [p.resolve() for job in jobs for p in [job[1], *job[3], *job[4], job[5]]]
    if report in protected or report == Path(__file__).resolve():
        parser.error("Report path must differ from scripts, inputs, outputs and logs.")
    # Refuse overwrites before starting any expensive training.
    for name, entry, arguments, outputs, inputs, log in jobs:
        for path in [*outputs, log]:
            if path.exists():
                parser.error(f"Output already exists: {path}. Choose a new output root.")
        if args.stage == "defense":
            for path in inputs:
                if not path.is_file():
                    parser.error(f"Missing attack file: {path}")
    if args.stage in ("attack", "all") and "DPGBA" in args.attacks:
        print("Note: original DPGBA also writes project/model_weights.pth.", flush=True)
        if (PROJECT / "model_weights.pth").exists():
            parser.error("Back up or move existing model_weights.pth before generating DPGBA.")
    # Use the same interpreter and environment as training; CUDA failure stops early.
    started = datetime.now().astimezone().isoformat(timespec="seconds")
    records = {}
    overall = "运行中"
    write_report(report, args, jobs, records, started, overall)
    try:
        cpu_reference_only = args.stage == 'defense' and set(args.methods) <= {'abp_reference','abp_feature_reference'} and args.reference_device == 'cpu'
        preflight = "import torch; print('torch:', torch.__version__); " + (
            "print('Device: CPU')" if cpu_reference_only else
            "assert torch.cuda.is_available(), 'CUDA unavailable in this Python environment'; print('GPU:', torch.cuda.get_device_name(0))")
        subprocess.run([sys.executable, "-c", preflight],
                       cwd=str(PROJECT), env=env, check=True)
        for job in jobs:
            name = job[0]
            record = records[name] = {"status": "运行中"}
            write_report(report, args, jobs, records, started, overall)
            tick = time.monotonic()
            try:
                run_job(job, env)
                record["status"] = "成功"
            except BaseException as error:
                record.update(status="中断" if isinstance(error, KeyboardInterrupt) else "失败",
                              error=str(error) or type(error).__name__)
                raise
            finally:
                record["seconds"] = time.monotonic() - tick
                record["metrics"] = parse_metrics(job[5], name.split("/")[1])
                required = {'CA', 'ASR'} if name.endswith('/attack') else {'CA', 'ASR', 'Precision', 'Recall', 'FPR', 'Seconds'}
                if record["status"] == "成功" and not required <= record["metrics"].keys():
                    record["status"] = "成功（结果指标不完整，请检查日志）"
                write_report(report, args, jobs, records, started, overall)
        overall = "全部完成"
    except BaseException as error:
        overall = ("用户中断" if isinstance(error, KeyboardInterrupt) else "失败") + ": " + (str(error) or type(error).__name__)
        raise
    finally:
        write_report(report, args, jobs, records, started, overall)
    print("\nAll requested experiments completed.")
    print(f"Markdown report saved: {report}")


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, subprocess.CalledProcessError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        sys.exit(1)
