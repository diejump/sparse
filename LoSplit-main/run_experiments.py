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
SETTINGS = {
    "stage": "all",                 # attack / defense / all
    "attacks": ["GTA", "UGBA", "DPGBA"],
    "methods": ["none", "prune", "rigbd", "abp", "abp_dual", "abp_assisted"],
    "seed": 10,
    "gpu": "0",                     # physical GPU; child uses visible device 0
    "attack_root": "runs/my_cora_attacks3",
    "defense_root": "runs/my_cora_defenses3",
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
}
METHOD_ARGS = {
    "none": ["--defense_mode=none", "--skip_rigbd"],
    "prune": ["--defense_mode=prune", "--skip_rigbd"],
    "rigbd": ["--defense_mode=none", "--rigbd_k=20", "--rigbd_b=0.5"],
    "abp": ["--defense_mode=abp"],
    "abp_dual": ["--defense_mode=abp_dual"],
    "abp_assisted": ["--defense_mode=abp_assisted"],
}


def project_path(value):
    path = Path(value).expanduser()
    return (PROJECT / path).resolve()


def build_jobs(args):
    attack_root = project_path(args.attack_root)
    defense_root = project_path(args.defense_root)
    common = ["--deterministic", "--dataset=Cora", "--model=GCN", "--test_model=GCN",
              "--device_id=0", f"--seed={args.seed}", "--hidden=32",
              "--epochs=200", "--target_class=0"]
    jobs = []
    for attack in args.attacks:
        prefix = "DPGBA/" if attack == "DPGBA" else ""
        generator = attack_root / attack / "generator.pth"
        snapshot = attack_root / attack / "poison.pt"
        inputs = [f"--trigger_generator_address={generator}", f"--pre_train_param={snapshot}"]
        if args.stage in ("attack", "all"):
            jobs.append((f"{attack}/attack", PROJECT / f"{prefix}{attack}_LoSplit.py",
                         common + ATTACK_ARGS[attack] + inputs,
                         [generator, snapshot], [], attack_root / attack / "attack.log"))
        if args.stage in ("defense", "all"):
            for method in args.methods:
                output = defense_root / attack / f"{method}.pth"
                extra = METHOD_ARGS[method].copy()
                if method == "prune":
                    extra.append(f"--prune_thr={0.2 if attack == 'UGBA' else 0.8}")
                jobs.append((f"{attack}/{method}", PROJECT / f"{prefix}{attack}_RIGBD.py",
                             common + ["--train_lr=0.015" if attack == "DPGBA" else "--train_lr=0.01"] + inputs + extra
                             + [f"--model_save_path={output}"],
                             [output], [generator, snapshot], output.with_suffix(".log")))
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
        marker = "FINAL_RIGBD_RESULT" if method == "rigbd" else "FINAL_DEFENSE_RESULT"
        lines = [line for line in content.splitlines() if line.startswith(marker + " ")]
        if lines:
            metrics.update({key: float(value) for key, value in re.findall(
                r"\b(CA|ASR|Precision|Recall|FPR)=(" + number + r")", lines[-1])})
    metrics = {key: value for key, value in metrics.items() if math.isfinite(value) and 0 <= value <= 1}
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
        return f"{metrics[key] * 100:.4f}" if key in metrics else "—"

    lines = ["# 攻击与防御实验结果", "", f"- 开始时间：{started}",
             f"- 报告更新时间：{datetime.now().astimezone().isoformat(timespec='seconds')}",
             f"- 运行状态：{cell(overall)}", f"- 阶段：{args.stage}；数据集：Cora；模型：GCN；seed：{args.seed}",
             f"- Python：`{sys.executable}`；物理 GPU：`{args.gpu}`",
             f"- 攻击目录：`{project_path(args.attack_root)}`",
             f"- 防御目录：`{project_path(args.defense_root)}`", "",
             "CA 为干净测试准确率，ASR 为非目标类攻击节点上的攻击成功率；下表指标单位均为 %。",
             "攻击入口与 none 分别训练分类器；仅当分类器构造前随机状态、训练输入和超参数均对齐时，才应得到一致结果。缺失指标显示 —，不视为零。", "",
             "| 攻击 | 阶段/方法 | 状态 | CA (%) | ASR (%) | Precision (%) | Recall (%) | FPR (%) | 耗时(s) | 日志 |",
             "|---|---|---|---:|---:|---:|---:|---:|---:|---|"]
    for job in jobs:
        name, entry, arguments, outputs, inputs, log = job
        attack, method = name.split("/")
        record = records.get(name, {"status": "未运行"})
        metrics = record.get("metrics", {})
        elapsed = f"{record['seconds']:.2f}" if "seconds" in record else "—"
        lines.append("| " + " | ".join([attack, method, cell(record["status"]),
                     *[percent(metrics, key) for key in ("CA", "ASR", "Precision", "Recall", "FPR")],
                     elapsed, link(log) if log.exists() else "—"]) + " |")
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
    parser.add_argument("--seed", type=int, default=SETTINGS["seed"])
    parser.add_argument("--gpu", default=SETTINGS["gpu"])
    parser.add_argument("--attack-root", default=SETTINGS["attack_root"])
    parser.add_argument("--defense-root", default=SETTINGS["defense_root"])
    parser.add_argument("--report", default=SETTINGS["report"], help="Markdown report path")
    parser.add_argument("--dry-run", action="store_true", help="Print jobs without importing torch or writing files")
    args = parser.parse_args()
    jobs = build_jobs(args)
    report = project_path(args.report) if args.report else project_path(
        args.attack_root if args.stage == "attack" else args.defense_root) / "results.md"
    env = os.environ.copy()
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
        subprocess.run([sys.executable, "-c", "import torch; print('torch:', torch.__version__); "
                        "assert torch.cuda.is_available(), 'CUDA unavailable in this Python environment'; "
                        "print('GPU:', torch.cuda.get_device_name(0))"],
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
                if record["status"] == "成功" and not {"CA", "ASR"} <= record["metrics"].keys():
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
