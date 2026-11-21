"""Diagnose GTA baseline RNG with isolated outputs; original entry files stay intact."""
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "runs" / "baseline_rng_audit_20261121"


def child(mode):
    import numpy as np
    import torch
    from run_experiments import ATTACK_ARGS

    def tensor_hash(tensor):
        data = tensor.detach().cpu().contiguous()
        return hashlib.sha256(str((str(data.dtype), tuple(data.shape))).encode()
                              + data.numpy().tobytes()).hexdigest()

    def state_hash(model):
        return {key: tensor_hash(value) for key, value in model.state_dict().items()}

    observed = {}

    def before(scope):
        state = {"cpu": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all(),
                 "numpy": np.random.get_state(), "python": random.getstate()}
        observed["rng_before_replay"] = {"cpu": tensor_hash(state["cpu"]),
                                         "cuda": [tensor_hash(s) for s in state["cuda"]]}
        if mode == "attack":
            torch.save(state, OUT / "classifier_rng.pt")
        elif mode == "replay":
            state = torch.load(OUT / "classifier_rng.pt", map_location="cpu")
            torch.set_rng_state(state["cpu"])
            torch.cuda.set_rng_state_all(state["cuda"])
            np.random.set_state(state["numpy"])
            random.setstate(state["python"])
        observed["training_inputs"] = {key: tensor_hash(scope[key]) for key in (
            "poison_x", "poison_edge_index", "poison_edge_weights", "poison_labels",
            "bkd_tn_nodes", "idx_val", "idx_clean_test", "idx_atk", "mask_edge_index")}

    entry = ROOT / ("GTA_LoSplit.py" if mode == "attack" else "GTA_RIGBD.py")
    common = ["--deterministic", "--dataset=Cora", "--model=GCN", "--test_model=GCN",
              "--device_id=0", "--seed=10", "--hidden=32", "--epochs=200", "--target_class=0",
              f"--trigger_generator_address={OUT / 'generator.pth'}",
              f"--pre_train_param={OUT / 'poison.pt'}"]
    sys.argv = [str(entry), *common, *(ATTACK_ARGS["GTA"] if mode == "attack" else [
        "--train_lr=0.01", "--defense_mode=none", "--skip_rigbd",
        f"--model_save_path={OUT / (mode + '.pth')}"])]
    source = entry.read_text(encoding="utf-8-sig")
    needle = "test_model = model_construct(args,args.test_model,data,device).to(device)"
    changed = []
    count = 0
    for line in source.splitlines():
        if line.strip() == needle:
            indent = line[:len(line) - len(line.lstrip())]
            changed.extend([indent + "_audit_before(globals())", line,
                            indent + "_audit_observed['initial_parameters'] = _audit_hash(test_model)"])
            count += 1
        else:
            changed.append(line)
    assert count == 1, count
    scope = {"__name__": "__main__", "__file__": str(entry), "_audit_before": before,
             "_audit_observed": observed, "_audit_hash": state_hash}
    try:
        exec(compile("\n".join(changed), str(entry), "exec"), scope)
    except SystemExit as error:
        if error.code not in (None, 0):
            raise
    observed["final_parameters"] = state_hash(scope["test_model"])
    observed["CA"] = float(scope["clean_acc"])
    observed["ASR"] = float(scope["flip_asr"])
    (OUT / (mode + ".json")).write_text(json.dumps(observed, indent=2), encoding="utf-8")


def main():
    if len(sys.argv) > 1:
        child(sys.argv[1])
        return
    OUT.mkdir(parents=True, exist_ok=False)
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES="0", PYTHONHASHSEED="10", CUBLAS_WORKSPACE_CONFIG=":4096:8",
               OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", MPLBACKEND="Agg", PYTHONUTF8="1",
               PYTHONIOENCODING="utf-8")
    for mode in ("attack", "none", "replay"):
        with (OUT / (mode + ".log")).open("w", encoding="utf-8") as log:
            subprocess.run([sys.executable, str(Path(__file__).resolve()), mode], cwd=ROOT,
                           env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        print(mode, "completed", flush=True)
    data = {mode: json.loads((OUT / (mode + ".json")).read_text())
            for mode in ("attack", "none", "replay")}
    summary = {}
    for mode in ("none", "replay"):
        summary[mode] = {key + "_equal": data["attack"][key] == data[mode][key]
                         for key in ("training_inputs", "initial_parameters", "final_parameters")}
    summary["metrics"] = {mode: {key: row[key] for key in ("CA", "ASR")}
                          for mode, row in data.items()}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
