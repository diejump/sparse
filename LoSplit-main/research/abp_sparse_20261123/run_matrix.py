"""Run the fixed protocol in isolated processes, retaining every log/result."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--generation-seeds',nargs='+',type=int,default=[10,11,12])
    p.add_argument('--classifier-seeds',nargs='+',type=int,default=[10,20,30])
    p.add_argument('--kinds',nargs='+',choices=['GTA','UGBA','DPGBA'],default=['GTA','UGBA','DPGBA'])
    p.add_argument('--methods',nargs='+',default=['none','abp','abp_all_candidates','abp_sparse'])
    p.add_argument('--stress',choices=['none','no_internal'],default='none')
    p.add_argument('--device',default='cpu')
    p.add_argument('--output-name',default='validation_v2')
    args=p.parse_args()
    for seed in args.generation_seeds:
        for kind in args.kinds:
            if seed==10:
                src=ROOT/'runs'/('dpgba_official_fix_20261123/inputs' if kind=='DPGBA' else 'my_cora_attacks3')/kind/'poison.pt'
            else:
                src=ROOT/'runs/abp_sparse_20261123'/f'seed{seed}'/'inputs'/kind/'poison.pt'
            if not src.exists():
                raise FileNotFoundError(src)
            dest=ROOT/'runs/abp_sparse_20261123'/args.output_name/f'gen{seed}'/kind
            dest.mkdir(parents=True,exist_ok=True)
            if (dest/'results.json').exists():
                raise FileExistsError(dest/'results.json')
            command=[sys.executable,'-u',str(Path(__file__).with_name('run_study.py')),
                '--snapshot',str(src),'--output',str(dest),'--methods',*args.methods,
                '--seeds',*map(str,args.classifier_seeds),'--stress',args.stress,'--device',args.device]
            env=os.environ.copy()
            env.update(PYTHONHASHSEED=str(seed),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',MPLBACKEND='Agg',CUBLAS_WORKSPACE_CONFIG=':4096:8')
            print(f'RUN generation={seed} kind={kind} log={dest / "run.log"}',flush=True)
            start=time.monotonic()
            with (dest/'run.log').open('w',encoding='utf-8') as log:
                result=subprocess.run(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=1800)
            print(f'FINISHED generation={seed} kind={kind} return={result.returncode} seconds={time.monotonic()-start:.1f}',flush=True)
            if result.returncode:
                raise SystemExit(result.returncode)

if __name__=='__main__':
    main()
