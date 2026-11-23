"""Run the preselected settings and retain one log per batch."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'runs/abp_sparse_20261123/parameter_study'

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--phase',choices=['confirmation','validation'],required=True)
    p.add_argument('--kind',choices=['GTA','UGBA','DPGBA'],required=True)
    args=p.parse_args()
    configs=json.loads((OUT/'screen_selection.json').read_text())['confirmation_configurations']
    if args.phase=='validation':
        assert (OUT/'confirmation_selection.json').exists()
    env=os.environ.copy();env.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONHASHSEED='10')
    log_path=OUT/f'{args.phase}_{args.kind}.log'
    with log_path.open('w',encoding='utf-8') as log:
        for config in configs:
            cmd=[sys.executable,'-u',str(Path(__file__).with_name('tune_sparse.py')),
                '--phase',args.phase,'--kinds',args.kind,
                '--multipliers',str(config['multiplier']),'--penalties',str(config['penalty_weight']),
                '--seeds',*(['20','30'] if args.phase=='confirmation' else ['10','20','30']),
                '--generations',*(['10'] if args.phase=='confirmation' else ['11','12','13'])]
            if args.phase=='validation':cmd+=['--evaluate']
            print(json.dumps({'kind':args.kind,'phase':args.phase,'configuration':config}),flush=True)
            result=subprocess.run(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=1800)
            if result.returncode:
                raise SystemExit(result.returncode)

if __name__=='__main__':main()
