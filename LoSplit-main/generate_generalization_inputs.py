"""Generate registered paired samples in separate per-job working directories."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from types import SimpleNamespace
from run_experiments import build_jobs

ROOT=Path(__file__).resolve().parent

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',choices=['Cora','Citeseer'],required=True)
    p.add_argument('--seeds',type=int,nargs='+',default=[41]);p.add_argument('--kinds',nargs='+',default=['GTA','UGBA','DPGBA','SPEAR'])
    p.add_argument('--output',required=True);p.add_argument('--timeout',type=int,default=1800)
    a=p.parse_args();root=Path(a.output).resolve()
    if (root/'generation.json').exists():raise FileExistsError(root)
    root.mkdir(parents=True,exist_ok=True)
    source=ROOT/'data'/a.dataset
    if not source.is_dir():
        source=ROOT.parent.parent/'abp_based_on_losplit/data'/a.dataset
    if not source.is_dir():raise FileNotFoundError(source)
    jobs=[]
    for seed in a.seeds:
        config=SimpleNamespace(dataset=a.dataset,model='GCN',stage='attack',seed=seed,deterministic=True,
            attacks=a.kinds,methods=['none'],attack_root=str(root/f'seed{seed}'),defense_root=str(root/f'seed{seed}/unused'))
        for name,entry,arguments,outputs,inputs,log in build_jobs(config):
            if any(q.exists() for q in [*outputs,log]):raise FileExistsError(log)
            jobs.append(dict(seed=seed,name=name,entry=str(entry),command=[sys.executable,'-u',str(entry),*arguments],
                outputs=list(map(str,outputs)),log=str(log),status='planned'))
    record=dict(created_utc=datetime.now(timezone.utc).isoformat(),dataset=a.dataset,
        data_source=str(source),generation_protocol='existing launcher settings; separate writable working directories',jobs=jobs)
    def save():
        (root/'generation.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
    save()
    for job in jobs:
        folder=Path(job['log']).parent;folder.mkdir(parents=True,exist_ok=True)
        workspace=folder/'workspace';workspace.mkdir()
        shutil.copytree(source,workspace/'data'/a.dataset)
        # SPEAR selection keeps its source-defined repository cache paths.
        env=dict(os.environ,PYTHONPATH=str(ROOT),CUDA_VISIBLE_DEVICES='0',PYTHONHASHSEED=str(job['seed']),
            CUBLAS_WORKSPACE_CONFIG=':4096:8',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',MPLBACKEND='Agg',PYTHONUTF8='1',PYTHONIOENCODING='utf-8')
        job['cwd']=str(workspace);job['source_sha256']=hashlib.sha256(Path(job['entry']).read_bytes()).hexdigest()
        job['status']='running';save();print(job['seed'],job['name'],'running',flush=True)
        start=time.monotonic()
        try:
            with Path(job['log']).open('w',encoding='utf-8') as stream:
                process=subprocess.run(job['command'],cwd=workspace,env=env,stdout=stream,stderr=subprocess.STDOUT,timeout=a.timeout)
            job['status']='completed' if process.returncode==0 and all(Path(q).is_file() for q in job['outputs']) else 'failed'
            job['returncode']=process.returncode
        except subprocess.TimeoutExpired:job['status']='timeout'
        job['seconds']=time.monotonic()-start
        if job['status']=='completed':job['output_sha256']={q:hashlib.sha256(Path(q).read_bytes()).hexdigest() for q in job['outputs']}
        save();print(job['seed'],job['name'],job['status'],round(job['seconds'],1),flush=True)
    if any(j['status']!='completed' for j in jobs):raise SystemExit(1)

if __name__=='__main__':main()
