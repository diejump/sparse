"""Registered paired studies, bounded subprocesses, and incremental records."""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parent
CASES={'box':['--repair','box'],'median':['--repair','median'],
    'ordinary':['--repair','median','--variant','no_reference'],
    'radial':['--repair','median','--score','radial'],
    'no_repair':['--repair','none'],'no_structure':['--repair','median','--no-structure'],
    'range':['--repair','median','--score','range'],
    'range_ordinary':['--repair','median','--score','range','--variant','no_reference'],
    'old':[],'none':[]}

def save(root,study):
    temp=root/'study.json.tmp';temp.write_text(json.dumps(study,indent=2,allow_nan=False),encoding='utf-8');temp.replace(root/'study.json')

def execute(job,timeout):
    folder=Path(job['output']).parent;folder.mkdir(parents=True,exist_ok=True)
    env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONUTF8='1',PYTHONIOENCODING='utf-8',
        PYTHONHASHSEED=str(job['seed']),CUBLAS_WORKSPACE_CONFIG=':4096:8',MPLBACKEND='Agg')
    start=time.monotonic()
    try:
        with Path(job['log']).open('w',encoding='utf-8') as stream:
            proc=subprocess.run(job['command'],cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,timeout=timeout)
        if proc.returncode:return dict(status='failed',returncode=proc.returncode,seconds=time.monotonic()-start)
        result=json.loads(Path(job['output']+'.meta.json').read_text(encoding='utf-8'))
        return dict(status='completed',seconds=time.monotonic()-start,metrics={k:result.get(k) for k in
            ['CA','ASR','Precision','Recall','FPR','TP','FP','FN','TN','CleanCorrect','CleanCount','ASRHits','ASRCount',
             'SameGraphCA','AffectedAccuracy','AffectedTargetAccuracy','ReloadVerified']})
    except subprocess.TimeoutExpired:return dict(status='timeout',seconds=time.monotonic()-start)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase',choices=['development','evaluation'],required=True)
    p.add_argument('--output',required=True);p.add_argument('--input-root')
    p.add_argument('--dataset',default='Cora');p.add_argument('--seeds',nargs='+',type=int,default=[12])
    p.add_argument('--kinds',nargs='+',default=['GTA','UGBA','DPGBA','SPEAR'])
    p.add_argument('--architectures',nargs='+',default=['GCN'])
    p.add_argument('--cases',nargs='+',choices=list(CASES),default=['box','median','ordinary','none','old'])
    p.add_argument('--gamma',type=float,default=1.);p.add_argument('--rho',type=float,default=.2)
    p.add_argument('--alpha',type=float,default=.005);p.add_argument('--device',default='cpu')
    p.add_argument('--workers',type=int,default=2);p.add_argument('--timeout',type=int,default=1800)
    p.add_argument('--execute',action='store_true');a=p.parse_args()
    root=Path(a.output).resolve()
    if (root/'study.json').exists():p.error('Choose a new study directory')
    root.mkdir(parents=True,exist_ok=True);jobs=[]
    for seed in a.seeds:
        for kind in a.kinds:
            if a.phase=='development':
                snap=ROOT/('runs/spear_losplit_20261126/seed12/inputs/SPEAR/poison.pt' if kind=='SPEAR'
                    else f'runs/abp_sparse_20261123/seed16/inputs/{kind}/poison.pt')
            else:
                if not a.input_root:p.error('Evaluation requires --input-root')
                snap=Path(a.input_root.format(seed=seed,dataset=a.dataset))/kind/'poison.pt'
                if not snap.is_absolute():snap=ROOT/snap
            for arch in a.architectures:
                for case in a.cases:
                    output=root/f'seed{seed}'/kind/arch/case/'model.pt'
                    if case in ('none','old'):
                        flags=[] if case=='old' else ['--no-feature-detection','--no-structure','--no-reference','--feature-repair','none']
                        cmd=[sys.executable,'-u',str(ROOT/'feature_reference_experiment.py'),'--pre_train_param',str(snap),
                            '--model_save_path',str(output),'--dataset',a.dataset,'--test_model',arch,'--seed',str(seed),
                            '--device',a.device,'--members','1','--deterministic',*flags]
                    else:
                        cmd=[sys.executable,'-u',str(ROOT/'envelope_experiment.py'),'--snapshot',str(snap),'--output',str(output),
                            '--architecture',arch,'--seed',str(seed),'--device',a.device,'--gamma',str(a.gamma),
                            '--rho',str(a.rho),'--alpha',str(a.alpha),*CASES[case]]
                    jobs.append(dict(id=f'{seed}/{kind}/{arch}/{case}',seed=seed,kind=kind,architecture=arch,
                        case=case,snapshot=str(snap),output=str(output),log=str(output.with_suffix('.log')),
                        command=cmd,status='planned'))
    sources=['models/EmpiricalEnvelope.py','envelope_experiment.py','reference_training.py',
        'feature_reference_experiment.py','models/ABP_Reference.py','run_generalization_study.py']
    study=dict(phase=a.phase,dataset=a.dataset,created_utc=datetime.now(timezone.utc).isoformat(),
        configuration=vars(a),source_sha256={s:hashlib.sha256((ROOT/s).read_bytes()).hexdigest() for s in sources},jobs=jobs)
    save(root,study)
    print(f'Registered {len(jobs)} jobs: {root}',flush=True)
    if not a.execute:return
    if any(not Path(j['snapshot']).is_file() for j in jobs):raise FileNotFoundError('Some registered snapshots are missing')
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        futures={pool.submit(execute,j,a.timeout):j for j in jobs}
        for future in as_completed(futures):
            job=futures[future]
            try:job.update(future.result())
            except Exception as error:job.update(status='failed',error=repr(error))
            save(root,study)
            print(job['id'],job['status'],job.get('metrics',{}),flush=True)
    if any(j['status']!='completed' for j in jobs):raise SystemExit(1)

if __name__=='__main__':main()
