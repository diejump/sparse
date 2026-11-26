"""Run recorded feature-method comparisons without choosing from test results."""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
from dataclasses import asdict
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from feature_reference_experiment import FeatureReferenceConfig

ROOT=Path(__file__).resolve().parent
CASES={
    'full':[],
    'members3':['--members','3'],
    'detection_only':['--feature-repair','none'],
    'no_structure':['--no-structure'],
    'no_reference':['--no-reference'],
    'clip_with_detection':['--feature-repair','clip_normalize'],
    'ungated_structure':['--structural-gate','all_edges'],
    'clip_only':['--no-feature-detection','--no-structure','--no-reference','--feature-repair','clip_normalize'],
    'repair_only':['--no-feature-detection','--no-structure','--no-reference'],
    'none':['--no-feature-detection','--no-structure','--no-reference','--feature-repair','none'],
    'original_reference':[],
}


def make_jobs(args):
    jobs=[]
    for seed in args.seeds:
        for kind in args.kinds:
            if args.phase=='development':
                path=(ROOT/'runs/spear_losplit_20261126/seed12/inputs/SPEAR/poison.pt' if kind=='SPEAR'
                      else ROOT/f'runs/abp_sparse_20261123/seed16/inputs/{kind}/poison.pt')
            else:
                path=ROOT/f'runs/feature_reference_20261126/confirmation_inputs/seed{seed}/{kind}/poison.pt'
            for case in args.cases:
                folder=Path(args.output)/f'seed{seed}'/kind/case
                if not folder.is_absolute():folder=ROOT/folder
                entry=ROOT/('reference_experiment.py' if case=='original_reference' else 'feature_reference_experiment.py')
                command=[sys.executable,'-u',str(entry),'--pre_train_param',str(path),
                    '--trigger_generator_address',str(path.with_name('generator.pth')),
                    '--model_save_path',str(folder/'model.pt'),'--dataset','Cora','--test_model','GCN',
                    '--hidden','32','--epochs','200','--train_lr','0.01','--seed',str(seed),
                    '--device',args.device,'--deterministic']
                command += (['--abp_reference_members','1'] if case=='original_reference' else ['--members','1',*CASES[case]])
                jobs.append(dict(id=f'{seed}/{kind}/{case}',seed=seed,kind=kind,case=case,snapshot=str(path),
                    command=command,output=str(folder/'model.pt'),log=str(folder/'run.log'),status='planned'))
    return jobs


def save(root,jobs,protocol):
    root.mkdir(parents=True,exist_ok=True)
    counts={state:sum(j['status']==state for j in jobs) for state in ('planned','completed','failed')}
    temporary=root/'study.json.tmp'
    temporary.write_text(json.dumps(dict(protocol=protocol,counts=counts,jobs=jobs),indent=2,allow_nan=False),encoding='utf-8')
    temporary.replace(root/'study.json')
    lines=['# 特征一致性扩展：实验记录','',f'状态：{counts}','',
        '| 种子 | 方式 | 设置 | 状态 | CA (%) | ASR (%) | Precision (%) | Recall (%) | FPR (%) | 模型数 |',
        '|---:|---|---|---|---:|---:|---:|---:|---:|---:|']
    for job in jobs:
        m=job.get('metrics',{})
        values=[f'{100*m[k]:.4f}' if m.get(k) is not None else '—' for k in ['CA','ASR','Precision','Recall','FPR']]
        lines.append('| '+' | '.join(map(str,[job['seed'],job['kind'],job['case'],job['status'],*values,m.get('ModelCount','—')]))+' |')
    (root/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def execute(job,timeout):
    output=Path(job['output']);log=Path(job['log'])
    if any(p.exists() for p in [output,log,Path(str(output)+'.meta.json')]):
        return dict(status='failed',error='Output already exists')
    output.parent.mkdir(parents=True,exist_ok=True)
    env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONHASHSEED=str(job['seed']),
        CUBLAS_WORKSPACE_CONFIG=':4096:8',MPLBACKEND='Agg',PYTHONUTF8='1',PYTHONIOENCODING='utf-8')
    start=time.monotonic()
    try:
        with log.open('w',encoding='utf-8') as stream:
            run=subprocess.run(job['command'],cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,timeout=timeout)
        if run.returncode:return dict(status='failed',returncode=run.returncode)
        result=json.loads(Path(str(output)+'.meta.json').read_text(encoding='utf-8'))
        metrics={k:result.get(k) for k in ['CA','ASR','Precision','Recall','FPR','TP','FP','FN','TN',
            'CleanCorrect','CleanCount','ASRHits','ASRCount','ModelCount','ReloadVerified','Seconds']}
        if job['case'] in ('none','clip_only','repair_only'):
            for name in ('Precision','Recall','FPR'):metrics[name]=None
        return dict(status='completed',wall_seconds=time.monotonic()-start,metrics=metrics)
    except subprocess.TimeoutExpired:return dict(status='failed',error='Configured timeout exceeded')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase',choices=['development','confirmation'],required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--seeds',nargs='+',type=int,default=[10])
    p.add_argument('--kinds',nargs='+',choices=['GTA','UGBA','DPGBA','SPEAR'],default=['GTA','UGBA','DPGBA','SPEAR'])
    p.add_argument('--cases',nargs='+',choices=list(CASES),default=list(CASES))
    p.add_argument('--device',default='cpu')
    p.add_argument('--workers',type=int,default=2)
    p.add_argument('--timeout',type=int,default=1800)
    p.add_argument('--execute',action='store_true')
    args=p.parse_args();root=Path(args.output)
    if not root.is_absolute():root=ROOT/root
    if (root/'study.json').exists():p.error('Choose a new output directory')
    jobs=make_jobs(args)
    if args.execute:
        missing=sorted({j['snapshot'] for j in jobs if not Path(j['snapshot']).exists()})
        if missing:p.error('Missing input snapshots: '+', '.join(missing))
    protocol=dict(phase=args.phase,created=datetime.now().astimezone().isoformat(),dataset='Cora',
        configuration=asdict(FeatureReferenceConfig(members=1)),
        metrics=['CA','ASR','Precision','Recall','FPR'],selection='all_registered_cases_reported',
        source_sha256={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in
            ['models/FeatureConsistency.py','feature_reference_experiment.py','reference_training.py','models/ABP_Reference.py']})
    save(root,jobs,protocol)
    print(f'{len(jobs)} jobs; {root}',flush=True)
    if not args.execute:return
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures={pool.submit(execute,j,args.timeout):j for j in jobs}
        for f in as_completed(futures):
            job=futures[f]
            try:job.update(f.result())
            except Exception as error:job.update(status='failed',error=str(error))
            save(root,jobs,protocol);print(job['id']+' '+job['status'],flush=True)
    if any(j['status']!='completed' for j in jobs):raise SystemExit(1)


if __name__=='__main__':main()
