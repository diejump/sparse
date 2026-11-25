"""Build or run a paired one-factor reference-method study on saved snapshots."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent


def cases():
    base = dict(multiplier=2., ratio=.2, risk=.005, members=3, count=None, seed=20261124, variant='full')
    result = [('base', dict(base))]
    for key, levels in [('multiplier',[1.,3.]),('ratio',[.1,.3]),
                        ('risk',[.001,.009,.02]),('members',[1,5])]:
        for value in levels:
            result.append((f'{key}_{value}', dict(base, **{key:value})))
    result.append(('count_120', dict(base, count=120)))
    result.append(('reference_seed_20261125', dict(base, seed=20261125)))
    for variant in ['no_reference','full_projection','keep_detected_labels']:
        result.append((variant, dict(base, variant=variant)))
    return result


def make_jobs(args):
    configurations = cases()
    if args.cases:
        available = dict(configurations)
        unknown = set(args.cases)-set(available)
        if unknown:
            raise ValueError('Unknown cases: ' + ', '.join(sorted(unknown)))
        configurations = [(name,available[name]) for name in args.cases]
    jobs=[]
    for dataset in args.datasets:
        for model in args.models:
            for generation in args.seeds:
                source = Path(args.input_root.format(dataset=dataset, model=model, seed=generation))
                if not source.is_absolute():
                    source = ROOT/source
                for kind in args.kinds:
                    for classifier_seed in args.classifier_seeds:
                        for name, parameters in configurations:
                            folder=Path(args.output_root)/dataset/model/f'gen{generation}'/kind/f'model{classifier_seed}'/name
                            if not folder.is_absolute():
                                folder=ROOT/folder
                            output=folder/'model.pt'
                            command=[sys.executable,'-u',str(ROOT/'reference_experiment.py'),
                                '--pre_train_param',str(source/kind/'poison.pt'),
                                '--trigger_generator_address',str(source/kind/'generator.pth'),
                                '--model_save_path',str(output),'--dataset',dataset,'--test_model',model,
                                '--seed',str(classifier_seed),'--epochs',str(args.epochs),'--device',args.device,
                                '--train_lr',str(.015 if kind=='DPGBA' else .01),'--deterministic']
                            for key,value in parameters.items():
                                if value is not None:
                                    command += ['--abp_reference_'+key,str(value)]
                            jobs.append(dict(id=f'{dataset}/{model}/gen{generation}/{kind}/model{classifier_seed}/{name}',
                                dataset=dataset,model=model,generation=generation,kind=kind,classifier_seed=classifier_seed,
                                case=name,parameters=parameters,command=command,output=str(output),
                                log=str(folder/'run.log'),status='planned'))
    return jobs


def write_reports(root,jobs,protocol):
    root.mkdir(parents=True,exist_ok=True)
    counts={state:sum(j['status']==state for j in jobs) for state in ('planned','running','completed','failed')}
    payload=dict(protocol=protocol,counts=counts,jobs=jobs)
    temporary=root/'study.json.tmp'
    temporary.write_text(json.dumps(payload,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    temporary.replace(root/'study.json')
    lines=['# 参考判别版消融实验','',f'状态：{counts}', '',
        '各行保存同一快照上的五项指标；比例单位为 %。本表用于开发分析，参数按预先列出的方案逐项变化。',
        '重复运行按完整配置和输出路径核对；失败结果保留。', '',
        '| 数据集 | 模型 | 图种子 | 方式 | 模型种子 | 配置 | 状态 | CA | ASR | Precision | Recall | FPR | 秒 |',
        '|---|---|---:|---|---:|---|---|---:|---:|---:|---:|---:|---:|']
    for job in jobs:
        m=job.get('metrics',{})
        values=[f'{100*m[k]:.4f}' if m.get(k) is not None else '—' for k in ('CA','ASR','Precision','Recall','FPR')]
        lines.append('| '+' | '.join(map(str,[job['dataset'],job['model'],job['generation'],job['kind'],
            job['classifier_seed'],job['case'],job['status'],*values,f"{m['Seconds']:.2f}" if m.get('Seconds') else '—']))+' |')
    (root/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def execute(job,timeout):
    log=Path(job['log']); output=Path(job['output'])
    log.parent.mkdir(parents=True,exist_ok=True)
    if output.exists() or Path(str(output)+'.meta.json').exists() or log.exists():
        return dict(status='failed',error='Output already exists; use a new output root')
    env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',MPLBACKEND='Agg',
        PYTHONHASHSEED=str(job['classifier_seed']),CUBLAS_WORKSPACE_CONFIG=':4096:8',
        PYTHONUTF8='1',PYTHONIOENCODING='utf-8')
    started=time.monotonic()
    with log.open('w',encoding='utf-8') as stream:
        try:
            completed=subprocess.run(job['command'],cwd=ROOT,env=env,stdout=stream,
                stderr=subprocess.STDOUT,timeout=timeout)
            if completed.returncode:
                return dict(status='failed',returncode=completed.returncode,seconds=time.monotonic()-started)
        except subprocess.TimeoutExpired:
            return dict(status='failed',error=f'Exceeded configured {timeout}s timeout',seconds=time.monotonic()-started)
    metadata=json.loads(Path(str(output)+'.meta.json').read_text(encoding='utf-8'))
    return dict(status='completed',seconds=time.monotonic()-started,
        metrics={key:metadata[key] for key in ('CA','ASR','Precision','Recall','FPR','Seconds',
            'ReferenceCount','CalibrationCount','ModelCount','ReloadVerified')})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-root',default='runs/comparison_inputs_GCN4/{dataset}/{model}/seed{seed}')
    parser.add_argument('--output-root',required=True)
    parser.add_argument('--datasets',nargs='+',choices=['Cora','Citeseer','Pubmed'],default=['Cora'])
    parser.add_argument('--models',nargs='+',choices=['GCN','GAT','GraphSage'],default=['GCN'])
    parser.add_argument('--seeds',type=int,nargs='+',default=[16])
    parser.add_argument('--classifier-seeds',type=int,nargs='+',default=[10])
    parser.add_argument('--kinds',nargs='+',choices=['GTA','UGBA','DPGBA'],default=['GTA','UGBA','DPGBA'])
    parser.add_argument('--cases',nargs='+')
    parser.add_argument('--epochs',type=int,default=200)
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--workers',type=int,default=1)
    parser.add_argument('--timeout',type=int,default=1800)
    parser.add_argument('--execute',action='store_true')
    args=parser.parse_args()
    if min(args.epochs,args.workers,args.timeout)<1:
        parser.error('Epochs, workers and timeout must be positive')
    jobs=make_jobs(args)
    root=Path(args.output_root)
    if not root.is_absolute(): root=ROOT/root
    if (root/'study.json').exists(): parser.error('Choose a new output root')
    if args.execute:
        missing=sorted({str(Path(j['command'][j['command'].index('--pre_train_param')+1])) for j in jobs
            if not Path(j['command'][j['command'].index('--pre_train_param')+1]).is_file()})
        if missing: parser.error('Missing snapshots: '+', '.join(missing))
    protocol=dict(created=datetime.now().astimezone().isoformat(),epochs=args.epochs,
        design='one_factor_at_a_time_with_component_ablations',purpose='development_sensitivity',
        parameter_selection='report_all_configurations',metrics=['CA','ASR','Precision','Recall','FPR'],
        reference_pool='heldout_retained_training_plus_disjoint_validation',
        fixed_split=True,timeout_seconds=args.timeout,requested_execution=args.execute)
    write_reports(root,jobs,protocol)
    print(f'{len(jobs)} jobs; manifest: {root / "study.json"}',flush=True)
    if not args.execute:return
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        future_to_job={pool.submit(execute,job,args.timeout):job for job in jobs}
        for future in as_completed(future_to_job):
            job=future_to_job[future]
            try: job.update(future.result())
            except Exception as error: job.update(status='failed',error=str(error))
            write_reports(root,jobs,protocol)
            print(job['id']+' '+job['status'],flush=True)
    if any(job['status']=='failed' for job in jobs):raise SystemExit(1)


if __name__=='__main__':main()
