"""Fit and score a reference-learned envelope with three method parameters."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
os.environ.setdefault('MPLBACKEND','Agg')
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time
import torch

from experiment_defenses import confusion
from reference_training import ReferenceConfig,fit_reference_method
from reference_experiment import observed_inputs,load_evaluation_generator,evaluate_reference
from models.ABP_Reference import ABPReferencePredictor,build_reference_model
from models.EmpiricalEnvelope import EmpiricalEnvelopeDetector
from models.RangeEnvelope import RangeEnvelopeDetector

ROOT=Path(__file__).resolve().parent

def load_envelope_bundle(path,device='cpu'):
    b=torch.load(path,map_location=device)
    if b['version'] not in (4,5):raise ValueError('Expected envelope bundle version 4 or 5')
    config=ReferenceConfig(**b['configuration'])
    detector_type=RangeEnvelopeDetector if b['version']==5 else EmpiricalEnvelopeDetector
    detector=detector_type.from_metadata(b['detector'],device)
    normal=[];recovery=[];thresholds=[]
    for member in b['members']:
        for key,target in [('normal',normal),('recovery',recovery)]:
            if member[key] is None:continue
            m=build_reference_model(config,b['architecture'],b['nfeat'],b['nclass'],device)
            m.load_state_dict(member[key]);m.eval();target.append(m)
        if member['recovery'] is not None:thresholds.append(member['threshold'])
    return ABPReferencePredictor(normal,recovery,detector,b['training_nodes'],b['inferred_class'],
        thresholds,b['graph_policy'],b['decision_rule'],detector.repair).to(device).eval()


def extra_metrics(model,snapshot,clean,changed,test_graph,detector):
    y=snapshot['original_labels'];target=int(snapshot['target_class'])
    affected=snapshot['idx_atk'];normal=snapshot['idx_clean_test']
    same=model.predict(*test_graph)
    tnodes=affected[y[affected]==target]
    clean_t=normal[y[normal]==target]
    scored=torch.cat([normal,affected]).unique()
    selected,_=detector.detect(*test_graph,model.training_nodes)
    return dict(SameGraphCA=float((same[normal]==y[normal]).double().mean()),
        AffectedAccuracy=float((changed[affected]==y[affected]).double().mean()),
        AffectedCorrect=int((changed[affected]==y[affected]).sum()),AffectedCount=len(affected),
        AffectedTargetAccuracy=float((changed[tnodes]==target).double().mean()) if len(tnodes) else None,
        AffectedTargetCorrect=int((changed[tnodes]==target).sum()),AffectedTargetCount=len(tnodes),
        CleanTargetAccuracy=float((clean[clean_t]==target).double().mean()) if len(clean_t) else None,
        TestDetection=confusion(selected,affected,scored))


def run(path,output,architecture='GCN',seed=31,device='cpu',gamma=1.0,rho=.2,alpha=.005,
        repair='box',score='joint',variant='full',epochs=200,structure=True):
    output=Path(output);meta=Path(str(output)+'.meta.json')
    if output.exists() or meta.exists():raise FileExistsError(output)
    s=torch.load(path,map_location=device)
    if s.get('snapshot_version')!=2:raise ValueError('Expected version-2 snapshot')
    x,e,w,y,a,v,ce,cw=observed_inputs(s)
    detector=(RangeEnvelopeDetector(gamma,v,repair=repair,use_structure=structure) if score=='range' else
              EmpiricalEnvelopeDetector(gamma,v,repair=repair,score=score,use_structure=structure))
    config=ReferenceConfig(multiplier=gamma,ratio=rho,risk=alpha,members=1,
        architecture=architecture,classifier_seed=seed,epochs=epochs,variant=variant)
    start=time.perf_counter()
    m,b,audit=fit_reference_method(x,e,w,y,a,v,config,ce,cw,detector=detector,feature_transform=detector.repair)
    generator=load_evaluation_generator(s,x.device)
    result,clean,changed,test_graph=evaluate_reference(m,s,ce,cw,generator)
    chosen,_=detector.detect(x,e,w,a,a)
    result.update(confusion(chosen,s['idx_attach'],a))
    with torch.no_grad():result.update(extra_metrics(m,s,clean,changed,test_graph,detector))
    result.update(Method='empirical_envelope_reference',Dataset=s['dataset'],Setting=s['attack'],
        Architecture=architecture,Seed=seed,Configuration=asdict(config),
        Gamma=gamma,Rho=rho,Alpha=alpha,Repair=repair,Score=score,Structure=structure,
        Seconds=time.perf_counter()-start,Audit=audit,
        SnapshotSHA256=hashlib.sha256(Path(path).read_bytes()).hexdigest())
    b['version']=5 if score=='range' else 4
    output.parent.mkdir(parents=True,exist_ok=True)
    torch.save(b,output)
    loaded=load_envelope_bundle(output,device)
    with torch.no_grad():
        if not torch.equal(clean,loaded.predict(x,ce,cw)) or not torch.equal(changed,loaded.predict(*test_graph)):
            raise RuntimeError('Reloaded predictions differ')
    torch.save(dict(clean=clean.cpu(),changed=changed.cpu(),selected=chosen.cpu()),output.with_suffix('.predictions.pt'))
    result['ReloadVerified']=True
    result['SourceSHA256']={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in
        ['envelope_experiment.py','models/EmpiricalEnvelope.py','models/RangeEnvelope.py','reference_training.py','models/ABP_Reference.py','models/ABP_Local.py']}
    meta.write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps({k:result[k] for k in ['Setting','Architecture','CA','ASR','Precision','Recall','SameGraphCA','AffectedAccuracy','AffectedTargetAccuracy','Seconds']},allow_nan=False),flush=True)
    return result

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--snapshot',required=True);p.add_argument('--output',required=True)
    p.add_argument('--architecture',choices=['GCN','GAT','GraphSage'],default='GCN')
    p.add_argument('--seed',type=int,default=31);p.add_argument('--device',default='cpu')
    p.add_argument('--gamma',type=float,default=1.0);p.add_argument('--rho',type=float,default=.2)
    p.add_argument('--alpha',type=float,default=.005)
    p.add_argument('--repair',choices=['box','median','none'],default='box')
    p.add_argument('--score',choices=['joint','radial','range'],default='joint')
    p.add_argument('--variant',choices=['full','no_reference'],default='full')
    p.add_argument('--epochs',type=int,default=200);p.add_argument('--no-structure',action='store_true')
    a=p.parse_args();torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    run(a.snapshot,a.output,a.architecture,a.seed,a.device,a.gamma,a.rho,a.alpha,
        a.repair,a.score,a.variant,a.epochs,not a.no_structure)

if __name__=='__main__':main()
