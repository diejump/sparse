"""Reference learning with explicit normalized-binary feature restoration."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
os.environ.setdefault('MPLBACKEND','Agg')
import argparse
from dataclasses import asdict,dataclass
from functools import partial
import hashlib
import json
from pathlib import Path
import time

import torch

from experiment_defenses import confusion,stamp
from models.ABP_Reference import load_reference_bundle
from models.FeatureConsistency import FeatureBranchDetector,restore_features
from reference_training import ReferenceConfig,fit_reference_method
from reference_experiment import observed_inputs,load_evaluation_generator,evaluate_reference

ROOT=Path(__file__).resolve().parent


@dataclass(frozen=True)
class FeatureReferenceConfig(ReferenceConfig):
    feature_tolerance: float = 1e-6
    feature_repair: str = 'unique_mode'
    feature_detection: bool = True
    structure_detection: bool = True
    structural_gate: str = 'abp'

    def validate(self):
        super().validate()
        import math
        if not math.isfinite(self.feature_tolerance) or self.feature_tolerance<=0:
            raise ValueError('Feature tolerance must be positive and finite')
        if self.feature_repair not in ('mode','unique_mode','none','clip_normalize'):
            raise ValueError('Unknown feature restoration rule')
        if self.structural_gate not in ('abp','all_edges'):
            raise ValueError('Unknown structural gate')


def run_feature_snapshot(snapshot_path,output_path,config,device='cpu',generator_path=None):
    config.validate()
    output_path=Path(output_path)
    meta_path=Path(str(output_path)+'.meta.json')
    if output_path.exists() or meta_path.exists():raise FileExistsError(output_path)
    snapshot_path=Path(snapshot_path)
    s=torch.load(snapshot_path,map_location=device)
    if s['snapshot_version']!=2 or s['dataset']!='Cora':
        raise ValueError('This protocol uses version-2 Cora snapshots with normalized binary features')
    if generator_path:
        external=torch.load(generator_path,map_location='cpu');saved=s['trigger_generator_state']
        if set(external)!=set(saved) or any(not torch.equal(external[k].cpu(),saved[k].cpu()) for k in saved):
            raise ValueError('Paired generator differs from the snapshot')
    sha=hashlib.sha256(snapshot_path.read_bytes()).hexdigest()
    print('SNAPSHOT_SHA256: '+sha,flush=True)
    x,e,w,y,anchors,val,ce,cw=observed_inputs(s)
    device=x.device;started=stamp(device)
    detector=FeatureBranchDetector(config.multiplier,feature_tolerance=config.feature_tolerance,
        use_structure=config.structure_detection,use_features=config.feature_detection,
        structural_gate=config.structural_gate)
    transform=partial(restore_features,tolerance=config.feature_tolerance,repair=config.feature_repair)
    model,bundle,audit=fit_reference_method(x,e,w,y,anchors,val,config,ce,cw,
        detector=detector,feature_transform=transform)
    bundle.update(version=3,feature_tolerance=config.feature_tolerance,feature_repair=config.feature_repair)
    # Labels and target identities used below belong to the scoring layer.
    generator=load_evaluation_generator(s,device)
    result,clean,changed,test_graph=evaluate_reference(model,s,ce,cw,generator)
    selected,info=detector.detect(x,e,w,anchors,anchors)
    result.update(confusion(selected,s['idx_attach'],anchors))
    result.update(Method='abp_feature_reference',Dataset=s['dataset'],Model=config.architecture,
        Configuration=asdict(config),ModelCount=audit['ModelCount'],ReferenceCount=audit['ReferenceCount'],
        CalibrationCount=audit['CalibrationCount'],InferredClass=audit['InferredClass'],
        Device=str(device),Deterministic=torch.are_deterministic_algorithms_enabled(),
        Seconds=stamp(device)-started,DetectionSeconds=audit['DetectionSeconds'],
        DetectionPopulation='training_anchors',snapshot_sha256=sha,Audit=audit,
        StructureDetection=confusion(info['structural_nodes'],s['idx_attach'],anchors),
        FeatureDetection=confusion(info['feature_nodes'],s['idx_attach'],anchors))
    with torch.no_grad():
        test_selected,_=detector.detect(*test_graph,anchors)
    test_universe=torch.cat([s['idx_clean_test'],s['idx_atk']]).unique()
    result['TestDetectionAudit']=confusion(test_selected,s['idx_atk'],test_universe)
    result['SourceSHA256']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in
        [Path(__file__).resolve(),ROOT/'models/FeatureConsistency.py',ROOT/'reference_training.py',ROOT/'models/ABP_Reference.py']}
    output_path.parent.mkdir(parents=True,exist_ok=True)
    torch.save(bundle,output_path)
    reloaded=load_reference_bundle(output_path,device)
    with torch.no_grad():
        if not torch.equal(clean,reloaded.predict(x,ce,cw)) or not torch.equal(changed,reloaded.predict(*test_graph)):
            raise RuntimeError('Reloaded predictions differ')
    result['ReloadVerified']=True
    meta_path.write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    print('FINAL_COMPARISON_JSON '+json.dumps(result,allow_nan=False),flush=True)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pre_train_param',required=True)
    p.add_argument('--model_save_path',required=True)
    p.add_argument('--trigger_generator_address')
    p.add_argument('--dataset',choices=['Cora'],default='Cora')
    p.add_argument('--model',default='GCN')
    p.add_argument('--test_model',choices=['GCN','GAT','GraphSage'],default='GCN')
    p.add_argument('--defense_mode',default='abp_feature_reference')
    p.add_argument('--target_class',type=int,default=0)
    p.add_argument('--device_id',type=int,default=0)
    p.add_argument('--device',default='cpu')
    p.add_argument('--seed',type=int,default=10)
    p.add_argument('--hidden',type=int,default=32)
    p.add_argument('--epochs',type=int,default=200)
    p.add_argument('--train_lr',type=float,default=.01)
    p.add_argument('--members',type=int,default=1)
    p.add_argument('--deterministic',action='store_true')
    p.add_argument('--feature-tolerance',type=float,default=1e-6)
    p.add_argument('--feature-repair',choices=['mode','unique_mode','none','clip_normalize'],default='unique_mode')
    p.add_argument('--no-feature-detection',action='store_true')
    p.add_argument('--no-structure',action='store_true')
    p.add_argument('--structural-gate',choices=['abp','all_edges'],default='abp')
    p.add_argument('--no-reference',action='store_true')
    a=p.parse_args();torch.set_num_threads(1);torch.use_deterministic_algorithms(a.deterministic)
    cfg=FeatureReferenceConfig(classifier_seed=a.seed,hidden=a.hidden,epochs=a.epochs,train_lr=a.train_lr,
        members=a.members,architecture=a.test_model,feature_tolerance=a.feature_tolerance,
        feature_repair=a.feature_repair,feature_detection=not a.no_feature_detection,
        structure_detection=not a.no_structure,structural_gate=a.structural_gate,
        variant='no_reference' if a.no_reference else 'full')
    run_feature_snapshot(a.pre_train_param,a.model_save_path,cfg,a.device,a.trigger_generator_address)


if __name__=='__main__':main()
