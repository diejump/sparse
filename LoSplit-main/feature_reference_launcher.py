"""Launcher adapter for the Cora/GCN feature-consistency reference protocol."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse
import torch
from feature_reference_experiment import FeatureReferenceConfig,run_feature_snapshot


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pre_train_param',required=True)
    p.add_argument('--trigger_generator_address',required=True)
    p.add_argument('--model_save_path',required=True)
    p.add_argument('--dataset',choices=['Cora'],default='Cora')
    p.add_argument('--model',default='GCN')
    p.add_argument('--test_model',choices=['GCN'],default='GCN')
    p.add_argument('--defense_mode',choices=['abp_feature_reference'],default='abp_feature_reference')
    p.add_argument('--target_class',type=int,default=0)
    p.add_argument('--device_id',type=int,default=0)
    p.add_argument('--device')
    p.add_argument('--seed',type=int,default=10)
    p.add_argument('--hidden',type=int,default=32)
    p.add_argument('--epochs',type=int,default=200)
    p.add_argument('--train_lr',type=float,default=.01)
    p.add_argument('--members',type=int,default=1)
    p.add_argument('--feature-repair',choices=['unique_mode','none','clip_normalize'],default='unique_mode')
    p.add_argument('--feature-tolerance',type=float,default=1e-6)
    p.add_argument('--abp_reference_multiplier',type=float,default=2.)
    p.add_argument('--abp_reference_ratio',type=float,default=.2)
    p.add_argument('--abp_reference_count',type=int)
    p.add_argument('--abp_reference_risk',type=float,default=.005)
    p.add_argument('--abp_reference_seed',type=int,default=20261124)
    p.add_argument('--abp_reference_variant',choices=['full','no_reference','full_projection','keep_detected_labels'],default='full')
    p.add_argument('--deterministic',action='store_true')
    a=p.parse_args();torch.set_num_threads(1);torch.use_deterministic_algorithms(a.deterministic)
    if a.deterministic:
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
        torch.backends.cudnn.benchmark=False
    cfg=FeatureReferenceConfig(multiplier=a.abp_reference_multiplier,ratio=a.abp_reference_ratio,
        count=a.abp_reference_count,risk=a.abp_reference_risk,reference_seed=a.abp_reference_seed,
        variant=a.abp_reference_variant,members=a.members,classifier_seed=a.seed,hidden=a.hidden,
        epochs=a.epochs,train_lr=a.train_lr,feature_repair=a.feature_repair,feature_tolerance=a.feature_tolerance)
    run_feature_snapshot(a.pre_train_param,a.model_save_path,cfg,a.device or f'cuda:{a.device_id}',a.trigger_generator_address)


if __name__=='__main__':main()
