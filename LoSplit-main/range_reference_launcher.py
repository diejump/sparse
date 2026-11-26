"""Adapter for the three-parameter empirical-range research method."""
import argparse
import json
from pathlib import Path
import torch
from envelope_experiment import run

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('pre_train_param','trigger_generator_address','model_save_path','dataset'):p.add_argument('--'+key,required=True)
    p.add_argument('--model',default='GCN');p.add_argument('--test_model',choices=['GCN','GAT','GraphSage'],default='GCN')
    p.add_argument('--defense_mode',choices=['range_reference'],default='range_reference')
    p.add_argument('--seed',type=int,default=41);p.add_argument('--epochs',type=int,default=200)
    p.add_argument('--hidden',type=int,choices=[32],default=32);p.add_argument('--train_lr',type=float,choices=[.01],default=.01)
    p.add_argument('--target_class',type=int,default=0);p.add_argument('--device_id',type=int,default=0)
    p.add_argument('--device',default='cpu');p.add_argument('--deterministic',action='store_true')
    for key,value in [('gamma',2.),('rho',.2),('alpha',.005)]:p.add_argument('--range_'+key,type=float,default=value)
    a=p.parse_args();torch.set_num_threads(1);torch.use_deterministic_algorithms(a.deterministic)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    s=torch.load(a.pre_train_param,map_location='cpu')
    if s['dataset']!=a.dataset:raise ValueError('Dataset differs from the saved snapshot')
    state=torch.load(a.trigger_generator_address,map_location='cpu');saved=s['trigger_generator_state']
    if set(state)!=set(saved) or any(not torch.equal(state[k].cpu(),saved[k].cpu()) for k in saved):raise ValueError('Paired generator differs from the snapshot')
    del s,state,saved
    result=run(a.pre_train_param,a.model_save_path,a.test_model,a.seed,a.device,a.range_gamma,a.range_rho,a.range_alpha,
        repair='median',score='range',variant='full',epochs=a.epochs)
    result.update(Method='range_reference',Model=a.test_model,Device=a.device,
        ModelCount=result['Audit']['ModelCount'],DetectionSeconds=result['Audit']['DetectionSeconds'])
    Path(a.model_save_path+'.meta.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    print('FINAL_COMPARISON_JSON '+json.dumps(result,allow_nan=False),flush=True)

if __name__=='__main__':main()
