"""Run parameter experiments with validation-only parameter selection."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
os.environ.setdefault('MPLBACKEND','Agg')
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace
import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from models.GCN import GCN
from models.ABP_Sparse import project_graph
from models.ABP_Sparse_Training import TunableSparseBranchDetector,train_with_detected_labels
from deterministic_runtime import seed_classifier
from research.abp_sparse_20261123.metric_reaudit import recount
from research.abp_sparse_20261123.run_study import load_generator
OUT=ROOT/'runs/abp_sparse_20261123/parameter_study'


def snapshot_path(generation,kind):
    if generation==10:
        return ROOT/'runs'/('dpgba_official_fix_20261123/inputs' if kind=='DPGBA' else 'my_cora_attacks3')/kind/'poison.pt'
    return ROOT/'runs/abp_sparse_20261123'/f'seed{generation}'/'inputs'/kind/'poison.pt'


def tag(multiplier,penalty):
    return 'mult'+str(float(multiplier)).replace('.','p')+'_pen'+str(float(penalty)).replace('.','p')


def prepare(generation,kind):
    path=snapshot_path(generation,kind)
    s=torch.load(path,map_location='cpu')
    x,e,w=s['poison_x'],s['poison_edge_index'],s['poison_edge_weights']
    training=torch.cat([s['idx_train'],s['idx_attach']]).unique(sorted=True)
    validation=s['idx_val'].sort().values
    labels=torch.full((len(x),),-1,dtype=torch.long)
    labels[training]=s['poison_labels'][training]
    labels[validation]=s['poison_labels'][validation]
    reference=torch.cat([training,validation]).unique(sorted=True)
    pe,pw,_=project_graph(e,w,len(x),training)
    screening=TunableSparseBranchDetector(.9).fit(x,reference)
    screen_nodes,_=screening.detect(x,e,w,training,training)
    common_validation=validation[~torch.isin(validation,screen_nodes)]
    cfg=SimpleNamespace(dataset='Cora',hidden=32,dropout=.5,train_lr=.015 if kind=='DPGBA' else .01,weight_decay=.0005)
    return dict(snapshot=s,path=path,x=x,e=e,w=w,training=training,validation=validation,
        labels=labels,reference=reference,pe=pe,pw=pw,common_validation=common_validation,config=cfg)


def model_from_config(data):
    c=data['config']
    return GCN(c,data['x'].size(1),c.hidden,7,dropout=c.dropout,lr=c.train_lr,
               weight_decay=c.weight_decay,device=torch.device('cpu'))


def validation_metrics(output,labels,nodes,inferred_class):
    pred=output[nodes].argmax(1);true=labels[nodes]
    nonclass=true!=inferred_class;inclass=~nonclass
    return {'accuracy':float(pred.eq(true).double().mean()),'count':len(nodes),
        'unwanted_class_rate':float(pred[nonclass].eq(inferred_class).double().mean()) if nonclass.any() else None,
        'class_accuracy':float(pred[inclass].eq(inferred_class).double().mean()) if inclass.any() else None,
        'nonclass_count':int(nonclass.sum()),'class_count':int(inclass.sum())}


def evaluate_saved(data,model,detector):
    s=data['snapshot'];x=data['x'];training=data['training']
    ce=torch.cat([data['e'],s['mask_edge_index']],1)
    cw=torch.cat([data['w'],torch.ones(s['mask_edge_index'].size(1))])
    gen=load_generator(s,torch.device('cpu'))
    with torch.no_grad():
        tx,te,tw=gen.inject_trigger(s['idx_atk'],x,ce,cw,torch.device('cpu'))
    projected_clean,clean_weights,_=project_graph(ce,cw,len(x),training)
    projected_trigger,trigger_weights,_=project_graph(te,tw,len(tx),training)
    model.eval()
    with torch.no_grad():
        clean=model(x,projected_clean,clean_weights)
        triggered=model(tx,projected_trigger,trigger_weights)
    clean_nodes=s['idx_clean_test']
    target_nodes=s['idx_atk'][s['original_labels'][s['idx_atk']]!=int(s['target_class'])]
    all_test=torch.cat([s['idx_clean_test'],s['idx_atk']]).unique()
    target_class=int(s['target_class'])
    target_clean=all_test[s['original_labels'][all_test]==target_class]
    ca_count=int(clean[clean_nodes].argmax(1).eq(s['original_labels'][clean_nodes]).sum())
    asr_count=int(triggered[target_nodes].argmax(1).eq(target_class).sum())
    detected,_=detector.detect(tx,te,tw,training)
    original=set(range(int(s['num_original_nodes'])))
    detected_original=set(detected.tolist())&original
    truth=set(s['idx_attach'].tolist())|set(s['idx_atk'].tolist())
    all_detection=recount(detected_original,truth,original)
    return {'CA':ca_count/len(clean_nodes),'ASR':asr_count/len(target_nodes),
        'clean_correct':ca_count,'clean_count':len(clean_nodes),'target_count':asr_count,'asr_count':len(target_nodes),
        'all_test_clean_accuracy':float(clean[all_test].argmax(1).eq(s['original_labels'][all_test]).double().mean()),
        'normal_target_class_accuracy':float(clean[target_clean].argmax(1).eq(target_class).double().mean()),
        'normal_target_class_count':len(target_clean),
        'target_prediction_changes':int((clean[:len(x)].argmax(1)!=triggered[:len(x)].argmax(1)).sum()),
        'original_output_max_difference':float((clean[:int(s['num_original_nodes'])]-triggered[:int(s['num_original_nodes'])]).abs().max()),
        'all_original_node_detection':all_detection}


def run(args):
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    phase_dir=OUT/args.phase
    for generation in args.generations:
        for kind in args.kinds:
            data=prepare(generation,kind)
            for multiplier in args.multipliers:
                detector=TunableSparseBranchDetector(multiplier).fit(data['x'],data['reference'])
                selected,_=detector.detect(data['x'],data['e'],data['w'],data['training'],data['training'])
                retained=data['training'][~torch.isin(data['training'],selected)]
                validation=data['validation'][~torch.isin(data['validation'],selected)]
                inferred_class=int(torch.bincount(data['labels'][selected],minlength=7).argmax()) if len(selected) else int(torch.bincount(data['labels'][retained],minlength=7).argmax())
                detection=recount(selected.tolist(),data['snapshot']['idx_attach'].tolist(),data['training'].tolist())
                for penalty in args.penalties:
                    for seed in args.seeds:
                        folder=phase_dir/f'gen{generation}'/kind/tag(multiplier,penalty)
                        folder.mkdir(parents=True,exist_ok=True)
                        path=folder/f'seed{seed}.json'
                        weights=folder/f'seed{seed}.pt'
                        if path.exists():
                            if args.skip_complete:
                                continue
                            raise FileExistsError(path)
                        start=time.monotonic()
                        seed_classifier(seed,False)
                        model=model_from_config(data)
                        training_result=train_with_detected_labels(model,data['x'],data['pe'],data['pw'],data['labels'],
                            retained,validation,selected,penalty,200)
                        with torch.no_grad():
                            output=model(data['x'],data['pe'],data['pw'])
                        row={'generation_seed':generation,'kind':kind,'classifier_seed':seed,
                            'multiplier':multiplier,'penalty_weight':penalty,'detection':detection,
                            'validation':validation_metrics(output,data['labels'],validation,inferred_class),
                            'common_validation':validation_metrics(output,data['labels'],data['common_validation'],inferred_class),
                            'inferred_class':inferred_class,'training_count':len(retained),'validation_count':len(validation),
                            'selected_nodes':selected.tolist(),'training':training_result,'detector':detector.metadata(),
                            'snapshot':str(data['path']),'snapshot_sha256':hashlib.sha256(data['path'].read_bytes()).hexdigest(),
                            'source_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__).resolve(),ROOT/'models/ABP_Sparse.py',ROOT/'models/ABP_Sparse_Training.py']},
                            'phase':args.phase,'test_metrics_used_for_selection':False}
                        if args.evaluate:
                            row['test']=evaluate_saved(data,model,detector)
                        torch.save({'state_dict':model.state_dict(),'configuration':vars(data['config']),
                            'training_nodes':data['training'],'selected_nodes':selected,'multiplier':multiplier,
                            'penalty_weight':penalty},weights)
                        row['seconds']=time.monotonic()-start
                        path.write_text(json.dumps(row,indent=2),encoding='utf-8')
                        print(json.dumps({k:row[k] for k in ['generation_seed','kind','classifier_seed','multiplier','penalty_weight','detection','common_validation','seconds']}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--phase',default='screen')
    p.add_argument('--generations',type=int,nargs='+',default=[10])
    p.add_argument('--kinds',choices=['GTA','UGBA','DPGBA'],nargs='+',default=['GTA','UGBA','DPGBA'])
    p.add_argument('--multipliers',type=float,nargs='+',default=[.9,.95,2.])
    p.add_argument('--penalties',type=float,nargs='+',default=[0.,.05,.1,.25,.5,1.,2.,4.])
    p.add_argument('--seeds',type=int,nargs='+',default=[10])
    p.add_argument('--evaluate',action='store_true')
    p.add_argument('--skip-complete',action='store_true')
    run(p.parse_args())
