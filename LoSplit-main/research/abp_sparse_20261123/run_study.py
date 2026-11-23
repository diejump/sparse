"""Reproducible fixed-snapshot study; detector inputs are separated from truth."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('MPLBACKEND', 'Agg')
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from deterministic_runtime import seed_classifier
from models.GCN import GCN
from models.ABP import filter_edges, branch_profiles
from models.ABP_Sparse import SparseBranchDetector, effective_support, project_graph


def confusion(predicted, truth, universe):
    p, t, u = map(lambda z: set(z.detach().cpu().tolist()), (predicted, truth, universe))
    p, t = p & u, t & u
    tp, fp, fn, tn = len(p&t), len(p-t), len(t-p), len(u-(p|t))
    return {'tp':tp,'fp':fp,'fn':fn,'tn':tn,
            'precision':tp/(tp+fp) if tp+fp else None,
            'recall':tp/(tp+fn) if tp+fn else None,
            'fpr':fp/(fp+tn) if fp+tn else None}


def load_generator(snapshot, device):
    kind = snapshot['attack']
    filename = {'GTA':'models/GTA.py','UGBA':'models/backdoor_UGBA.py',
                'DPGBA':'DPGBA/models/backdoor.py'}[kind]
    spec = importlib.util.spec_from_file_location('study_generator_'+kind, ROOT/filename)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    args = SimpleNamespace(trigger_size=int(snapshot['trigger_size']),
                           thrd=float(snapshot['trigger_threshold']))
    generator = mod.Backdoor(args, device)
    generator.trojan = mod.GraphTrojanNet(device, snapshot['poison_x'].size(1), args.trigger_size, layernum=2).to(device)
    generator.trojan.load_state_dict(snapshot['trigger_generator_state'])
    generator.trojan.eval()
    return generator


def logits(value):
    return value[0] if isinstance(value, tuple) else value


def run(args):
    started = time.monotonic()
    started_utc = datetime.now(timezone.utc).isoformat()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(args.device)
    path = Path(args.snapshot).resolve()
    s = torch.load(path, map_location=device)
    if s['dataset'] != 'Cora' or s['snapshot_version'] != 2:
        raise ValueError('This study is scoped to Cora version-2 snapshots')
    output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    x, e, w = s['poison_x'], s['poison_edge_index'], s['poison_edge_weights']
    # Sorting removes provenance carried by the original concatenation order.
    anchors = torch.cat([s['idx_train'],s['idx_attach']]).unique(sorted=True)
    val = s['idx_val'].sort().values
    observed = torch.full((x.size(0),), -1, dtype=torch.long, device=device)
    observed[anchors] = s['poison_labels'][anchors]
    observed[val] = s['poison_labels'][val]
    model_args = SimpleNamespace(dataset='Cora', hidden=32, dropout=.5,
        train_lr=.015 if s['attack']=='DPGBA' else .01, weight_decay=.0005)
    if args.lr is not None:
        model_args.train_lr = args.lr
    generator = load_generator(s, device)
    eval_e = torch.cat([e,s['mask_edge_index']],dim=1)
    eval_w = torch.cat([w,torch.ones(s['mask_edge_index'].size(1),device=device)])
    with torch.no_grad():
        trigger_x, trigger_e, trigger_w = generator.inject_trigger(s['idx_atk'],x,eval_e,eval_w,device)
    if args.stress == 'no_internal':
        # Controlled topology stress in training and evaluation graphs.
        n = int(s['num_original_nodes'])
        train_keep = ~((e[0]>=n)&(e[1]>=n))
        e,w = e[:,train_keep],w[train_keep]
        eval_e = torch.cat([e,s['mask_edge_index']],dim=1)
        eval_w = torch.cat([w,torch.ones(s['mask_edge_index'].size(1),device=device)])
        eval_keep = ~((trigger_e[0]>=n)&(trigger_e[1]>=n))
        trigger_e, trigger_w = trigger_e[:,eval_keep], trigger_w[eval_keep]
    reference_nodes = torch.cat([anchors,val]).unique(sorted=True)
    detector = SparseBranchDetector(args.margin).fit(x,reference_nodes)
    selected_sparse, sparse_info = detector.detect(x,e,w,anchors,anchors)
    pe,pw,blocked = project_graph(e,w,x.size(0),anchors)
    largest,_ = branch_profiles(e[:,w>0],x.size(0),anchors)
    receiver_candidates = blocked[1].unique(sorted=True)
    receiver_candidates = receiver_candidates[torch.isin(receiver_candidates,anchors)]
    diagnostic = {'detector':detector.metadata(), 'effective_branch_candidates':confusion(receiver_candidates,s['idx_attach'],anchors),
        'minimum_two_candidates':confusion(anchors[largest[anchors]>=2],s['idx_attach'],anchors),
        'sparse_detection':confusion(selected_sparse,s['idx_attach'],anchors),
        'selected_nodes':selected_sparse.tolist(), 'candidate_nodes':receiver_candidates.tolist(),
        'stored_arcs':e.size(1),'positive_arcs':int((w>0).sum()),
        'feature_support':{'reference_max':detector.reference_max,
            'original_max':float(effective_support(x[:s['num_original_nodes']]).max()),
            'added_min':float(effective_support(x[s['num_original_nodes']:]).min())},
        'attach_branch_sizes':{str(int(k)):int(v) for k,v in zip(*torch.unique(largest[s['idx_attach']],return_counts=True))}}
    (output_dir/'diagnostics.json').write_text(json.dumps(diagnostic,indent=2),encoding='utf-8')
    for seed in args.seeds:
        for method in args.methods:
            outpath=output_dir/f'{method}_seed{seed}.json'
            if outpath.exists():
                raise FileExistsError(str(outpath))
            t=time.monotonic()
            fit_nodes = anchors
            fit_val = val
            train_e,train_w=e,w
            selected=None
            if method in ('abp','abp_positive','abp_all_candidates','abp_sparse','abp_sparse_no_projection'):
                if method=='abp':
                    train_e,train_w=filter_edges(e,w,x.size(0),anchors)
                elif method!='abp_sparse_no_projection':
                    train_e,train_w=pe,pw
                if method=='abp_all_candidates':
                    selected=receiver_candidates
                elif method.startswith('abp_sparse'):
                    selected=selected_sparse
                if selected is not None:
                    fit_nodes=anchors[~torch.isin(anchors,selected)]
                    fit_val=val[~torch.isin(val,selected)]
            seed_classifier(seed,device.type=='cuda')
            if method.startswith('abp_dual') or method.startswith('abp_assisted'):
                from models.ABP_Learning import ABPLearning
                cfg=SimpleNamespace(**vars(model_args),seed=seed,
                    defense_mode='abp_dual' if method.startswith('abp_dual') else 'abp_assisted',
                    abp_min_branch=1 if method.endswith('_min1') else 2,
                    abp_recovery_epochs=400,abp_negative_weight=3.,
                    abp_teacher_hidden=32,abp_teacher_epochs=200,abp_confidence=.8)
                descriptor=SimpleNamespace(x=x,y=torch.arange(7,device=device))
                model=ABPLearning(cfg,'GCN',descriptor,device).to(device)
            else:
                model=GCN(model_args,x.size(1),32,7,dropout=.5,lr=model_args.train_lr,
                          weight_decay=.0005,device=device).to(device)
            if torch.unique(observed[fit_nodes]).numel()!=7:
                raise ValueError('Filtered training labels do not cover all seven classes')
            if fit_val.numel() == 0:
                raise ValueError('No validation nodes remain after filtering')
            model.fit(x,train_e,train_w,observed,fit_nodes,fit_val,train_iters=args.epochs,verbose=False)
            model.eval()
            if hasattr(model,'defense_metadata'):
                selected=torch.tensor(model.defense_metadata['selected_nodes'],device=device,dtype=torch.long)
            def evaluation_graph(edges,weights,nodes):
                if method=='abp':
                    return filter_edges(edges,weights,nodes,anchors)
                if method in ('abp_positive','abp_all_candidates','abp_sparse'):
                    return project_graph(edges,weights,nodes,anchors)[:2]
                return edges,weights
            ce,cw=evaluation_graph(eval_e,eval_w,x.size(0))
            te,tw=evaluation_graph(trigger_e,trigger_w,trigger_x.size(0))
            with torch.no_grad():
                clean_output=logits(model(x,ce,cw))
                trigger_output=logits(model(trigger_x,te,tw))
            ca_nodes=s['idx_clean_test']
            atk_nodes=s['idx_atk'][s['original_labels'][s['idx_atk']]!=int(s['target_class'])]
            if not torch.isfinite(clean_output).all() or not torch.isfinite(trigger_output).all():
                raise RuntimeError('Nonfinite model output')
            ca=int((clean_output[ca_nodes].argmax(1)==s['original_labels'][ca_nodes]).sum())
            asr=int((trigger_output[atk_nodes].argmax(1)==int(s['target_class'])).sum())
            base_target=int((clean_output[atk_nodes].argmax(1)==int(s['target_class'])).sum())
            n=int(s['num_original_nodes'])
            row={'method':method,'classifier_seed':seed,'generation_seed':s['generation_seed'],
                'kind':s['attack'],'snapshot':str(path),'snapshot_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                'stress':args.stress,'CA':ca/ca_nodes.numel(),'ASR':asr/atk_nodes.numel(),
                'clean_correct':ca,'clean_count':ca_nodes.numel(),'target_count':asr,'asr_count':atk_nodes.numel(),
                'untriggered_target_error':base_target/atk_nodes.numel(),
                'max_original_logit_delta':float((clean_output[:n]-trigger_output[:n]).abs().max()),
                'prediction_changes':int((clean_output[:n].argmax(1)!=trigger_output[:n].argmax(1)).sum()),
                'fit_count':fit_nodes.numel(),'validation_count':fit_val.numel(),
                'protocol_version':2,'epochs':args.epochs,'learning_rate':model_args.train_lr,
                'detection':confusion(selected,s['idx_attach'],anchors) if selected is not None else None,
                'seconds':time.monotonic()-t,'detector':detector.metadata() if method.startswith('abp_sparse') else None}
            if selected is not None:
                row['selected_nodes']=selected.tolist()
            if method=='abp_sparse':
                eval_selected,_=detector.detect(trigger_x,trigger_e,trigger_w,anchors)
                truth=torch.cat([s['idx_attach'],s['idx_atk']]).unique()
                row['triggered_receiver_detection']=confusion(eval_selected,truth,torch.arange(n,device=device))
            if hasattr(model,'defense_metadata'):
                row['legacy_metadata']=model.defense_metadata
            torch.save({'state_dict':model.state_dict(),'method':method,'anchors':anchors.cpu(),
                        'detector':detector.metadata(),'configuration':vars(model_args)},output_dir/f'{method}_seed{seed}.pt')
            outpath.write_text(json.dumps(row,indent=2),encoding='utf-8')
            records.append(row)
            print(json.dumps({k:v for k,v in row.items() if k not in ('selected_nodes','legacy_metadata')},ensure_ascii=False),flush=True)
            del model
            if device.type=='cuda':
                torch.cuda.empty_cache()
    manifest={'started_utc':started_utc,'completed_utc':datetime.now(timezone.utc).isoformat(),'seconds':time.monotonic()-started,
        'command':sys.argv,'torch':torch.__version__,'device':str(device),'deterministic':True,
        'source_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in
            [Path(__file__).resolve(),ROOT/'models/ABP.py',ROOT/'models/ABP_Sparse.py',ROOT/'models/GCN.py',ROOT/'models/ABP_Learning.py']},
        'records':records,'diagnostics':diagnostic}
    (output_dir/'results.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--snapshot',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--methods',nargs='+',default=['none','abp','abp_sparse'],
        choices=['none','abp','abp_positive','abp_all_candidates','abp_sparse','abp_sparse_no_projection',
                 'abp_dual','abp_assisted','abp_dual_min1','abp_assisted_min1'])
    p.add_argument('--seeds',nargs='+',type=int,default=[10])
    p.add_argument('--margin',type=float,default=2.)
    p.add_argument('--epochs',type=int,default=200)
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--stress',choices=['none','no_internal'],default='none')
    p.add_argument('--lr',type=float)
    run(p.parse_args())
