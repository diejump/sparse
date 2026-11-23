"""Independently recount detection and test metrics from saved inputs and weights."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np
import torch
from sklearn.metrics import confusion_matrix

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from models.ABP_Sparse import SparseBranchDetector,project_graph
from models.GCN import GCN
from research.abp_sparse_20261123.run_study import load_generator
OUT=ROOT/'runs/abp_sparse_20261123/parameter_study'

def recount(selected,truth,population):
    selected,truth,population=map(lambda t:set(map(int,t)),(selected,truth,population))
    if not selected<=population or not truth<=population:
        raise ValueError('Detection sets must be inside the stated population')
    order=sorted(population)
    true=np.array([int(v in truth) for v in order])
    pred=np.array([int(v in selected) for v in order])
    tn,fp,fn,tp=map(int,confusion_matrix(true,pred,labels=[0,1]).ravel())
    return dict(tp=tp,fp=fp,fn=fn,tn=tn,
        precision=tp/(tp+fp) if tp+fp else None,
        recall=tp/(tp+fn) if tp+fn else None,
        fpr=fp/(fp+tn) if fp+tn else None)

def main():
    torch.set_num_threads(1)
    OUT.mkdir(parents=True,exist_ok=True)
    records=[]
    threshold_rows=[]
    for gen in [10,11,12]:
        for kind in ['GTA','UGBA','DPGBA']:
            directory=ROOT/'runs/abp_sparse_20261123/validation_v2'/f'gen{gen}'/kind
            first=json.loads((directory/'abp_sparse_seed10.json').read_text())
            s=torch.load(first['snapshot'],map_location='cpu')
            x,e,w=s['poison_x'],s['poison_edge_index'],s['poison_edge_weights']
            train=torch.cat([s['idx_train'],s['idx_attach']]).unique(sorted=True)
            val=s['idx_val'].sort().values
            refs=torch.cat([train,val]).unique(sorted=True)
            detector=SparseBranchDetector().fit(x,refs)
            selected,info=detector.detect(x,e,w,train,train)
            detection=recount(selected.tolist(),s['idx_attach'].tolist(),train.tolist())
            blocked=info['blocked_arcs']
            maximum=torch.zeros(x.size(0),dtype=torch.float64)
            for src,dst in blocked.t().tolist():
                maximum[dst]=max(maximum[dst],info['support_scores'][src])
            for m in [.25,.5,.65,.7,.75,.8,.85,.875,.9,.925,.95,.975,1,1.25,1.5,2,4,8,9,10,12,16,24,32]:
                selected_m=train[maximum[train]>m*detector.reference_max*(1+1e-12)]
                threshold_rows.append(dict(generation_seed=gen,kind=kind,multiplier=m,
                    threshold=m*detector.reference_max,
                    **recount(selected_m.tolist(),s['idx_attach'].tolist(),train.tolist())))
            clean_edges=torch.cat([e,s['mask_edge_index']],1)
            clean_weights=torch.cat([w,torch.ones(s['mask_edge_index'].size(1))])
            generator=load_generator(s,torch.device('cpu'))
            with torch.no_grad():
                tx,te,tw=generator.inject_trigger(s['idx_atk'],x,clean_edges,clean_weights,torch.device('cpu'))
            ce,cw,_=project_graph(clean_edges,clean_weights,len(x),train)
            te,tw,_=project_graph(te,tw,len(tx),train)
            full_test=torch.cat([s['idx_clean_test'],s['idx_atk']]).unique()
            normal_test=s['idx_clean_test']
            target_test=s['idx_atk']
            selected_target_test=target_test[s['original_labels'][target_test]!=int(s['target_class'])]
            all_sets={'train':set(train.tolist()),'validation':set(val.tolist()),
                'clean_test':set(normal_test.tolist()),'trigger_test':set(target_test.tolist())}
            assert not all_sets['train']&all_sets['clean_test']
            assert not all_sets['train']&all_sets['trigger_test']
            assert not all_sets['validation']&all_sets['clean_test']
            assert not all_sets['validation']&all_sets['trigger_test']
            for seed in [10,20,30]:
                old=json.loads((directory/f'abp_sparse_seed{seed}.json').read_text())
                saved=torch.load(directory/f'abp_sparse_seed{seed}.pt',map_location='cpu')
                cfg=SimpleNamespace(**saved['configuration'])
                model=GCN(cfg,x.size(1),cfg.hidden,7,dropout=cfg.dropout,lr=cfg.train_lr,
                    weight_decay=cfg.weight_decay,device=torch.device('cpu'))
                model.load_state_dict(saved['state_dict']);model.eval()
                with torch.no_grad():
                    clean_prediction=model(x,ce,cw).argmax(1)
                    trigger_prediction=model(tx,te,tw).argmax(1)
                ca_count=int(clean_prediction[normal_test].eq(s['original_labels'][normal_test]).sum())
                asr_count=int(trigger_prediction[selected_target_test].eq(int(s['target_class'])).sum())
                ca=ca_count/len(normal_test);asr=asr_count/len(selected_target_test)
                assert detection==old['detection']
                assert set(old['selected_nodes'])==set(selected.tolist())
                assert (ca_count,len(normal_test),asr_count,len(selected_target_test))==(old['clean_correct'],old['clean_count'],old['target_count'],old['asr_count'])
                assert ca==old['CA'] and asr==old['ASR']
                records.append(dict(generation_seed=gen,kind=kind,classifier_seed=seed,
                    detection=detection,CA=ca,ASR=asr,
                    clean_test_correct=ca_count,clean_test_count=len(normal_test),
                    triggered_non_target_success=asr_count,triggered_non_target_count=len(selected_target_test),
                    all_test_clean_CA=float(clean_prediction[full_test].eq(s['original_labels'][full_test]).double().mean()),
                    all_test_count=len(full_test),
                    ASR_including_original_target_class=float(trigger_prediction[target_test].eq(int(s['target_class'])).double().mean()),
                    trigger_test_count=len(target_test),
                    train_validation_overlap=len(all_sets['train']&all_sets['validation']),
                    matches_saved_results=True))
    result={'status':'VERIFIED','models_checked':len(records),
        'formulas':{'Recall':'正确识别的目标节点数 / 全部真实目标节点数',
                    'Precision':'正确识别的目标节点数 / 全部识别候选节点数',
                    'FPR':'误报的干净节点数 / 全部干净节点数'},
        'main_detection_population':'参与训练的有标签节点；真实目标为实际挂接触发器的节点',
        'records':records}
    (OUT/'metric_audit.json').write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8')
    (OUT/'threshold_sweep.json').write_text(json.dumps(threshold_rows,indent=2),encoding='utf-8')
    admissible=[]
    for m in sorted(set(r['multiplier'] for r in threshold_rows)):
        rows=[r for r in threshold_rows if r['generation_seed']==10 and r['multiplier']==m]
        if all(r['precision'] is not None and r['precision']>.9 and r['recall']>.9 for r in rows):
            admissible.append({'multiplier':m,'minimum_precision':min(r['precision'] for r in rows),'minimum_recall':min(r['recall'] for r in rows),
                'counts':[(r['kind'],r['tp'],r['fp'],r['fn']) for r in rows]})
    print(json.dumps({'status':'VERIFIED','models':len(records),'development_admissible_multipliers':admissible},indent=2))

if __name__=='__main__':
    main()
