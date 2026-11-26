"""Audit frozen graph changes and every receiver-score threshold for ABP."""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.utils import to_undirected

from experiment_defenses import confusion
from models.ABP_Sparse import effective_support, project_graph
from reference_experiment import observed_inputs, load_evaluation_generator

ROOT=Path(__file__).resolve().parent


def quantiles(x):
    if x.numel()==0:return None
    values=x.double().quantile(torch.tensor([0.,.25,.5,.75,1.],dtype=torch.float64))
    return dict(zip(('min','q25','median','q75','max'),map(float,values)))


def edge_keys(e,n):
    return set((e[0]*n+e[1]).tolist())


def receiver_scores(x,e,w,anchors):
    pe,pw,blocked=project_graph(e,w,len(x),anchors)
    support=effective_support(x)
    score=np.full(len(x),-np.inf)
    if blocked.numel():
        np.maximum.at(score,blocked[1].numpy(),support[blocked[0]].numpy())
    return torch.from_numpy(score),blocked,support


def counts_at(q,threshold,truth,universe):
    selected=(q>threshold*(1+1e-12)).nonzero().flatten()
    return dict(threshold=float(threshold),SelectedCount=int(torch.isin(selected,universe).sum()),
                **confusion(selected,truth,universe))


def envelope(q,truth,universe,reference_max):
    # For nonnegative thresholds, decisions change only at finite observed scores.
    levels=sorted({0.,*[float(v)/(1+1e-12) for v in q[universe] if torch.isfinite(v) and v>=0]})
    # Compare at the effective threshold itself to avoid cancelling floating errors.
    rows=[]
    for effective in sorted({0.,*[float(v) for v in q[universe] if torch.isfinite(v) and v>=0]}):
        selected=(q>effective).nonzero().flatten()
        rows.append(dict(threshold=effective/(1+1e-12),effective_threshold=effective,
            multiplier=effective/((1+1e-12)*reference_max),
            SelectedCount=int(torch.isin(selected,universe).sum()),**confusion(selected,truth,universe)))
    structural=blocked_nodes=(torch.isfinite(q)).nonzero().flatten()
    bound=confusion(structural,truth,universe)
    valid=[r for r in rows if r['threshold']>=reference_max]
    # Include the exact lower endpoint gamma=1 in the current parameter domain.
    valid.append(dict(multiplier=1.,**counts_at(q,reference_max,truth,universe)))
    def best(items):
        return max(items,key=lambda r:(r['Recall'] or 0,-(r['FPR'] or 0),r['Precision'] or 0)) if items else None
    constrained=[r for r in rows if r['Precision'] is not None and r['Precision']>=.9 and r['FPR'] is not None and r['FPR']<=.01]
    return dict(structural_ceiling=bound,best_nonnegative_threshold=best(rows),
        best_valid_multiplier=best(valid),best_precision90_fpr1=best(constrained),
        exhaustive_rows=rows)


def inspect_snapshot(path,clean):
    s=torch.load(path,map_location='cpu')
    x,e,w,labels,anchors,val,ce,cw=observed_inputs(s)
    pool=torch.cat([anchors,val]).unique(sorted=True)
    reference_max=float(effective_support(x[pool]).max())
    with torch.no_grad():
        generator=load_evaluation_generator(s,torch.device('cpu'))
        tx,te,tw=generator.inject_trigger(s['idx_atk'],x,ce,cw,torch.device('cpu'))
    train_q,blocked,support=receiver_scores(x,e,w,anchors)
    test_q,test_blocked,test_support=receiver_scores(tx,te,tw,anchors)
    clean_q,_,_=receiver_scores(x,ce,cw,anchors)
    original=int(s['num_original_nodes']); targets=s['idx_attach']
    changed_train=(x[:original]!=clean.x).any(1).nonzero().flatten()
    changed_test=(tx[:len(x)]!=x).any(1).nonzero().flatten()
    modified_dimensions=(x[:original]!=clean.x).sum(1)
    structural_target=targets[torch.isin(targets,blocked[1])]
    non_target=anchors[~torch.isin(anchors,targets)]
    test_population=torch.cat([s['idx_clean_test'],s['idx_atk']]).unique(sorted=True)
    asr_nodes=s['idx_atk'][s['original_labels'][s['idx_atk']]!=int(s['target_class'])]
    train_envelope=envelope(train_q,targets,anchors,reference_max)
    test_envelope=envelope(test_q,s['idx_atk'],test_population,reference_max)
    asr_structural=int(torch.isin(asr_nodes,test_blocked[1]).sum())
    n=max(len(x),len(clean.x),len(tx))
    stored_keys=edge_keys(ce,n); original_keys=edge_keys(clean.edge_index,n)
    source_changed=torch.isin(blocked[0],changed_train).sum().item()
    fixed_points=[]
    for multiplier in (0.,.05,.1,.2,.25,.3,.4,.5,.75,.9,.95,1.,1.25,1.5,2.,3.,4.):
        threshold=multiplier*reference_max
        row=dict(multiplier=multiplier,training=counts_at(train_q,threshold,targets,anchors),
            test=counts_at(test_q,threshold,s['idx_atk'],test_population),
            asr_detected=int((test_q[asr_nodes]>threshold*(1+1e-12)).sum()),asr_population=len(asr_nodes))
        fixed_points.append(row)
    row=dict(snapshot=str(path.relative_to(ROOT)),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        kind=s['attack'],generation_seed=s['generation_seed'],original_nodes=original,
        observed_nodes=len(x),added_nodes=len(x)-original,training_count=len(anchors),target_count=len(targets),
        train_changed_original_count=len(changed_train),test_changed_count=len(changed_test),
        train_changed_identity_matches_targets=set(changed_train.tolist())==set(targets.tolist()),
        test_changed_identity_matches_targets=set(changed_test.tolist())==set(s['idx_atk'].tolist()),
        changed_dimensions=quantiles(modified_dimensions[changed_train]),
        selected_dimensions=s.get('selected_feature_indices',torch.tensor([])).tolist(),
        completed_graph_added_arcs=len(stored_keys-original_keys),completed_graph_removed_arcs=len(original_keys-stored_keys),
        test_edges_equal=torch.equal(te,ce),test_weights_equal=torch.equal(tw,cw),
        reference_max=reference_max,default_threshold=2*reference_max,
        support_original=quantiles(effective_support(clean.x)),support_target_before=quantiles(effective_support(clean.x[targets])),
        support_target_after=quantiles(support[targets]),support_clean_training=quantiles(support[non_target]),
        support_blocked_senders=quantiles(support[blocked[0].unique()]),
        structural_arcs=blocked.size(1),structural_receivers=len(blocked[1].unique()),
        modified_training_nodes_as_blocked_senders=source_changed,
        structural_target_nodes=structural_target.tolist(),
        asr_structural_coverage_count=asr_structural,asr_count=len(asr_nodes),
        train_envelope=train_envelope,test_envelope=test_envelope,fixed_points=fixed_points)
    if s['attack']=='SPEAR':
        before_q,_,_=receiver_scores(clean.x,e,w,anchors)
        row['fixed_threshold_receiver_scores_unchanged']=torch.equal(before_q,train_q)
        row['blocked_training_anchor_sender_count']=int(torch.isin(blocked[0],anchors).sum())
        row['selected_dimensions_match_changes']=set((x!=clean.x).any(0).nonzero().flatten().tolist())==set(s['selected_feature_indices'].tolist())
        # Effective support is blind to coordinate identity, even when every changed coordinate is known offline.
        delta=x-clean.x
        row['feature_difference_l2']=quantiles(delta[targets].norm(dim=1))
        row['changed_value_range']=quantiles(x[targets][:,s['selected_feature_indices']].flatten())
    print(json.dumps({k:row[k] for k in ('kind','generation_seed','target_count','added_nodes','reference_max','default_threshold','structural_target_nodes','asr_structural_coverage_count','asr_count')},ensure_ascii=False),flush=True)
    return row


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',default='runs/spear_reference_diagnosis_20261126')
    args=parser.parse_args();torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    out=ROOT/args.output;out.mkdir(parents=True,exist_ok=True)
    raw=torch.load(ROOT/'data/Cora/processed/data.pt',map_location='cpu')[0]
    clean=NormalizeFeatures()(Data.from_dict(raw) if isinstance(raw,dict) else raw)
    clean.edge_index=to_undirected(clean.edge_index)
    paths=[ROOT/'runs/spear_losplit_20261126'/seed/'inputs/SPEAR/poison.pt' for seed in ('seed12','seed10_repeat')]
    paths += [ROOT/f'runs/abp_sparse_20261123/seed{seed}/inputs/{kind}/poison.pt' for seed in (12,16) for kind in ('GTA','UGBA','DPGBA')]
    rows=[]
    for path in paths:
        rows.append(inspect_snapshot(path,clean))
        (out/'diagnostics.json').write_text(json.dumps(dict(status='running',snapshots=rows),indent=2,allow_nan=False),encoding='utf-8')
    result=dict(status='complete',purpose='frozen_snapshot_detection_diagnosis',
        scoring_truth_used_only_for_audit=True,production_parameters_changed=False,snapshots=rows)
    (out/'diagnostics.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    print('Completed',len(rows),'snapshots; exhaustive receiver-score thresholds saved',flush=True)


if __name__=='__main__':main()
