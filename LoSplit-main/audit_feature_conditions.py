"""Check theorem conditions on generated snapshots, outside model fitting."""
import hashlib
import json
from pathlib import Path

import torch
from torch_geometric.data import Data
from torch_geometric.transforms import NormalizeFeatures

from models.FeatureConsistency import feature_consistency
from experiment_defenses import confusion
from reference_experiment import observed_inputs,load_evaluation_generator

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'runs/feature_reference_20261126'


def sign_mass_nodes(features,tolerance=1e-6):
    nonzero=features.abs().amax(1)>tolerance
    negative=features.amin(1)<-tolerance
    mass=(features.double().sum(1)-1).abs()>tolerance
    return (negative|(nonzero&mass)).nonzero().flatten()


def recovery_conditions(observed,original,nodes,tolerance=1e-6):
    result=feature_consistency(observed,tolerance)
    records=[]
    for node in nodes.tolist():
        x=original[node];z=observed[node];restored=result['features'][node]
        support=x>0;k=int(support.sum())
        if k==0:continue
        level=float(x[support][0])
        unchanged=(x-z).abs()<=tolerance
        kept=support&unchanged;h=int(kept.sum());r=k-h
        values,counts=torch.unique(z[z>tolerance],return_counts=True)
        other=counts[(values-level).abs()>tolerance]
        maximal_other=int(other.max()) if len(other) else 0
        mode_matches=abs(float(result['levels'][node])-level)<=tolerance
        support_matches=torch.equal(restored>0,kept)
        condition=h>0 and maximal_other<h and mode_matches and support_matches
        row=dict(node=node,k=k,lost=r,remaining=h,other_max_frequency=maximal_other,
            conditions_hold=condition,ambiguous_mode=not bool(result['unique_mode'][node]),
            extra_coordinates=int(((restored>0)&~support).sum()))
        if condition:
            row.update(l1_error=float((restored-x).double().abs().sum()),l1_formula=2*r/k,
                       l2_squared=float((restored-x).double().square().sum()),l2_formula=r/(k*h))
        records.append(row)
    qualified=[r for r in records if r['conditions_hold']]
    return dict(count=len(records),conditions_hold=len(qualified),
        ambiguous=sum(r['ambiguous_mode'] for r in records),extra_coordinates=sum(r['extra_coordinates'] for r in records),
        max_l1_formula_error=max((abs(r['l1_error']-r['l1_formula']) for r in qualified),default=0),
        max_l2_formula_error=max((abs(r['l2_squared']-r['l2_formula']) for r in qualified),default=0),rows=records)


def main():
    torch.set_num_threads(1)
    raw=torch.load(ROOT/'data/Cora/processed/data.pt',map_location='cpu')[0]
    original=NormalizeFeatures()(Data.from_dict(raw) if isinstance(raw,dict) else raw).x
    normal=feature_consistency(original)
    frozen=json.loads((OUT/'frozen_config.json').read_text(encoding='utf-8'))
    result=dict(normal_count=len(original),normal_feature_detections=len(normal['nodes']),
        normal_repair_exact=torch.equal(original,normal['features']),
        frozen_source_match={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==digest for p,digest in frozen['source_sha256'].items()},
        snapshots=[])
    for seed in (31,32,33):
        for kind in ('GTA','UGBA','DPGBA','SPEAR'):
            path=OUT/f'confirmation_inputs/seed{seed}/{kind}/poison.pt'
            s=torch.load(path,map_location='cpu')
            x,e,w,y,a,v,ce,cw=observed_inputs(s)
            with torch.no_grad():tx,te,tw=load_evaluation_generator(s,torch.device('cpu')).inject_trigger(s['idx_atk'],x,ce,cw,torch.device('cpu'))
            base=feature_consistency(x);test=feature_consistency(tx)
            tolerance_checks=[]
            for epsilon in (1e-7,1e-6,1e-5):
                train_variant=feature_consistency(x,epsilon);test_variant=feature_consistency(tx,epsilon)
                tolerance_checks.append(dict(tolerance=epsilon,
                    training_decisions_same=torch.equal(base['nodes'],train_variant['nodes']),
                    test_decisions_same=torch.equal(test['nodes'],test_variant['nodes']),
                    training_repair_same=torch.equal(base['features'],train_variant['features']),
                    test_repair_same=torch.equal(test['features'],test_variant['features'])))
            row=dict(kind=kind,seed=seed,tolerance_checks=tolerance_checks)
            if kind=='SPEAR':
                row['training_recovery']=recovery_conditions(x,original,s['idx_attach'])
                row['test_recovery']=recovery_conditions(tx,original,s['idx_atk'])
                clipped_train=feature_consistency(x,repair='clip_normalize')['features']
                clipped_test=feature_consistency(tx,repair='clip_normalize')['features']
                cp=feature_consistency(clipped_train);ct=feature_consistency(clipped_test)
                universe=torch.cat([s['idx_atk'],s['idx_clean_test']]).unique()
                row['clipped_normalized_feature_detection']=dict(
                    training=confusion(cp['nodes'],s['idx_attach'],a),test=confusion(ct['nodes'],s['idx_atk'],universe))
                row['simple_sign_mass_detection']=dict(
                    raw_training=confusion(sign_mass_nodes(x),s['idx_attach'],a),
                    raw_test=confusion(sign_mass_nodes(tx),s['idx_atk'],universe),
                    normalized_training=confusion(sign_mass_nodes(clipped_train),s['idx_attach'],a),
                    normalized_test=confusion(sign_mass_nodes(clipped_test),s['idx_atk'],universe))
            result['snapshots'].append(row)
            print(seed,kind,'conditions checked',flush=True)
    (OUT/'condition_audit.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')


if __name__=='__main__':main()
