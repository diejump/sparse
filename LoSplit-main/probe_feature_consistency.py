"""Evaluate representation consistency using frozen development snapshots."""
import hashlib
import json
from pathlib import Path

import torch
from torch_geometric.data import Data
from torch_geometric.transforms import NormalizeFeatures

from models.FeatureConsistency import feature_consistency,FeatureBranchDetector
from reference_experiment import observed_inputs,load_evaluation_generator
from experiment_defenses import confusion

ROOT=Path(__file__).resolve().parent


def main():
    torch.set_num_threads(1)
    out=ROOT/'runs/feature_reference_20261126/development'
    out.mkdir(parents=True,exist_ok=True)
    raw=torch.load(ROOT/'data/Cora/processed/data.pt',map_location='cpu')[0]
    clean=NormalizeFeatures()(Data.from_dict(raw) if isinstance(raw,dict) else raw).x
    normal=feature_consistency(clean)
    paths=[ROOT/'runs/spear_losplit_20261126'/seed/'inputs/SPEAR/poison.pt' for seed in ('seed12','seed10_repeat')]
    paths += [ROOT/f'runs/abp_sparse_20261123/seed16/inputs/{kind}/poison.pt' for kind in ('GTA','UGBA','DPGBA')]
    rows=[]
    for path in paths:
        s=torch.load(path,map_location='cpu')
        x,e,w,y,anchors,val,ce,cw=observed_inputs(s)
        detector=FeatureBranchDetector(2.).fit(x,torch.cat([anchors,val]).unique())
        selected,info=detector.detect(x,e,w,anchors,anchors)
        repair=feature_consistency(x)
        with torch.no_grad():tx,te,tw=load_evaluation_generator(s,torch.device('cpu')).inject_trigger(s['idx_atk'],x,ce,cw,torch.device('cpu'))
        test_selected,test_info=detector.detect(tx,te,tw,anchors)
        test_u=torch.cat([s['idx_clean_test'],s['idx_atk']]).unique()
        row=dict(snapshot=str(path.relative_to(ROOT)),kind=s['attack'],seed=s['generation_seed'],
            training=confusion(selected,s['idx_attach'],anchors),
            feature_only=confusion(info['feature_nodes'],s['idx_attach'],anchors),
            structure_only=confusion(info['structural_nodes'],s['idx_attach'],anchors),
            test=confusion(test_selected,s['idx_atk'],test_u),
            retained_feature_count_min=int(repair['retained_counts'][s['idx_attach']].min()),
            snapshot_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        if s['attack']=='SPEAR':
            restored=repair['features'][s['idx_attach']]
            original=clean[s['idx_attach']]
            original_support=original>0
            restored_support=restored>0
            missing=(original_support&~restored_support).sum(1)
            k=original_support.sum(1)
            extra=(~original_support&restored_support).sum(1)
            observed=(restored-original).abs().sum(1).double()
            expected=2*missing.double()/k
            row['repair_audit']=dict(added_coordinates=int(extra.sum()),lost_coordinates=missing.tolist(),
                original_support=k.tolist(),max_l1_formula_error=float((observed-expected).abs().max()),
                exact_rows=int((missing==0).sum()),total_rows=len(missing))
        rows.append(row)
        print(s['attack'],s['generation_seed'],row['training'],row.get('repair_audit'),flush=True)
    (out/'feature_probe.json').write_text(json.dumps(dict(normal_feature_false_positives=len(normal['nodes']),
        normal_repair_exact=torch.equal(normal['features'],clean),rows=rows),indent=2),encoding='utf-8')


if __name__=='__main__':main()
