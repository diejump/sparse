"""Development comparison of observed-feature reference scores."""
import json
from pathlib import Path
import numpy as np
import torch
from scipy.stats import rankdata
from models.ABP_Sparse import effective_support
from reference_experiment import observed_inputs
from experiment_defenses import confusion

ROOT=Path(__file__).resolve().parent

def main():
    torch.set_num_threads(1)
    results=[]
    for kind in ['GTA','UGBA','DPGBA','SPEAR']:
        path=(ROOT/'runs/spear_losplit_20261126/seed12/inputs/SPEAR/poison.pt'
              if kind=='SPEAR' else ROOT/f'runs/abp_sparse_20261123/seed16/inputs/{kind}/poison.pt')
        s=torch.load(path,map_location='cpu')
        x,e,w,y,anchors,val,ce,cw=observed_inputs(s)
        order=torch.as_tensor(np.random.RandomState(20261126).permutation(len(val)))
        fit=val[order[:len(val)//2]];cal=val[order[len(val)//2:]]
        z=x.double(); norms=z.norm(dim=1)
        distance=torch.cdist(z,z[fit])
        distance[fit,torch.arange(len(fit))]=torch.inf
        support=effective_support(z)
        eps=torch.finfo(torch.float64).eps
        descriptor=torch.stack([torch.log(norms.clamp_min(eps)),torch.log(support.clamp_min(eps))],1)
        center=descriptor[fit].median(0).values
        scale=(descriptor[fit]-center).abs().median(0).values.clamp_min(1e-6)
        scores={'euclidean':distance.min(1).values,
                'radial':((descriptor[:,0]-center[0])/scale[0]).abs(),
                'radial_support':((descriptor-center)/scale).abs().max(1).values}
        for name,score in scores.items():
            threshold=score[cal].max()
            odd=score>threshold
            selected=odd.nonzero().flatten()
            # Incident receiver expansion exposes the effect of abnormal neighbours.
            expanded=torch.cat([selected,e[1,(w>0)&odd[e[0]]]]).unique()
            row=dict(kind=kind,score=name,threshold=float(threshold),
                reference_count=len(fit),calibration_count=len(cal),
                self=confusion(selected,s['idx_attach'],anchors),
                incident=confusion(expanded,s['idx_attach'],anchors),
                normal_range=[float(score[anchors[~torch.isin(anchors,s['idx_attach'])]].min()),float(score[anchors[~torch.isin(anchors,s['idx_attach'])]].max())],
                changed_range=[float(score[s['idx_attach']].min()),float(score[s['idx_attach']].max())],
                generated_range=[float(score[s['num_original_nodes']:].min()),float(score[s['num_original_nodes']:].max())] if len(x)>s['num_original_nodes'] else None)
            results.append(row)
            print(kind,name,'self',row['self'],'incident',row['incident'],flush=True)
    out=ROOT/'runs/generalization_20261126';out.mkdir(parents=True,exist_ok=True)
    (out/'score_probe.json').write_text(json.dumps(results,indent=2),encoding='utf-8')

if __name__=='__main__':main()
