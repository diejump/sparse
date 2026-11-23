"""Post-freeze diagnostics: do not change any fitted detection rule."""
import json
from pathlib import Path
import sys
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from models.ABP_Sparse import SparseBranchDetector,effective_support,project_graph
from research.abp_sparse_20261123.run_study import confusion


def main():
    torch.set_num_threads(1)
    records=[]
    for seed in [10,11,12]:
        for kind in ['GTA','UGBA','DPGBA']:
            if seed==10:
                p=ROOT/'runs'/('dpgba_official_fix_20261123/inputs' if kind=='DPGBA' else 'my_cora_attacks3')/kind/'poison.pt'
            else:
                p=ROOT/'runs/abp_sparse_20261123'/f'seed{seed}'/'inputs'/kind/'poison.pt'
            s=torch.load(p,map_location='cpu')
            x,e,w=s['poison_x'],s['poison_edge_index'],s['poison_edge_weights']
            a=torch.cat([s['idx_train'],s['idx_attach']]).unique(sorted=True)
            refs=torch.cat([a,s['idx_val']]).unique(sorted=True)
            det=SparseBranchDetector().fit(x,refs)
            selected,info=det.detect(x,e,w,a,a)
            blocked=info['blocked_arcs']
            r=info['support_scores']
            scores=np.zeros(len(x))
            np.maximum.at(scores,blocked[1].numpy(),r[blocked[0]].numpy())
            clean=a[~torch.isin(a,s['idx_attach'])]
            max_clean=float(scores[clean.numpy()].max())
            min_changed=float(scores[s['idx_attach'].numpy()].min())
            n=int(s['num_original_nodes'])
            internal=(e[0]>=n)&(e[1]>=n)&(w>0)
            roots=torch.arange(n,len(x),int(s['trigger_size']))
            deg=torch.bincount(e[1,w>0],minlength=len(x))
            record={'generation_seed':seed,'kind':kind,'threshold':det.threshold,
                'reference_max':det.reference_max,'original_support_max':float(r[:n].max()),
                'generated_support_min':float(r[n:].min()),'root_support_min':float(r[roots].min()),
                'clean_receiver_score_max':max_clean,'changed_receiver_score_min':min_changed,
                'perfect_threshold_interval_left_inclusive_right_exclusive':[max_clean,min_changed],
                'internal_positive_undirected_edges':int(internal.sum())//2,
                'fully_isolated_generated_nodes':int((deg[n:]==0).sum()),
                'validation_changed_labels':int(torch.isin(s['idx_val'],s['idx_attach']).sum()),
                'detection':confusion(selected,s['idx_attach'],a),'margin_sensitivity':[]}
            for margin in [1.,1.25,1.5,2.,4.,8.,16.,32.]:
                candidate=a[torch.from_numpy(scores[a.numpy()])>margin*det.reference_max*(1+1e-12)]
                record['margin_sensitivity'].append({'margin':margin,**confusion(candidate,s['idx_attach'],a)})
            records.append(record)
    out=ROOT/'runs/abp_sparse_20261123/detection_analysis.json'
    out.write_text(json.dumps(records,indent=2),encoding='utf-8')
    print(json.dumps([{k:v for k,v in r.items() if k!='margin_sensitivity'} for r in records],indent=2))

if __name__=='__main__':
    main()
