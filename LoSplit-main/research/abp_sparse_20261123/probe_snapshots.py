"""Read-only snapshot diagnostics; truth is used only for reported evaluation."""
import json
import sys
from pathlib import Path
import torch
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from models.ABP import branch_profiles

def summary(a):
    a = np.asarray(a)
    return dict(zip(['min', 'q05', 'median', 'q95', 'max'], np.quantile(a, [0,.05,.5,.95,1]).tolist())) if a.size else None

def main():
    records = []
    for name in ['my_cora_attacks', 'my_cora_attacks1', 'my_cora_attacks2']:
        for kind in ['GTA', 'UGBA', 'DPGBA']:
            path = ROOT / 'runs' / name / kind / 'poison.pt'
            if not path.exists():
                continue
            s = torch.load(path, map_location='cpu')
            x, e, w = s['poison_x'], s['poison_edge_index'], s['poison_edge_weights']
            train = torch.cat([s['idx_train'], s['idx_attach']]).sort().values
            largest, removed = branch_profiles(e[:, w > 0], x.size(0), train)
            n = s['num_original_nodes']
            feats = {}
            for group, xx in [('original', x[:n]), ('new', x[n:])]:
                feats[group] = {
                    'support_1e-6': summary((xx.abs()>1e-6).sum(1).numpy()),
                    'effective_support': summary((xx.abs().sum(1).square()/xx.square().sum(1).clamp_min(1e-30)).numpy()),
                    'negative_fraction': summary((xx<0).float().mean(1).numpy()),
                    'norm': summary(xx.norm(dim=1).numpy()),
                }
            rec = {'path':str(path.relative_to(ROOT)), 'keys':list(s.keys()), 'shape':list(x.shape), 'train':train.numel(), 'val':s['idx_val'].numel(), 'attach':s['idx_attach'].numel(), 'positive_arcs':int((w>0).sum()), 'stored_arcs':e.size(1), 'features':feats}
            for minimum in [1,2]:
                selected = train[largest[train]>=minimum]
                tp = int(torch.isin(selected, s['idx_attach']).sum())
                rec[f'branch_{minimum}']={'selected':selected.numel(), 'tp':tp, 'precision':tp/max(1,selected.numel()), 'recall':tp/s['idx_attach'].numel()}
            rec['attach_branch_histogram'] = {str(int(a)):int(b) for a,b in zip(*torch.unique(largest[s['idx_attach']],return_counts=True))}
            records.append(rec)
            print(json.dumps(rec,ensure_ascii=False),flush=True)
    (Path(__file__).parent/'snapshot_diagnostics.json').write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding='utf-8')

if __name__ == '__main__':
    main()
