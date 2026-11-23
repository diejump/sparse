"""Evaluation-only accounting of snapshot split overlap and observed labels."""
import json
from pathlib import Path
import torch

ROOT=Path(__file__).resolve().parents[3]
paths=[ROOT/'runs/my_cora_attacks3'/kind/'poison.pt' for kind in ['GTA','UGBA','DPGBA']]
paths += list((ROOT/'runs/abp_sparse_20261123').glob('seed*/inputs/*/poison.pt'))
rows=[]
for path in paths:
    if not path.exists():
        continue
    s=torch.load(path,map_location='cpu')
    keys=['idx_train','idx_attach','idx_val','idx_clean_test','idx_atk']
    sets={k:set(s[k].tolist()) for k in keys}
    intersections={a+'&'+b:len(sets[a]&sets[b]) for i,a in enumerate(keys) for b in keys[i+1:]}
    val=s['idx_val']
    val_wrong=val[s['poison_labels'][val]!=s['original_labels'][val]]
    row={'path':str(path.relative_to(ROOT)),'counts':{k:len(v) for k,v in sets.items()},
         'overlap':intersections,'validation_label_changes':int(val_wrong.numel()),
         'changed_validation_ids':val_wrong.tolist()}
    rows.append(row)
    print(json.dumps(row,ensure_ascii=False))
(Path(__file__).parent/'snapshot_overlap.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
