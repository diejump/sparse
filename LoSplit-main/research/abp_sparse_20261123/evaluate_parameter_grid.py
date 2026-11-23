"""After parameter selection, evaluate saved development models on test nodes."""
import hashlib
import json
from pathlib import Path
import sys
import torch
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from research.abp_sparse_20261123.tune_sparse import prepare,model_from_config,evaluate_saved
from models.ABP_Sparse_Training import TunableSparseBranchDetector
OUT=ROOT/'runs/abp_sparse_20261123/parameter_study'

def main():
    torch.set_num_threads(1)
    selection=OUT/'confirmation_selection.json'
    assert selection.exists()
    digest=hashlib.sha256(selection.read_bytes()).hexdigest()
    rows=[];data_cache={}
    for phase in ['screen','threshold_ablation']:
        for path in sorted((OUT/phase).glob('gen10/*/mult*/seed*.json')):
            row=json.loads(path.read_text())
            kind=row['kind']
            if kind not in data_cache:data_cache[kind]=prepare(10,kind)
            data=data_cache[kind]
            model=model_from_config(data)
            saved=torch.load(path.with_suffix('.pt'),map_location='cpu')
            model.load_state_dict(saved['state_dict']);model.eval()
            detector=TunableSparseBranchDetector(row['multiplier']).fit(data['x'],data['reference'])
            row['test']=evaluate_saved(data,model,detector)
            row['parameter_selection_sha256_before_test_evaluation']=digest
            rows.append(row)
    assert len(rows)==75
    (OUT/'development_test_metrics.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
    print(json.dumps({'evaluated_models':len(rows),'selected_configuration':json.loads(selection.read_text())['winner'],
        'selection_sha256_unchanged':hashlib.sha256(selection.read_bytes()).hexdigest()==digest},indent=2))

if __name__=='__main__':main()
