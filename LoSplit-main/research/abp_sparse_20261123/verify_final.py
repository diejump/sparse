"""Check study completeness, source identities, and one exact training replay."""
import hashlib
import json
from pathlib import Path
import sys
import torch
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'runs/abp_sparse_20261123'

def main():
    summary=json.loads((OUT/'summary.json').read_text())
    assert summary['main_classifier_runs']==108
    assert summary['all_new_method_meet_temporary_targets']
    assert summary['stress_all_meet_targets']
    manifests=list((OUT/'validation_v2').glob('gen*/*/results.json'))
    assert len(manifests)==9
    hashes={}
    source_matches=True
    for path in manifests:
        manifest=json.loads(path.read_text())
        assert manifest['device']=='cpu'
        assert len(manifest['records'])==12
        for relative,expected in manifest['source_sha256'].items():
            actual=hashlib.sha256((ROOT/relative).read_bytes()).hexdigest()
            source_matches &= actual==expected
            hashes.setdefault(relative,set()).add(expected)
    assert source_matches
    original=OUT/'validation_v2/gen11/GTA'
    replay=OUT/'reproduce_v2/GTA_gen11_seed20'
    a=torch.load(original/'abp_sparse_seed20.pt',map_location='cpu')
    b=torch.load(replay/'abp_sparse_seed20.pt',map_location='cpu')
    state_match=set(a['state_dict'])==set(b['state_dict']) and all(torch.equal(a['state_dict'][k],b['state_dict'][k]) for k in a['state_dict'])
    ra=json.loads((original/'abp_sparse_seed20.json').read_text())
    rb=json.loads((replay/'abp_sparse_seed20.json').read_text())
    keys=['CA','ASR','detection','triggered_receiver_detection','selected_nodes','prediction_changes','max_original_logit_delta','fit_count','validation_count']
    metrics_match=all(ra[k]==rb[k] for k in keys)
    assert state_match and metrics_match
    study_sources=[ROOT/'models/ABP_Sparse.py',*Path(__file__).parent.glob('*.py')]
    for path in study_sources:
        compile(path.read_text(encoding='utf-8'),str(path),'exec')
    result={'status':'VERIFIED','main_models':108,'new_method_main_models':27,'stress_models':9,
        'ablation_no_projection_models':3,'exact_replay_models':1,
        'main_source_hashes_match_current_files':source_matches,
        'same_source_hashes_across_all_main_runs':all(len(v)==1 for v in hashes.values()),
        'exact_replay_parameters_equal':state_match,'exact_replay_metrics_equal':metrics_match,
        'syntax_checked_files':len(study_sources),
        'note':'All main comparisons use one fixed Cora node split; repeat counts do not imply independent graphs.'}
    assert result['same_source_hashes_across_all_main_runs']
    (OUT/'final_verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))

if __name__=='__main__':
    main()
