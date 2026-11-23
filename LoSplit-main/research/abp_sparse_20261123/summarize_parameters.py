"""Aggregate complete parameter experiments without changing the selection."""
import csv
import hashlib
import json
from pathlib import Path
import statistics
import torch
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'runs/abp_sparse_20261123/parameter_study'

def average(rows,key,section='test'):
    return statistics.mean(r[section][key] for r in rows)

def main():
    selection=json.loads((OUT/'confirmation_selection.json').read_text())
    winner=selection['winner']
    rows=[json.loads(p.read_text()) for p in (OUT/'validation').glob('gen*/*/mult*/seed*.json')]
    assert len(rows)==108,f'Validation fits incomplete: {len(rows)}/108'
    assert len({(r['generation_seed'],r['kind'],r['classifier_seed'],r['multiplier'],r['penalty_weight']) for r in rows})==108
    summary=[]
    for m,l in sorted({(r['multiplier'],r['penalty_weight']) for r in rows}):
        for kind in ['GTA','UGBA','DPGBA']:
            rr=[r for r in rows if r['multiplier']==m and r['penalty_weight']==l and r['kind']==kind]
            assert len(rr)==9
            summary.append({'multiplier':m,'penalty_weight':l,'kind':kind,'models':len(rr),
                'CA':average(rr,'CA'),'ASR':average(rr,'ASR'),
                'minimum_ASR':min(r['test']['ASR'] for r in rr),'maximum_ASR':max(r['test']['ASR'] for r in rr),
                'precision':average(rr,'precision','detection'),'recall':average(rr,'recall','detection'),
                'FPR':average(rr,'fpr','detection'),
                'minimum_precision':min(r['detection']['precision'] for r in rr),'minimum_recall':min(r['detection']['recall'] for r in rr),
                'normal_target_class_accuracy':average(rr,'normal_target_class_accuracy'),
                'all_test_clean_accuracy':average(rr,'all_test_clean_accuracy')})
    final=[r for r in rows if r['multiplier']==winner['multiplier'] and r['penalty_weight']==winner['penalty_weight']]
    assert len(final)==27
    counts=[]
    for generation in [11,12,13]:
        for kind in ['GTA','UGBA','DPGBA']:
            rr=[r for r in final if r['generation_seed']==generation and r['kind']==kind]
            assert all(r['detection']==rr[0]['detection'] for r in rr)
            counts.append({'generation_seed':generation,'kind':kind,**rr[0]['detection']})
    base_reproduction=[]
    for kind in ['GTA','UGBA','DPGBA']:
        for seed in [10,20,30]:
            phase='screen' if seed==10 else 'confirmation'
            new_path=OUT/phase/'gen10'/kind/'mult2p0_pen0p0'/f'seed{seed}.pt'
            old_path=ROOT/'runs/abp_sparse_20261123/validation_v2/gen10'/kind/f'abp_sparse_seed{seed}.pt'
            a=torch.load(new_path,map_location='cpu')['state_dict']
            b=torch.load(old_path,map_location='cpu')['state_dict']
            same=set(a)==set(b) and all(torch.equal(a[k],b[k]) for k in a)
            assert same,f'Zero-penalty training differs: {kind} {seed}'
            base_reproduction.append({'kind':kind,'seed':seed,'weights_equal':same})
    for r in rows:
        assert r['test']['original_output_max_difference']<1e-5
        assert not r['test_metrics_used_for_selection']
        for file,expected in r['source_sha256'].items():
            assert hashlib.sha256((ROOT/file).read_bytes()).hexdigest()==expected
    priority=[json.loads(p.read_text()) for p in (OUT/'asr_priority').glob('gen13/*/mult*/seed*.json')]
    assert len(priority)==18
    priority_summary=[]
    for kind in ['GTA','UGBA','DPGBA']:
        for penalty in [0.,.25,.5,2.,4.]:
            if penalty in [2.,4.]:
                rr=[r for r in priority if r['kind']==kind and r['penalty_weight']==penalty]
            else:
                multiplier=.95 if penalty==.25 else 2.
                rr=[r for r in rows if r['generation_seed']==13 and r['kind']==kind and r['penalty_weight']==penalty and r['multiplier']==multiplier]
            assert len(rr)==3
            priority_summary.append({'kind':kind,'generation_seed':13,'multiplier':rr[0]['multiplier'],
                'penalty_weight':penalty,'CA':average(rr,'CA'),'ASR':average(rr,'ASR'),
                'precision':average(rr,'precision','detection'),'recall':average(rr,'recall','detection'),
                'FPR':average(rr,'fpr','detection'),'normal_target_class_accuracy':average(rr,'normal_target_class_accuracy')})
    payload={'status':'VERIFIED','metric_audit_models':27,'detection_threshold_settings':24,
        'screening_models':72,'confirmation_models':24,'additional_threshold_models':3,'validation_models':108,
        'asr_priority_models':18,'total_new_classifier_fits':225,'winner':winner,'summary':summary,
        'priority_summary':priority_summary,'selected_detection_counts':counts,
        'zero_penalty_reproduction':base_reproduction,'all_selected_precision_recall_above_90':all(r['detection']['precision']>.9 and r['detection']['recall']>.9 for r in final),
        'all_selected_ASR_range':[min(r['test']['ASR'] for r in final),max(r['test']['ASR'] for r in final)],
        'generation_seeds':[11,12,13],'classifier_seeds':[10,20,30],
        'test_selection':False,'normal_target_class_metric_scope':'All untriggered test nodes whose true class is the designated class'}
    (OUT/'parameter_summary.json').write_text(json.dumps(payload,indent=2),encoding='utf-8')
    flat=[]
    for r in rows:
        item={k:r[k] for k in ['generation_seed','kind','classifier_seed','multiplier','penalty_weight']}
        item.update({k:r['detection'][k] for k in ['tp','fp','fn','tn','precision','recall','fpr']})
        item.update({k:r['test'][k] for k in ['CA','ASR','clean_correct','clean_count','target_count','asr_count','normal_target_class_accuracy','normal_target_class_count']})
        flat.append(item)
    with (OUT/'validation_metrics.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(flat[0]));writer.writeheader();writer.writerows(flat)
    print(json.dumps({'winner':winner,'summary':summary,'selected_ASR_range':payload['all_selected_ASR_range'],'all_selected_precision_recall_above_90':payload['all_selected_precision_recall_above_90']},indent=2))

if __name__=='__main__':main()
