"""Choose settings using development detection and validation results only."""
import argparse
import json
from pathlib import Path
import statistics

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'runs/abp_sparse_20261123/parameter_study'

def collect(folder):
    return [json.loads(p.read_text()) for p in folder.glob('gen10/*/mult*/seed*.json')]

def main():
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['screen','confirmation'],default='screen')
    args=p.parse_args()
    rows=collect(OUT/'screen')
    assert len(rows)==72, f'Expected 72 screening fits, found {len(rows)}'
    if args.phase=='confirmation':
        extra=collect(OUT/'confirmation');rows+=extra
        configs=json.loads((OUT/'screen_selection.json').read_text())['confirmation_configurations']
        allowed={(r['multiplier'],r['penalty_weight']) for r in configs}
        rows=[r for r in rows if (r['multiplier'],r['penalty_weight']) in allowed]
        assert len(rows)==len(allowed)*9
    baseline=[r for r in rows if r['multiplier']==2 and r['penalty_weight']==0]
    avg=lambda rs,key: statistics.mean(r['common_validation'][key] for r in rs)
    baseline_acc=avg(baseline,'accuracy');baseline_class=avg(baseline,'class_accuracy')
    settings=[]
    for m,l in sorted({(r['multiplier'],r['penalty_weight']) for r in rows}):
        rs=[r for r in rows if r['multiplier']==m and r['penalty_weight']==l]
        record={'multiplier':m,'penalty_weight':l,'fits':len(rs),
            'validation_accuracy':avg(rs,'accuracy'),
            'validation_unwanted_class_rate':avg(rs,'unwanted_class_rate'),
            'validation_class_accuracy':avg(rs,'class_accuracy'),
            'minimum_precision':min(r['detection']['precision'] for r in rs),
            'minimum_recall':min(r['detection']['recall'] for r in rs)}
        record['accuracy_change']=record['validation_accuracy']-baseline_acc
        record['class_accuracy_change']=record['validation_class_accuracy']-baseline_class
        record['admissible']=record['minimum_precision']>.9 and record['minimum_recall']>.9 and record['accuracy_change']>=-.02 and record['class_accuracy_change']>=-.05
        settings.append(record)
    eligible=[r for r in settings if r['admissible']]
    eligible.sort(key=lambda r:(r['validation_unwanted_class_rate'],-r['validation_accuracy'],r['penalty_weight'],abs(r['multiplier']-2)))
    assert eligible
    winner=eligible[0]
    result={'phase':args.phase,'selection_uses_test_metrics':False,'winner':winner,'settings':settings}
    if args.phase=='screen':
        finalists=[{'multiplier':2.,'penalty_weight':0.}]
        for r in eligible:
            config={k:r[k] for k in ['multiplier','penalty_weight']}
            if config not in finalists:
                finalists.append(config)
            if len(finalists)==4:
                break
        result['confirmation_configurations']=finalists
    (OUT/f'{args.phase}_selection.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result if args.phase=='confirmation' else {'winner':winner,'confirmation_configurations':result['confirmation_configurations'],'admissible_settings':eligible},indent=2))

if __name__=='__main__':main()
