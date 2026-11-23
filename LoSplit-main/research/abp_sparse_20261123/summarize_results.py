"""Validate the completed fixed design and generate traceable result tables."""
import csv
import json
from pathlib import Path
import statistics

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'runs/abp_sparse_20261123'
DOC=Path(__file__).parent
KINDS=['GTA','UGBA','DPGBA']
METHODS=['none','abp','abp_all_candidates','abp_sparse']


def metric(values):
    return {'mean':statistics.mean(values),'sd':statistics.stdev(values) if len(values)>1 else None,
            'min':min(values),'max':max(values),'count':len(values)}


def pct(x):
    return '—' if x is None else f'{100*x:.2f}'


def table(rows):
    header='| 生成方式 | 方法 | 分类器运行数 | CA均值 (%) | ASR均值 (%) | ASR范围 (%) | Precision (%) | Recall (%) |\n|---|---|---:|---:|---:|---:|---:|---:|'
    lines=[header]
    for kind in KINDS:
        for method in METHODS:
            rs=[r for r in rows if r['kind']==kind and r['method']==method]
            if not rs:
                continue
            ps=[r['detection']['precision'] for r in rs if r['detection'] is not None]
            recalls=[r['detection']['recall'] for r in rs if r['detection'] is not None]
            values=[r['ASR'] for r in rs]
            lines.append(f'| {kind} | {method} | {len(rs)} | {pct(statistics.mean(r["CA"] for r in rs))} | {pct(statistics.mean(values))} | {pct(min(values))}–{pct(max(values))} | {pct(statistics.mean(ps)) if ps else "—"} | {pct(statistics.mean(recalls)) if recalls else "—"} |')
    return '\n'.join(lines)


def main():
    rows=[]
    for generation in [10,11,12]:
        for kind in KINDS:
            for seed in [10,20,30]:
                for method in METHODS:
                    path=OUT/'validation_v2'/f'gen{generation}'/kind/f'{method}_seed{seed}.json'
                    r=json.loads(path.read_text(encoding='utf-8'))
                    assert r['protocol_version']==2
                    assert r['kind']==kind and r['generation_seed']==generation and r['classifier_seed']==seed
                    r['result_file']=str(path.relative_to(ROOT)).replace('\\','/')
                    rows.append(r)
    assert len(rows)==108
    new=[r for r in rows if r['method']=='abp_sparse']
    for r in new:
        assert r['detection']['tp']>0 and r['detection']['fp']==0 and r['detection']['fn']==0
        assert r['prediction_changes']==0 and r['max_original_logit_delta']<1e-5
        assert r['ASR']==r['untriggered_target_error']
    summary={'fixed_design_completed':True,'main_classifier_runs':len(rows),'new_method_runs':len(new),
        'all_new_method_meet_temporary_targets':all(r['ASR']<=.1 and r['detection']['precision']>=.9 and r['detection']['recall']>=.9 for r in new),
        'development_generation_seed':10,'heldout_generation_seeds':[11,12],
        'classifier_seeds':[10,20,30],'same_node_split':True,'by_scope':{}}
    for name,generations in [('development',[10]),('heldout',[11,12]),('all',[10,11,12])]:
        summaries=[]
        for kind in KINDS:
            for method in METHODS:
                rs=[r for r in rows if r['generation_seed'] in generations and r['kind']==kind and r['method']==method]
                per_gen=[statistics.mean(r['ASR'] for r in rs if r['generation_seed']==g) for g in generations]
                summaries.append({'kind':kind,'method':method,'CA':metric([r['CA'] for r in rs]),
                    'ASR':metric([r['ASR'] for r in rs]),'ASR_after_averaging_classifiers_per_generation':metric(per_gen),
                    'detection_once_per_generation':[next(r['detection'] for r in rs if r['generation_seed']==g) for g in generations]})
        summary['by_scope'][name]=summaries
    stress=[]
    for generation in [10,11,12]:
        for kind in KINDS:
            r=json.loads((OUT/'no_internal_v2'/f'gen{generation}'/kind/'abp_sparse_seed10.json').read_text())
            assert r['stress']=='no_internal'
            baseline=next(x for x in rows if x['generation_seed']==generation and x['kind']==kind and x['method']=='abp_sparse' and x['classifier_seed']==10)
            r['same_CA_ASR_as_original_topology']=r['CA']==baseline['CA'] and r['ASR']==baseline['ASR']
            stress.append(r)
    summary['no_internal_stress_runs']=len(stress)
    summary['stress_all_meet_targets']=all(r['ASR']<=.1 and r['detection']['precision']==1 and r['detection']['recall']==1 for r in stress)
    summary['stress_same_predictions_summary']=all(r['same_CA_ASR_as_original_topology'] for r in stress)
    flat=[]
    for r in rows:
        rec={k:r[k] for k in ['generation_seed','kind','classifier_seed','method','CA','ASR','clean_correct','clean_count','target_count','asr_count','untriggered_target_error','prediction_changes','max_original_logit_delta','fit_count','validation_count','result_file','snapshot_sha256']}
        rec.update({k:r['detection'][k] if r['detection'] else None for k in ['tp','fp','fn','tn','precision','recall','fpr']})
        flat.append(rec)
    with (OUT/'main_results.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=list(flat[0]));writer.writeheader();writer.writerows(flat)
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    from plain_reports import write_reports
    write_reports()
    print(json.dumps({'main_runs':len(rows),'new_runs':len(new),'targets_met':summary['all_new_method_meet_temporary_targets'],
        'new_CA':metric([r['CA'] for r in new]),'new_ASR':metric([r['ASR'] for r in new]),
        'heldout':[x for x in summary['by_scope']['heldout'] if x['method']=='abp_sparse']},indent=2))

if __name__=='__main__':
    main()
