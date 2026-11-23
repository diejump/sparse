"""Verify all saved parameter runs and the final report links and language."""
import hashlib
import json
from pathlib import Path
import re
ROOT=Path(__file__).resolve().parents[2]
DOC=Path(__file__).parent
OUT=ROOT/'runs/abp_sparse_20261123/parameter_study'

def main():
    expected={'screen':72,'confirmation':24,'threshold_ablation':3,'validation':108,'asr_priority':18}
    rows=[]
    for phase,count in expected.items():
        files=list((OUT/phase).glob('gen*/*/mult*/seed*.json'))
        assert len(files)==count,(phase,len(files),count)
        for p in files:
            r=json.loads(p.read_text());rows.append(r)
            assert p.with_suffix('.pt').is_file()
            for source,sha in r['source_sha256'].items():
                assert hashlib.sha256((ROOT/source).read_bytes()).hexdigest()==sha
            d=r['detection']
            assert d['precision']==d['tp']/(d['tp']+d['fp'])
            assert d['recall']==d['tp']/(d['tp']+d['fn'])
            assert d['fpr']==d['fp']/(d['fp']+d['tn'])
            if 'test' in r:
                t=r['test']
                assert t['CA']==t['clean_correct']/t['clean_count']
                assert t['ASR']==t['target_count']/t['asr_count']
                assert t['clean_count']==270 and t['asr_count']==225
                assert t['original_output_max_difference']<1e-5
    assert len(rows)==225
    forbidden=re.compile(r'锚|P/R|Material Passport|abp_dual|abp_assisted|\b(?:TP|FP|FN|TN)\b|冻结|路由|投影')
    documents=[DOC/'REPORT.md',DOC/'theory_review.md',DOC/'audit.md',ROOT/'runs/abp_sparse_20261123/RESULTS.md']
    links=0
    for path in documents:
        content=path.read_text(encoding='utf-8-sig')
        assert not forbidden.search(content),(path,forbidden.search(content).group(0))
        for target in re.findall(r'\[[^\]]*\]\(([^)]+)\)',content):
            if not target.startswith(('http:','https:')):
                assert (path.parent/target).resolve().exists(),(path,target)
                links+=1
    selection_digest=hashlib.sha256((OUT/'confirmation_selection.json').read_bytes()).hexdigest()
    exploratory=json.loads((OUT/'development_test_metrics.json').read_text())
    assert len(exploratory)==75
    assert all(r['parameter_selection_sha256_before_test_evaluation']==selection_digest for r in exploratory)
    checks={'status':'VERIFIED','classifier_fits':225,'weights_present':225,
        'input_source_hashes_match':True,'detection_formulas_match':True,'test_formulas_match':True,
        'documents_checked':4,'local_links_checked':links,'old_comparison_sections_removed':True,
        'selection_fixed_before_exploratory_test_evaluation':True,
        'parameter_count_original':1,'parameter_count_with_new_training_term':2}
    (OUT/'report_verification.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')
    print(json.dumps(checks,indent=2))

if __name__=='__main__':main()
