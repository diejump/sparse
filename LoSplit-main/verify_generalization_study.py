"""Reload every model and independently recount task and recovery metrics."""
import argparse
import hashlib
import json
import time
from pathlib import Path
import torch
from scipy.stats import beta
from experiment_defenses import confusion
from reference_experiment import observed_inputs,load_evaluation_generator,evaluate_reference
from models.ABP_Reference import load_reference_bundle
from envelope_experiment import load_envelope_bundle,extra_metrics

def upper(hits,count):
    return float(beta.ppf(.95,hits+1,count-hits)) if hits<count else 1.

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('study')
    p.add_argument('--follow',action='store_true');a=p.parse_args()
    root=Path(a.study);study=json.loads((root/'study.json').read_text(encoding='utf-8'))
    out=root/'verification.json'
    if out.exists():raise FileExistsError(out)
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    rows=[]
    for job in study['jobs']:
        if a.follow and job['status']=='planned':
            started=time.monotonic()
            while job['status']=='planned' and time.monotonic()-started<1800:
                current=json.loads((root/'study.json').read_text(encoding='utf-8'))
                job=next(j for j in current['jobs'] if j['id']==job['id'])
                if job['status']=='planned':time.sleep(5)
        if job['status']!='completed':
            rows.append(dict(id=job['id'],verified=False,status=job['status']));continue
        s=torch.load(job['snapshot'],map_location='cpu')
        model=(load_envelope_bundle if job['case'] not in ('old','none') else load_reference_bundle)(job['output'])
        x,e,w,y,anchors,val,ce,cw=observed_inputs(s)
        with torch.no_grad():
            generator=load_evaluation_generator(s,x.device)
            result,clean,changed,graph=evaluate_reference(model,s,ce,cw,generator)
            selected,_=model.detector.detect(x,e,w,anchors,anchors)
            result.update(confusion(selected,s['idx_attach'],anchors))
            result.update(extra_metrics(model,s,clean,changed,graph,model.detector))
        saved=json.loads(Path(job['output']+'.meta.json').read_text(encoding='utf-8'))
        keys=['CA','ASR','Precision','Recall','FPR','TP','FP','FN','TN','CleanCorrect','CleanCount','ASRHits','ASRCount']
        errors={k:[saved.get(k),result[k]] for k in keys if saved.get(k)!=result[k]}
        nodes=s['idx_atk'];labels=s['original_labels'];nclass=int(labels.max())+1
        result['AffectedConfusion']=torch.bincount(labels[nodes]*nclass+changed[nodes],minlength=nclass*nclass).reshape(nclass,nclass).tolist()
        result['ASR95UpperUnderIndependentBernoulli']=upper(result['ASRHits'],result['ASRCount'])
        result['ReferenceInfiniteCount']=sum(bool(h['threshold_is_infinite']) for h in saved['Audit']['TrainingHistory'])
        row=dict(id=job['id'],verified=not errors,errors=errors,case=job['case'],kind=job['kind'],
            architecture=job['architecture'],seed=job['seed'],metrics=result,
            snapshot_sha256=hashlib.sha256(Path(job['snapshot']).read_bytes()).hexdigest())
        rows.append(row)
        out.write_text(json.dumps(dict(status='running',jobs=rows),indent=2,allow_nan=False),encoding='utf-8')
        print(job['id'],'verified' if not errors else 'DIFFERENCE',flush=True)
    verdict=all(r['verified'] for r in rows) and len(rows)==len(study['jobs'])
    out.write_text(json.dumps(dict(status='verified' if verdict else 'incomplete',jobs=rows,
        statistical_assumption='Binomial upper bounds assume independent Bernoulli evaluation; shared graph identities are reported separately.'),indent=2,allow_nan=False),encoding='utf-8')
    print('Verified',sum(r['verified'] for r in rows),'of',len(rows),flush=True)
    if not verdict:raise SystemExit(1)

if __name__=='__main__':main()
