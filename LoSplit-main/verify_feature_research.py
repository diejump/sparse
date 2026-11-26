"""Reload study outputs and independently recount the requested five metrics."""
import argparse
import hashlib
import json
import math
from pathlib import Path

import torch

from models.ABP_Reference import load_reference_bundle,log_output
from models.ABP_Local import graph_views
from reference_experiment import observed_inputs,load_evaluation_generator


def verify(path):
    study=json.loads(path.read_text(encoding='utf-8'))
    if any(j['status']!='completed' for j in study['jobs']):raise ValueError('Study has unfinished jobs')
    rows=[]
    for job in study['jobs']:
        snapshot=torch.load(job['snapshot'],map_location='cpu')
        metadata=json.loads(Path(job['output']+'.meta.json').read_text(encoding='utf-8'))
        model=load_reference_bundle(job['output'],'cpu')
        x,e,w,y,anchors,val,ce,cw=observed_inputs(snapshot)
        with torch.no_grad():
            generator=load_evaluation_generator(snapshot,torch.device('cpu'))
            tx,te,tw=generator.inject_trigger(snapshot['idx_atk'],x,ce,cw,torch.device('cpu'))
            clean=model.predict(x,ce,cw);changed=model.predict(tx,te,tw)
            selected,_=model.detector.detect(x,e,w,anchors,anchors)
        labels=snapshot['original_labels'];t=int(snapshot['target_class'])
        c=snapshot['idx_clean_test'];a=snapshot['idx_atk'];a=a[labels[a]!=t]
        correct=int((clean[c]==labels[c]).sum());hits=int((changed[a]==t).sum())
        universe=set(anchors.tolist());truth=set(snapshot['idx_attach'].tolist())&universe
        selected=set(selected.tolist())&universe
        tp=len(truth&selected);fp=len(selected-truth);fn=len(truth-selected);tn=len(universe-(truth|selected))
        values=dict(CA=correct/len(c),ASR=hits/len(a),Precision=tp/(tp+fp) if tp+fp else None,
            Recall=tp/(tp+fn) if tp+fn else None,FPR=fp/(fp+tn) if fp+tn else None)
        counts=dict(CleanCorrect=correct,CleanCount=len(c),ASRHits=hits,ASRCount=len(a),TP=tp,FP=fp,FN=fn,TN=tn)
        for key,value in values.items():
            if value is None:assert metadata[key] is None,(job['id'],key)
            else:assert math.isclose(metadata[key],value,abs_tol=1e-12),(job['id'],key,metadata[key],value)
        assert all(metadata[k]==v for k,v in counts.items()),job['id']
        audit=metadata['Audit']
        assert not set(audit['RecoveryTrainingNodes'])&set(audit['ReferenceNodes']),job['id']
        assert not set(anchors.tolist())&set(val.tolist()),job['id']
        threshold_checks=[]
        if model.recovery_models:
            views=graph_views(model.detector,x,ce,cw,anchors);pe,pw=views[model.graph_policy]
            transformed=model.feature_transform(x) if model.feature_transform else x
            reference=torch.tensor(audit['ReferenceNodes']);reference=reference[y[reference]!=model.inferred_class]
            n=len(reference);alpha=metadata['Configuration']['risk'];k=math.ceil((n+1)*(1-alpha))
            for member,threshold in zip(model.recovery_models,model.thresholds):
                with torch.no_grad():z=log_output(member(transformed,pe,pw))
                other=[j for j in range(z.size(1)) if j!=model.inferred_class]
                score=z[reference,model.inferred_class]-z[reference][:,other].max(1).values
                recomputed=math.inf if k>n else max(0.,float(score.sort().values[k-1]))
                assert math.isclose(float(threshold),recomputed,abs_tol=2e-4),(job['id'],'threshold')
                threshold_checks.append(dict(n=n,k=k,finite=math.isfinite(float(threshold))))
        rows.append(dict(id=job['id'],case=job['case'],kind=job['kind'],seed=job['seed'],metrics=values,counts=counts,
            thresholds=threshold_checks,reference_labels_separated=True,
            snapshot_sha256=hashlib.sha256(Path(job['snapshot']).read_bytes()).hexdigest(),
            bundle_sha256=hashlib.sha256(Path(job['output']).read_bytes()).hexdigest()))
        if len(rows)%10==0:print(f'{path.parent.name}: {len(rows)}/{len(study["jobs"])} verified',flush=True)
    result=dict(status='verified',count=len(rows),device='cpu',rows=rows)
    path.with_name('verification.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    print(path.parent.name,len(rows),'verified',flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('studies',nargs='+',type=Path);a=p.parse_args()
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    for path in a.studies:verify(path)


if __name__=='__main__':main()
