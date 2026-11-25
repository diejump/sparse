"""Recount saved reference studies from independently reloaded predictions."""
import argparse
import hashlib
import json
import math
from pathlib import Path

import torch

from models.ABP_Local import graph_views
from models.ABP_Reference import load_reference_bundle, log_output
from reference_experiment import observed_inputs, load_evaluation_generator


def verify_study(path):
    study=json.loads(path.read_text(encoding='utf-8'))
    if any(j['status']!='completed' for j in study['jobs']):
        raise ValueError('Every study job must finish before final verification')
    rows=[]
    for job in study['jobs']:
        model_path=Path(job['output'])
        meta=json.loads(Path(str(model_path)+'.meta.json').read_text(encoding='utf-8'))
        source=Path(job['command'][job['command'].index('--pre_train_param')+1])
        snapshot=torch.load(source,map_location='cpu')
        x,e,w,labels,anchors,val,ce,cw=observed_inputs(snapshot)
        model=load_reference_bundle(model_path,'cpu')
        with torch.no_grad():
            clean=model.predict(x,ce,cw)
            generator=load_evaluation_generator(snapshot,torch.device('cpu'))
            tx,te,tw=generator.inject_trigger(snapshot['idx_atk'],x,ce,cw,torch.device('cpu'))
            changed=model.predict(tx,te,tw)
            selected,_=model.detector.detect(x,e,w,anchors,anchors)
        truth=snapshot['original_labels']; target=int(snapshot['target_class'])
        c=snapshot['idx_clean_test']; a=snapshot['idx_atk']; a=a[truth[a]!=target]
        correct=int((clean[c]==truth[c]).sum()); hits=int((changed[a]==target).sum())
        universe=set(anchors.tolist()); positive=set(snapshot['idx_attach'].tolist())&universe
        predicted=set(selected.tolist())&universe
        tp=len(positive&predicted); fp=len(predicted-positive); fn=len(positive-predicted)
        tn=len(universe-(positive|predicted))
        ratios=dict(CA=correct/len(c),ASR=hits/len(a),Precision=tp/(tp+fp) if tp+fp else None,
                    Recall=tp/(tp+fn) if tp+fn else None,FPR=fp/(fp+tn) if fp+tn else None)
        for key,value in ratios.items():
            if value is None: assert meta[key] is None,(job['id'],key)
            else: assert math.isclose(meta[key],value,abs_tol=1e-12),(job['id'],key)
        counts=dict(CleanCorrect=correct,CleanCount=len(c),ASRHits=hits,ASRCount=len(a),TP=tp,FP=fp,FN=fn,TN=tn)
        assert all(meta[key]==value for key,value in counts.items()),job['id']
        audit=meta['Audit']
        assert not set(audit['RecoveryTrainingNodes'])&set(audit['ReferenceNodes']),job['id']
        assert not set(anchors.tolist())&set(val.tolist()),job['id']
        threshold_checks=[]
        if model.recovery_models:
            view=graph_views(model.detector,x,ce,cw,anchors)
            pe,pw=view[model.graph_policy]
            reference=torch.tensor(audit['ReferenceNodes'])
            other_nodes=reference[labels[reference]!=model.inferred_class]
            for member,recorded in zip(model.recovery_models,model.thresholds):
                with torch.no_grad(): out=log_output(member(x,pe,pw))
                others=[k for k in range(out.size(1)) if k!=model.inferred_class]
                scores=out[other_nodes,model.inferred_class]-out[other_nodes][:,others].max(1).values
                n=len(scores); alpha=meta['Configuration']['risk']
                k=n if alpha is None else math.ceil((n+1)*(1-alpha))
                expected=math.inf if k>n else max(0.,float(scores.sort().values[k-1]))
                assert math.isclose(float(recorded),expected,abs_tol=1e-7),(job['id'],'threshold')
                threshold_checks.append(dict(n=n,k=k,infinite=math.isinf(expected)))
        rows.append(dict(id=job['id'],metrics=ratios,counts=counts,thresholds=threshold_checks,
            snapshot_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            bundle_sha256=hashlib.sha256(model_path.read_bytes()).hexdigest(),
            reload_verified=True,label_partitions_disjoint=True))
        if len(rows)%10==0:print(f'{path.parent.name}: verified {len(rows)}/{len(study["jobs"])}',flush=True)
    output=dict(status='verified',count=len(rows),torch=torch.__version__,device='cpu',rows=rows)
    path.with_name('verification.json').write_text(json.dumps(output,indent=2,allow_nan=False),encoding='utf-8')
    print(f'{path.parent.name}: verified {len(rows)}',flush=True)
    return output


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('studies',nargs='+',type=Path)
    args=parser.parse_args();torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    for path in args.studies:verify_study(path)


if __name__=='__main__':main()
