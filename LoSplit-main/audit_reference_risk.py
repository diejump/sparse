"""Recalibrate alpha on fixed saved reference models and recount all outcomes."""
import argparse
import json
import math
from pathlib import Path
import torch
from envelope_experiment import load_envelope_bundle,extra_metrics
from reference_experiment import observed_inputs,load_evaluation_generator,evaluate_reference
from models.ABP_Local import graph_views
from models.ABP_Reference import log_output,fit_reference_threshold

@torch.no_grad()
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('study');a=p.parse_args()
    root=Path(a.study);study=json.loads((root/'study.json').read_text(encoding='utf-8'))
    out=root/'alpha_sensitivity.json'
    if out.exists():raise FileExistsError(out)
    torch.set_num_threads(1);rows=[]
    for job in study['jobs']:
        if job['case']!='range' or job['architecture']!='GCN' or job['status']!='completed':continue
        model=load_envelope_bundle(job['output'])
        bundle=torch.load(job['output'],map_location='cpu')
        s=torch.load(job['snapshot'],map_location='cpu')
        x,e,w,y,anchors,val,ce,cw=observed_inputs(s)
        view=graph_views(model.detector,x,ce,cw,anchors)
        ve,vw=view[model.graph_policy]
        outputs=[log_output(m(model.feature_transform(x),ve,vw)) for m in model.recovery_models]
        generator=load_evaluation_generator(s,'cpu')
        for alpha in (.002,.005,.01,.02,.05):
            thresholds=[fit_reference_threshold(o,y,bundle['reference_nodes'],model.inferred_class,alpha,small_sample='infinite')[0] for o in outputs]
            model.thresholds.copy_(torch.tensor(thresholds,dtype=model.thresholds.dtype))
            result,clean,changed,graph=evaluate_reference(model,s,ce,cw,generator)
            result.update(extra_metrics(model,s,clean,changed,graph,model.detector))
            rows.append(dict(id=job['id'],alpha=alpha,thresholds=[t if math.isfinite(t) else None for t in thresholds],
                infinite_count=sum(math.isinf(t) for t in thresholds),metrics=result))
        print(job['id'],'alpha audit completed',flush=True)
    out.write_text(json.dumps(rows,indent=2,allow_nan=False),encoding='utf-8')

if __name__=='__main__':main()
