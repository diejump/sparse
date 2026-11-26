"""Detection sensitivities and information-preserving representation controls."""
import argparse
import json
from pathlib import Path
import torch
from models.RangeEnvelope import RangeEnvelopeDetector
from reference_experiment import observed_inputs
from experiment_defenses import confusion

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input-root',required=True)
    p.add_argument('--output',required=True);a=p.parse_args();torch.set_num_threads(1)
    out=Path(a.output)
    if out.exists():raise FileExistsError(out)
    rows=[]
    for kind in ['GTA','UGBA','DPGBA','SPEAR']:
        s=torch.load(Path(a.input_root)/kind/'poison.pt',map_location='cpu')
        x,e,w,y,anchors,val,ce,cw=observed_inputs(s)
        for gamma in (1.,1.5,2.,3.,4.):
            d=RangeEnvelopeDetector(gamma,val).fit(x,val)
            rows.append(dict(setting=kind,representation='original',gamma=gamma,
                metrics=confusion(d.detect(x,e,w,anchors,anchors)[0],s['idx_attach'],anchors),
                stat_min=d.stat_min.tolist(),stat_max=d.stat_max.tolist()))
        views={'common_scaling':x*7,'row_l2_normalization':torch.nn.functional.normalize(x,p=2,dim=1)}
        # A Householder reflection is orthogonal and gives dense coordinates.
        v=torch.ones(x.shape[1],dtype=x.dtype);v/=v.norm()
        views['orthogonal_householder']=x-2*(x@v)[:,None]*v[None,:]
        for name,z in views.items():
            d=RangeEnvelopeDetector(2.,val).fit(z,val)
            rows.append(dict(setting=kind,representation=name,gamma=2.,
                metrics=confusion(d.detect(z,e,w,anchors,anchors)[0],s['idx_attach'],anchors),
                max_norm_change=float((z.double().norm(dim=1)-x.double().norm(dim=1)).abs().max())))
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(rows,indent=2),encoding='utf-8')
    for row in rows:print(row['setting'],row['representation'],row['gamma'],row['metrics'],flush=True)

if __name__=='__main__':main()
