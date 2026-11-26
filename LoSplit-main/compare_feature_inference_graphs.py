"""Measure the same clean evaluation nodes before and after test-graph changes."""
import json
from pathlib import Path
import torch
from models.ABP_Reference import load_reference_bundle
from reference_experiment import observed_inputs,load_evaluation_generator

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'runs/feature_reference_20261126'


def main():
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--study',default='confirmation');cli=p.parse_args()
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    study=json.loads((OUT/cli.study/'study.json').read_text(encoding='utf-8'))
    rows=[]
    for seed in sorted({j['seed'] for j in study['jobs']}):
        for kind in ('GTA','UGBA','DPGBA','SPEAR'):
            jobs=[j for j in study['jobs'] if j['seed']==seed and j['kind']==kind and j['case'] in ('full','detection_only','none','clip_with_detection')]
            snapshot=torch.load(jobs[0]['snapshot'],map_location='cpu')
            x,e,w,y,a,v,ce,cw=observed_inputs(snapshot)
            with torch.no_grad():
                tx,te,tw=load_evaluation_generator(snapshot,torch.device('cpu')).inject_trigger(snapshot['idx_atk'],x,ce,cw,torch.device('cpu'))
            clean_nodes=snapshot['idx_clean_test'];truth=snapshot['original_labels']
            assert torch.equal(x[clean_nodes],tx[clean_nodes])
            for job in jobs:
                model=load_reference_bundle(job['output'],'cpu')
                with torch.no_grad():
                    before=model.predict(x,ce,cw);after=model.predict(tx,te,tw)
                before_count=int((before[clean_nodes]==truth[clean_nodes]).sum())
                after_count=int((after[clean_nodes]==truth[clean_nodes]).sum())
                row=dict(seed=seed,kind=kind,case=job['case'],clean_count=len(clean_nodes),
                    standard_CA=before_count/len(clean_nodes),same_modified_graph_CA=after_count/len(clean_nodes),
                    before_correct=before_count,after_correct=after_count,clean_features_unchanged=True)
                rows.append(row)
                if kind=='SPEAR':print(row,flush=True)
    filename='same_graph_clean_accuracy.json' if cli.study=='confirmation' else f'{cli.study}_same_graph_clean_accuracy.json'
    (OUT/filename).write_text(json.dumps(dict(
        purpose='secondary_CA_on_the_same_unchanged_nodes_in_the_modified_graph',rows=rows),indent=2),encoding='utf-8')


if __name__=='__main__':main()
