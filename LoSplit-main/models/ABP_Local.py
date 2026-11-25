"""Use a separate recovery predictor only at detected receiver nodes."""
import torch
from .ABP_Sparse import project_graph


def graph_views(detector,features,edges,weights,training_nodes):
    detected,info=detector.detect(features,edges,weights,training_nodes)
    blocked=info['blocked_arcs']
    exceptional=info['support_scores'][blocked[0]]>detector.threshold*(1+1e-12)
    remove=blocked[:,exceptional]
    if weights is None:
        weights=torch.ones(edges.size(1),device=edges.device)
    positive=weights>0
    e,w=edges[:,positive],weights[positive]
    n=features.size(0)
    removed_keys=remove[0]*n+remove[1]
    keep=~torch.isin(e[0]*n+e[1],removed_keys)
    return {'detected':detected,
        'raw':(edges,weights),
        'abp':(info['projected_edges'],info['projected_weights']),
        'selective':(e[:,keep],w[keep]),
        'selected_removed_arcs':remove}


def use_recovery_at_detected(normal_output,recovery_output,detected):
    if normal_output.shape!=recovery_output.shape:
        raise ValueError('Both models must predict the same nodes and classes')
    output=normal_output.clone()
    output[detected]=recovery_output[detected]
    return output
