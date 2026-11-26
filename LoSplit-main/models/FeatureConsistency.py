"""Consistency and restoration for normalized binary node features."""
from dataclasses import dataclass
import math

import torch

from .ABP_Sparse import SparseBranchDetector, effective_support


def feature_consistency(features, tolerance=1e-6, repair='unique_mode'):
    """Use only each observed row; return decisions and a restored feature matrix.

    The public representation is a nonnegative normalized binary vector or zero.
    The most frequent positive value estimates the retained common feature level.
    """
    if features.ndim != 2 or not torch.isfinite(features).all():
        raise ValueError('Finite two-dimensional features are required')
    if not math.isfinite(tolerance) or tolerance <= 0:
        raise ValueError('Tolerance must be positive and finite')
    if repair not in ('mode','unique_mode','none','clip_normalize'):
        raise ValueError('Unknown feature restoration rule')
    x=features.detach().cpu()
    restored=x.clone()
    flagged=torch.zeros(len(x),dtype=torch.bool)
    scores=torch.zeros(len(x),dtype=torch.float64)
    levels=torch.zeros(len(x),dtype=x.dtype)
    retained_counts=torch.zeros(len(x),dtype=torch.long)
    inconsistent_counts=torch.zeros(len(x),dtype=torch.long)
    unique_modes=torch.zeros(len(x),dtype=torch.bool)
    for i,row in enumerate(x):
        positive=row>tolerance
        if positive.any():
            values,counts=torch.unique(row[positive],sorted=True,return_counts=True)
            level=values[counts.argmax()]
            unique_modes[i]=int((counts==counts.max()).sum())==1
            keep=(row-level).abs()<=tolerance
            keep &= positive
            residual=torch.where(keep,(row-level).abs(),row.abs())
            score=max(float(residual.max()),abs(float(row.double().sum())-1.))
            levels[i]=level
            retained_counts[i]=int(keep.sum())
        else:
            keep=torch.zeros_like(positive)
            residual=row.abs()
            score=float(residual.max())
        inconsistent_counts[i]=int((residual>tolerance).sum())
        scores[i]=score
        flagged[i]=score>tolerance
        if flagged[i] and repair in ('mode','unique_mode'):
            restored[i].zero_()
            if keep.any() and (repair=='mode' or unique_modes[i]):restored[i,keep]=1./int(keep.sum())
        elif repair=='clip_normalize':
            restored[i]=row.clamp_min(0)
            total=restored[i].sum()
            if total>0:restored[i]/=total
    return dict(nodes=flagged.nonzero().flatten().to(features.device),
        features=restored.to(features.device),scores=scores.to(features.device),
        levels=levels.to(features.device),retained_counts=retained_counts.to(features.device),
        unique_mode=unique_modes.to(features.device),
        inconsistent_counts=inconsistent_counts.to(features.device))


def restore_features(features,tolerance=1e-6,repair='unique_mode'):
    return feature_consistency(features,tolerance,repair)['features']


@dataclass
class FeatureBranchDetector(SparseBranchDetector):
    feature_tolerance: float = 1e-6
    use_structure: bool = True
    use_features: bool = True
    structural_gate: str = 'abp'

    def detect(self,features,edges,weights,anchors,candidates=None):
        if self.use_structure and self.structural_gate=='abp':
            selected,info=super().detect(features,edges,weights,anchors,None)
        else:
            if weights is None:weights=features.new_ones(edges.size(1))
            active=weights>0
            selected=anchors[:0]
            info=dict(projected_edges=edges[:,active],projected_weights=weights[active],
                blocked_arcs=edges[:,:0],support_scores=effective_support(features),anomalous_senders=anchors[:0])
            if self.use_structure:
                blocked=edges[:,active & (edges[0]!=edges[1])]
                exceptional=info['support_scores'][blocked[0]]>self.threshold*(1+1e-12)
                selected=blocked[1,exceptional].unique(sorted=True)
                info['blocked_arcs']=blocked
                info['anomalous_senders']=blocked[0,exceptional].unique(sorted=True)
        feature_nodes=(feature_consistency(features,self.feature_tolerance,'none')['nodes']
                       if self.use_features else anchors[:0])
        info['structural_nodes']=selected
        info['feature_nodes']=feature_nodes
        selected=torch.cat([selected,feature_nodes]).unique(sorted=True)
        if candidates is not None:selected=selected[torch.isin(selected,candidates)]
        return selected,info

    def metadata(self):
        return dict(super().metadata(),method='abp_feature_reference',feature_tolerance=self.feature_tolerance,
            use_structure=self.use_structure,use_features=self.use_features,
            structural_gate=self.structural_gate,
            feature_representation='nonnegative_normalized_binary_or_zero')
