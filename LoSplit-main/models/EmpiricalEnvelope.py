"""Reference-learned feature envelopes and coordinate-box projection.

No dataset name, equal-value test, class identity, or evaluation labels enter
this detector. Definitions and conditional results are in ENVELOPE_THEORY.md.
"""
import math
import numpy as np
import torch
from .ABP_Sparse import effective_support, project_graph


def lower_median0(values):
    """The same lower-median convention, with a deterministic sorting kernel."""
    return values.sort(dim=0).values[(len(values)-1)//2]


def descriptors(features):
    if features.ndim != 2 or not torch.isfinite(features).all():
        raise ValueError('Expected a finite feature matrix')
    x=features.detach().double()
    eps=torch.finfo(x.dtype).eps
    return torch.stack([x.norm(dim=1).clamp_min(eps).log(),
                        effective_support(x).clamp_min(eps).log()],1)


class EmpiricalEnvelopeDetector:
    def __init__(self,margin=1.0,reference_nodes=None,split_seed=20261126,
                 repair='box',score='joint',use_structure=True):
        if not math.isfinite(margin) or margin<1:raise ValueError('margin must be finite and >= 1')
        if repair not in ('box','median','none'):raise ValueError('Unknown repair')
        if score not in ('joint','radial'):raise ValueError('Unknown score')
        self.margin=margin
        self.reference_nodes=reference_nodes
        self.split_seed=split_seed
        self.repair_rule=repair
        self.score_rule=score
        self.use_structure=use_structure
        self.threshold=None

    def fit(self,features,reference_nodes):
        refs=(self.reference_nodes if self.reference_nodes is not None else reference_nodes).unique(sorted=True)
        if len(refs)<4:raise ValueError('At least four reference identities required')
        refs=refs.to(features.device)
        order=torch.as_tensor(np.random.RandomState(self.split_seed).permutation(len(refs)),device=refs.device)
        self.fit_nodes=refs[order[:len(refs)//2]]
        self.calibration_nodes=refs[order[len(refs)//2:]]
        z=descriptors(features)
        positive_fit=self.fit_nodes[features[self.fit_nodes].double().norm(dim=1)>0]
        if len(positive_fit)==0:raise ValueError('Reference fitting requires a nonzero feature row')
        self.center=lower_median0(z[positive_fit])
        mad=lower_median0((z[positive_fit]-self.center).abs())
        numerical_floor=torch.finfo(z.dtype).eps*(1+self.center.abs())
        self.scale=torch.maximum(mad,numerical_floor)
        self.reference_max=float(self.scores(features)[self.calibration_nodes].max())
        self.threshold=self.margin*self.reference_max
        self.reference_count=len(self.calibration_nodes)
        self.lower=features[refs].min(0).values.detach().clone()
        self.upper=features[refs].max(0).values.detach().clone()
        self.median=lower_median0(features[refs]).detach().clone()
        return self

    def scores(self,features):
        z=descriptors(features)
        value=(z-self.center.to(z.device)).abs()/self.scale.to(z.device)
        result=value[:,0] if self.score_rule=='radial' else value.max(1).values
        # The point mass at zero carries no directional feature information.
        # It stays separate from the positive-magnitude distribution.
        result[features.detach().double().norm(dim=1)==0]=0
        return result

    def detect(self,features,edges,weights,anchors,candidates=None):
        if self.threshold is None:raise RuntimeError('Fit the detector first')
        scores=self.scores(features)
        odd=scores>self.threshold
        if weights is None:weights=features.new_ones(edges.size(1))
        if self.use_structure:
            pe,pw,blocked=project_graph(edges,weights,len(features),anchors)
            structural=blocked[1,odd[blocked[0]]].unique(sorted=True)
        else:
            active=weights>0
            pe,pw,blocked=edges[:,active],weights[active],edges[:,:0]
            structural=anchors[:0]
        own=odd.nonzero().flatten()
        selected=torch.cat([own,structural]).unique(sorted=True)
        if candidates is not None:selected=selected[torch.isin(selected,candidates)]
        return selected,dict(projected_edges=pe,projected_weights=pw,blocked_arcs=blocked,
            support_scores=scores,structural_nodes=structural,feature_nodes=own,
            anomalous_senders=blocked[0,odd[blocked[0]]].unique(sorted=True))

    def repair(self,features):
        if self.repair_rule=='none':return features
        selected=self.scores(features)>self.threshold
        result=features.clone()
        if self.repair_rule=='box':
            result[selected]=torch.maximum(torch.minimum(features[selected],self.upper.to(features.device)),self.lower.to(features.device))
        else:
            bad=(features[selected]<self.lower.to(features.device))|(features[selected]>self.upper.to(features.device))
            result[selected]=torch.where(bad,self.median.to(features.device),features[selected])
        return result

    def metadata(self):
        return dict(method='empirical_envelope_reference',margin=self.margin,threshold=self.threshold,
            reference_max=self.reference_max,reference_count=self.reference_count,
            reference_source='validation_features',score=self.score_rule,repair=self.repair_rule,
            zero_row_policy='separate_point_mass_score_zero',
            use_structure=self.use_structure,split_seed=self.split_seed,
            fit_nodes=self.fit_nodes.tolist(),calibration_nodes=self.calibration_nodes.tolist(),
            center=self.center.tolist(),scale=self.scale.tolist(),
            lower=self.lower.tolist(),upper=self.upper.tolist(),median=self.median.tolist())

    @classmethod
    def from_metadata(cls,state,device='cpu'):
        obj=cls(state['margin'],split_seed=state['split_seed'],repair=state['repair'],
                score=state['score'],use_structure=state['use_structure'])
        for name in ('threshold','reference_max','reference_count'):setattr(obj,name,state[name])
        for name in ('center','scale'):setattr(obj,name,torch.tensor(state[name],dtype=torch.float64,device=device))
        for name in ('lower','upper'):setattr(obj,name,torch.tensor(state[name],device=device))
        obj.median=torch.tensor(state.get('median',state['lower']),device=device)
        for name in ('fit_nodes','calibration_nodes'):setattr(obj,name,torch.tensor(state[name],dtype=torch.long,device=device))
        return obj
