"""Two empirical coordinate ranges for amplitude and effective dimension."""
import torch
from .ABP_Sparse import effective_support
from .EmpiricalEnvelope import EmpiricalEnvelopeDetector,lower_median0

class RangeEnvelopeDetector(EmpiricalEnvelopeDetector):
    def __init__(self,margin=2.,reference_nodes=None,repair='median',use_structure=True):
        super().__init__(margin,reference_nodes,repair=repair,use_structure=use_structure)
        self.score_rule='range'

    @staticmethod
    def statistics(features):
        x=features.detach().double()
        return torch.stack([x.norm(dim=1),effective_support(x)],1)

    def fit(self,features,reference_nodes):
        refs=(self.reference_nodes if self.reference_nodes is not None else reference_nodes).unique(sorted=True).to(features.device)
        if len(refs)<2:raise ValueError('At least two reference identities required')
        stat=self.statistics(features[refs]);positive=stat[:,0]>0
        if not positive.any():raise ValueError('Reference features require a nonzero row')
        self.stat_min=stat[positive].min(0).values
        self.stat_max=stat[positive].max(0).values
        self.threshold=self.margin;self.reference_max=1.;self.reference_count=len(refs)
        self.fit_nodes=refs[:0];self.calibration_nodes=refs
        self.center=stat.new_zeros(2);self.scale=stat.new_ones(2)
        self.lower=features[refs].min(0).values.detach().clone()
        self.upper=features[refs].max(0).values.detach().clone()
        self.median=lower_median0(features[refs]).detach().clone()
        return self

    def scores(self,features):
        stat=self.statistics(features)
        positive=stat[:,0]>0
        score=stat.new_zeros(len(stat))
        z=stat[positive]
        score[positive]=torch.maximum(z/self.stat_max.to(z.device),self.stat_min.to(z.device)/z).max(1).values
        return score

    def metadata(self):
        state=super().metadata()
        state.update(method='range_envelope_reference',score='range',stat_min=self.stat_min.tolist(),
            stat_max=self.stat_max.tolist(),calibration='full_reference_coordinate_extrema',split_seed=None)
        return state

    @classmethod
    def from_metadata(cls,state,device='cpu'):
        obj=cls(state['margin'],repair=state['repair'],use_structure=state['use_structure'])
        for name in ('threshold','reference_max','reference_count'):setattr(obj,name,state[name])
        for name in ('center','scale','stat_min','stat_max'):setattr(obj,name,torch.tensor(state[name],dtype=torch.float64,device=device))
        for name in ('lower','upper','median'):setattr(obj,name,torch.tensor(state[name],device=device))
        for name in ('fit_nodes','calibration_nodes'):setattr(obj,name,torch.tensor(state[name],dtype=torch.long,device=device))
        return obj
