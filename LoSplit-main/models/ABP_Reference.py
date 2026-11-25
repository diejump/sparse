"""A held-out class decision threshold used only for detected receivers."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .ABP_Local import graph_views


def log_output(value):
    return value[0] if isinstance(value, tuple) else value


def fit_fixed_epochs(model,features,edges,weights,labels,training_nodes,epochs=200):
    """Fit without reading any validation or reference labels."""
    if len(training_nodes)==0 or epochs<1:
        raise ValueError('Nonempty training nodes and positive epochs required')
    optimizer=torch.optim.Adam(model.parameters(),lr=model.lr,weight_decay=model.weight_decay)
    for _ in range(epochs):
        model.train();optimizer.zero_grad()
        output=log_output(model(features,edges,weights))
        loss=F.nll_loss(output[training_nodes],labels[training_nodes])
        if not torch.isfinite(loss):raise RuntimeError('Nonfinite recovery loss')
        loss.backward();optimizer.step()
    model.eval()
    return {'epochs':epochs,'checkpoint':'last fixed epoch','validation_labels_read':False}

def class_margin(log_probabilities,class_index):
    alternatives=log_probabilities.clone()
    alternatives[:,class_index]=-torch.inf
    return log_probabilities[:,class_index]-alternatives.max(1).values

def fit_reference_threshold(log_probabilities,labels,reference_nodes,class_index,risk_level=None,
                            small_sample='error'):
    reference_nodes = reference_nodes.unique(sorted=True)
    nonclass=reference_nodes[labels[reference_nodes]!=class_index]
    if len(nonclass)==0:
        raise ValueError('Reference set needs nodes outside the inferred class')
    scores=class_margin(log_probabilities,class_index)[nonclass]
    if risk_level is None:
        threshold=max(0.,float(scores.max()))
    else:
        if not 0<risk_level<1:
            raise ValueError('risk_level must be between zero and one')
        rank=math.ceil((len(nonclass)+1)*(1-risk_level))
        if rank>len(nonclass):
            if small_sample == 'infinite':
                return math.inf, len(nonclass)
            raise ValueError('Not enough reference nodes for this risk level')
        threshold=max(0.,float(scores.detach().cpu().kthvalue(rank).values))
    return threshold,len(nonclass)

def threshold_prediction(log_probabilities,class_index,threshold):
    alternatives=log_probabilities.clone()
    alternatives[:,class_index]=-torch.inf
    prediction=alternatives.argmax(1)
    prediction[class_margin(log_probabilities,class_index)>threshold]=class_index
    return prediction

def local_threshold_prediction(primary_output,recovery_output,detected,class_index,threshold):
    prediction=primary_output.argmax(1)
    recovered=threshold_prediction(recovery_output,class_index,threshold)
    prediction[detected]=recovered[detected]
    return prediction


class ABPReferencePredictor(nn.Module):
    """Selective graph cleanup plus a reference decision for detected receivers.

    Normal and recovery ensembles are trained separately with configurable size.
    """
    def __init__(self,normal_models,recovery_models,detector,training_nodes,
                 inferred_class,thresholds,graph_policy='selective',decision_rule='all'):
        super().__init__()
        if not normal_models or len(recovery_models)!=len(thresholds):
            raise ValueError('Models and reference thresholds must be provided')
        self.normal_models=nn.ModuleList(normal_models)
        self.recovery_models=nn.ModuleList(recovery_models)
        self.detector=detector
        self.inferred_class=int(inferred_class)
        self.graph_policy=graph_policy
        self.decision_rule=decision_rule
        self.register_buffer('training_nodes',training_nodes.detach().clone())
        self.register_buffer('thresholds',torch.as_tensor(thresholds,dtype=torch.float64))

    @torch.no_grad()
    def predict(self,features,edges,weights=None):
        self.eval()
        views=graph_views(self.detector,features,edges,weights,self.training_nodes)
        e,w=views[self.graph_policy]
        normal=[log_output(model(features,e,w)) for model in self.normal_models]
        ordinary=torch.stack([z.exp() for z in normal]).mean(0)
        if self.inferred_class < 0 or not self.recovery_models or self.decision_rule == 'ordinary':
            return ordinary.argmax(1)
        recovery=[log_output(model(features,e,w)) for model in self.recovery_models]
        recovered=torch.stack([z.exp() for z in recovery]).mean(0)
        accepted=torch.stack([class_margin(z,self.inferred_class)>threshold
            for z,threshold in zip(recovery,self.thresholds)]).all(0)
        recovered[:,self.inferred_class]=-torch.inf
        recovered_prediction=recovered.argmax(1)
        recovered_prediction[accepted]=self.inferred_class
        prediction=ordinary.argmax(1)
        prediction[views['detected']]=recovered_prediction[views['detected']]
        return prediction


def load_reference_bundle(path,device='cpu'):
    """Load a locally trained reference bundle with its original node identities."""
    from types import SimpleNamespace
    from .ABP_Sparse import SparseBranchDetector
    bundle=torch.load(path,map_location=device)
    if bundle.get('version') not in (1, 2):
        raise ValueError('Unsupported reference bundle version')
    config=SimpleNamespace(**bundle['configuration'])
    normal=[];recovery=[];thresholds=[]
    for member in bundle['members']:
        for key,target in [('normal',normal),('recovery',recovery)]:
            state=member[key]
            if state is None:
                continue
            features = bundle.get('nfeat', state.get('convs.0.lin.weight', torch.empty(0,0)).size(1))
            classes = bundle.get('nclass', state.get('gc2.lin.weight', torch.empty(0,0)).size(0))
            model=build_reference_model(config, bundle.get('architecture','GCN'), features, classes, device)
            model.load_state_dict(state);model.eval();target.append(model)
        if member['recovery'] is not None:
            thresholds.append(member['threshold'])
    det=bundle['detector']
    detector=SparseBranchDetector(det['margin'],det['threshold'])
    detector.reference_max=det['reference_max'];detector.reference_count=det['reference_count']
    return ABPReferencePredictor(normal,recovery,detector,bundle['training_nodes'],
        bundle['inferred_class'],thresholds,bundle.get('graph_policy','selective'),
        bundle.get('decision_rule','all')).to(device).eval()


def build_reference_model(config, architecture, nfeat, nclass, device):
    """Construct from public dimensions, including the two local GCN signatures."""
    import inspect
    from .GCN import GCN
    from .GAT import GAT
    from .SAGE import GraphSage
    classes = {'GCN': GCN, 'GAT': GAT, 'GraphSage': GraphSage}
    if architecture not in classes:
        raise ValueError('Choose GCN, GAT or GraphSage')
    cls = classes[architecture]
    kwargs = dict(nfeat=nfeat, nhid=config.hidden, nclass=nclass,
                  dropout=config.dropout, lr=config.train_lr,
                  weight_decay=config.weight_decay, device=torch.device(device))
    if 'args' in inspect.signature(cls).parameters:
        kwargs['args'] = config
    return cls(**kwargs).to(device)
