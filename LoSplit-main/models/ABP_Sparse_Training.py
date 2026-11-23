"""Threshold exploration and complementary-label training for ABP-Sparse."""
from copy import deepcopy
from dataclasses import dataclass
import math
import torch
import torch.nn.functional as F
from .ABP_Sparse import SparseBranchDetector,effective_support


@dataclass
class TunableSparseBranchDetector(SparseBranchDetector):
    """Use the same detection rule while allowing positive multipliers below 1."""
    def fit(self,features,reference_nodes):
        if not math.isfinite(self.margin) or self.margin<=0:
            raise ValueError('The threshold multiplier must be finite and positive')
        nodes=reference_nodes.unique(sorted=True)
        if nodes.numel()==0:
            raise ValueError('Reference nodes cannot be empty')
        self.reference_max=float(effective_support(features[nodes]).max())
        self.threshold=self.margin*self.reference_max
        self.reference_count=nodes.numel()
        return self


def complementary_label_loss(log_probabilities,observed_labels):
    """Stable -log(1-p_observed), calculated without probability clipping."""
    if log_probabilities.ndim!=2 or log_probabilities.size(1)<2:
        raise ValueError('At least two output classes are required')
    alternative=log_probabilities.clone()
    alternative.scatter_(1,observed_labels[:,None],-torch.inf)
    return (torch.logsumexp(log_probabilities,1)-torch.logsumexp(alternative,1)).mean()


def train_with_detected_labels(model,features,edges,weights,labels,training_nodes,
                              validation_nodes,detected_nodes,penalty_weight=0.,epochs=200):
    """Train on retained labels and lower the recorded class of detected nodes.

    With penalty_weight=0 this follows the original GCN training/update order.
    """
    if penalty_weight<0 or not math.isfinite(penalty_weight):
        raise ValueError('The penalty weight must be finite and nonnegative')
    if training_nodes.numel()==0 or validation_nodes.numel()==0:
        raise ValueError('Training and validation nodes are required')
    optimizer=torch.optim.Adam(model.parameters(),lr=model.lr,weight_decay=model.weight_decay)
    best_accuracy=0.
    best_state=None
    history=[]
    best_epoch=None
    for epoch in range(epochs):
        model.train();optimizer.zero_grad()
        output=model(features,edges,weights)
        ordinary=F.nll_loss(output[training_nodes],labels[training_nodes],reduction='none').mean()
        loss=ordinary
        penalty=output.new_zeros(())
        if penalty_weight>0 and detected_nodes.numel()>0:
            penalty=complementary_label_loss(output[detected_nodes],labels[detected_nodes])
            loss=loss+penalty_weight*penalty
        if not torch.isfinite(loss):
            raise RuntimeError('Nonfinite training loss')
        loss.backward();optimizer.step()
        model.eval()
        with torch.no_grad():
            current=model(features,edges,weights)
            accuracy=float(current[validation_nodes].argmax(1).eq(labels[validation_nodes]).double().mean())
        if accuracy>best_accuracy:
            best_accuracy=accuracy
            best_state=deepcopy(model.state_dict())
            best_epoch=epoch+1
        if epoch%20==0 or epoch==epochs-1:
            history.append(dict(epoch=epoch+1,ordinary_loss=float(ordinary.detach()),
                complementary_loss=float(penalty.detach()),validation_accuracy=accuracy))
    if best_state is None:
        raise RuntimeError('No positive-accuracy validation checkpoint was found')
    model.load_state_dict(best_state);model.eval()
    return {'best_epoch':best_epoch,'best_validation_accuracy':best_accuracy,'history':history}
