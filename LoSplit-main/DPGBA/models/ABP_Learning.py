"""ABP双模型/教师扩展；复用作者骨架和fit，结构判据使用正权有效边。"""
from copy import deepcopy
from types import SimpleNamespace
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .ABP import filter_edges, branch_profiles
from .construct import model_construct

PUBLIC_CLASSES = {'Cora': 7, 'Citeseer': 6, 'Pubmed': 3, 'Physics': 5}


def logits(value):
    return value[0] if isinstance(value, tuple) else value


def positive_graph(edges, weights):
    if weights is None:
        weights = torch.ones(edges.size(1), dtype=torch.float, device=edges.device)
    mask = weights > 0
    return edges[:, mask], weights[mask]


class ABPLearning(nn.Module):
    def __init__(self, args, model_name, data, device, tuple_output=False):
        super().__init__()
        self.device = torch.device(device)
        self.model_name = model_name
        self.method = args.defense_mode
        self.tuple_output = tuple_output
        self.seed = args.seed
        self.base_args = SimpleNamespace(**{key: getattr(args, key) for key in
                                           ('dataset', 'hidden', 'dropout', 'train_lr', 'weight_decay')})
        self.classes = PUBLIC_CLASSES[args.dataset]
        self.descriptor = SimpleNamespace(x=data.x, y=torch.arange(self.classes, device=device))
        self.minimum = args.abp_min_branch
        self.recovery_epochs = args.abp_recovery_epochs
        self.negative_weight = args.abp_negative_weight
        self.teacher_hidden = args.abp_teacher_hidden
        self.teacher_epochs = args.abp_teacher_epochs
        self.confidence = args.abp_confidence
        if self.minimum < 1 or self.recovery_epochs < 0 or self.negative_weight < 0:
            raise ValueError('Invalid ABP learning parameters')
        if not 0 <= self.confidence <= 1 or self.teacher_epochs < 1:
            raise ValueError('Invalid teacher parameters')
        self.primary = self._make()
        self.reviewer = None
        self.teacher = None
        self.register_buffer('anchors', torch.empty(0, dtype=torch.long, device=device))
        self.defense_metadata = {}

    def _make(self, model_name=None, hidden=None):
        args = deepcopy(self.base_args)
        if hidden is not None:
            args.hidden = hidden
        return model_construct(args, model_name or self.model_name, self.descriptor, self.device).to(self.device)

    def _reseed(self):
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if self.device.type == 'cuda':
            torch.cuda.manual_seed_all(self.seed)

    def _recover(self, x, edges, weights, labels, val, positive, positive_labels, selected, fixed_weights):
        if selected.numel() == 0 or self.recovery_epochs == 0 or float(fixed_weights.sum()) == 0:
            return {'epochs': 0}
        optimizer = torch.optim.Adam(self.reviewer.parameters(), lr=self.base_args.train_lr,
                                     weight_decay=self.base_args.weight_decay)
        best_acc, best_nll = -1., float('inf')
        best_state = deepcopy(self.reviewer.state_dict())
        for epoch in range(self.recovery_epochs):
            self.reviewer.train()
            optimizer.zero_grad()
            output = logits(self.reviewer(x, edges, weights))
            positive_loss = F.nll_loss(output[positive], positive_labels[positive])
            probability = output[selected].softmax(-1).gather(1, labels[selected, None]).flatten()
            penalty = -torch.log1p(-probability.clamp(max=1-1e-6))
            negative_loss = (penalty * fixed_weights).sum() / fixed_weights.sum().clamp(min=1e-6)
            loss = positive_loss + self.negative_weight * negative_loss
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite ABP recovery loss')
            loss.backward()
            optimizer.step()
            self.reviewer.eval()
            with torch.no_grad():
                validation = logits(self.reviewer(x, edges, weights))[val]
                accuracy = float(validation.argmax(-1).eq(labels[val]).float().mean())
                nll = float(F.nll_loss(validation, labels[val]))
            if accuracy > best_acc or (accuracy == best_acc and nll < best_nll):
                best_acc, best_nll = accuracy, nll
                best_state = deepcopy(self.reviewer.state_dict())
            if epoch % 100 == 0:
                print('ABP_RECOVERY epoch={} loss={:.6f} val_acc={:.6f}'.format(epoch, float(loss), accuracy), flush=True)
        self.reviewer.load_state_dict(best_state)
        self.reviewer.eval()
        return {'epochs': self.recovery_epochs, 'validation_accuracy': best_acc, 'validation_nll': best_nll}

    def fit(self, features, edge_index, edge_weight, labels, idx_train, idx_val=None,
            train_iters=200, verbose=False, **kwargs):
        if idx_val is None:
            raise ValueError('ABP learning needs the provided validation indices')
        self.anchors = idx_train.detach().clone()
        observed = torch.full_like(labels, -1)
        visible = torch.cat([idx_train, idx_val])
        observed[visible] = labels[visible]
        active_edges, active_weights = positive_graph(edge_index, edge_weight)
        largest, _ = branch_profiles(active_edges, features.size(0), idx_train)
        candidates = idx_train[largest[idx_train] >= self.minimum]
        trusted = idx_train[largest[idx_train] < self.minimum]
        if trusted.numel() == 0 or torch.unique(observed[trusted]).numel() != self.classes:
            raise ValueError('Retained training subset does not cover all public classes')
        edges, weights = filter_edges(active_edges, active_weights, features.size(0), idx_train)
        self.primary.fit(features, edge_index, edge_weight, observed, idx_train, idx_val,
                         train_iters=train_iters, verbose=verbose)
        self.primary.eval()
        self._reseed()
        corrected = candidates[:0]
        if self.method == 'abp_dual':
            self.reviewer = self._make()
            self.reviewer.fit(features, edges, weights, observed, trusted, idx_val,
                              train_iters=train_iters, verbose=verbose)
            self.reviewer.eval()
            with torch.no_grad():
                probability = logits(self.reviewer(features, edges, weights)).softmax(-1)
            positive, positive_labels = trusted, observed
        elif self.method == 'abp_assisted':
            self.teacher = self._make('GCN', self.teacher_hidden)
            self.teacher.fit(features, edges, weights, observed, trusted, idx_val,
                             train_iters=self.teacher_epochs, verbose=verbose)
            self.teacher.eval()
            with torch.no_grad():
                probability = logits(self.teacher(features, edges, weights)).softmax(-1)
            confidence, prediction = probability.max(-1)
            corrected = candidates[confidence[candidates] >= self.confidence]
            positive_labels = observed.clone()
            positive_labels[corrected] = prediction[corrected]
            positive = torch.cat([trusted, corrected]).sort().values
            self._reseed()
            self.reviewer = self._make()
            self.reviewer.fit(features, edges, weights, positive_labels, positive, idx_val,
                              train_iters=train_iters, verbose=verbose)
            self.reviewer.eval()
        else:
            raise ValueError(self.method)
        candidate_counts = torch.bincount(observed[candidates].detach().cpu(), minlength=self.classes).to(device=features.device, dtype=torch.float)
        trusted_counts = torch.bincount(observed[trusted].detach().cpu(), minlength=self.classes).to(device=features.device, dtype=torch.float)
        enrichment = candidate_counts / max(candidates.numel(), 1) - trusted_counts / trusted.numel()
        risk_class = int(enrichment.argmax()) if candidates.numel() else None
        selected = candidates[observed[candidates] == risk_class] if risk_class is not None else candidates
        fixed_weights = 1 - probability[selected].gather(1, observed[selected, None]).flatten()
        recovery = self._recover(features, edges, weights, observed, idx_val, positive,
                                 positive_labels, selected, fixed_weights)
        self.defense_metadata = {'method': self.method, 'candidate_nodes': candidates.cpu().tolist(),
            'trusted_nodes': trusted.cpu().tolist(), 'selected_nodes': selected.cpu().tolist(),
            'corrected_nodes': corrected.cpu().tolist(), 'inferred_risk_class': risk_class,
            'label_enrichment': enrichment.cpu().tolist(), 'minimum_branch': self.minimum,
            'topology': 'positive edge weights only', 'recovery': recovery,
            'negative_weight': self.negative_weight, 'teacher_hidden': self.teacher_hidden,
            'teacher_epochs': self.teacher_epochs, 'confidence': self.confidence,
            'training_reads_attack_target_or_test_truth': False}
        print('ABP_LEARNING_SUMMARY:', json.dumps({'method': self.method, 'candidates': candidates.numel(),
              'trusted': trusted.numel(), 'selected': selected.numel(), 'risk_class': risk_class}), flush=True)
        self.eval()
        return self

    def forward(self, x, edge_index, edge_weight=None):
        ordinary_result = self.primary(x, edge_index, edge_weight)
        ordinary = logits(ordinary_result)
        active_edges, active_weights = positive_graph(edge_index, edge_weight)
        edges, weights = filter_edges(active_edges, active_weights, x.size(0), self.anchors)
        reviewed = logits(self.reviewer(x, edges, weights))
        largest, _ = branch_profiles(active_edges, x.size(0), self.anchors)
        routing = largest >= self.minimum
        output = ordinary.clone()
        output[routing] = reviewed[routing]
        if not torch.isfinite(output).all():
            raise RuntimeError('Nonfinite ABP learning output')
        if self.tuple_output:
            return output, ordinary_result[1]
        return output

    @torch.no_grad()
    def test(self, features, edges, weights, labels, indices):
        self.eval()
        output = logits(self(features, edges, weights))
        return float(output[indices].argmax(-1).eq(labels[indices]).double().mean())

    def load_state_dict(self, state_dict, strict=True):
        if self.reviewer is None and any(key.startswith('reviewer.') for key in state_dict):
            self.reviewer = self._make()
        if self.teacher is None and any(key.startswith('teacher.') for key in state_dict):
            self.teacher = self._make('GCN', self.teacher_hidden)
        if 'anchors' in state_dict:
            self.anchors = state_dict['anchors'].to(self.device).clone()
        return super().load_state_dict(state_dict, strict=strict)
