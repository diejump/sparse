"""Reference fitting with observed labels and explicit node partitions."""
from dataclasses import asdict, dataclass
import math
import time
from types import SimpleNamespace

import numpy as np
import torch

from deterministic_runtime import seed_classifier
from models.ABP_Local import graph_views
from models.ABP_Reference import (ABPReferencePredictor, build_reference_model,
    fit_fixed_epochs, fit_reference_threshold, log_output)
from models.ABP_Sparse import SparseBranchDetector
from models.ABP_Sparse_Training import train_with_detected_labels


@dataclass(frozen=True)
class ReferenceConfig:
    multiplier: float = 2.0
    ratio: float = 0.2
    count: int = None
    risk: float = 0.005
    members: int = 3
    reference_seed: int = 20261124
    classifier_seed: int = 10
    epochs: int = 200
    hidden: int = 32
    dropout: float = 0.5
    train_lr: float = 0.01
    weight_decay: float = 0.0005
    architecture: str = 'GCN'
    variant: str = 'full'

    def validate(self):
        if not math.isfinite(self.multiplier) or self.multiplier < 1:
            raise ValueError('Reference multiplier must be finite and at least one')
        if not math.isfinite(self.ratio) or not 0 < self.ratio < 1:
            raise ValueError('Reference ratio must lie strictly between zero and one')
        if self.count is not None and self.count < 1:
            raise ValueError('Reference count must be positive')
        if self.risk is not None and not (math.isfinite(self.risk) and 0 < self.risk < 1):
            raise ValueError('Reference risk must lie strictly between zero and one')
        if self.members < 1 or self.epochs < 1 or self.hidden < 1:
            raise ValueError('Members, epochs and hidden dimensions must be positive')
        if not 0 <= self.dropout < 1 or not math.isfinite(self.dropout):
            raise ValueError('Dropout must be finite and in [0, 1)')
        if not math.isfinite(self.train_lr) or self.train_lr <= 0:
            raise ValueError('Learning rate must be finite and positive')
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError('Weight decay must be finite and nonnegative')
        if self.architecture not in ('GCN', 'GAT', 'GraphSage'):
            raise ValueError('Choose GCN, GAT or GraphSage')
        if self.variant not in ('full', 'no_reference', 'full_projection', 'keep_detected_labels'):
            raise ValueError('Unknown reference ablation variant')


def split_reference(kept, ratio, count, seed):
    """Uniform nested draws; size depends only on available training identities."""
    kept = kept.unique(sorted=True)
    size = count if count is not None else math.ceil(ratio * kept.numel())
    if not 1 <= size < kept.numel():
        raise ValueError(f'Reference size {size} must leave at least one of {kept.numel()} training nodes')
    order = torch.as_tensor(np.random.RandomState(seed).permutation(kept.numel()), device=kept.device)
    return kept[order[:size]], kept[order[size:]].sort().values


class TensorModel(torch.nn.Module):
    def __init__(self, base):
        super().__init__()
        self.base = base
        self.lr, self.weight_decay = base.lr, base.weight_decay

    def forward(self, *inputs):
        return log_output(self.base(*inputs))


def fit_reference_method(features, edges, weights, observed_labels, training_nodes,
                         validation_nodes, config, calibration_edges=None,
                         calibration_weights=None):
    """Fit without evaluation labels, designated class or affected-node identities.

    observed_labels contains values only at training_nodes and validation_nodes.
    Calibration may use public unlabelled graph structure from the transductive graph.
    """
    config.validate()
    device = features.device
    if device.type == 'cuda': torch.cuda.synchronize(device)
    preprocessing_started = time.perf_counter()
    anchors = training_nodes.unique(sorted=True)
    validation = validation_nodes.unique(sorted=True)
    if anchors.numel() == 0 or validation.numel() == 0:
        raise ValueError('Nonempty training and validation partitions are required')
    if torch.isin(anchors, validation).any():
        raise ValueError('Training and validation partitions must be disjoint')
    public_nodes = torch.cat([anchors, validation]).unique(sorted=True)
    public_labels = observed_labels[public_nodes]
    if (public_labels < 0).any():
        raise ValueError('Observed labels are required for the public labelled partitions')
    nclass = int(public_labels.max()) + 1
    if nclass < 2:
        raise ValueError('At least two public classes are required')
    # Materialize the allowed label view; values elsewhere are never used.
    labels = torch.full((len(features),), -1, dtype=torch.long, device=device)
    labels[public_nodes] = public_labels
    detector = SparseBranchDetector(config.multiplier).fit(features, public_nodes)
    views = graph_views(detector, features, edges, weights, anchors)
    selected = views['detected'][torch.isin(views['detected'], anchors)]
    retained = anchors[~torch.isin(anchors, selected)]
    clean_validation = validation[~torch.isin(validation, views['detected'])]
    if retained.numel() < 2 or clean_validation.numel() == 0:
        raise ValueError('Detection left too few training or validation nodes')
    inferred = int(torch.bincount(labels[selected].cpu(), minlength=nclass).argmax()) if len(selected) else -1
    heldout, recovery_training = split_reference(retained, config.ratio, config.count, config.reference_seed)
    reference = torch.cat([heldout, clean_validation]).unique(sorted=True)
    if config.variant == 'keep_detected_labels':
        normal_training = anchors
        recovery_training = torch.cat([recovery_training, selected]).unique(sorted=True)
    else:
        normal_training = retained
    graph_policy = 'abp' if config.variant == 'full_projection' else 'selective'
    e, w = views[graph_policy]
    visible = ~(torch.isin(e[0], heldout) | torch.isin(e[1], heldout))
    if calibration_edges is None:
        calibration_edges, calibration_weights = edges, weights
    calibration_view = graph_views(detector, features, calibration_edges, calibration_weights, anchors)
    calibration_e, calibration_w = calibration_view[graph_policy]
    if device.type == 'cuda': torch.cuda.synchronize(device)
    detection_seconds = time.perf_counter() - preprocessing_started
    normal_models, recovery_models, thresholds = [], [], []
    members, training_history = [], []
    for member_index in range(config.members):
        seed = config.classifier_seed + 10 * member_index
        seed_classifier(seed, device.type == 'cuda')
        normal = build_reference_model(config, config.architecture, features.size(1), nclass, device)
        normal_history = train_with_detected_labels(TensorModel(normal), features, e, w,
            labels, normal_training, clean_validation, anchors[:0], 0., config.epochs)
        normal_models.append(normal)
        recovery = None
        threshold = None
        reference_count = None
        recovery_history = None
        if inferred >= 0 and config.variant != 'no_reference':
            seed_classifier(seed, device.type == 'cuda')
            recovery = build_reference_model(config, config.architecture, features.size(1), nclass, device)
            recovery_history = fit_fixed_epochs(recovery, features, e[:, visible], w[visible],
                                                 labels, recovery_training, config.epochs)
            with torch.no_grad():
                output = log_output(recovery(features, calibration_e, calibration_w))
            threshold, reference_count = fit_reference_threshold(output, labels, reference, inferred,
                config.risk, small_sample='infinite')
            recovery_models.append(recovery)
            thresholds.append(threshold)
        members.append(dict(normal={k: v.detach().cpu().clone() for k,v in normal.state_dict().items()},
            recovery=({k: v.detach().cpu().clone() for k,v in recovery.state_dict().items()}
                      if recovery is not None else None), threshold=threshold, seed=seed))
        training_history.append(dict(seed=seed, normal=normal_history, recovery=recovery_history,
            threshold=threshold if threshold is None or math.isfinite(threshold) else None,
            threshold_is_infinite=threshold is not None and math.isinf(threshold),
            reference_nonclass_count=reference_count))
    model = ABPReferencePredictor(normal_models, recovery_models, detector, anchors, inferred,
        thresholds, graph_policy=graph_policy,
        decision_rule='ordinary' if config.variant == 'no_reference' else 'all').to(device).eval()
    bundle = dict(version=2, architecture=config.architecture, nfeat=features.size(1), nclass=nclass,
        configuration=asdict(config), members=members, detector=detector.metadata(),
        training_nodes=anchors.cpu(), inferred_class=inferred, graph_policy=graph_policy,
        decision_rule=model.decision_rule, reference_nodes=reference.cpu())
    metadata = dict(Configuration=asdict(config), InferredClass=inferred,
        DetectionSeconds=detection_seconds,
        ClassCount=nclass, ClassCountSource='observed_training_and_validation_label_vocabulary',
        ModelCount=len(normal_models)+len(recovery_models), Detector=detector.metadata(),
        SelectedNodes=selected.tolist(), TrainingNodes=anchors.tolist(),
        RetainedTrainingNodes=retained.tolist(), NormalTrainingNodes=normal_training.tolist(),
        RecoveryTrainingNodes=recovery_training.tolist(), HeldoutTrainingNodes=heldout.tolist(),
        ValidationNodes=clean_validation.tolist(), ReferenceNodes=reference.tolist(),
        ReferenceCount=len(heldout), CalibrationCount=len(reference),
        RecoveryUsesReferenceLabels=False, RecoveryUsesValidationCheckpoint=False,
        TrainingHistory=training_history,
        CalibrationGraph='public_transductive_graph' if calibration_edges is not edges else 'training_graph',
        Fallback='ordinary_ensemble_no_detected_training_nodes' if inferred < 0 else None)
    return model, bundle, metadata
