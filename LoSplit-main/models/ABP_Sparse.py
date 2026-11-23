"""ABP with a calibrated effective-support test on blocked incoming messages.

The detector sees only the observed graph and calibration node identities.
Its conditional guarantees are documented in research/abp_sparse_20261123.
"""
from dataclasses import dataclass
import math
import torch
from .ABP import build_tree, find_training_connections


def effective_support(x):
    """Return ||x||_1^2 / ||x||_2^2, with r(0)=0 and stable rescaling."""
    if x.ndim != 2 or not torch.isfinite(x).all():
        raise ValueError('Features must be a finite two-dimensional tensor')
    z = x.detach().double().abs()
    scale = z.amax(dim=1, keepdim=True)
    z = z / torch.where(scale > 0, scale, torch.ones_like(scale))
    denominator = z.square().sum(dim=1)
    return z.sum(dim=1).square() / torch.where(
        denominator > 0, denominator, torch.ones_like(denominator))


def project_graph(edges, weights, node_count, anchors):
    """Original ABP rule on positive-weight edges; also return blocked arcs."""
    if anchors.numel() == 0:
        raise ValueError('At least one supervised anchor is required')
    if weights is None:
        weights = torch.ones(edges.size(1), device=edges.device)
    if not torch.isfinite(weights).all() or (weights < 0).any():
        raise ValueError('ABP requires finite nonnegative edge weights')
    active = weights > 0
    edges, weights = edges[:, active], weights[active]
    tree, edge_part, pairs = build_tree(edges, node_count)
    core, parent = find_training_connections(tree, anchors.sort().values)
    keep = []
    for sender, receiver in pairs:
        if sender == receiver:
            keep.append(True)
            continue
        part = edge_part[(sender, receiver)]
        keep.append(part in parent and
                    (part in core or receiver != parent[part]))
    keep = torch.tensor(keep, dtype=torch.bool, device=edges.device)
    return edges[:, keep], weights[keep], edges[:, ~keep]


@dataclass
class SparseBranchDetector:
    """A receiver is flagged if a blocked sender exceeds the reference envelope.

    margin=2 is a fixed conservative expansion, not a fitted class threshold.
    No distribution-free precision or recall claim follows from calibration.
    """
    margin: float = 2.0
    threshold: float = None

    def fit(self, features, reference_nodes):
        if not math.isfinite(self.margin) or self.margin < 1:
            raise ValueError('margin must be finite and at least one')
        reference_nodes = reference_nodes.unique(sorted=True)
        if reference_nodes.numel() == 0:
            raise ValueError('Calibration references cannot be empty')
        self.reference_max = float(effective_support(features[reference_nodes]).max())
        self.threshold = self.margin * self.reference_max
        self.reference_count = reference_nodes.numel()
        return self

    def detect(self, features, edges, weights, anchors, candidates=None):
        if self.threshold is None:
            raise RuntimeError('Fit the calibration envelope first')
        pe, pw, blocked = project_graph(edges, weights, features.size(0), anchors)
        scores = effective_support(features)
        anomalous = scores[blocked[0]] > self.threshold * (1 + 1e-12)
        flagged = blocked[1, anomalous].unique(sorted=True)
        if candidates is not None:
            flagged = flagged[torch.isin(flagged, candidates)]
        return flagged, {'projected_edges': pe, 'projected_weights': pw,
                         'blocked_arcs': blocked, 'support_scores': scores,
                         'anomalous_senders': blocked[0, anomalous].unique(sorted=True)}

    def metadata(self):
        return {'method': 'abp_sparse', 'score': 'L1_squared_over_L2_squared',
                'margin': self.margin, 'threshold': self.threshold,
                'reference_max': self.reference_max,
                'reference_count': self.reference_count,
                'minimum_branch_size': 1,
                'calibration_is_distribution_free_guarantee': False}
