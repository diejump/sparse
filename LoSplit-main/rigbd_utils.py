"""RIGBD graph sampling, label decisions, and LoSplit recovery objective.

PyG source_to_target messages aggregate at edge_index[1]. Candidate selection
uses observed training labels by default; evaluation truth is never required.
"""
import torch
import torch.nn.functional as F


def sample_edges(edge_index, edge_weight, num_nodes, drop_ratio, generator=None,
                 restore_axis=1, keep=None):
    if not 0 <= drop_ratio <= 1:
        raise ValueError('drop_ratio must be in [0, 1]')
    if restore_axis not in (0, 1):
        raise ValueError('restore_axis must be 0 or 1')
    if edge_weight is None:
        edge_weight = torch.ones(edge_index.size(1), device=edge_index.device)
    active = edge_weight > 0
    edge_index, edge_weight = edge_index[:, active], edge_weight[active]
    if keep is None:
        probabilities = torch.full((edge_weight.numel(),), 1 - drop_ratio,
                                   dtype=torch.float32, device='cpu')
        keep = torch.bernoulli(probabilities, generator=generator).bool().to(edge_index.device)
    else:
        keep = torch.as_tensor(keep, dtype=torch.bool, device=edge_index.device)
        if keep.numel() != edge_weight.numel():
            raise ValueError('keep must refer to the positive-weight edge list')
    sampled_e, sampled_w = edge_index[:, keep], edge_weight[keep]
    degree = edge_weight.new_zeros(num_nodes)
    degree.index_add_(0, sampled_e[restore_axis], edge_weight.new_ones(sampled_e.size(1)))
    restore = degree[edge_index[restore_axis]] == 0
    # Keep original weights. In particular, never resurrect a zero-weight edge.
    return (torch.cat([sampled_e, edge_index[:, restore]], dim=1),
            torch.cat([sampled_w, edge_weight[restore]]))


def select_candidates(scores, labels, anchors, rule='label_boundary',
                      target_class=None, count=None):
    if scores.ndim != 1 or scores.numel() != anchors.numel():
        raise ValueError('scores must contain one value per training anchor')
    if not torch.isfinite(scores).all():
        raise ValueError('RIGBD scores are not finite')
    if rule not in ('label_boundary', 'known_label', 'fixed_count'):
        raise ValueError('Unknown RIGBD selection rule: ' + rule)
    order = scores.argsort(descending=True, stable=True)
    ranked = anchors[order]
    if ranked.numel() == 0:
        return ranked, dict(Rule=rule, InferredTarget=None, Boundary=0)
    target = int(labels[ranked[0]]) if target_class is None else int(target_class)
    if rule == 'fixed_count':
        if count is None or not 0 <= count <= anchors.numel():
            raise ValueError('fixed_count requires a valid explicit count')
        return ranked[:count], dict(Rule=rule, InferredTarget=target, Boundary=int(count), UsesOracleCount=True)
    if rule == 'label_boundary':
        target = int(labels[ranked[0]])
    elif target_class is None:
        raise ValueError('known_label requires target_class')
    sequence = labels[ranked]
    boundaries = ((sequence[:-1] != target) & (sequence[1:] != target)).nonzero().flatten()
    if float(scores.max()) <= 0:
        selected, boundary, threshold = ranked[:0], 0, float(scores.max())
    elif boundaries.numel():
        boundary = int(boundaries[0])
        threshold = float(scores[order[boundary]])
        selected = ranked[scores[order] > scores[order[boundary]]]
    else:
        boundary, threshold, selected = None, None, ranked
    return selected, dict(Rule=rule, InferredTarget=target, Boundary=boundary,
                          Threshold=threshold, UsesOracleCount=False)


def split_training_nodes(idx_train, idx_selected, device):
    train = torch.as_tensor(idx_train, dtype=torch.long, device=device).flatten()
    selected = torch.as_tensor([] if idx_selected is None else idx_selected,
                               dtype=torch.long, device=device).flatten().unique(sorted=True)
    if selected.numel() and not torch.isin(selected, train).all():
        raise ValueError('RIGBD selected nodes must belong to the training population')
    return train[~torch.isin(train, selected)], selected


def recovery_loss(output, labels, retained, selected):
    """LoSplit's two mean terms, including well-defined empty-group behavior."""
    loss = output.sum() * 0
    if retained.numel():
        loss = loss + F.nll_loss(output[retained], labels[retained])
    if selected.numel():
        probabilities = F.softmax(output[selected], dim=-1)
        loss = loss + probabilities.gather(1, labels[selected, None]).mean()
    return loss
