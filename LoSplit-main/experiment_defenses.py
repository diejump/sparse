"""Shared comparison protocol for the five launcher methods."""
import json
import time

import torch
import torch.nn.functional as F

from deterministic_runtime import seed_classifier


ORIGINAL_GRIDS = {
    'GTA': ([10, 20, 25, 50], [0.3, 0.5, 0.7, 0.9]),
    'UGBA': ([25, 50, 100], [0.3, 0.5, 0.7, 0.9]),
    'DPGBA': ([2, 5, 25, 50, 100], [0.1, 0.3, 0.5, 0.7, 0.9]),
}


def add_arguments(parser):
    parser.add_argument('--abp_sparse_multiplier', type=float, default=0.95)
    parser.add_argument('--abp_sparse_strength', type=float, default=0.25)
    parser.add_argument('--rigbd_grid', action='store_true')


def log_probabilities(value):
    return value[0] if isinstance(value, tuple) else value


def rigbd_selection_key(row):
    """Maximize this key to choose minimum ASR, then lower K and b."""
    return (-row['ASR'], -row['K'], -row['b'])


def stamp(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    return time.perf_counter()


def confusion(selected, truth, universe):
    selected, truth, universe = [set(v.detach().cpu().tolist()) for v in (selected, truth, universe)]
    selected, truth = selected & universe, truth & universe
    tp, fp, fn = len(selected & truth), len(selected - truth), len(truth - selected)
    tn = len(universe - (selected | truth))
    return dict(Precision=tp / (tp + fp) if tp + fp else None,
                Recall=tp / (tp + fn) if tp + fn else None,
                TP=tp, FP=fp, FN=fn, TN=tn)


def run_comparison(args, data, device, snapshot, generator, construct, prune):
    """Use training anchors as the common detection population.

    Detection truth is used only for reporting, apart from RIGBD's original
    fixed-count ranking rule, retained here to match the supplied implementation.
    Grid selection uses minimum evaluated ASR, with lower K and b for ties.
    Seconds excludes loading/serialization and includes training and evaluation.
    DetectionSeconds separately ends at the node decision / graph filter.
    """
    method = 'rigbd' if args.defense_mode == 'none' and not args.skip_rigbd else args.defense_mode
    x, edges, weights, labels = [snapshot[k] for k in
        ('poison_x', 'poison_edge_index', 'poison_edge_weights', 'poison_labels')]
    anchors = torch.cat([snapshot['idx_train'], snapshot['idx_attach']]).unique(sorted=True)
    val = snapshot['idx_val']
    truth = snapshot['idx_attach']
    seed = args.seed if args.classifier_seed is None else args.classifier_seed
    start = stamp(device)
    # Zero-weight stored arcs are absent edges for topology and random deletion.
    active = weights > 0
    edges, weights = edges[:, active], weights[active]
    from models.ABP import filter_edges
    if method == 'abp_sparse':
        from models.ABP_Sparse import project_graph
        from models.ABP_Sparse_Training import TunableSparseBranchDetector, train_with_detected_labels

    def new_model(self_loops=True):
        seed_classifier(seed, args.cuda)
        return construct(args, args.test_model, data, device, add_self_loops=self_loops).to(device)

    def forward(model, features, e, w):
        return log_probabilities(model(features, e, w))

    def graph(e, w, features):
        active = w > 0
        e, w = e[:, active], w[active]
        if method == 'abp':
            return filter_edges(e, w, features.size(0), anchors)
        if method == 'abp_sparse':
            return project_graph(e, w, features.size(0), anchors)[:2]
        if method == 'prune':
            return prune(args, e, w, features, device, large_graph=args.dataset in ('Pubmed', 'Physics'))
        return e, w

    def evaluate(model):
        model.eval()
        with torch.no_grad():
            ce = torch.cat([edges, snapshot['mask_edge_index']], 1)
            cw = torch.cat([weights, x.new_ones(snapshot['mask_edge_index'].size(1))])
            pe, pw = graph(ce, cw, x)
            clean = forward(model, x, pe, pw)
            clean_nodes = snapshot['idx_clean_test']
            ca = float(clean[clean_nodes].argmax(1).eq(data.y[clean_nodes]).float().mean())
            tx, te, tw = generator.inject_trigger(snapshot['idx_atk'], x, ce, cw, device)
            te, tw = graph(te.detach(), tw.detach(), tx)
            output = forward(model, tx.detach(), te, tw)
            nodes = snapshot['idx_atk'][data.y[snapshot['idx_atk']] != args.target_class]
            if nodes.numel() == 0:
                raise ValueError('No non-target evaluation nodes')
            asr = float(output[nodes].argmax(1).eq(args.target_class).float().mean())
            validation = forward(model, x, train_e, train_w)[val]
            score = float(validation.argmax(1).eq(labels[val]).float().mean())
        return dict(CA=ca, ASR=asr, ValidationAccuracy=score)

    selected = anchors[:0]
    train_e, train_w = graph(edges, weights, x)
    if method == 'rigbd':
        detector_model = new_model(False)
        detector_model.fit(x, edges, weights, labels, anchors, val,
                           train_iters=args.epochs, verbose=False)
        detector_model.eval()
        with torch.no_grad():
            original = forward(detector_model, x, edges, weights)[anchors].exp()
        shared_seconds = stamp(device) - start
        ks, bs = ORIGINAL_GRIDS[snapshot['attack']] if args.rigbd_grid else ([args.rigbd_k], [args.rigbd_b])
        candidates, best, best_state = [], None, None
        for b in bs:
            for k in ks:
                tick = stamp(device)
                seed_classifier(seed, args.cuda)
                deviations = torch.zeros_like(original)
                with torch.no_grad():
                    for _ in range(k):
                        keep = torch.distributions.Bernoulli(1 - b).sample(weights.size()).bool()
                        noisy_e, noisy_w = edges[:, keep], weights[keep]
                        degree = torch.zeros(x.size(0), device=device)
                        degree.index_add_(0, noisy_e[0], x.new_ones(noisy_e.size(1)))
                        restore = degree[edges[0]] == 0
                        noisy_e = torch.cat([noisy_e, edges[:, restore]], 1)
                        noisy_w = torch.cat([noisy_w, x.new_ones(int(restore.sum()))])
                        probability = forward(detector_model, x, noisy_e, noisy_w)[anchors].exp()
                        deviations += F.kl_div((probability + 1e-8).log(), original, reduction='none')
                selected = anchors[deviations.mean(-1).argsort(descending=True)[:truth.numel()]]
                detection_seconds = shared_seconds + stamp(device) - tick
                classifier = new_model()
                # Python integer IDs ensure the original set difference compares values.
                classifier.fit(x, edges, weights, labels, anchors.tolist(), val, train_iters=400,
                               verbose=False, finetune1=True, attach=selected.tolist())
                row = dict(K=k, b=b, DetectionSeconds=detection_seconds, **evaluate(classifier))
                row.update(confusion(selected, truth, anchors))
                row['Seconds'] = shared_seconds + stamp(device) - tick
                candidates.append(row)
                # Stable tie-break: lower K, then lower drop ratio.
                rank = rigbd_selection_key(row)
                if best is None or rank > best[0]:
                    best = (rank, row)
                    best_state = {key: value.detach().cpu().clone() for key, value in classifier.state_dict().items()}
        result = dict(best[1], GridSeconds=stamp(device) - start,
                      SelectionRule='min_ASR_then_lower_K_then_lower_b',
                      DetectionRule='original_fixed_count', GridResults=candidates)
        state = best_state
    else:
        detector = None
        if method == 'abp_sparse':
            detector = TunableSparseBranchDetector(args.abp_sparse_multiplier).fit(
                x, torch.cat([anchors, val]).unique(sorted=True))
            selected, _ = detector.detect(x, edges, weights, anchors, anchors)
        detection_seconds = stamp(device) - start
        classifier = new_model()
        if detector is not None:
            retained = anchors[~torch.isin(anchors, selected)]
            validation = val[~torch.isin(val, selected)]
            # DPGBA GCN exposes a tuple; use a forwarding adapter only for training.
            class TensorModel(torch.nn.Module):
                def __init__(self, base):
                    super().__init__()
                    self.base = base
                    self.lr, self.weight_decay = base.lr, base.weight_decay
                def forward(self, *inputs):
                    return log_probabilities(self.base(*inputs))
            train_with_detected_labels(TensorModel(classifier), x, train_e, train_w,
                labels, retained, validation, selected, args.abp_sparse_strength, args.epochs)
        else:
            classifier.fit(x, train_e, train_w, labels, anchors, val,
                           train_iters=args.epochs, verbose=False)
        result = evaluate(classifier)
        # ABP/prune are graph filters without a node detector. Their P/R are N/A.
        if detector is not None:
            result.update(confusion(selected, truth, anchors))
            result.update(Multiplier=args.abp_sparse_multiplier, Strength=args.abp_sparse_strength,
                          Detector=detector.metadata())
        else:
            result.update(Precision=None, Recall=None)
        result['DetectionSeconds'] = None if method == 'none' else detection_seconds
        result['Seconds'] = stamp(device) - start
        state = classifier.state_dict()
    result.update(Method=method, Dataset=args.dataset, Model=args.test_model,
                  DetectionPopulation='training_anchors', TimeScope='preprocessing_through_training_and_evaluation')
    if args.model_save_path:
        torch.save(state, args.model_save_path)
        with open(args.model_save_path + '.meta.json', 'w', encoding='utf-8') as stream:
            json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
    print('FINAL_COMPARISON_JSON ' + json.dumps(result, allow_nan=False), flush=True)
    return result
