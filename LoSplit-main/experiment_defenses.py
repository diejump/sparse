"""Shared comparison protocol for the launcher methods."""
import json
import time
import hashlib

import torch
import torch.nn.functional as F

from deterministic_runtime import seed_classifier
from rigbd_utils import sample_edges, select_candidates, split_training_nodes, recovery_loss


from run_experiments import SETTINGS, RIGBD_GRIDS

ORIGINAL_GRIDS = RIGBD_GRIDS


def add_arguments(parser):
    for name, kind in (('split_lr', float), ('split_epoch', int), ('gamma', float)):
        if '--' + name not in parser._option_string_actions:
            parser.add_argument('--' + name, type=kind, default=None)
        else:
            parser.set_defaults(**{name: None})
    parser.add_argument('--abp_sparse_multiplier', type=float, default=SETTINGS['abp_sparse_multiplier'])
    parser.add_argument('--abp_sparse_strength', type=float, default=SETTINGS['abp_sparse_strength'])
    parser.add_argument('--rigbd_grid', action='store_true')
    parser.add_argument('--rigbd_probe_self_loops', action='store_true',
                        help='Explicit self-loop probe variant; not original neighbor-only RIGBD')
    parser.add_argument('--rigbd_sampling', choices=('receiver', 'legacy_source'), default=SETTINGS['rigbd_sampling'])
    parser.add_argument('--rigbd_selection', choices=('label_boundary', 'losplit', 'fixed_count'),
                        default=SETTINGS['rigbd_selection'], help='Infer target and threshold from training labels by default')
    parser.add_argument('--rigbd_grid_selection', choices=('validation', 'min_asr'), default=SETTINGS['rigbd_grid_selection'],
                        help='min_asr uses test outcomes and is for retrospective diagnostics only')


def log_probabilities(value):
    return value[0] if isinstance(value, tuple) else value


def state_hash(model):
    digest = hashlib.sha256()
    for key, value in model.state_dict().items():
        digest.update(key.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def rigbd_selection_key(row):
    """Maximize this key to choose minimum ASR, then lower K and b."""
    return (-row['ASR'], -row['K'], -row['b'])


def rigbd_validation_key(row):
    return (row['ValidationAccuracy'], -row['K'], -row['b'])


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
                FPR=fp / (fp + tn) if fp + tn else None,
                TP=tp, FP=fp, FN=fn, TN=tn)


def run_comparison(args, data, device, snapshot, generator, construct, prune):
    """Use training anchors as the common detection population.

    Detection truth is used only for reporting under the default label-boundary
    rule. Explicit fixed-count/LoSplit reproduction modes record oracle use.
    Grid selection uses validation accuracy unless min_asr is explicitly requested.
    Seconds excludes loading/serialization and includes training and evaluation.
    DetectionSeconds separately ends at the node decision / graph filter.
    """
    method = 'rigbd' if args.defense_mode == 'none' and not args.skip_rigbd else args.defense_mode
    x, edges, weights, labels = [snapshot[k] for k in
        ('poison_x', 'poison_edge_index', 'poison_edge_weights', 'poison_labels')]
    anchors = torch.cat([snapshot['idx_train'], snapshot['idx_attach']])
    if method != 'losplit':
        anchors = anchors.unique(sorted=True)
    val = snapshot['idx_val']
    truth = snapshot['idx_attach']
    seed = args.seed if args.classifier_seed is None else args.classifier_seed
    if method == 'rigbd' and getattr(args, 'rigbd_probe_self_loops', False) and args.test_model != 'GCN':
        raise ValueError('rigbd_selfloop is an explicitly audited GCN probe variant; use --test_model=GCN')
    start = stamp(device)
    # Zero-weight stored arcs are absent edges for topology and random deletion.
    if method != 'losplit':
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
        if method == 'losplit':
            return e, w
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
            target_nodes = clean_nodes[data.y[clean_nodes] == args.target_class]
            target_ca = (float(clean[target_nodes].argmax(1).eq(data.y[target_nodes]).float().mean())
                         if target_nodes.numel() else None)
            tx, te, tw = generator.inject_trigger(snapshot['idx_atk'], x, ce, cw, device)
            te, tw = graph(te.detach(), tw.detach(), tx)
            output = forward(model, tx.detach(), te, tw)
            nodes = snapshot['idx_atk'][data.y[snapshot['idx_atk']] != args.target_class]
            if nodes.numel() == 0:
                raise ValueError('No non-target evaluation nodes')
            asr = float(output[nodes].argmax(1).eq(args.target_class).float().mean())
            validation = forward(model, x, train_e, train_w)[val]
            score = float(validation.argmax(1).eq(labels[val]).float().mean())
        return dict(CA=ca, ASR=asr, ValidationAccuracy=score, TargetClassCA=target_ca,
                    CleanCount=int(clean_nodes.numel()), ASRCount=int(nodes.numel()))

    selected = anchors[:0]
    train_e, train_w = graph(edges, weights, x)
    if method == 'losplit':
        from run_experiments import LOSPLIT_SETTINGS
        defaults = LOSPLIT_SETTINGS[snapshot['attack']][args.dataset]
        split_lr, split_epoch, gamma = [default if getattr(args, key, None) is None else getattr(args, key)
            for key, default in zip(('split_lr', 'split_epoch', 'gamma'), defaults)]
        if split_epoch < 2 or split_lr <= 0 or not 0 <= gamma <= 1:
            raise ValueError('LoSplit requires split_epoch >= 2, split_lr > 0 and gamma in [0,1]')
        seed_classifier(seed, args.cuda)
        splitter = construct(args, 'LoSplit', data, device, split_lr=split_lr).to(device)
        poison_found, clean_found, deltas, targets = splitter.fit(
            x, train_e, train_w, labels, anchors, val, train_iters=args.epochs,
            verbose=False, split_epoch=split_epoch, num_attach=len(truth))
        # Author entry: discard the initial (untrained) split, maximize the gap.
        best_delta = max(deltas[1:])
        best_t = next(i for i, delta in enumerate(deltas) if delta == best_delta)
        selected, retained = poison_found[best_t], clean_found[best_t]
        target = int(targets[best_t])
        detection_seconds = stamp(device) - start
        # Continue the RNG stream from the splitting stage, as in the author entry.
        classifier = construct(args, args.test_model, data, device).to(device)
        classifier.fit(x, train_e, train_w, labels.clone(), anchors, val,
            train_iters=args.epochs, verbose=False, finetune2=True,
            attach=selected, clean=retained, target_label=target, gamma=gamma)
        result = evaluate(classifier)
        result.update(confusion(selected, truth, anchors))
        result.update(SplitLR=split_lr, SplitEpochs=split_epoch, Gamma=gamma,
            BestSplitEpoch=best_t, BestDelta=float(best_delta), InferredClass=target,
            SelectedNodes=selected.detach().cpu().tolist(),
            RetainedNodes=retained.detach().cpu().tolist(),
            SplitHistory=[dict(Epoch=i, Delta=float(delta), InferredClass=int(targets[i]),
                               SelectedCount=int(poison_found[i].numel()))
                          for i, delta in enumerate(deltas)],
            ClassifierSeed=seed, Protocol='author_RCE_GMM_then_finetune2',
            DetectionSeconds=detection_seconds, Seconds=stamp(device) - start)
        state = classifier.state_dict()
    elif method == 'rigbd':
        probe_self_loops = getattr(args, 'rigbd_probe_self_loops', False)
        sampling = getattr(args, 'rigbd_sampling', 'receiver')
        selection = getattr(args, 'rigbd_selection', 'label_boundary')
        grid_selection = getattr(args, 'rigbd_grid_selection', 'validation')
        detector_model = new_model(probe_self_loops)
        detector_model.fit(x, edges, weights, labels, anchors, val,
                           train_iters=args.epochs, verbose=False)
        detector_model.eval()
        with torch.no_grad():
            original = forward(detector_model, x, edges, weights)[anchors].exp()
        detector_accuracy = float(original.argmax(1).eq(labels[anchors]).float().mean())
        detector_hash = state_hash(detector_model)
        shared_seconds = stamp(device) - start
        ks, bs = ORIGINAL_GRIDS[snapshot['attack']] if args.rigbd_grid else ([args.rigbd_k], [args.rigbd_b])
        candidates, best, best_state = [], None, None
        for b in bs:
            for k in ks:
                tick = stamp(device)
                # Independent CPU stream: every K uses the same sampling prefix.
                sampling_generator = torch.Generator(device='cpu').manual_seed(seed)
                deviations = torch.zeros_like(original)
                with torch.no_grad():
                    for _ in range(k):
                        noisy_e, noisy_w = sample_edges(
                            edges, weights, x.size(0), b, sampling_generator,
                            restore_axis=1 if sampling == 'receiver' else 0)
                        probability = forward(detector_model, x, noisy_e, noisy_w)[anchors].exp()
                        deviations += F.kl_div((probability + 1e-8).log(), original, reduction='none')
                scores = deviations.mean(-1)
                rule = selection
                if rule == 'losplit':
                    rule = 'fixed_count' if snapshot['attack'] == 'GTA' else 'known_label'
                selected, decision = select_candidates(
                    scores, labels, anchors, rule=rule,
                    target_class=args.target_class if rule == 'known_label' else None,
                    count=int(truth.numel()) if rule == 'fixed_count' else None)
                detection_seconds = shared_seconds + stamp(device) - tick
                classifier = new_model()
                retained, selected_training = split_training_nodes(anchors, selected, device)
                optimizer = torch.optim.Adam(classifier.parameters(), lr=classifier.lr,
                                             weight_decay=classifier.weight_decay)
                for _ in range(400):
                    classifier.train()
                    optimizer.zero_grad()
                    output = forward(classifier, x, edges, weights)
                    loss = recovery_loss(output, labels, retained, selected_training)
                    loss.backward()
                    optimizer.step()
                classifier.eval()
                row = dict(K=k, b=b, DetectionSeconds=detection_seconds, **evaluate(classifier))
                row.update(confusion(selected, truth, anchors))
                row['FPR'] = row['FP'] / (row['FP'] + row['TN']) if row['FP'] + row['TN'] else None
                row.update(Decision=decision, SelectedNodes=selected.tolist(),
                           SelectedCount=int(selected.numel()), ClassifierStateSHA256=state_hash(classifier))
                row['Seconds'] = shared_seconds + stamp(device) - tick
                candidates.append(row)
                if state_hash(detector_model) != detector_hash:
                    raise RuntimeError('RIGBD detector changed during sampling or recovery')
                # Stable tie-break: lower K, then lower drop ratio.
                rank = (rigbd_validation_key(row) if grid_selection == 'validation'
                        else rigbd_selection_key(row))
                if best is None or rank > best[0]:
                    best = (rank, row)
                    best_state = {key: value.detach().cpu().clone() for key, value in classifier.state_dict().items()}
        result = dict(best[1], GridSeconds=stamp(device) - start,
                      SelectionRule=('fixed_K_b' if not args.rigbd_grid else
                          'max_validation_accuracy_then_lower_K_then_lower_b' if grid_selection == 'validation'
                          else 'min_test_ASR_then_lower_K_then_lower_b'),
                      DetectionRule=selection, GridResults=candidates,
                      ProbeSelfLoops=probe_self_loops, SamplingRule=sampling,
                      DetectorTrainingAccuracy=detector_accuracy,
                      DetectorStateSHA256=detector_hash, DetectorUnchanged=True,
                      RecoveryObjective='losplit_mean_NLL_plus_mean_selected_label_probability',
                      ClassifierSeed=seed, SamplingSeed=seed,
                      Protocol='selfloop_probe_variant' if probe_self_loops else 'neighbor_only_probe')
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
            result.update(Precision=None, Recall=None, FPR=None)
        result['DetectionSeconds'] = None if method == 'none' else detection_seconds
        result['Seconds'] = stamp(device) - start
        state = classifier.state_dict()
    result.update(Method='rigbd_selfloop' if method == 'rigbd' and probe_self_loops else method,
                  Dataset=args.dataset, Model=args.test_model,
                  DetectionPopulation='training_anchors', TimeScope='preprocessing_through_training_and_evaluation')
    if args.model_save_path:
        torch.save(state, args.model_save_path)
        with open(args.model_save_path + '.meta.json', 'w', encoding='utf-8') as stream:
            json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
    print('FINAL_COMPARISON_JSON ' + json.dumps(result, allow_nan=False), flush=True)
    return result
