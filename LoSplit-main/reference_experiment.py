"""Train and evaluate the reference method on a saved, paired graph snapshot."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('MPLBACKEND', 'Agg')
import argparse
from dataclasses import asdict
import hashlib
import importlib.util
import json
from pathlib import Path
import time
from types import SimpleNamespace

import torch

from experiment_defenses import confusion, stamp
from reference_training import ReferenceConfig, fit_reference_method
from models.ABP_Reference import load_reference_bundle

ROOT = Path(__file__).resolve().parent


def load_evaluation_generator(snapshot, device):
    filename = {'GTA': 'models/GTA.py', 'UGBA': 'models/backdoor_UGBA.py',
                'DPGBA': 'DPGBA/models/backdoor.py'}[snapshot['attack']]
    spec = importlib.util.spec_from_file_location('reference_eval_' + snapshot['attack'], ROOT / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    args = SimpleNamespace(trigger_size=int(snapshot['trigger_size']), thrd=float(snapshot['trigger_threshold']))
    generator = module.Backdoor(args, device)
    generator.trojan = module.GraphTrojanNet(device, snapshot['poison_x'].size(1), args.trigger_size, layernum=2).to(device)
    generator.trojan.load_state_dict(snapshot['trigger_generator_state'])
    generator.trojan.eval()
    return generator


def observed_inputs(snapshot):
    """Merge labelled identities before passing data into the fitting function."""
    x = snapshot['poison_x']
    anchors = torch.cat([snapshot['idx_train'], snapshot['idx_attach']]).unique(sorted=True)
    val = snapshot['idx_val'].unique(sorted=True)
    # A snapshot may add labelled nodes from the original validation population.
    # Assign each observed identity to exactly one partition before fitting.
    val = val[~torch.isin(val, anchors)]
    evaluated = torch.cat([snapshot['idx_clean_test'], snapshot['idx_atk']]).unique(sorted=True)
    if torch.isin(torch.cat([anchors, val]), evaluated).any():
        raise ValueError('Labelled partitions overlap evaluation nodes')
    if torch.isin(snapshot['idx_clean_test'], snapshot['idx_atk']).any():
        raise ValueError('Clean and designated evaluation populations overlap')
    labels = torch.full((len(x),), -1, dtype=torch.long, device=x.device)
    public = torch.cat([anchors, val]).unique(sorted=True)
    labels[public] = snapshot['poison_labels'][public]
    edges, weights = snapshot['poison_edge_index'], snapshot['poison_edge_weights']
    complete_edges = torch.cat([edges, snapshot['mask_edge_index']], 1)
    complete_weights = torch.cat([weights, x.new_ones(snapshot['mask_edge_index'].size(1))])
    return x, edges, weights, labels, anchors, val, complete_edges, complete_weights


def classification_counts(clean_prediction, changed_prediction, truth, clean_nodes, changed_nodes, designated_class):
    population = changed_nodes[truth[changed_nodes] != designated_class]
    if clean_nodes.numel() == 0 or population.numel() == 0:
        raise ValueError('CA and ASR require nonempty evaluation populations')
    correct = int(clean_prediction[clean_nodes].eq(truth[clean_nodes]).sum())
    hits = int(changed_prediction[population].eq(designated_class).sum())
    return dict(CA=correct / len(clean_nodes), ASR=hits / len(population),
        CleanCorrect=correct, CleanCount=len(clean_nodes), ASRHits=hits, ASRCount=len(population))


@torch.no_grad()
def evaluate_reference(model, snapshot, complete_edges, complete_weights, generator):
    x = snapshot['poison_x']
    clean = model.predict(x, complete_edges, complete_weights)
    tx, te, tw = generator.inject_trigger(snapshot['idx_atk'], x, complete_edges, complete_weights, x.device)
    changed = model.predict(tx, te, tw)
    result = classification_counts(clean, changed, snapshot['original_labels'], snapshot['idx_clean_test'],
        snapshot['idx_atk'], int(snapshot['target_class']))
    return result, clean, changed, (tx, te, tw)


def run_snapshot(snapshot_path, output_path, config, device='cpu', generator_path=None,
                 expected_dataset=None, verify_reload=True):
    snapshot_path, output_path = Path(snapshot_path), Path(output_path)
    meta_path = Path(str(output_path) + '.meta.json')
    if output_path.exists() or meta_path.exists():
        raise FileExistsError('Choose a new model output path')
    snapshot = torch.load(snapshot_path, map_location=device)
    if snapshot.get('snapshot_version') != 2:
        raise ValueError('A version-2 graph snapshot is required')
    if expected_dataset and snapshot['dataset'] != expected_dataset:
        raise ValueError('Dataset differs from saved snapshot')
    if generator_path is not None:
        state = torch.load(generator_path, map_location='cpu')
        saved = snapshot['trigger_generator_state']
        if set(state) != set(saved) or any(not torch.equal(state[k].cpu(), saved[k].cpu()) for k in saved):
            raise ValueError('Generator differs from saved snapshot')
    sha = hashlib.sha256(snapshot_path.read_bytes()).hexdigest()
    print('SNAPSHOT_SHA256: ' + sha, flush=True)
    device = torch.device(device)
    started = stamp(device)
    x, e, w, labels, anchors, val, ce, cw = observed_inputs(snapshot)
    model, bundle, metadata = fit_reference_method(x, e, w, labels, anchors, val, config, ce, cw)
    fitted = stamp(device)
    # These identities and true labels enter scoring only after fitting is complete.
    generator = load_evaluation_generator(snapshot, device)
    result, clean, changed, changed_graph = evaluate_reference(model, snapshot, ce, cw, generator)
    result.update(confusion(torch.as_tensor(metadata['SelectedNodes'], device=device), snapshot['idx_attach'], anchors))
    result.update(Method='abp_reference', Dataset=snapshot['dataset'], Model=config.architecture,
        Device=str(device), Deterministic=torch.are_deterministic_algorithms_enabled(),
        DetectionPopulation='training_anchors', Seconds=stamp(device)-started,
        FitSeconds=fitted-started, DetectionSeconds=metadata['DetectionSeconds'], Configuration=asdict(config),
        ReferenceCount=metadata['ReferenceCount'], CalibrationCount=metadata['CalibrationCount'],
        ModelCount=metadata['ModelCount'], InferredClass=metadata['InferredClass'], snapshot_sha256=sha,
        Audit=metadata)
    result['Audit']['RemovedValidationOverlap'] = snapshot['idx_val'][torch.isin(snapshot['idx_val'], anchors)].tolist()
    result['SourceSHA256'] = {name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in
        ('reference_training.py','reference_experiment.py','models/ABP_Reference.py','models/ABP_Sparse.py','models/ABP_Local.py')}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bundle, output_path)
    if verify_reload:
        reloaded = load_reference_bundle(output_path, device)
        with torch.no_grad():
            same = torch.equal(clean, reloaded.predict(x, ce, cw)) and torch.equal(changed, reloaded.predict(*changed_graph))
        if not same:
            raise RuntimeError('Reloaded bundle predictions differ')
        result['ReloadVerified'] = True
    meta_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    print('FINAL_COMPARISON_JSON ' + json.dumps(result, ensure_ascii=False, allow_nan=False), flush=True)
    return result


def add_reference_arguments(parser, prefix='--abp_reference_'):
    parser.add_argument(prefix+'multiplier', type=float, default=2.0)
    parser.add_argument(prefix+'ratio', type=float, default=.2)
    parser.add_argument(prefix+'count', type=int)
    parser.add_argument(prefix+'risk', type=float, default=.005)
    parser.add_argument(prefix+'members', type=int, default=3)
    parser.add_argument(prefix+'seed', type=int, default=20261124)
    parser.add_argument(prefix+'variant', choices=['full','no_reference','full_projection','keep_detected_labels'], default='full')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pre_train_param', required=True)
    parser.add_argument('--trigger_generator_address')
    parser.add_argument('--model_save_path', required=True)
    parser.add_argument('--dataset')
    parser.add_argument('--model', default='GCN')
    parser.add_argument('--test_model', choices=['GCN','GAT','GraphSage'], default='GCN')
    parser.add_argument('--defense_mode', choices=['abp_reference'], default='abp_reference')
    parser.add_argument('--target_class', type=int, default=0)  # Evaluator uses saved metadata.
    parser.add_argument('--device_id', type=int, default=0)
    parser.add_argument('--device', default=None)
    parser.add_argument('--seed', type=int, default=10)
    parser.add_argument('--classifier_seed', type=int)
    parser.add_argument('--hidden', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--dropout', type=float, default=.5)
    parser.add_argument('--train_lr', type=float, default=.01)
    parser.add_argument('--weight_decay', type=float, default=.0005)
    parser.add_argument('--deterministic', action='store_true')
    add_reference_arguments(parser)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(args.deterministic)
    if args.deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    cfg = ReferenceConfig(multiplier=args.abp_reference_multiplier, ratio=args.abp_reference_ratio,
        count=args.abp_reference_count, risk=args.abp_reference_risk, members=args.abp_reference_members,
        reference_seed=args.abp_reference_seed, classifier_seed=args.seed if args.classifier_seed is None else args.classifier_seed,
        epochs=args.epochs, hidden=args.hidden, dropout=args.dropout, train_lr=args.train_lr,
        weight_decay=args.weight_decay, architecture=args.test_model, variant=args.abp_reference_variant)
    cfg.validate()
    requested_device = args.device or f'cuda:{args.device_id}'
    effective_device = requested_device
    if cfg.architecture == 'GAT' and args.deterministic and torch.device(requested_device).type == 'cuda':
        try:
            torch.zeros(2, device=requested_device).scatter_reduce_(0,
                torch.tensor([0,1], device=requested_device), torch.ones(2,device=requested_device), reduce='amax')
        except RuntimeError as error:
            if 'deterministic implementation' not in str(error):
                raise
            effective_device = 'cpu'
            print('REFERENCE_DEVICE: requested=' + requested_device + '; effective=cpu; '
                  'reason=GAT grouped maximum lacks a deterministic CUDA kernel in this runtime', flush=True)
    run_snapshot(args.pre_train_param, args.model_save_path, cfg,
        device=effective_device, generator_path=args.trigger_generator_address,
        expected_dataset=args.dataset)


if __name__ == '__main__':
    main()
